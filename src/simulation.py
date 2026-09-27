"""The simulation loop and replication manager: drives one shift forward in timesteps, advancing patients, executing routes, and re-solving on the AND-gate."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

from src.config import DEPOT_COORDS, ROLLING_WINDOW_SIZE, SimulationConfig
from src.data_generator import PARAMETER_NAMES, SyntheticCohort
from src.ml_pipeline import RiskClassifier
from src.models import Patient, SimulationResult, Worker
from src.routing import (
    SECONDS_PER_MINUTE,
    AbstractRouter,
    DistanceMatrix,
    compute_churn,
)
from src.trigger import ANDGateTrigger

logger = logging.getLogger(__name__)

SCENARIOS = ("unprioritised", "static_priority", "ai_integrated")
AI_INTEGRATED = "ai_integrated"


class SimulationState:
    """Mutable state of one shift in progress: patients, workers and routes."""

    def __init__(
        self,
        config: SimulationConfig,
        patients: list[Patient],
        workers: list[Worker],
        distance_matrix: DistanceMatrix,
    ) -> None:
        """Args:
        config: Validated simulation configuration.
        patients: The cohort, at shift start.
        workers: The fleet, at shift start.
        distance_matrix: Supplies travel times between coordinates.
        """
        self._config = config
        self.patients = patients
        self.workers = workers
        self._distance_matrix = distance_matrix
        self.worker_route_assignment: dict[int, list[int]] = {
            worker.id: [] for worker in workers
        }

        self.simulation_time_mins = 0.0
        self.total_travel_distance_km = 0.0
        self.total_busy_mins = 0.0
        # Patient id mapped to the time a worker arrived, for arrivals that
        # happened at or after the patient first became high risk.
        self.arrival_time_mins: dict[int, float] = {}

    @property
    def patients_by_id(self) -> dict[int, Patient]:
        """Patients indexed by id."""
        return {patient.id: patient for patient in self.patients}

    def pending_routes(self) -> dict[int, list[int]]:
        """Routes stripped of visits already completed, since churn is only meaningful over visits still outstanding.

        Returns:
            Worker id mapped to the patient ids they have still to visit.
        """
        visited = {patient.id for patient in self.patients if patient.visited}
        return {
            worker_id: [patient_id for patient_id in route if patient_id not in visited]
            for worker_id, route in self.worker_route_assignment.items()
        }

    def advance_workers_to(self, target_time_mins: float) -> None:
        """Executes every visit a worker can complete by the target time; only visits finishing inside the shift count, matching the routing model's own horizon.

        Args:
            target_time_mins: Simulation clock to advance to.
        """
        patients_by_id = self.patients_by_id
        service_mins = float(self._config.service_duration_mins)
        shift_end_mins = float(self._config.shift_duration_mins)

        for worker in self.workers:
            route = self.worker_route_assignment.get(worker.id, [])
            while True:
                remaining = [
                    patient_id
                    for patient_id in route
                    if not patients_by_id[patient_id].visited
                ]
                if not remaining:
                    break
                patient = patients_by_id[remaining[0]]
                travel_mins, distance_km = self._journey(
                    worker.current_location, patient.coords
                )
                arrival_mins = worker.current_time_mins + travel_mins
                completion_mins = arrival_mins + service_mins
                if (
                    completion_mins > target_time_mins
                    or completion_mins > shift_end_mins
                ):
                    break

                patient.visited = True
                patient.time_since_last_visit_mins = 0.0
                if (
                    patient.time_first_high_risk_mins is not None
                    and arrival_mins >= patient.time_first_high_risk_mins
                ):
                    self.arrival_time_mins.setdefault(patient.id, arrival_mins)

                worker.current_location = patient.coords
                worker.current_time_mins = completion_mins
                worker.visits_completed.append(patient.id)
                self.total_travel_distance_km += distance_km
                self.total_busy_mins += travel_mins + service_mins

    def _journey(
        self, origin: tuple[float, float], destination: tuple[float, float]
    ) -> tuple[float, float]:
        """Travel time and distance between two points.

        Args:
            origin: Starting coordinates.
            destination: Ending coordinates.

        Returns:
            Travel time in minutes and distance in kilometres.
        """
        distance_km, travel_secs = self._distance_matrix.pair_travel(
            origin, destination
        )
        return travel_secs / SECONDS_PER_MINUTE, distance_km

    def other_patients_travel_mins(
        self, routes: dict[int, list[int]], excluded_patient_ids: set[int]
    ) -> float:
        """Total travel time across the fleet, ignoring excluded patients.

        Measures the cost a re-solve imposes on other patients: called once before and once after, with the triggered patients excluded from both.

        Args:
            routes: Worker id mapped to ordered patient ids still to visit.
            excluded_patient_ids: Patients to skip when walking each route.

        Returns:
            Summed travel time in minutes across every worker.
        """
        patients_by_id = self.patients_by_id
        total_mins = 0.0
        for worker in self.workers:
            location = worker.current_location
            for patient_id in routes.get(worker.id, []):
                if patient_id in excluded_patient_ids:
                    continue
                destination = patients_by_id[patient_id].coords
                travel_mins, _ = self._journey(location, destination)
                total_mins += travel_mins
                location = destination
        return total_mins

    def workforce_utilisation(self) -> float:
        """Proportion of available worker time spent travelling or delivering care.

        Returns:
            Utilisation between 0 and 1.
        """
        available_mins = len(self.workers) * self._config.shift_duration_mins
        return min(self.total_busy_mins / available_mins, 1.0)

    def response_times(self) -> tuple[list[float], int]:
        """Response time for every patient who reached high risk.

        Measured from first high-risk label to first worker arrival after it; a patient never reached scores the remainder of the shift, the worst attainable value.

        Returns:
            Response times in minutes, and the number of high-risk patients who
            were never reached after deteriorating.
        """
        shift_end_mins = float(self._config.shift_duration_mins)
        response_times = []
        unreached = 0
        for patient in self.patients:
            onset = patient.time_first_high_risk_mins
            if onset is None:
                continue
            arrival = self.arrival_time_mins.get(patient.id)
            if arrival is None:
                response_times.append(shift_end_mins - onset)
                unreached += 1
            else:
                response_times.append(arrival - onset)
        return response_times, unreached


class Simulation:
    """Runs one replication of one scenario."""

    def __init__(
        self,
        config: SimulationConfig,
        cohort: SyntheticCohort,
        router: AbstractRouter,
        distance_matrix: DistanceMatrix,
        scenario: str,
        risk_classifier: RiskClassifier | None = None,
        trigger: ANDGateTrigger | None = None,
    ) -> None:
        """Components are injected so a scenario is defined by what it is given
        rather than by branching inside the loop.

        Args:
            config: Validated simulation configuration.
            cohort: Supplies patients, deterioration and labelling.
            router: The routing strategy under test.
            distance_matrix: Supplies travel times.
            scenario: One of SCENARIOS, recorded on the result.
            risk_classifier: Required for the AI-integrated scenario only.
            trigger: Required for the AI-integrated scenario only.

        Raises:
            ValueError: If the scenario is unknown, or the AI-integrated
                scenario is missing its classifier or trigger.
        """
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario}")
        if scenario == AI_INTEGRATED and (risk_classifier is None or trigger is None):
            raise ValueError(
                "The ai_integrated scenario requires both a classifier and a trigger."
            )
        self._config = config
        self._cohort = cohort
        self._router = router
        self._distance_matrix = distance_matrix
        self._scenario = scenario
        self._risk_classifier = risk_classifier
        self._trigger = trigger

    def run(self, replication_id: int, seed: int) -> SimulationResult:
        """Run one complete shift.

        Args:
            replication_id: Index of this replication.
            seed: Seed shared with the other scenarios in this replication.

        Returns:
            The validated outcome record.
        """
        rng = np.random.default_rng(seed)
        patients = self._cohort.initialise_patients(rng)
        workers = [
            Worker(
                id=worker_id,
                current_location=DEPOT_COORDS,
                shift_duration_mins=self._config.shift_duration_mins,
            )
            for worker_id in range(self._config.n_workers)
        ]
        state = SimulationState(self._config, patients, workers, self._distance_matrix)

        state.worker_route_assignment = self._router.get_initial_routes(
            patients, workers
        )
        initial_solution = self._router.last_solution
        solve_times = [initial_solution.solve_time_secs] if initial_solution else []
        reroute_times: list[float] = []
        solver_feasible = initial_solution.is_feasible if initial_solution else True

        recent_observations: list[pd.DataFrame] = []
        n_timesteps = self._config.shift_duration_mins // self._config.timestep_mins

        for timestep_index in range(n_timesteps):
            state.simulation_time_mins = float(
                timestep_index * self._config.timestep_mins
            )
            observations = self._observe(state, timestep_index, rng)
            recent_observations.append(observations)
            recent_observations[:] = recent_observations[-ROLLING_WINDOW_SIZE:]

            if self._scenario == AI_INTEGRATED:
                reroute_time = self._evaluate_and_reroute(state, recent_observations)
                if reroute_time is not None:
                    reroute_times.append(reroute_time)
                    solve_times.append(reroute_time)

            state.advance_workers_to(
                state.simulation_time_mins + self._config.timestep_mins
            )
            self._accrue_time_since_visit(state)

        state.simulation_time_mins = float(self._config.shift_duration_mins)
        if self._trigger is not None:
            self._trigger.resolve_due_events(state.patients, state.simulation_time_mins)

        return self._build_result(
            state, replication_id, seed, solver_feasible, reroute_times
        )

    def _observe(
        self, state: SimulationState, timestep_index: int, rng: np.random.Generator
    ) -> pd.DataFrame:
        """Advance deterioration and record one observation per patient.

        Args:
            state: Simulation state, mutated in place.
            timestep_index: Index of the current timestep.
            rng: Seeded generator driving deterioration and measurement.

        Returns:
            One row per patient, in the shape the feature engineer expects.
        """
        records = []
        for patient in state.patients:
            if timestep_index > 0:
                self._cohort.advance_patient(patient, rng)
            observation = self._cohort.observe_patient(patient, rng)
            label = self._cohort.label_observation(observation)
            if label == "high" and patient.time_first_high_risk_mins is None:
                patient.time_first_high_risk_mins = state.simulation_time_mins
            records.append(
                {
                    "patient_id": patient.id,
                    "timestep": timestep_index,
                    **dict(zip(PARAMETER_NAMES, observation, strict=True)),
                    **{
                        f"baseline_{parameter}": patient.personal_baseline_mean[
                            parameter
                        ]
                        for parameter in PARAMETER_NAMES
                    },
                    "time_since_last_visit_mins": patient.time_since_last_visit_mins,
                    "age": patient.age,
                    "diagnosis": patient.diagnosis,
                }
            )
        return pd.DataFrame.from_records(records)

    def _evaluate_and_reroute(
        self, state: SimulationState, recent_observations: list[pd.DataFrame]
    ) -> float | None:
        """Score the cohort, evaluate the AND-gate, and re-solve if it fires.

        Only the trailing observation window is scored, since the rolling features look back three observations and scoring the full history would be quadratic for no benefit.

        Args:
            state: Simulation state, mutated in place.
            recent_observations: Trailing observation frames, oldest first.

        Returns:
            Re-solve time in seconds, or None if the gate did not fire.
        """
        window = pd.concat(recent_observations, ignore_index=True)
        predicted_labels, probabilities = self._risk_classifier.run_inference(
            window, threshold=self._trigger.theta
        )
        window["predicted_risk_label"] = predicted_labels
        window["high_risk_probability"] = probabilities
        current = window[window["timestep"] == window["timestep"].max()]

        label_of_patient = dict(
            zip(current["patient_id"], current["predicted_risk_label"], strict=True)
        )
        probability_of_patient = dict(
            zip(current["patient_id"], current["high_risk_probability"], strict=True)
        )
        triggered = self._trigger.evaluate_cohort(
            state.patients,
            label_of_patient,
            probability_of_patient,
            state.simulation_time_mins,
        )
        self._trigger.resolve_due_events(state.patients, state.simulation_time_mins)
        if not triggered:
            return None

        triggered_ids = {patient.id for patient in triggered}
        routes_before = state.pending_routes()
        travel_before = state.other_patients_travel_mins(routes_before, triggered_ids)

        state.worker_route_assignment = self._router.get_updated_routes(
            state.patients,
            state.workers,
            state.worker_route_assignment,
            triggered,
            state.simulation_time_mins,
        )
        solution = self._router.last_solution
        if solution is not None and not solution.is_feasible:
            # An infeasible re-solve leaves the schedule unchanged, so the patient must stay eligible to re-trigger; the event is not recorded.
            logger.info(
                "Re-solve at %.0f mins was infeasible; not recording an event, "
                "so the triggered patient(s) remain eligible to re-trigger",
                state.simulation_time_mins,
            )
            return solution.solve_time_secs

        routes_after = state.pending_routes()
        churn = compute_churn(routes_before, routes_after)
        travel_after = state.other_patients_travel_mins(routes_after, triggered_ids)
        # Only lengthening counts as a cost; the field itself forbids negative values.
        additional_travel_mins = max(0.0, travel_after - travel_before)
        self._trigger.record_events(
            triggered,
            probability_of_patient,
            state.simulation_time_mins,
            churn,
            additional_travel_mins,
        )
        return solution.solve_time_secs if solution else 0.0

    def _accrue_time_since_visit(self, state: SimulationState) -> None:
        """Add one timestep to the wait of every patient still unvisited.

        Args:
            state: Simulation state, mutated in place.
        """
        for patient in state.patients:
            if not patient.visited:
                patient.time_since_last_visit_mins += self._config.timestep_mins

    def _build_result(
        self,
        state: SimulationState,
        replication_id: int,
        seed: int,
        solver_feasible: bool,
        reroute_times: list[float],
    ) -> SimulationResult:
        """Assemble the validated outcome record for this replication.

        Args:
            state: Final simulation state.
            replication_id: Index of this replication.
            seed: Seed used.
            solver_feasible: Whether the initial solve produced a solution.
            reroute_times: Mid-shift re-solve times in seconds.

        Returns:
            The outcome record.
        """
        response_times, unreached_high_risk = state.response_times()
        unvisited = [patient for patient in state.patients if not patient.visited]

        return SimulationResult(
            replication_id=replication_id,
            seed=seed,
            scenario=self._scenario,
            n_patients=self._config.n_patients,
            mean_response_time_high_risk=float(np.mean(response_times))
            if response_times
            else 0.0,
            total_travel_distance_km=state.total_travel_distance_km,
            workforce_utilisation=state.workforce_utilisation(),
            rerouting_events_per_shift=len(self._trigger.events)
            if self._trigger
            else 0,
            false_trigger_rate=self._trigger.false_trigger_rate
            if self._trigger
            else 0.0,
            rerouting_churn_proportion=self._trigger.mean_rerouting_churn
            if self._trigger
            else 0.0,
            mean_additional_travel_per_false_trigger=self._trigger.mean_additional_travel_per_false_trigger
            if self._trigger
            else 0.0,
            unvisited_patients=len(unvisited),
            unreached_high_risk_patients=unreached_high_risk,
            solver_feasible=solver_feasible,
            mean_reroute_time_secs=float(np.mean(reroute_times))
            if reroute_times
            else 0.0,
            confidence_interval_half_width=0.0,
        )


class ReplicationManager:
    """Runs replications across scenarios until the precision target is met."""

    def __init__(self, config: SimulationConfig) -> None:
        """Args:
        config: Supplies the batch size, precision target and confidence level.
        """
        self._config = config

    def base_seeds(self, n_replications: int) -> list[int]:
        """Seeds for each replication, shared across all scenarios.

        Args:
            n_replications: How many seeds are needed.

        Returns:
            One seed per replication.
        """
        return list(range(1, n_replications + 1))

    def _critical_value(self, n: int) -> float:
        """Two-sided t critical value at the configured confidence level.

        Args:
            n: Number of observations.

        Returns:
            The critical value for n - 1 degrees of freedom.
        """
        return float(stats.t.ppf(1 - (1 - self._config.confidence_level) / 2, n - 1))

    def confidence_interval_half_width(self, values: list[float]) -> float:
        """Half-width of the confidence interval around a mean.

        Args:
            values: Observed values across replications.

        Returns:
            Half-width in the same units, or infinity below two observations.
        """
        if len(values) < 2:
            return float("inf")
        critical_value = self._critical_value(len(values))
        return float(critical_value * np.std(values, ddof=1) / np.sqrt(len(values)))

    def required_replications(self, values: list[float]) -> int:
        """Replications needed to reach the relative precision target, via the confidence interval half-width method (Law, 2015).

        Args:
            values: Observed values from the initial batch.

        Returns:
            The required replication count.
        """
        if len(values) < 2:
            return self._config.initial_batch_reps
        mean_value = float(np.mean(values))
        if mean_value <= 0:
            return len(values)
        critical_value = self._critical_value(len(values))
        standard_deviation = float(np.std(values, ddof=1))
        target = self._config.target_precision * mean_value
        return max(
            len(values),
            int(np.ceil((critical_value * standard_deviation / target) ** 2)),
        )
