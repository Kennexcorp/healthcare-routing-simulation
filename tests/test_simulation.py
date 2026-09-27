"""Tests for the simulation loop and replication manager, using tiny configs and
a stub classifier so the tests check the loop's own behaviour rather than
OR-Tools convergence or the trained model's predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import DEPOT_COORDS, SimulationConfig
from src.data_generator import (
    BiometricGenerator,
    CTMCModel,
    NEWS2Scorer,
    SyntheticCohort,
)
from src.models import Patient, RoutingSolution, Worker
from src.routing import (
    AbstractRouter,
    BaselineRouter,
    DistanceMatrix,
    RoutingModel,
    StaticPriorityRouter,
)
from src.simulation import (
    SCENARIOS,
    ReplicationManager,
    Simulation,
    SimulationState,
)
from src.trigger import ANDGateTrigger


class StubClassifier:
    """A classifier that labels every patient high risk at a fixed probability."""

    def __init__(self, probability: float = 0.99) -> None:
        self._probability = probability

    def run_inference(self, features, threshold=None):
        """Return a constant prediction for every row."""
        return (
            np.array(["high"] * len(features)),
            np.full(len(features), self._probability),
        )


class InfeasibleRerouteRouter(AbstractRouter):
    """A router whose re-solve always reports infeasible, for testing the
    suppression-on-failure fix without needing OR-Tools to actually fail."""

    def get_initial_routes(self, patients, workers):
        routes = {worker.id: [] for worker in workers}
        self._last_solution = RoutingSolution(
            worker_route_assignment=routes,
            solver_status="STUB",
            solve_time_secs=0.0,
            is_feasible=True,
        )
        return routes

    def get_updated_routes(
        self,
        patients,
        workers,
        current_route_assignments,
        triggered_patients,
        simulation_time_mins,
    ):
        self._last_solution = RoutingSolution(
            worker_route_assignment=current_route_assignments,
            solver_status="STUB_INFEASIBLE",
            solve_time_secs=0.5,
            is_feasible=False,
        )
        return current_route_assignments


class ScriptedRerouteRouter(AbstractRouter):
    """Returns a pre-scripted, feasible set of routes on re-solve, so the
    additional-travel-mins computation can be tested against known
    coordinates instead of whatever OR-Tools happens to find."""

    def __init__(self, config: SimulationConfig, post_routes: dict[int, list[int]]):
        super().__init__(config)
        self._post_routes = post_routes

    def get_initial_routes(self, patients, workers):
        raise NotImplementedError("Not exercised by these tests")

    def get_updated_routes(
        self,
        patients,
        workers,
        current_route_assignments,
        triggered_patients,
        simulation_time_mins,
    ):
        self._last_solution = RoutingSolution(
            worker_route_assignment=self._post_routes,
            solver_status="STUB",
            solve_time_secs=0.1,
            is_feasible=True,
        )
        return self._post_routes


def make_config(**overrides) -> SimulationConfig:
    """Build a small, fast configuration."""
    defaults = {
        "n_patients": 6,
        "n_workers": 2,
        "shift_duration_mins": 120,
        "initial_solve_limit": 5,
        "reroute_solve_limit": 1,
    }
    defaults.update(overrides)
    return SimulationConfig(**defaults)


def make_cohort(config: SimulationConfig) -> SyntheticCohort:
    return SyntheticCohort(config, CTMCModel(), BiometricGenerator(), NEWS2Scorer())


def make_patient(patient_id: int = 0, **overrides) -> Patient:
    fields = {
        "id": patient_id,
        "age": 70,
        "diagnosis": "COPD",
        "coords": (5.0, 5.0),
        "personal_baseline_mean": {"spo2": 97.0, "hr": 72.0, "sbp": 125.0},
    }
    fields.update(overrides)
    return Patient(**fields)


def make_state(config: SimulationConfig, patients: list[Patient]) -> SimulationState:
    workers = [
        Worker(
            id=worker_id,
            current_location=DEPOT_COORDS,
            shift_duration_mins=config.shift_duration_mins,
        )
        for worker_id in range(config.n_workers)
    ]
    return SimulationState(
        config, patients, workers, DistanceMatrix(config.travel_speed_kmh)
    )


class TestSimulationState:
    def test_pending_routes_exclude_completed_visits(self):
        config = make_config()
        patients = [make_patient(index) for index in range(4)]
        patients[0].visited = True
        state = make_state(config, patients)
        state.worker_route_assignment = {0: [0, 1], 1: [2, 3]}
        assert state.pending_routes() == {0: [1], 1: [2, 3]}

    def test_a_visit_completes_only_once_travel_and_service_have_elapsed(self):
        config = make_config(service_duration_mins=30)
        patient = make_patient(0, coords=DEPOT_COORDS)  # no travel time
        state = make_state(config, [patient])
        state.worker_route_assignment = {0: [0], 1: []}

        state.advance_workers_to(29.0)
        assert patient.visited is False
        state.advance_workers_to(30.0)
        assert patient.visited is True

    def test_a_visit_that_would_overrun_the_shift_is_not_made(self):
        """A worker cannot abandon a patient at the end of the day, and the
        routing model enforces the same horizon."""
        config = make_config(shift_duration_mins=60, service_duration_mins=30)
        patient = make_patient(0, coords=DEPOT_COORDS)
        state = make_state(config, [patient])
        state.worker_route_assignment = {0: [0], 1: []}
        state.workers[0].current_time_mins = 40.0

        state.advance_workers_to(120.0)
        assert patient.visited is False

    def test_completing_a_visit_resets_the_wait_and_moves_the_worker(self):
        config = make_config(service_duration_mins=30)
        patient = make_patient(0, coords=(1.0, 1.0), time_since_last_visit_mins=90.0)
        state = make_state(config, [patient])
        state.worker_route_assignment = {0: [0], 1: []}

        state.advance_workers_to(120.0)
        assert patient.time_since_last_visit_mins == 0.0
        assert state.workers[0].current_location == (1.0, 1.0)
        assert state.workers[0].visits_completed == [0]
        assert state.total_travel_distance_km > 0

    def test_utilisation_is_the_share_of_fleet_time_spent_working(self):
        config = make_config(n_workers=2, shift_duration_mins=120)
        state = make_state(config, [])
        state.total_busy_mins = 60.0
        assert state.workforce_utilisation() == pytest.approx(0.25)

    def test_patients_who_never_deteriorate_have_no_response_time(self):
        config = make_config()
        state = make_state(config, [make_patient(0)])
        assert state.response_times() == ([], 0)

    def test_response_is_measured_from_the_high_risk_label(self):
        config = make_config(shift_duration_mins=480)
        patient = make_patient(0, time_first_high_risk_mins=100.0)
        state = make_state(config, [patient])
        state.arrival_time_mins[0] = 160.0
        assert state.response_times() == ([60.0], 0)

    def test_an_unreached_patient_scores_the_rest_of_the_shift(self):
        config = make_config(shift_duration_mins=480)
        patient = make_patient(0, time_first_high_risk_mins=300.0)
        state = make_state(config, [patient])
        assert state.response_times() == ([180.0], 1)

    def test_a_visit_before_deterioration_does_not_count_as_a_response(self):
        """The clinical failure the metric captures: a patient seen early, who
        then deteriorates and is never returned to, was not responded to."""
        config = make_config(shift_duration_mins=480, service_duration_mins=30)
        patient = make_patient(0, coords=DEPOT_COORDS)
        state = make_state(config, [patient])
        state.worker_route_assignment = {0: [0], 1: []}
        state.advance_workers_to(60.0)
        assert patient.visited is True

        patient.time_first_high_risk_mins = 200.0
        response_times, unreached = state.response_times()
        assert response_times == [280.0]
        assert unreached == 1


class TestSimulationConstruction:
    def test_an_unknown_scenario_is_rejected(self):
        config = make_config()
        with pytest.raises(ValueError, match="Unknown scenario"):
            Simulation(
                config=config,
                cohort=make_cohort(config),
                router=BaselineRouter(config, DistanceMatrix(config.travel_speed_kmh)),
                distance_matrix=DistanceMatrix(config.travel_speed_kmh),
                scenario="something_else",
            )

    def test_the_ai_scenario_requires_a_classifier_and_a_trigger(self):
        config = make_config()
        with pytest.raises(ValueError, match="requires both"):
            Simulation(
                config=config,
                cohort=make_cohort(config),
                router=BaselineRouter(config, DistanceMatrix(config.travel_speed_kmh)),
                distance_matrix=DistanceMatrix(config.travel_speed_kmh),
                scenario="ai_integrated",
            )


class TestScenarioRuns:
    def make_simulation(self, config: SimulationConfig, scenario: str) -> Simulation:
        distance_matrix = DistanceMatrix(config.travel_speed_kmh)
        if scenario == "unprioritised":
            router = BaselineRouter(config, distance_matrix)
        else:
            router = StaticPriorityRouter(
                config, distance_matrix, RoutingModel(config, distance_matrix)
            )
        is_ai = scenario == "ai_integrated"
        return Simulation(
            config=config,
            cohort=make_cohort(config),
            router=router,
            distance_matrix=distance_matrix,
            scenario=scenario,
            risk_classifier=StubClassifier() if is_ai else None,
            trigger=ANDGateTrigger(config) if is_ai else None,
        )

    @pytest.mark.parametrize("scenario", SCENARIOS)
    def test_every_scenario_produces_a_valid_result(self, scenario):
        config = make_config()
        result = self.make_simulation(config, scenario).run(replication_id=0, seed=1)
        assert result.scenario == scenario
        assert result.mean_response_time_high_risk >= 0
        assert 0.0 <= result.workforce_utilisation <= 1.0
        assert result.unvisited_patients <= config.n_patients

    def test_only_the_ai_scenario_records_triggers(self):
        config = make_config()
        for scenario in ("unprioritised", "static_priority"):
            result = self.make_simulation(config, scenario).run(0, 1)
            assert result.rerouting_events_per_shift == 0

    def test_the_same_seed_reproduces_a_replication(self):
        config = make_config()
        first = self.make_simulation(config, "unprioritised").run(0, 7)
        second = self.make_simulation(config, "unprioritised").run(0, 7)
        assert first.to_csv_row() == second.to_csv_row()

    def test_different_seeds_produce_different_replications(self):
        config = make_config()
        first = self.make_simulation(config, "unprioritised").run(0, 7)
        second = self.make_simulation(config, "unprioritised").run(0, 8)
        assert first.total_travel_distance_km != second.total_travel_distance_km

    def test_scenarios_sharing_a_seed_share_their_deterioration(self):
        """The precondition for paired testing: within a replication, only the
        routing strategy may differ between scenarios, not the deterioration
        trajectories."""
        config = make_config()
        trajectories = {}
        for scenario in ("unprioritised", "static_priority"):
            simulation = self.make_simulation(config, scenario)
            rng = np.random.default_rng(11)
            patients = simulation._cohort.initialise_patients(rng)
            states = []
            for _ in range(12):
                for patient in patients:
                    simulation._cohort.advance_patient(patient, rng)
                    simulation._cohort.observe_patient(patient, rng)
                states.append([p.current_clinical_state for p in patients])
            trajectories[scenario] = states
        assert trajectories["unprioritised"] == trajectories["static_priority"]


class TestReplicationManager:
    def test_all_scenarios_share_the_same_seeds(self):
        seeds = ReplicationManager(make_config()).base_seeds(5)
        assert seeds == [1, 2, 3, 4, 5]

    def test_half_width_is_undefined_below_two_observations(self):
        manager = ReplicationManager(make_config())
        assert manager.confidence_interval_half_width([10.0]) == float("inf")

    def test_half_width_shrinks_as_replications_accumulate(self):
        manager = ReplicationManager(make_config())
        few = manager.confidence_interval_half_width([10.0, 12.0, 8.0])
        many = manager.confidence_interval_half_width([10.0, 12.0, 8.0] * 10)
        assert many < few

    def test_identical_observations_need_no_further_replications(self):
        manager = ReplicationManager(make_config())
        assert manager.confidence_interval_half_width([10.0] * 5) == pytest.approx(0.0)
        assert manager.required_replications([10.0] * 5) == 5

    def test_a_noisier_metric_requires_more_replications(self):
        """The stopping rule that decides the experiment's size."""
        manager = ReplicationManager(make_config())
        steady = manager.required_replications([100.0, 101.0, 99.0, 100.0])
        volatile = manager.required_replications([100.0, 180.0, 40.0, 130.0])
        assert volatile > steady

    def test_the_required_count_never_falls_below_what_has_been_run(self):
        manager = ReplicationManager(make_config())
        assert manager.required_replications([100.0, 101.0, 99.0, 100.0]) >= 4


# A biometric history three standard deviations below baseline on SpO2, so
# AND-gate condition two holds.
_DEVIATING_HISTORY = [[97.0 - 3 * 1.5, 72.0, 125.0]] * 3


def make_triggering_patient(patient_id: int = 0, **overrides) -> Patient:
    """Build a patient who satisfies every AND-gate condition against the
    default theta/tau, given a classifier that always predicts high risk."""
    fields = {
        "id": patient_id,
        "age": 70,
        "diagnosis": "COPD",
        "coords": (5.0, 6.0),
        "personal_baseline_mean": {"spo2": 97.0, "hr": 72.0, "sbp": 125.0},
        "time_since_last_visit_mins": 90.0,
        "biometric_history": _DEVIATING_HISTORY,
    }
    fields.update(overrides)
    return Patient(**fields)


class TestEvaluateAndReroute:
    """Direct tests of Simulation._evaluate_and_reroute using a controllable
    router, covering an infeasible re-solve and one with a known effect on
    other patients' travel time."""

    def make_simulation(self, config: SimulationConfig, router: AbstractRouter):
        return Simulation(
            config=config,
            cohort=make_cohort(config),
            router=router,
            distance_matrix=DistanceMatrix(config.travel_speed_kmh),
            scenario="ai_integrated",
            risk_classifier=StubClassifier(),
            trigger=ANDGateTrigger(config),
        )

    def make_observations(self, patients: list[Patient]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "patient_id": [patient.id for patient in patients],
                "timestep": [0] * len(patients),
            }
        )

    def test_infeasible_reroute_does_not_suppress_the_triggered_patient(self):
        """The schedule never changed, so treating an infeasible re-solve as if
        it had fired would leave a deteriorating patient unprioritised for nothing."""
        config = make_config(n_patients=1, n_workers=1)
        patient = make_triggering_patient(0)
        state = make_state(config, [patient])
        state.worker_route_assignment = {0: [0]}

        simulation = self.make_simulation(config, InfeasibleRerouteRouter(config))
        simulation._evaluate_and_reroute(state, [self.make_observations([patient])])

        assert simulation._trigger.events == []
        assert simulation._trigger.is_suppressed(patient, 100.0) is False

    def test_a_reroute_that_leaves_other_routes_unchanged_costs_nothing(self):
        config = make_config(n_patients=1, n_workers=1)
        patient = make_triggering_patient(0)
        state = make_state(config, [patient])
        state.worker_route_assignment = {0: [0]}

        router = ScriptedRerouteRouter(config, post_routes={0: [0]})
        simulation = self.make_simulation(config, router)
        simulation._evaluate_and_reroute(state, [self.make_observations([patient])])

        assert simulation._trigger.events[0].additional_travel_mins == 0.0

    def test_a_reroute_that_lengthens_other_routes_records_the_extra_travel(self):
        """Reversing the visit order from [1, 2] to [2, 1] costs a known 2 km
        more, entirely attributable to the re-solve."""
        config = make_config(n_patients=3, n_workers=1)
        triggered = make_triggering_patient(0, coords=DEPOT_COORDS)
        other_one = make_patient(1, coords=(5.0, 6.0))
        other_two = make_patient(2, coords=(5.0, 8.0))
        state = make_state(config, [triggered, other_one, other_two])
        state.worker_route_assignment = {0: [1, 2]}

        router = ScriptedRerouteRouter(config, post_routes={0: [2, 1]})
        simulation = self.make_simulation(config, router)
        observations = self.make_observations([triggered, other_one, other_two])
        simulation._evaluate_and_reroute(state, [observations])

        expected_km = 2.0
        expected_mins = expected_km / config.travel_speed_kmh * 60
        recorded = simulation._trigger.events[0].additional_travel_mins
        assert recorded == pytest.approx(expected_mins, rel=1e-3)

    def test_a_reroute_that_shortens_other_routes_clamps_to_zero(self):
        """The mirror image of the lengthening case: since additional_travel_mins
        can't be negative, a saving must be clamped to zero rather than raise
        mid-replication."""
        config = make_config(n_patients=3, n_workers=1)
        triggered = make_triggering_patient(0, coords=DEPOT_COORDS)
        other_one = make_patient(1, coords=(5.0, 6.0))
        other_two = make_patient(2, coords=(5.0, 8.0))
        state = make_state(config, [triggered, other_one, other_two])
        state.worker_route_assignment = {0: [2, 1]}

        router = ScriptedRerouteRouter(config, post_routes={0: [1, 2]})
        simulation = self.make_simulation(config, router)
        observations = self.make_observations([triggered, other_one, other_two])
        simulation._evaluate_and_reroute(state, [observations])

        assert simulation._trigger.events[0].additional_travel_mins == 0.0
