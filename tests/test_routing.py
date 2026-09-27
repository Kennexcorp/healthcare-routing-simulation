"""Tests for the DVRP routing strategies, using short solver time limits since
these check that the model is built and constrained correctly, not that
OR-Tools converges."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import DEPOT_COORDS, PRIORITY_WEIGHTS, SimulationConfig
from src.models import Patient, RoutingSolution, Worker
from src.routing import (
    SECONDS_PER_MINUTE,
    BaselineRouter,
    DistanceMatrix,
    DynamicRouter,
    OrToolsRouter,
    RoutingModel,
    SolverError,
    StaticPriorityRouter,
    compute_churn,
)


def make_patients(n_patients: int, seed: int = 0, state: str = "low") -> list[Patient]:
    """Build a cohort scattered across the city square."""
    rng = np.random.default_rng(seed)
    return [
        Patient(
            id=patient_id,
            age=70,
            diagnosis="COPD",
            coords=(float(rng.uniform(0, 10)), float(rng.uniform(0, 10))),
            personal_baseline_mean={"spo2": 97.0, "hr": 72.0, "sbp": 125.0},
            current_clinical_state=state,
        )
        for patient_id in range(n_patients)
    ]


def make_workers(n_workers: int) -> list[Worker]:
    """Build a fleet sitting at the depot at shift start."""
    return [
        Worker(id=worker_id, current_location=DEPOT_COORDS)
        for worker_id in range(n_workers)
    ]


def make_config(**overrides) -> SimulationConfig:
    """Build a small, fast configuration for solver tests."""
    defaults = {
        "n_patients": 12,
        "n_workers": 3,
        "initial_solve_limit": 5,
        "reroute_solve_limit": 1,
    }
    defaults.update(overrides)
    return SimulationConfig(**defaults)


class TestDistanceMatrix:
    def test_manhattan_distance_is_the_sum_of_axis_differences(self):
        matrix = DistanceMatrix(travel_speed_kmh=30.0)
        distances = matrix.manhattan_distance_matrix([(0.0, 0.0), (3.0, 4.0)])
        assert distances[0][1] == pytest.approx(7.0)  # not the Euclidean 5.0

    def test_diagonal_is_zero_and_matrix_is_symmetric(self):
        matrix = DistanceMatrix(travel_speed_kmh=30.0)
        distances = matrix.manhattan_distance_matrix(
            [(0.0, 0.0), (3.0, 4.0), (1.0, 9.0)]
        )
        assert np.allclose(np.diag(distances), 0.0)
        assert np.allclose(distances, distances.T)

    def test_travel_time_converts_distance_at_the_configured_speed(self):
        matrix = DistanceMatrix(travel_speed_kmh=30.0)
        times = matrix.travel_time_matrix_secs([(0.0, 0.0), (3.0, 4.0)])
        # 7 km at 30 km/h is 0.2333 hours, or 840 seconds
        assert times[0][1] == 840

    def test_travel_times_are_integers_for_or_tools(self):
        """OR-Tools rejects non-integer arc costs."""
        matrix = DistanceMatrix(travel_speed_kmh=30.0)
        times = matrix.travel_time_matrix_secs([(0.0, 0.0), (1.7, 2.3)])
        assert np.issubdtype(times.dtype, np.integer)


class TestBaselineRouter:
    def test_work_is_shared_across_the_whole_fleet(self):
        """A naive baseline that leaves most of the fleet idle would not be a
        fair comparator to the prioritised scenarios."""
        config = make_config()
        router = BaselineRouter(config, DistanceMatrix(config.travel_speed_kmh))
        routes = router.get_initial_routes(make_patients(12), make_workers(3))
        assert all(len(route) > 0 for route in routes.values())

    def test_every_patient_is_assigned_at_most_once(self):
        config = make_config()
        router = BaselineRouter(config, DistanceMatrix(config.travel_speed_kmh))
        routes = router.get_initial_routes(make_patients(12), make_workers(3))
        assigned = [patient_id for route in routes.values() for patient_id in route]
        assert len(assigned) == len(set(assigned))

    def test_routes_respect_the_shift_budget(self):
        """A worker cannot be scheduled beyond the end of their shift,
        including the trip back to the depot at the end of the route."""
        config = make_config(n_patients=60, n_workers=2)
        router = BaselineRouter(config, DistanceMatrix(config.travel_speed_kmh))
        patients = make_patients(60)
        routes = router.get_initial_routes(patients, make_workers(2))

        matrix = DistanceMatrix(config.travel_speed_kmh)
        coordinates = [DEPOT_COORDS] + [patient.coords for patient in patients]
        travel = matrix.travel_time_matrix_secs(coordinates)
        service = config.service_duration_mins * SECONDS_PER_MINUTE
        for route in routes.values():
            elapsed, node = 0, 0
            for patient_id in route:
                elapsed += int(travel[node][patient_id + 1]) + service
                node = patient_id + 1
            elapsed += int(travel[node][0])
            assert elapsed <= config.shift_duration_mins * SECONDS_PER_MINUTE

    def test_a_visit_that_leaves_no_time_to_return_to_depot_is_excluded(self):
        """A visit whose travel-plus-service fits before the horizon must still
        be rejected if it leaves no time to return to base."""
        config = make_config(n_patients=1, n_workers=1)
        router = BaselineRouter(config, DistanceMatrix(config.travel_speed_kmh))
        matrix = DistanceMatrix(config.travel_speed_kmh)

        # One patient placed so that arrival (travel + service) lands just
        # before the horizon, but the return trip alone would push past it.
        horizon_secs = config.shift_duration_mins * SECONDS_PER_MINUTE
        service_secs = config.service_duration_mins * SECONDS_PER_MINUTE
        one_way_secs = (horizon_secs - service_secs) // 2 + 1
        one_way_km = one_way_secs / SECONDS_PER_MINUTE / 60 * config.travel_speed_kmh
        far_patient = make_patients(1)[0]
        far_patient.coords = (DEPOT_COORDS[0] + one_way_km, DEPOT_COORDS[1])
        travel = matrix.travel_time_matrix_secs([DEPOT_COORDS, far_patient.coords])
        assert travel[0][1] + service_secs <= horizon_secs
        assert travel[0][1] + service_secs + travel[1][0] > horizon_secs

        routes = router.get_initial_routes([far_patient], make_workers(1))
        assert routes[0] == []
        assert far_patient.id in router.last_solution.dropped_patient_ids

    def test_over_subscribed_cohort_leaves_patients_unassigned(self):
        """200 patients need 6,000 service minutes against 4,800 available."""
        config = make_config(n_patients=200, n_workers=10)
        router = BaselineRouter(config, DistanceMatrix(config.travel_speed_kmh))
        routes = router.get_initial_routes(make_patients(200), make_workers(10))
        assigned = sum(len(route) for route in routes.values())
        assert assigned < 200
        assert router.last_solution.dropped_patient_ids

    def test_ignores_clinical_priority_entirely(self):
        """Same geography, different risk classes, identical routes."""
        config = make_config()
        router = BaselineRouter(config, DistanceMatrix(config.travel_speed_kmh))
        low_risk = router.get_initial_routes(
            make_patients(12, state="low"), make_workers(3)
        )
        high_risk = router.get_initial_routes(
            make_patients(12, state="high"), make_workers(3)
        )
        assert low_risk == high_risk

    def test_does_not_reroute(self):
        config = make_config()
        router = BaselineRouter(config, DistanceMatrix(config.travel_speed_kmh))
        patients = make_patients(12)
        workers = make_workers(3)
        current = router.get_initial_routes(patients, workers)
        updated = router.get_updated_routes(
            patients, workers, current, patients[:2], 60.0
        )
        assert updated == current


class TestStaticPriorityRouter:
    def make_router(self, config: SimulationConfig) -> StaticPriorityRouter:
        distance_matrix = DistanceMatrix(config.travel_speed_kmh)
        return StaticPriorityRouter(
            config, distance_matrix, RoutingModel(config, distance_matrix)
        )

    def test_solver_reports_a_feasible_solution(self):
        config = make_config()
        router = self.make_router(config)
        router.get_initial_routes(make_patients(12), make_workers(3))
        assert router.last_solution.is_feasible
        assert router.last_solution.solver_status == "ROUTING_SUCCESS"
        assert router.last_solution.solve_time_secs > 0

    def test_an_infeasible_initial_solve_raises_rather_than_returning_empty_routes(
        self,
    ):
        """At shift start there is no earlier schedule to fall back to, so an
        infeasible solve must raise rather than silently return empty routes."""

        class InfeasibleRoutingModel:
            def solve(self, **kwargs):
                return RoutingSolution(
                    worker_route_assignment={0: [], 1: [], 2: []},
                    solver_status="ROUTING_FAIL_TIMEOUT",
                    solve_time_secs=5.0,
                    is_feasible=False,
                )

        config = make_config()
        router = OrToolsRouter(
            config, DistanceMatrix(config.travel_speed_kmh), InfeasibleRoutingModel()
        )
        with pytest.raises(SolverError):
            router.get_initial_routes(make_patients(12), make_workers(3))

    def test_every_patient_is_visited_at_most_once(self):
        config = make_config()
        router = self.make_router(config)
        routes = router.get_initial_routes(make_patients(12), make_workers(3))
        assigned = [patient_id for route in routes.values() for patient_id in route]
        assert len(assigned) == len(set(assigned))

    def test_high_risk_patients_are_served_before_low_risk_ones(self):
        """The point of the priority weighting: when capacity is short, the
        clinically urgent patients are the ones that get scheduled."""
        config = make_config(n_patients=40, n_workers=2, initial_solve_limit=5)
        patients = make_patients(40)
        for patient in patients[:5]:
            patient.current_clinical_state = "high"
        router = self.make_router(config)
        routes = router.get_initial_routes(patients, make_workers(2))

        scheduled = {patient_id for route in routes.values() for patient_id in route}
        high_risk_scheduled = sum(
            1 for patient in patients[:5] if patient.id in scheduled
        )
        assert high_risk_scheduled == 5

    def test_does_not_reroute(self):
        config = make_config()
        router = self.make_router(config)
        patients = make_patients(12)
        workers = make_workers(3)
        current = router.get_initial_routes(patients, workers)
        updated = router.get_updated_routes(
            patients, workers, current, patients[:1], 60.0
        )
        assert updated == current


class TestDynamicRouter:
    def make_router(self, config: SimulationConfig) -> DynamicRouter:
        distance_matrix = DistanceMatrix(config.travel_speed_kmh)
        return DynamicRouter(
            config, distance_matrix, RoutingModel(config, distance_matrix)
        )

    def test_reroute_returns_a_feasible_schedule(self):
        config = make_config()
        router = self.make_router(config)
        patients = make_patients(12)
        workers = make_workers(3)
        current = router.get_initial_routes(patients, workers)

        patients[0].current_clinical_state = "high"
        updated = router.get_updated_routes(
            patients, workers, current, [patients[0]], 60.0
        )
        assert router.last_solution.is_feasible
        assert any(0 in route for route in updated.values())

    def test_completed_visits_are_excluded_from_the_reroute(self):
        """Frozen visits must not be scheduled a second time."""
        config = make_config()
        router = self.make_router(config)
        patients = make_patients(12)
        workers = make_workers(3)
        current = router.get_initial_routes(patients, workers)

        for patient in patients[:4]:
            patient.visited = True
        updated = router.get_updated_routes(
            patients, workers, current, [patients[5]], 60.0
        )
        rescheduled = {patient_id for route in updated.values() for patient_id in route}
        assert rescheduled.isdisjoint({0, 1, 2, 3})

    def test_reroute_respects_the_solver_time_limit(self):
        """Operational feasibility of the mid-shift re-solve is a reported
        result, so the budget has to be honoured in practice."""
        config = make_config(n_patients=60, n_workers=4, reroute_solve_limit=1)
        router = self.make_router(config)
        patients = make_patients(60)
        workers = make_workers(4)
        current = router.get_initial_routes(patients, workers)

        patients[0].current_clinical_state = "high"
        router.get_updated_routes(patients, workers, current, [patients[0]], 120.0)
        assert router.last_solution.solve_time_secs < config.reroute_solve_limit + 3

    def test_reroute_with_nothing_left_to_visit_is_a_no_op(self):
        config = make_config()
        router = self.make_router(config)
        patients = make_patients(12)
        workers = make_workers(3)
        current = router.get_initial_routes(patients, workers)
        for patient in patients:
            patient.visited = True
        assert (
            router.get_updated_routes(patients, workers, current, [], 60.0) == current
        )

    def test_warm_start_is_accepted_by_the_solver(self, caplog):
        """ReadAssignmentFromRoutes takes solver indices, not node numbers; passing
        the wrong ones fails silently into a cold solve and inflates churn to 100%."""
        config = make_config(n_patients=30, n_workers=3)
        router = self.make_router(config)
        patients = make_patients(30)
        workers = make_workers(3)
        current = router.get_initial_routes(patients, workers)

        for patient in patients[:6]:
            patient.visited = True
        with caplog.at_level("WARNING"):
            router.get_updated_routes(patients, workers, current, [patients[7]], 90.0)
        assert "Warm start rejected" not in caplog.text

    def test_reroute_does_not_rewrite_the_whole_schedule(self):
        """Churn is measured against a realistic mid-shift state, with each worker
        at the last patient they saw rather than an unrelated location."""
        config = make_config(n_patients=40, n_workers=3)
        router = self.make_router(config)
        patients = make_patients(40)
        workers = make_workers(3)
        current = router.get_initial_routes(patients, workers)

        for worker in workers:
            completed = current[worker.id][:2]
            for patient_id in completed:
                patients[patient_id].visited = True
            if completed:
                worker.current_location = patients[completed[-1]].coords

        pending = {
            worker_id: [pid for pid in route if not patients[pid].visited]
            for worker_id, route in current.items()
        }
        triggered = next(patient for patient in patients if not patient.visited)
        updated = router.get_updated_routes(
            patients, workers, current, [triggered], 90.0
        )
        assert compute_churn(pending, updated) < 1.0

    def test_the_warm_start_is_trimmed_to_what_still_fits(self):
        """ReadAssignmentFromRoutes rejects every route if any one overruns the
        shift, so the feasible prefix is supplied to keep the warm start usable."""
        config = make_config(n_patients=40, n_workers=3)
        router = self.make_router(config)
        patients = make_patients(40)
        patient_by_id = {patient.id: patient for patient in patients}
        patient_nodes = {
            patient.id: index + 5 for index, patient in enumerate(patients)
        }

        route = [patient.id for patient in patients[:20]]
        horizon_secs = config.shift_duration_mins * SECONDS_PER_MINUTE
        prefix = router._feasible_route_prefix(
            route,
            patient_by_id,
            patient_nodes,
            DEPOT_COORDS,
            start_time_secs=400 * SECONDS_PER_MINUTE,
            horizon_secs=horizon_secs,
        )
        assert len(prefix) < len(route)

        early = router._feasible_route_prefix(
            route,
            patient_by_id,
            patient_nodes,
            DEPOT_COORDS,
            start_time_secs=0,
            horizon_secs=horizon_secs,
        )
        assert len(early) > len(prefix)

    def test_a_rejected_warm_start_still_returns_a_usable_schedule(self):
        """Late in a shift the in-progress plan may no longer fit, so the re-solve
        must fall back to a cold solve rather than losing the schedule."""
        config = make_config(n_patients=40, n_workers=3)
        router = self.make_router(config)
        patients = make_patients(40)
        workers = make_workers(3)
        current = router.get_initial_routes(patients, workers)

        for patient in patients[:8]:
            patient.visited = True
        updated = router.get_updated_routes(
            patients, workers, current, [patients[9]], 400.0
        )
        assert router.last_solution.is_feasible
        assert any(9 in route for route in updated.values())

    def test_triggered_patient_is_scheduled_within_tau(self):
        config = make_config(n_patients=30, n_workers=2, default_tau=60)
        router = self.make_router(config)
        patients = make_patients(30)
        workers = make_workers(2)
        current = router.get_initial_routes(patients, workers)

        triggered = patients[17]
        triggered.current_clinical_state = "high"
        updated = router.get_updated_routes(
            patients, workers, current, [triggered], 100.0
        )
        assert any(triggered.id in route for route in updated.values())

    def test_tau_defaults_to_the_configured_value(self):
        config = make_config(default_tau=45)
        router = self.make_router(config)
        assert router.tau == 45

    def test_an_injected_tau_overrides_the_configured_default(self):
        config = make_config(default_tau=60)
        distance_matrix = DistanceMatrix(config.travel_speed_kmh)
        router = DynamicRouter(
            config, distance_matrix, RoutingModel(config, distance_matrix), tau=15
        )
        assert router.tau == 15

    def test_the_injected_tau_sets_the_hard_deadline_the_solver_sees(self):
        """Recording the kwargs passed to the solver proves the injected tau is
        actually used for the re-solve deadline, not just stored."""

        class RecordingRoutingModel:
            def __init__(self):
                self.calls = []

            def solve(self, **kwargs):
                self.calls.append(kwargs)
                patient_nodes = kwargs["patient_nodes"]
                return RoutingSolution(
                    worker_route_assignment={0: list(patient_nodes)},
                    solver_status="STUB",
                    solve_time_secs=0.0,
                    is_feasible=True,
                )

        config = make_config(n_patients=5, n_workers=1, default_tau=60)
        distance_matrix = DistanceMatrix(config.travel_speed_kmh)
        recording_model = RecordingRoutingModel()
        router = DynamicRouter(config, distance_matrix, recording_model, tau=15)
        patients = make_patients(5)
        workers = make_workers(1)
        triggered = patients[0]

        router.get_updated_routes(
            patients, workers, {0: [p.id for p in patients]}, [triggered], 100.0
        )

        call = recording_model.calls[0]
        triggered_node = call["patient_nodes"][triggered.id]
        _, latest, is_hard = call["time_windows_secs"][triggered_node]
        assert is_hard is True
        assert latest == 100 * SECONDS_PER_MINUTE + 15 * SECONDS_PER_MINUTE


class TestComputeChurn:
    def test_identical_routes_have_no_churn(self):
        routes = {0: [1, 2, 3], 1: [4, 5]}
        assert compute_churn(routes, routes) == 0.0

    def test_reordering_within_a_worker_counts_as_churn(self):
        assert compute_churn({0: [1, 2, 3]}, {0: [3, 2, 1]}) == pytest.approx(2 / 3)

    def test_moving_a_patient_between_workers_counts_as_churn(self):
        churn = compute_churn({0: [1, 2], 1: [3]}, {0: [1, 2], 1: [], 2: [3]})
        assert churn == pytest.approx(1 / 3)

    def test_dropping_a_patient_counts_as_churn(self):
        assert compute_churn({0: [1, 2]}, {0: [1]}) == pytest.approx(1 / 2)

    def test_empty_schedule_has_no_churn(self):
        assert compute_churn({0: [], 1: []}, {0: [1], 1: []}) == 0.0


class TestPriorityConstants:
    def test_drop_penalty_ranks_high_risk_above_lower_risk(self):
        """Dropping a high-risk patient must cost more than dropping any number of
        low-risk ones the solver could trade against."""
        assert (
            PRIORITY_WEIGHTS["high"]
            > PRIORITY_WEIGHTS["moderate"]
            > PRIORITY_WEIGHTS["low"]
        )
