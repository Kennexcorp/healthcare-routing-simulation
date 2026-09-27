"""Dynamic vehicle routing over the patient cohort: three strategies sharing one interface, from an unprioritised nearest-neighbour heuristic to an AND-gate-driven dynamic re-solve."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from src.config import (
    DEPOT_COORDS,
    INITIAL_TIME_WINDOWS,
    PRIORITY_SCALE,
    PRIORITY_WEIGHTS,
    SimulationConfig,
)
from src.models import Patient, RoutingSolution, Worker

logger = logging.getLogger(__name__)

SECONDS_PER_MINUTE = 60

# Kept well below PRIORITY_SCALE so lateness is always preferable to dropping a patient.
SOFT_WINDOW_PENALTY_SCALE = 1_000

# OR-Tools reports solve outcomes as integers; these are the names it documents.
SOLVER_STATUS_NAMES = {
    0: "ROUTING_NOT_SOLVED",
    1: "ROUTING_SUCCESS",
    2: "ROUTING_FAIL",
    3: "ROUTING_FAIL_TIMEOUT",
    4: "ROUTING_INVALID",
}


class SolverError(Exception):
    """Raised when the routing model cannot be constructed."""


def _depot_prefixed_node_map(
    patients: list[Patient],
) -> tuple[list[tuple[float, float]], dict[int, int]]:
    """Coordinates and node indices for a solve with the depot at node 0.

    Args:
        patients: The full cohort.

    Returns:
        Coordinates ordered depot-first, and patient id mapped to its node
        index in that list.
    """
    coordinates = [DEPOT_COORDS] + [patient.coords for patient in patients]
    patient_nodes = {patient.id: index + 1 for index, patient in enumerate(patients)}
    return coordinates, patient_nodes


class DistanceMatrix:
    """Manhattan distances between nodes, and their travel times."""

    def __init__(self, travel_speed_kmh: float) -> None:
        """Args:
        travel_speed_kmh: Constant worker travel speed.
        """
        self._travel_speed_kmh = travel_speed_kmh

    def manhattan_distance_matrix(
        self, coordinates: list[tuple[float, float]]
    ) -> np.ndarray:
        """Computes pairwise Manhattan distances in kilometres, since workers travel a road network rather than straight lines."""
        points = np.asarray(coordinates, dtype=float)
        return np.abs(points[:, None, :] - points[None, :, :]).sum(axis=2)

    def travel_time_matrix_secs(
        self, coordinates: list[tuple[float, float]]
    ) -> np.ndarray:
        """Converts Manhattan distances to integer travel times in seconds, since OR-Tools requires integer arc costs and seconds keep rounding error negligible."""
        distance_km = self.manhattan_distance_matrix(coordinates)
        travel_hours = distance_km / self._travel_speed_kmh
        return (travel_hours * 3600).astype(int)

    def pair_travel(
        self, origin: tuple[float, float], destination: tuple[float, float]
    ) -> tuple[float, int]:
        """Distance and travel time between exactly two points, in kilometres and whole seconds."""
        coordinates = [origin, destination]
        distance_km = self.manhattan_distance_matrix(coordinates)[0][1]
        travel_secs = self.travel_time_matrix_secs(coordinates)[0][1]
        return float(distance_km), int(travel_secs)


class RoutingModel:
    """Constructs, configures and solves the OR-Tools routing problem."""

    def __init__(
        self, config: SimulationConfig, distance_matrix: DistanceMatrix
    ) -> None:
        """Args:
        config: Supplies fleet size, shift length and solver time limits.
        distance_matrix: Builds the travel time matrix.
        """
        self._config = config
        self._distance_matrix = distance_matrix

    def solve(
        self,
        coordinates: list[tuple[float, float]],
        patient_nodes: dict[int, int],
        vehicle_start_nodes: list[int],
        vehicle_end_node: int,
        time_windows_secs: dict[int, tuple[int, int, bool]],
        priority_weights: dict[int, int],
        time_limit_seconds: int,
        start_time_secs: int = 0,
        warm_start_routes: list[list[int]] | None = None,
    ) -> RoutingSolution:
        """Build and solve one routing problem.

        Args:
            coordinates: Coordinates of every node, indexed by node.
            patient_nodes: Patient id mapped to its node index.
            vehicle_start_nodes: Starting node for each vehicle.
            vehicle_end_node: Node every vehicle returns to.
            time_windows_secs: Node mapped to (earliest, latest, is_hard).
            priority_weights: Patient id mapped to clinical priority weight.
            time_limit_seconds: Solver budget.
            start_time_secs: Simulation clock at which routing begins.
            warm_start_routes: Existing routes as node indices, per vehicle.

        Returns:
            The routes found, with solver status and elapsed time.

        Raises:
            SolverError: If the model cannot be constructed.
        """
        travel_time = self._distance_matrix.travel_time_matrix_secs(coordinates)
        node_to_patient = {node: patient for patient, node in patient_nodes.items()}
        service_secs = self._config.service_duration_mins * SECONDS_PER_MINUTE
        horizon_secs = self._config.shift_duration_mins * SECONDS_PER_MINUTE

        try:
            manager = pywrapcp.RoutingIndexManager(
                len(coordinates),
                self._config.n_workers,
                vehicle_start_nodes,
                [vehicle_end_node] * self._config.n_workers,
            )
            routing = pywrapcp.RoutingModel(manager)
        except Exception as error:  # OR-Tools raises bare exceptions on bad input
            raise SolverError(
                f"Could not construct the routing model: {error}"
            ) from error

        def transit_callback(from_index: int, to_index: int) -> int:
            """Travel time plus the service already owed at the origin node."""
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            service = service_secs if from_node in node_to_patient else 0
            return int(travel_time[from_node][to_node]) + service

        transit_index = routing.RegisterTransitCallback(transit_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_index)

        routing.AddDimension(
            transit_index,
            horizon_secs,  # generous slack: waiting is allowed but never forced
            horizon_secs,
            start_time_secs == 0,
            "Time",
        )
        time_dimension = routing.GetDimensionOrDie("Time")

        for node, (earliest, latest, is_hard) in time_windows_secs.items():
            index = manager.NodeToIndex(node)
            if index < 0:
                continue
            patient_id = node_to_patient[node]
            weight = priority_weights[patient_id]
            if is_hard:
                # A closed window would make the node unreachable rather than merely late; dropping is still allowed.
                time_dimension.CumulVar(index).SetRange(
                    max(earliest, start_time_secs), max(latest, start_time_secs)
                )
            else:
                time_dimension.SetCumulVarSoftUpperBound(
                    index, latest, weight * SOFT_WINDOW_PENALTY_SCALE
                )

        # Drop penalty scales with priority weight, so high-risk patients are served first while the model stays feasible.
        for patient_id, node in patient_nodes.items():
            index = manager.NodeToIndex(node)
            if index >= 0:
                routing.AddDisjunction(
                    [index], priority_weights[patient_id] * PRIORITY_SCALE
                )

        for vehicle in range(self._config.n_workers):
            start_index = routing.Start(vehicle)
            if start_time_secs > 0:
                time_dimension.CumulVar(start_index).SetRange(
                    start_time_secs, horizon_secs
                )
            routing.AddVariableMinimizedByFinalizer(
                time_dimension.CumulVar(start_index)
            )

        parameters = self._configure_solver(time_limit_seconds)

        started_at = time.perf_counter()
        if warm_start_routes is not None:
            # ReadAssignmentFromRoutes needs solver indices, not node numbers, which diverge once vehicles start from per-worker positions.
            warm_start_indices = [
                [manager.NodeToIndex(node) for node in route]
                for route in warm_start_routes
            ]
            initial = routing.ReadAssignmentFromRoutes(warm_start_indices, True)
            if initial is None:
                logger.warning("Warm start rejected; falling back to a cold solve")
                assignment = routing.SolveWithParameters(parameters)
            else:
                assignment = routing.SolveFromAssignmentWithParameters(
                    initial, parameters
                )
        else:
            assignment = routing.SolveWithParameters(parameters)
        solve_time_secs = time.perf_counter() - started_at

        status = SOLVER_STATUS_NAMES.get(
            routing.status(), f"UNKNOWN_{routing.status()}"
        )
        if assignment is None:
            logger.warning(
                "Solver returned no solution within %ds (status %s)",
                time_limit_seconds,
                status,
            )
            return RoutingSolution(
                worker_route_assignment={
                    vehicle: [] for vehicle in range(self._config.n_workers)
                },
                solver_status=status,
                solve_time_secs=solve_time_secs,
                is_feasible=False,
                dropped_patient_ids=sorted(patient_nodes),
            )

        routes = self._extract_routes(routing, manager, assignment, node_to_patient)
        served = {patient for route in routes.values() for patient in route}
        logger.info(
            "Solved in %.2fs (status %s): %d of %d patients scheduled",
            solve_time_secs,
            status,
            len(served),
            len(patient_nodes),
        )
        return RoutingSolution(
            worker_route_assignment=routes,
            solver_status=status,
            solve_time_secs=solve_time_secs,
            is_feasible=True,
            dropped_patient_ids=sorted(set(patient_nodes) - served),
        )

    def _configure_solver(
        self, time_limit_seconds: int
    ) -> pywrapcp.RoutingSearchParameters:
        """Builds search parameters using cheapest-arc construction and guided local search, the combination OR-Tools recommends for time-windowed routing at this scale (Google, 2024)."""
        parameters = pywrapcp.DefaultRoutingSearchParameters()
        parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        parameters.time_limit.seconds = time_limit_seconds
        return parameters

    def _extract_routes(
        self,
        routing: pywrapcp.RoutingModel,
        manager: pywrapcp.RoutingIndexManager,
        assignment,
        node_to_patient: dict[int, int],
    ) -> dict[int, list[int]]:
        """Reads patient visit sequences out of a solved assignment into worker id mapped to ordered patient ids."""
        routes: dict[int, list[int]] = {}
        for vehicle in range(self._config.n_workers):
            route: list[int] = []
            index = routing.Start(vehicle)
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                if node in node_to_patient:
                    route.append(node_to_patient[node])
                index = assignment.Value(routing.NextVar(index))
            routes[vehicle] = route
        return routes


class AbstractRouter(ABC):
    """Base class for all routing strategies.

    New strategies extend this rather than modifying existing ones, since the simulation depends only on this interface.
    """

    def __init__(self, config: SimulationConfig) -> None:
        """Args:
        config: Validated simulation configuration.
        """
        self._config = config
        self._last_solution: RoutingSolution | None = None

    @property
    def last_solution(self) -> RoutingSolution | None:
        """Diagnostics from the most recent solve, if any."""
        return self._last_solution

    @abstractmethod
    def get_initial_routes(
        self, patients: list[Patient], workers: list[Worker]
    ) -> dict[int, list[int]]:
        """Compute route assignments at shift start.

        Args:
            patients: The full cohort.
            workers: The full fleet.

        Returns:
            Worker id mapped to an ordered list of patient ids.
        """

    @abstractmethod
    def get_updated_routes(
        self,
        patients: list[Patient],
        workers: list[Worker],
        current_route_assignments: dict[int, list[int]],
        triggered_patients: list[Patient],
        simulation_time_mins: float,
    ) -> dict[int, list[int]]:
        """Recompute routes mid-shift.

        Args:
            patients: The full cohort, carrying current clinical state.
            workers: The fleet, carrying current positions and clock.
            current_route_assignments: Routes currently being executed.
            triggered_patients: Patients whose AND-gate fired at this timestep,
                batched into a single re-solve.
            simulation_time_mins: Current simulation clock.

        Returns:
            Updated worker id to ordered patient ids.
        """


class BaselineRouter(AbstractRouter):
    """Unprioritised nearest-neighbour heuristic with no re-routing."""

    def __init__(
        self, config: SimulationConfig, distance_matrix: DistanceMatrix
    ) -> None:
        """Args:
        config: Validated simulation configuration.
        distance_matrix: Supplies travel times between patients.
        """
        super().__init__(config)
        self._distance_matrix = distance_matrix

    def get_initial_routes(
        self, patients: list[Patient], workers: list[Worker]
    ) -> dict[int, list[int]]:
        """Assigns patients by repeated nearest-neighbour selection; no clinical priority enters, which is what this baseline is for.

        Returns:
            Worker id mapped to ordered patient ids.
        """
        coordinates, node_of_patient = _depot_prefixed_node_map(patients)
        travel_time = self._distance_matrix.travel_time_matrix_secs(coordinates)

        service_secs = self._config.service_duration_mins * SECONDS_PER_MINUTE
        horizon_secs = self._config.shift_duration_mins * SECONDS_PER_MINUTE

        unclaimed = {patient.id for patient in patients}
        routes: dict[int, list[int]] = {worker.id: [] for worker in workers}
        current_node = {worker.id: 0 for worker in workers}
        elapsed_secs = {worker.id: 0 for worker in workers}
        available = [worker.id for worker in workers]

        while unclaimed and available:
            for worker_id in list(available):
                if not unclaimed:
                    break
                nearest = min(
                    unclaimed,
                    key=lambda patient_id: travel_time[current_node[worker_id]][
                        node_of_patient[patient_id]
                    ],
                )
                arrival = (
                    elapsed_secs[worker_id]
                    + int(
                        travel_time[current_node[worker_id]][node_of_patient[nearest]]
                    )
                    + service_secs
                )
                # Reserves the same return-to-depot leg the OR-Tools scenarios enforce structurally, so capacity is compared fairly.
                return_leg_secs = int(travel_time[node_of_patient[nearest]][0])
                if arrival + return_leg_secs > horizon_secs:
                    available.remove(worker_id)
                    continue
                routes[worker_id].append(nearest)
                elapsed_secs[worker_id] = arrival
                current_node[worker_id] = node_of_patient[nearest]
                unclaimed.remove(nearest)

        self._last_solution = RoutingSolution(
            worker_route_assignment=routes,
            solver_status="NEAREST_NEIGHBOUR",
            solve_time_secs=0.0,
            is_feasible=True,
            dropped_patient_ids=sorted(unclaimed),
        )
        logger.info(
            "Nearest-neighbour baseline scheduled %d of %d patients",
            sum(len(route) for route in routes.values()),
            len(patients),
        )
        return routes

    def get_updated_routes(
        self,
        patients: list[Patient],
        workers: list[Worker],
        current_route_assignments: dict[int, list[int]],
        triggered_patients: list[Patient],
        simulation_time_mins: float,
    ) -> dict[int, list[int]]:
        """Returns the existing routes unchanged; this baseline never re-routes, its schedule is fixed at shift start.

        Returns:
            The routes exactly as supplied.
        """
        return current_route_assignments


class OrToolsRouter(AbstractRouter):
    """Shared OR-Tools machinery for the priority-aware routing strategies."""

    def __init__(
        self,
        config: SimulationConfig,
        distance_matrix: DistanceMatrix,
        routing_model: RoutingModel,
    ) -> None:
        """Args:
        config: Validated simulation configuration.
        distance_matrix: Builds travel time matrices.
        routing_model: Wraps model construction and solving.
        """
        super().__init__(config)
        self._distance_matrix = distance_matrix
        self._routing_model = routing_model

    def get_initial_routes(
        self, patients: list[Patient], workers: list[Worker]
    ) -> dict[int, list[int]]:
        """Solves once at shift start using the risk classes then in force.

        Returns:
            Worker id mapped to ordered patient ids.

        Raises:
            SolverError: If the solve does not find a feasible schedule.
        """
        coordinates, patient_nodes = _depot_prefixed_node_map(patients)

        time_windows = {}
        priority_weights = {}
        for patient in patients:
            risk = patient.current_clinical_state
            earliest, latest = INITIAL_TIME_WINDOWS[risk]
            time_windows[patient_nodes[patient.id]] = (
                earliest * SECONDS_PER_MINUTE,
                latest * SECONDS_PER_MINUTE,
                risk == "high",
            )
            priority_weights[patient.id] = PRIORITY_WEIGHTS[risk]

        solution = self._routing_model.solve(
            coordinates=coordinates,
            patient_nodes=patient_nodes,
            vehicle_start_nodes=[0] * self._config.n_workers,
            vehicle_end_node=0,
            time_windows_secs=time_windows,
            priority_weights=priority_weights,
            time_limit_seconds=self._config.initial_solve_limit,
        )
        self._last_solution = solution
        if not solution.is_feasible:
            # No earlier schedule exists to fall back to at shift start, so an infeasible initial solve must stop the run rather than proceed silently.
            raise SolverError(
                f"Initial solve was infeasible (status {solution.solver_status}); "
                "no schedule was produced for this replication."
            )
        return solution.worker_route_assignment

    def get_updated_routes(
        self,
        patients: list[Patient],
        workers: list[Worker],
        current_route_assignments: dict[int, list[int]],
        triggered_patients: list[Patient],
        simulation_time_mins: float,
    ) -> dict[int, list[int]]:
        """Returns the existing routes unchanged; overridden by DynamicRouter, since the static-priority strategy fixes its schedule at shift start.

        Returns:
            The routes exactly as supplied.
        """
        return current_route_assignments


class StaticPriorityRouter(OrToolsRouter):
    """NEWS2 priorities applied once at shift start, with no mid-shift updates."""


class DynamicRouter(OrToolsRouter):
    """AI-integrated routing that re-solves when the AND-gate fires."""

    def __init__(
        self,
        config: SimulationConfig,
        distance_matrix: DistanceMatrix,
        routing_model: RoutingModel,
        tau: int | None = None,
    ) -> None:
        """Tau is injected rather than read from config, so the sensitivity analysis can sweep it without mutating global state.

        Args:
            config: Validated simulation configuration.
            distance_matrix: Builds travel time matrices.
            routing_model: Wraps model construction and solving.
            tau: Minutes from now within which a triggered patient must be
                seen. Defaults to the configured value.
        """
        super().__init__(config, distance_matrix, routing_model)
        self._tau = config.default_tau if tau is None else tau

    @property
    def tau(self) -> int:
        """Minutes from now within which a triggered patient must be seen."""
        return self._tau

    def _feasible_route_prefix(
        self,
        route: list[int],
        patient_by_id: dict[int, Patient],
        patient_nodes: dict[int, int],
        worker_location: tuple[float, float],
        start_time_secs: int,
        horizon_secs: int,
    ) -> list[int]:
        """Trims a route to the visits that still fit inside the shift, since one overrunning route would reject the whole warm start and force a cold re-solve.

        Returns:
            Node indices for the prefix of the route that fits.
        """
        service_secs = self._config.service_duration_mins * SECONDS_PER_MINUTE
        elapsed_secs = start_time_secs
        location = worker_location
        prefix = []
        for patient_id in route:
            destination = patient_by_id[patient_id].coords
            _, travel_secs = self._distance_matrix.pair_travel(location, destination)
            arrival = elapsed_secs + travel_secs + service_secs
            _, return_leg_secs = self._distance_matrix.pair_travel(
                destination, DEPOT_COORDS
            )
            if arrival + return_leg_secs > horizon_secs:
                break
            prefix.append(patient_nodes[patient_id])
            elapsed_secs = arrival
            location = destination
        return prefix

    def get_updated_routes(
        self,
        patients: list[Patient],
        workers: list[Worker],
        current_route_assignments: dict[int, list[int]],
        triggered_patients: list[Patient],
        simulation_time_mins: float,
    ) -> dict[int, list[int]]:
        """Re-solves over the remaining patients from the workers' positions, raising triggered patients to a hard tau-minute deadline.

        Returns:
            Updated worker id to ordered patient ids.
        """
        remaining = [patient for patient in patients if not patient.visited]
        if not remaining:
            return current_route_assignments

        # Node layout: one node per worker's current position, then the depot
        # they return to, then every patient still to be seen.
        coordinates = [worker.current_location for worker in workers]
        depot_node = len(coordinates)
        coordinates.append(DEPOT_COORDS)
        patient_nodes = {
            patient.id: depot_node + 1 + index
            for index, patient in enumerate(remaining)
        }
        coordinates.extend(patient.coords for patient in remaining)

        start_time_secs = int(simulation_time_mins * SECONDS_PER_MINUTE)
        horizon_secs = self._config.shift_duration_mins * SECONDS_PER_MINUTE
        triggered_ids = {patient.id for patient in triggered_patients}

        time_windows = {}
        priority_weights = {}
        for patient in remaining:
            if patient.id in triggered_ids:
                # The trigger is the only source of hard constraints mid-shift, which is what makes the re-solve worth performing.
                risk = "high"
                latest = start_time_secs + self._tau * SECONDS_PER_MINUTE
                is_hard = True
            else:
                # Shift-start windows have often closed by now; re-imposing them as hard constraints would make patients unreachable, so they stay soft.
                risk = patient.current_clinical_state
                latest = max(
                    INITIAL_TIME_WINDOWS[risk][1] * SECONDS_PER_MINUTE, start_time_secs
                )
                is_hard = False
            time_windows[patient_nodes[patient.id]] = (
                start_time_secs,
                min(latest, horizon_secs),
                is_hard,
            )
            priority_weights[patient.id] = PRIORITY_WEIGHTS[risk]

        patient_by_id = {patient.id: patient for patient in remaining}
        warm_start = [
            self._feasible_route_prefix(
                [
                    patient_id
                    for patient_id in current_route_assignments.get(worker.id, [])
                    if patient_id in patient_nodes and patient_id not in triggered_ids
                ],
                patient_by_id,
                patient_nodes,
                worker.current_location,
                start_time_secs,
                horizon_secs,
            )
            for worker in workers
        ]

        logger.info(
            "Re-solving over %d remaining patients; warm start retains %d of %d "
            "planned visits",
            len(remaining),
            sum(len(route) for route in warm_start),
            sum(
                len(current_route_assignments.get(worker.id, [])) for worker in workers
            ),
        )
        solution = self._routing_model.solve(
            coordinates=coordinates,
            patient_nodes=patient_nodes,
            vehicle_start_nodes=[worker.id for worker in workers],
            vehicle_end_node=depot_node,
            time_windows_secs=time_windows,
            priority_weights=priority_weights,
            time_limit_seconds=self._config.reroute_solve_limit,
            start_time_secs=start_time_secs,
            warm_start_routes=warm_start,
        )
        self._last_solution = solution
        if not solution.is_feasible:
            # A failed re-solve must not discard a schedule that is already
            # running: keep executing the current plan.
            return current_route_assignments
        return solution.worker_route_assignment


def compute_churn(
    pre_routes: dict[int, list[int]], post_routes: dict[int, list[int]]
) -> float:
    """Proportion of planned visits whose worker or position changed, the operational cost of re-routing.

    Both arguments must describe the same set of pending visits, or every visit will wrongly report as moved.

    Returns:
        Proportion of pre-solve visits that moved, between 0 and 1.
    """
    planned_position = {
        patient_id: (worker_id, position)
        for worker_id, route in pre_routes.items()
        for position, patient_id in enumerate(route)
    }
    if not planned_position:
        return 0.0
    updated_position = {
        patient_id: (worker_id, position)
        for worker_id, route in post_routes.items()
        for position, patient_id in enumerate(route)
    }
    changed = sum(
        1
        for patient_id, before in planned_position.items()
        if updated_position.get(patient_id) != before
    )
    return changed / len(planned_position)
