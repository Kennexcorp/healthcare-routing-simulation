"""Capacity sweep for the unprioritised baseline: measures coverage across cohort sizes at a fixed 10-worker fleet, to choose a cohort size that operates under capacity.

Usage:
    uv run python scripts/capacity_sweep.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import SimulationConfig
from src.data_generator import (
    BiometricGenerator,
    CTMCModel,
    NEWS2Scorer,
    SyntheticCohort,
)
from src.routing import BaselineRouter, DistanceMatrix
from src.simulation import ReplicationManager, Simulation

RESULTS_DIRECTORY = Path(__file__).resolve().parent.parent / "results"
COHORT_SIZES = (130, 150, 160, 170, 190)
N_WORKERS = 10

logger = logging.getLogger(__name__)


def build_cohort(config: SimulationConfig) -> SyntheticCohort:
    """Assemble the synthetic cohort generator from its components."""
    return SyntheticCohort(
        config=config,
        ctmc_model=CTMCModel(),
        biometric_generator=BiometricGenerator(),
        news2_scorer=NEWS2Scorer(),
    )


def sweep() -> pd.DataFrame:
    """Run one unprioritised-scenario replication at each cohort size.

    Returns:
        One row per cohort size: served count, unvisited count, and coverage.
    """
    rows = []
    for n_patients in COHORT_SIZES:
        config = SimulationConfig(n_patients=n_patients, n_workers=N_WORKERS)
        distance_matrix = DistanceMatrix(config.travel_speed_kmh)
        seed = ReplicationManager(config).base_seeds(1)[0]
        simulation = Simulation(
            config=config,
            cohort=build_cohort(config),
            router=BaselineRouter(config, distance_matrix),
            distance_matrix=distance_matrix,
            scenario="unprioritised",
        )
        result = simulation.run(replication_id=0, seed=seed)
        served = n_patients - result.unvisited_patients
        coverage = served / n_patients
        logger.info(
            "n_patients=%d: %d of %d served (%.1f%% coverage)",
            n_patients,
            served,
            n_patients,
            coverage * 100,
        )
        rows.append(
            {
                "n_patients": n_patients,
                "n_workers": N_WORKERS,
                "seed": seed,
                "served_patients": served,
                "unvisited_patients": result.unvisited_patients,
                "coverage": coverage,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    frame = sweep()
    output_path = RESULTS_DIRECTORY / "capacity_sweep.csv"
    frame.to_csv(output_path, index=False)
    logger.info("Wrote %s", output_path)


if __name__ == "__main__":
    main()
