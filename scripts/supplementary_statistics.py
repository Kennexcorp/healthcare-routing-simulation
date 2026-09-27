"""Supplementary statistics computed from existing outputs.

Adds a confidence interval on the paired difference in mean response time,
and the number of patients who reach the high-risk state per shift. No
simulation or solver runs here: the cohort replay reuses the same generator
calls in the same order as the original run, so the same seed yields the
same patients.

Usage:
    uv run python scripts/supplementary_statistics.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.config import SimulationConfig
from src.data_generator import (
    BiometricGenerator,
    CTMCModel,
    NEWS2Scorer,
    SyntheticCohort,
)

RESULTS_DIRECTORY = Path(__file__).resolve().parent.parent / "results"
PRIMARY_METRIC = "mean_response_time_high_risk"
COMPARISONS = (
    ("ai_integrated", "static_priority"),
    ("ai_integrated", "unprioritised"),
    ("static_priority", "unprioritised"),
)
CONFIDENCE_LEVEL = 0.95

logger = logging.getLogger(__name__)


def paired_difference_interval(differences: np.ndarray) -> tuple[float, float, float]:
    """Mean of paired differences and its t-based confidence interval."""
    mean = float(np.mean(differences))
    standard_error = float(np.std(differences, ddof=1) / np.sqrt(len(differences)))
    critical_value = stats.t.ppf((1 + CONFIDENCE_LEVEL) / 2, len(differences) - 1)
    margin = float(critical_value * standard_error)
    return mean, mean - margin, mean + margin


def paired_intervals(simulation_results: pd.DataFrame) -> pd.DataFrame:
    """Interval on the paired difference for each primary comparison, one row
    per comparison with the count of pairs favouring the first scenario."""
    by_seed = simulation_results.pivot(
        index="seed", columns="scenario", values=PRIMARY_METRIC
    )
    rows = []
    for first, second in COMPARISONS:
        differences = (by_seed[first] - by_seed[second]).to_numpy()
        mean, lower, upper = paired_difference_interval(differences)
        rows.append(
            {
                "scenario_a": first,
                "scenario_b": second,
                "mean_difference": mean,
                "ci_lower": lower,
                "ci_upper": upper,
                "pairs_favouring_a": int((differences < 0).sum()),
                "n_pairs": len(differences),
            }
        )
    return pd.DataFrame(rows)


def high_risk_patients_per_shift(config: SimulationConfig, seed: int) -> int:
    """Count patients who receive a high-risk label at any step of one shift."""
    cohort = SyntheticCohort(
        config=config,
        ctmc_model=CTMCModel(),
        biometric_generator=BiometricGenerator(),
        news2_scorer=NEWS2Scorer(),
    )
    rng = np.random.default_rng(seed)
    patients = cohort.initialise_patients(rng)
    reached_high_risk: set[int] = set()
    for timestep_index in range(config.shift_duration_mins // config.timestep_mins):
        for patient in patients:
            if timestep_index > 0:
                cohort.advance_patient(patient, rng)
            observation = cohort.observe_patient(patient, rng)
            if cohort.label_observation(observation) == "high":
                reached_high_risk.add(patient.id)
    return len(reached_high_risk)


def high_risk_incidence(
    config: SimulationConfig, simulation_results: pd.DataFrame
) -> pd.DataFrame:
    """Per-seed high-risk patient counts and the unreached share per scenario."""
    counts = {
        seed: high_risk_patients_per_shift(config, seed)
        for seed in sorted(simulation_results["seed"].unique())
    }
    frame = simulation_results[
        ["seed", "scenario", "unreached_high_risk_patients"]
    ].copy()
    frame["high_risk_patients"] = frame["seed"].map(counts)
    frame["unreached_share"] = (
        frame["unreached_high_risk_patients"] / frame["high_risk_patients"]
    )
    return frame


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    config = SimulationConfig()
    simulation_results = pd.read_csv(RESULTS_DIRECTORY / "simulation_results.csv")

    intervals = paired_intervals(simulation_results)
    intervals.to_csv(RESULTS_DIRECTORY / "paired_intervals.csv", index=False)
    incidence = high_risk_incidence(config, simulation_results)
    incidence.to_csv(RESULTS_DIRECTORY / "high_risk_incidence.csv", index=False)

    logger.info("Paired intervals:\n%s", intervals.to_string(index=False))
    per_seed = incidence.drop_duplicates("seed")["high_risk_patients"]
    logger.info(
        "High-risk patients per shift: mean %.1f, SD %.1f",
        per_seed.mean(),
        per_seed.std(),
    )
    logger.info(
        "Unreached share by scenario:\n%s",
        incidence.groupby("scenario")["unreached_share"].mean().to_string(),
    )


if __name__ == "__main__":
    main()
