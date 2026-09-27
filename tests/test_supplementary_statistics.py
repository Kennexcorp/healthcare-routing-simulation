"""Tests for the supplementary statistics script."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import SimulationConfig

SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "supplementary_statistics.py"
)

# scripts/ is not a package, so the module is loaded from its path.
_spec = importlib.util.spec_from_file_location("supplementary_statistics", SCRIPT_PATH)
supplementary_statistics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(supplementary_statistics)


def test_interval_is_centred_on_the_mean_difference():
    mean, lower, upper = supplementary_statistics.paired_difference_interval(
        np.array([-1.0, -3.0, -2.0, -4.0])
    )
    assert mean == pytest.approx(-2.5)
    assert lower < mean < upper
    assert mean - lower == pytest.approx(upper - mean)


def test_interval_matches_a_hand_computed_case():
    """Hand-computed check for the paired-difference confidence interval."""
    _, lower, upper = supplementary_statistics.paired_difference_interval(
        np.array([1.0, 2.0, 3.0])
    )
    assert upper - 2.0 == pytest.approx(4.3027 / np.sqrt(3), abs=1e-4)
    assert lower == pytest.approx(2.0 - 4.3027 / np.sqrt(3), abs=1e-4)


def test_pairs_are_matched_by_seed_not_by_row_order():
    frame = pd.DataFrame(
        {
            "seed": [2, 1, 2, 1, 2, 1],
            "scenario": [
                "ai_integrated",
                "ai_integrated",
                "static_priority",
                "static_priority",
                "unprioritised",
                "unprioritised",
            ],
            "mean_response_time_high_risk": [90.0, 100.0, 110.0, 120.0, 130.0, 140.0],
        }
    )
    result = supplementary_statistics.paired_intervals(frame)
    first = result.iloc[0]
    assert first["mean_difference"] == pytest.approx(-20.0)
    assert first["pairs_favouring_a"] == 2
    assert first["n_pairs"] == 2


def test_high_risk_count_is_reproducible_and_bounded_by_the_cohort():
    config = SimulationConfig(n_patients=12)
    first = supplementary_statistics.high_risk_patients_per_shift(config, seed=3)
    second = supplementary_statistics.high_risk_patients_per_shift(config, seed=3)
    assert first == second
    assert 0 <= first <= config.n_patients


def test_unreached_share_uses_the_seeds_own_denominator():
    config = SimulationConfig(n_patients=12)
    frame = pd.DataFrame(
        {
            "seed": [1, 1],
            "scenario": ["ai_integrated", "static_priority"],
            "unreached_high_risk_patients": [0, 0],
        }
    )
    result = supplementary_statistics.high_risk_incidence(config, frame)
    assert (result["unreached_share"] == 0).all()
    assert result["high_risk_patients"].nunique() == 1
