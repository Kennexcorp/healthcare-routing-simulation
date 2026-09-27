"""Tests for the AND-gate diagnostics script.

The script's counting trigger must never change what the gate decides, or the
diagnostics would describe a different gate from the one the experiment used.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from src.config import SimulationConfig
from src.models import Patient
from src.trigger import ANDGateTrigger

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "gate_diagnostics.py"
BASELINE = {"spo2": 97.0, "hr": 72.0, "sbp": 125.0}
SPO2_SD = 1.5

# scripts/ is not a package, so the module is loaded from its path.
_spec = importlib.util.spec_from_file_location("gate_diagnostics", SCRIPT_PATH)
gate_diagnostics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate_diagnostics)


def make_patient(
    deviating: bool = True, minutes_since_visit: float = 90.0, visited: bool = False
) -> Patient:
    """A patient whose deviation and overdue conditions are set explicitly."""
    spo2 = 97.0 - 3 * SPO2_SD if deviating else 97.0
    return Patient(
        id=0,
        age=70,
        diagnosis="COPD",
        coords=(1.0, 2.0),
        personal_baseline_mean=dict(BASELINE),
        time_since_last_visit_mins=minutes_since_visit,
        biometric_history=[[spo2, 72.0, 125.0]] * 3,
        visited=visited,
    )


def make_trigger() -> gate_diagnostics.CountingANDGateTrigger:
    return gate_diagnostics.CountingANDGateTrigger(SimulationConfig())


@pytest.mark.parametrize("label", ["high", "moderate"])
@pytest.mark.parametrize("probability", [0.5, 0.9])
@pytest.mark.parametrize("deviating", [True, False])
@pytest.mark.parametrize("minutes_since_visit", [10.0, 90.0])
def test_decision_matches_the_uncounted_gate(
    label, probability, deviating, minutes_since_visit
):
    patient = make_patient(deviating, minutes_since_visit)
    config = SimulationConfig()
    expected = ANDGateTrigger(config).evaluate(patient, label, probability, 120.0)
    assert make_trigger().evaluate(patient, label, probability, 120.0) is expected


def test_a_firing_evaluation_is_counted_in_every_condition():
    trigger = make_trigger()
    assert trigger.evaluate(make_patient(), "high", 0.9, 120.0) is True
    counts = trigger.counts
    assert counts["eligible"] == 1
    assert counts["ml_prediction"] == counts["overdue"] == counts["deviating"] == 1
    assert counts["fired"] == 1


def test_a_condition_that_fails_alone_is_recorded_as_binding():
    trigger = make_trigger()
    trigger.evaluate(make_patient(deviating=False), "high", 0.9, 120.0)
    trigger.evaluate(make_patient(minutes_since_visit=10.0), "high", 0.9, 120.0)
    trigger.evaluate(
        make_patient(deviating=False, minutes_since_visit=10.0), "high", 0.9, 120.0
    )
    counts = trigger.counts
    assert counts["ml_blocked_by_deviation_only"] == 1
    assert counts["ml_blocked_by_overdue_only"] == 1
    assert counts["ml_blocked_by_both"] == 1
    assert counts["fired"] == 0


def test_conditions_are_tallied_even_when_an_earlier_one_fails():
    """The parent gate skips the deviation test when the classifier says low;
    the counter must still record it, or its marginal rate would be understated."""
    trigger = make_trigger()
    trigger.evaluate(make_patient(), "low", 0.1, 120.0)
    assert trigger.counts["deviating"] == 1
    assert trigger.counts["ml_prediction"] == 0


def test_visited_and_suppressed_patients_are_not_counted():
    trigger = make_trigger()
    trigger.evaluate(make_patient(visited=True), "high", 0.9, 120.0)
    patient = make_patient()
    trigger.record_events([patient], {patient.id: 0.9}, 100.0, 0.0)
    trigger.evaluate(patient, "high", 0.9, 110.0)
    assert trigger.counts["eligible"] == 0


def test_unused_counters_start_at_zero_so_every_column_exists():
    assert set(make_trigger().counts) == set(gate_diagnostics.COUNT_KEYS)


def test_pooled_rates_are_shares_of_the_right_denominators():
    frame = pd.DataFrame(
        [
            {
                "eligible": 100,
                "ml_prediction": 20,
                "overdue": 80,
                "deviating": 30,
                "fired": 10,
                "ml_blocked_by_overdue_only": 2,
                "ml_blocked_by_deviation_only": 6,
                "ml_blocked_by_both": 2,
            },
            {
                "eligible": 100,
                "ml_prediction": 20,
                "overdue": 80,
                "deviating": 30,
                "fired": 10,
                "ml_blocked_by_overdue_only": 2,
                "ml_blocked_by_deviation_only": 6,
                "ml_blocked_by_both": 2,
            },
        ]
    )
    rates = gate_diagnostics.pooled_rates(frame)
    assert rates["ml_prediction_rate"] == pytest.approx(0.2)
    assert rates["fired_rate"] == pytest.approx(0.1)
    assert rates["fired_share_of_ml_flagged"] == pytest.approx(0.5)
    assert rates["ml_flagged_blocked_by_deviation_only"] == pytest.approx(0.3)
