"""Tests for the AND-gate trigger, exercising each condition alone and in
combination since the claim is about the conjunction, not any single signal."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config import SimulationConfig
from src.models import Patient, TriggerEvent
from src.trigger import ANDGateTrigger

BASELINE = {"spo2": 97.0, "hr": 72.0, "sbp": 125.0}

# Population spreads from BASELINE_DISTRIBUTIONS, restated so the tests state
# what a two-standard-deviation departure actually is in clinical units.
SPO2_SD, HR_SD, SBP_SD = 1.5, 8.0, 10.0


def make_patient(history: list[list[float]] | None = None, **overrides) -> Patient:
    """Build a patient who satisfies conditions two and three by default, with
    oxygen saturation three SD below baseline so condition two holds unless
    overridden."""
    fields = {
        "id": 0,
        "age": 70,
        "diagnosis": "COPD",
        "coords": (1.0, 2.0),
        "personal_baseline_mean": dict(BASELINE),
        "time_since_last_visit_mins": 90.0,
        "biometric_history": history
        if history is not None
        else [[97.0 - 3 * SPO2_SD, 72.0, 125.0]] * 3,
    }
    fields.update(overrides)
    return Patient(**fields)


def make_trigger(**overrides) -> ANDGateTrigger:
    """Build a trigger on the default theta and tau unless overridden."""
    return ANDGateTrigger(SimulationConfig(), **overrides)


class TestConditionOnePredictedRisk:
    def test_fires_when_probability_is_at_the_threshold(self):
        trigger = make_trigger(theta=0.70)
        assert trigger.evaluate(make_patient(), "high", 0.70, 120.0) is True

    def test_does_not_fire_just_below_the_threshold(self):
        trigger = make_trigger(theta=0.70)
        assert trigger.evaluate(make_patient(), "high", 0.699, 120.0) is False

    @pytest.mark.parametrize("label", ["low", "moderate"])
    def test_does_not_fire_when_the_predicted_label_is_not_high(self, label):
        """A high probability alone is not enough: the label must agree."""
        trigger = make_trigger(theta=0.70)
        assert trigger.evaluate(make_patient(), label, 0.99, 120.0) is False


class TestConditionTwoBaselineDeviation:
    @pytest.mark.parametrize(
        ("observation", "expected", "reason"),
        [
            ([97.0 - 2 * SPO2_SD, 72.0, 125.0], True, "spo2 two sd below baseline"),
            ([97.0, 72.0 + 2 * HR_SD, 125.0], True, "heart rate two sd above"),
            ([97.0, 72.0, 125.0 - 2 * SBP_SD], True, "systolic bp two sd below"),
            ([97.0 - 1.9 * SPO2_SD, 72.0, 125.0], False, "spo2 just short of two sd"),
            ([97.0, 72.0, 125.0], False, "sitting exactly on baseline"),
            ([99.0, 60.0, 135.0], False, "comfortably better than baseline"),
        ],
    )
    def test_any_single_adverse_departure_counts(self, observation, expected, reason):
        patient = make_patient([observation] * 3)
        assert make_trigger().is_deviating_from_baseline(patient) is expected, reason

    def test_rising_systolic_pressure_is_not_adverse(self):
        """Rising pressure is the expected compensatory phase; only a fall
        marks decompensation."""
        patient = make_patient([[97.0, 72.0, 125.0 + 3 * SBP_SD]] * 3)
        assert make_trigger().is_deviating_from_baseline(patient) is False

    def test_the_comparison_is_against_the_patients_own_baseline(self):
        """Each patient carries an individual baseline: a heart rate of 88 is
        two SD above a typical patient's normal but unremarkable for one
        whose own normal is 90."""
        typical = make_patient([[97.0, 88.0, 125.0]] * 3)
        naturally_tachycardic = make_patient(
            [[97.0, 88.0, 125.0]] * 3,
            personal_baseline_mean={"spo2": 97.0, "hr": 90.0, "sbp": 125.0},
        )
        assert make_trigger().is_deviating_from_baseline(typical) is True
        assert make_trigger().is_deviating_from_baseline(naturally_tachycardic) is False

    def test_a_single_extreme_reading_does_not_fire_the_condition(self):
        """Averaging over the window is what stops one noisy observation
        triggering a re-route."""
        history = [
            [97.0, 72.0, 125.0],
            [97.0, 72.0, 125.0],
            [97.0 - 4 * SPO2_SD, 72.0, 125.0],
        ]
        assert make_trigger().is_deviating_from_baseline(make_patient(history)) is False

    def test_a_sustained_departure_does_fire_the_condition(self):
        history = [[97.0 - 2.5 * SPO2_SD, 72.0, 125.0]] * 3
        assert make_trigger().is_deviating_from_baseline(make_patient(history)) is True

    def test_empty_history_cannot_satisfy_the_condition(self):
        assert make_trigger().is_deviating_from_baseline(make_patient([])) is False

    def test_only_the_trailing_window_is_considered(self):
        """A departure long past should not keep the condition alive once the
        patient has returned to their own normal range."""
        history = [
            [90.0, 110.0, 95.0],  # older observations, severely deranged
            [90.0, 110.0, 95.0],
            [97.0, 72.0, 125.0],  # trailing window: back to baseline
            [97.0, 72.0, 125.0],
            [97.0, 72.0, 125.0],
        ]
        assert make_trigger().is_deviating_from_baseline(make_patient(history)) is False

    def test_zscore_is_computed_per_parameter(self):
        patient = make_patient([[97.0 - SPO2_SD, 72.0 + 2 * HR_SD, 125.0]] * 3)
        deviation = make_trigger().baseline_deviation_zscore(patient)
        assert deviation == pytest.approx([-1.0, 2.0, 0.0])


class TestConditionThreeTimeSinceVisit:
    def test_fires_exactly_at_tau(self):
        trigger = make_trigger(tau=60)
        patient = make_patient(time_since_last_visit_mins=60.0)
        assert trigger.evaluate(patient, "high", 0.9, 120.0) is True

    def test_does_not_fire_just_below_tau(self):
        trigger = make_trigger(tau=60)
        patient = make_patient(time_since_last_visit_mins=59.9)
        assert trigger.evaluate(patient, "high", 0.9, 120.0) is False


class TestConjunction:
    """The central claim: the gate fires only when all three conditions hold."""

    def test_all_three_conditions_fire_the_gate(self):
        assert make_trigger().evaluate(make_patient(), "high", 0.9, 120.0) is True

    @pytest.mark.parametrize("missing", ["ml", "trend", "time"])
    def test_any_single_condition_failing_blocks_the_gate(self, missing):
        trigger = make_trigger()
        patient = make_patient()
        label, probability = "high", 0.9
        if missing == "ml":
            probability = 0.1
        elif missing == "trend":
            patient = make_patient([[97.0, 72.0, 125.0]] * 3)
        else:
            patient = make_patient(time_since_last_visit_mins=10.0)
        assert trigger.evaluate(patient, label, probability, 120.0) is False

    def test_a_visited_patient_never_triggers(self):
        patient = make_patient(visited=True)
        assert make_trigger().evaluate(patient, "high", 0.99, 120.0) is False


class TestDuplicateSuppression:
    def test_a_repeat_trigger_is_suppressed_within_tau(self):
        """The patient already holds the highest priority and a hard deadline;
        re-solving again would churn routes without changing any constraint."""
        trigger = make_trigger(tau=60)
        patient = make_patient()
        assert trigger.evaluate(patient, "high", 0.9, 100.0) is True
        trigger.record_events([patient], {0: 0.9}, 100.0, 0.2)
        assert trigger.evaluate(patient, "high", 0.9, 130.0) is False

    def test_the_patient_becomes_eligible_again_once_the_deadline_passes(self):
        """If the deadline was missed and the patient is still waiting, the
        situation has changed and a fresh trigger is warranted."""
        trigger = make_trigger(tau=60)
        patient = make_patient()
        trigger.record_events([patient], {0: 0.9}, 100.0, 0.2)
        assert trigger.evaluate(patient, "high", 0.9, 160.0) is True

    def test_suppression_is_per_patient(self):
        trigger = make_trigger(tau=60)
        first, second = make_patient(id=0), make_patient(id=1)
        trigger.record_events([first], {0: 0.9}, 100.0, 0.2)
        assert trigger.evaluate(second, "high", 0.9, 110.0) is True


class TestBatching:
    def test_all_firing_patients_are_returned_together(self):
        """Batching is what keeps one timestep to a single re-solve."""
        trigger = make_trigger()
        patients = [make_patient(id=index) for index in range(4)]
        labels = {index: "high" for index in range(4)}
        probabilities = {index: 0.9 for index in range(4)}
        triggered = trigger.evaluate_cohort(patients, labels, probabilities, 120.0)
        assert [patient.id for patient in triggered] == [0, 1, 2, 3]

    def test_only_qualifying_patients_are_returned(self):
        trigger = make_trigger()
        patients = [make_patient(id=index) for index in range(4)]
        patients[1].time_since_last_visit_mins = 5.0
        patients[3].visited = True
        labels = {index: "high" for index in range(4)}
        probabilities = {0: 0.9, 1: 0.9, 2: 0.4, 3: 0.9}
        triggered = trigger.evaluate_cohort(patients, labels, probabilities, 120.0)
        assert [patient.id for patient in triggered] == [0]

    def test_a_batch_shares_one_churn_value(self):
        trigger = make_trigger()
        patients = [make_patient(id=index) for index in range(3)]
        events = trigger.record_events(patients, {0: 0.9, 1: 0.8, 2: 0.7}, 120.0, 0.35)
        assert [event.rerouting_churn_proportion for event in events] == [0.35] * 3


class TestFalseTriggerResolution:
    def test_an_event_is_unresolved_when_recorded(self):
        """The outcome depends on a state 30 minutes in the future, so it
        cannot be known when the trigger fires."""
        trigger = make_trigger()
        (event,) = trigger.record_events([make_patient()], {0: 0.9}, 120.0, 0.1)
        assert event.is_resolved is False
        assert event.is_false_trigger is None
        assert trigger.unresolved_event_count == 1

    def test_resolution_waits_for_the_full_lead_time(self):
        trigger = make_trigger()
        patient = make_patient()
        trigger.record_events([patient], {0: 0.9}, 120.0, 0.1)
        lead_time = SimulationConfig().prediction_lead_time_mins
        assert trigger.resolve_due_events([patient], 120.0 + lead_time - 5) == []
        assert len(trigger.resolve_due_events([patient], 120.0 + lead_time)) == 1

    def test_a_patient_who_deteriorates_is_a_true_trigger(self):
        """The patient was not high risk when the gate fired, but became so
        within the lead time."""
        trigger = make_trigger()
        patient = make_patient(current_clinical_state="moderate")
        trigger.record_events([patient], {0: 0.9}, 120.0, 0.1)
        patient.current_clinical_state = "high"
        (event,) = trigger.resolve_due_events([patient], 160.0)
        assert event.ctmc_ground_truth_state == "moderate"
        assert event.ctmc_state_at_lead_time == "high"
        assert event.is_false_trigger is False

    def test_a_patient_who_does_not_deteriorate_is_a_false_trigger(self):
        trigger = make_trigger()
        patient = make_patient(current_clinical_state="moderate")
        trigger.record_events([patient], {0: 0.9}, 120.0, 0.1)
        (event,) = trigger.resolve_due_events([patient], 160.0)
        assert event.is_false_trigger is True

    def test_a_patient_already_high_who_stays_high_is_a_true_trigger(self):
        trigger = make_trigger()
        patient = make_patient(current_clinical_state="high")
        trigger.record_events([patient], {0: 0.9}, 120.0, 0.1)
        (event,) = trigger.resolve_due_events([patient], 160.0)
        assert event.is_false_trigger is False

    def test_events_are_resolved_only_once(self):
        trigger = make_trigger()
        patient = make_patient()
        trigger.record_events([patient], {0: 0.9}, 120.0, 0.1)
        assert len(trigger.resolve_due_events([patient], 160.0)) == 1
        assert trigger.resolve_due_events([patient], 200.0) == []

    def test_late_triggers_stay_unresolved_and_are_counted_separately(self):
        """Triggers firing within one lead time of the shift end have no
        future state, so they are excluded from the rate."""
        trigger = make_trigger()
        patient = make_patient()
        trigger.record_events([patient], {0: 0.9}, 470.0, 0.1)
        trigger.resolve_due_events([patient], 480.0)
        assert trigger.unresolved_event_count == 1
        assert trigger.resolved_events == []


class TestQualityMetrics:
    def make_resolved(self, outcomes: list[str]) -> ANDGateTrigger:
        """Record and resolve one trigger per supplied outcome state."""
        trigger = make_trigger(tau=0)
        for index, state in enumerate(outcomes):
            patient = make_patient(id=index, current_clinical_state="moderate")
            trigger.record_events(
                [patient], {index: 0.9}, 0.0, 0.2, additional_travel_mins=12.0
            )
            patient.current_clinical_state = state
            trigger.resolve_due_events([patient], 60.0)
        return trigger

    def test_false_trigger_rate_counts_only_resolved_events(self):
        trigger = self.make_resolved(["high", "high", "moderate", "low"])
        assert trigger.false_trigger_rate == pytest.approx(0.5)

    def test_false_trigger_rate_is_zero_when_nothing_has_fired(self):
        assert make_trigger().false_trigger_rate == 0.0

    def test_mean_churn_averages_across_events(self):
        trigger = make_trigger(tau=0)
        trigger.record_events([make_patient(id=0)], {0: 0.9}, 0.0, 0.4)
        trigger.record_events([make_patient(id=1)], {1: 0.9}, 0.0, 0.2)
        assert trigger.mean_rerouting_churn == pytest.approx(0.3)

    def test_operational_cost_covers_only_false_triggers(self):
        """The metric is the cost of the re-routes that turned out to be
        unnecessary, not of every re-route."""
        trigger = self.make_resolved(["high", "moderate"])
        assert trigger.mean_additional_travel_per_false_trigger == pytest.approx(12.0)

    def test_operational_cost_is_zero_without_false_triggers(self):
        trigger = self.make_resolved(["high", "high"])
        assert trigger.mean_additional_travel_per_false_trigger == 0.0


class TestTriggerEventValidation:
    def test_an_invalid_lead_time_state_is_rejected(self):
        event = TriggerEvent(
            patient_id=0,
            timestamp_mins=100.0,
            ml_risk_label="high",
            high_risk_probability=0.9,
            biometric_deviation_detected=True,
            time_since_last_visit_mins=90.0,
            ctmc_ground_truth_state="moderate",
            rerouting_churn_proportion=0.2,
        )
        with pytest.raises(ValidationError):
            event.ctmc_state_at_lead_time = "critical"

    def test_thresholds_are_injected_not_read_globally(self):
        """The sensitivity analysis sweeps theta and tau, so a trigger must be
        constructable at any operating point without mutating configuration."""
        trigger = make_trigger(theta=0.55, tau=15)
        assert (trigger.theta, trigger.tau) == (0.55, 15)
