"""Tests for the shared Pydantic models, focused on validation rejecting invalid input."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models import Patient, SimulationResult, TriggerEvent, Worker


def make_patient(**overrides) -> Patient:
    """Build a valid Patient, overriding individual fields for a given test."""
    fields = {
        "id": 0,
        "age": 70,
        "diagnosis": "COPD",
        "coords": (1.0, 2.0),
        "personal_baseline_mean": {"spo2": 97.0, "hr": 72.0, "sbp": 125.0},
    }
    fields.update(overrides)
    return Patient(**fields)


def make_result(**overrides) -> SimulationResult:
    """Build a valid SimulationResult, overriding individual fields."""
    fields = {
        "replication_id": 0,
        "seed": 42,
        "scenario": "ai_integrated",
        "n_patients": 200,
        "mean_response_time_high_risk": 35.5,
        "total_travel_distance_km": 120.0,
        "workforce_utilisation": 0.82,
        "rerouting_events_per_shift": 7,
        "false_trigger_rate": 0.15,
        "rerouting_churn_proportion": 0.22,
        "mean_additional_travel_per_false_trigger": 4.5,
        "unvisited_patients": 12,
        "unreached_high_risk_patients": 2,
        "solver_feasible": True,
        "mean_reroute_time_secs": 8.4,
        "confidence_interval_half_width": 3.1,
    }
    fields.update(overrides)
    return SimulationResult(**fields)


class TestPatient:
    def test_defaults_match_shift_start_state(self):
        patient = make_patient()
        assert patient.current_clinical_state == "low"
        assert patient.elapsed_time_in_state_hours == 0.0
        assert patient.biometric_history == []
        assert patient.time_first_high_risk_mins is None
        assert patient.visited is False

    @pytest.mark.parametrize("age", [44, 86, 0, -1])
    def test_age_outside_cohort_bounds_is_rejected(self, age):
        with pytest.raises(ValidationError):
            make_patient(age=age)

    @pytest.mark.parametrize("age", [45, 85])
    def test_age_at_cohort_bounds_is_accepted(self, age):
        assert make_patient(age=age).age == age

    @pytest.mark.parametrize("state", ["critical", "High", "", "unknown"])
    def test_invalid_clinical_state_is_rejected(self, state):
        with pytest.raises(ValidationError):
            make_patient(current_clinical_state=state)

    @pytest.mark.parametrize("diagnosis", ["Asthma", "copd", "", "CANCER"])
    def test_invalid_diagnosis_is_rejected(self, diagnosis):
        with pytest.raises(ValidationError):
            make_patient(diagnosis=diagnosis)

    def test_valid_state_transition_is_accepted_on_assignment(self):
        patient = make_patient()
        patient.current_clinical_state = "moderate"
        assert patient.current_clinical_state == "moderate"

    def test_invalid_state_is_rejected_on_assignment(self):
        """The reason validate_assignment is set: a bad CTMC transition must
        fail where it happens, not silently corrupt the replication."""
        patient = make_patient()
        with pytest.raises(ValidationError):
            patient.current_clinical_state = "critical"
        assert patient.current_clinical_state == "low"

    def test_invalid_age_is_rejected_on_assignment(self):
        patient = make_patient()
        with pytest.raises(ValidationError):
            patient.age = 200


class TestWorker:
    def test_defaults_match_shift_start_state(self):
        worker = Worker(id=0, current_location=(5.0, 5.0))
        assert worker.current_time_mins == 0.0
        assert worker.visits_completed == []
        assert worker.shift_duration_mins == 480

    @pytest.mark.parametrize("current_time_mins", [-1.0, 480.1, 1000.0])
    def test_time_outside_the_shift_is_rejected(self, current_time_mins):
        with pytest.raises(ValidationError):
            Worker(
                id=0,
                current_location=(5.0, 5.0),
                current_time_mins=current_time_mins,
            )

    def test_time_beyond_shift_end_is_rejected_on_assignment(self):
        worker = Worker(id=0, current_location=(5.0, 5.0))
        with pytest.raises(ValidationError):
            worker.current_time_mins = 481.0

    def test_time_bound_follows_a_configured_shift_length(self):
        """The 480-minute bound is only the default: a worker built with a
        shorter or longer shift must be checked against its own length, not a
        hardcoded constant."""
        worker = Worker(
            id=0,
            current_location=(5.0, 5.0),
            shift_duration_mins=120,
            current_time_mins=119.0,
        )
        assert worker.current_time_mins == 119.0
        with pytest.raises(ValidationError):
            Worker(
                id=0,
                current_location=(5.0, 5.0),
                shift_duration_mins=120,
                current_time_mins=121.0,
            )

    def test_time_beyond_a_configured_shift_is_rejected_on_assignment(self):
        worker = Worker(id=0, current_location=(5.0, 5.0), shift_duration_mins=120)
        with pytest.raises(ValidationError):
            worker.current_time_mins = 121.0


class TestTriggerEvent:
    def make_event(self, **overrides) -> TriggerEvent:
        fields = {
            "patient_id": 3,
            "timestamp_mins": 120.0,
            "ml_risk_label": "high",
            "high_risk_probability": 0.85,
            "biometric_deviation_detected": True,
            "time_since_last_visit_mins": 90.0,
            "ctmc_ground_truth_state": "high",
            "is_false_trigger": False,
            "rerouting_churn_proportion": 0.3,
        }
        fields.update(overrides)
        return TriggerEvent(**fields)

    def test_valid_event_is_accepted(self):
        assert self.make_event().additional_travel_mins == 0.0

    @pytest.mark.parametrize("probability", [-0.01, 1.01, 2.0])
    def test_probability_outside_unit_interval_is_rejected(self, probability):
        with pytest.raises(ValidationError):
            self.make_event(high_risk_probability=probability)

    @pytest.mark.parametrize("label", ["severe", "HIGH", ""])
    def test_invalid_ml_risk_label_is_rejected(self, label):
        with pytest.raises(ValidationError):
            self.make_event(ml_risk_label=label)

    @pytest.mark.parametrize("state", ["severe", "HIGH", ""])
    def test_invalid_ground_truth_state_is_rejected(self, state):
        with pytest.raises(ValidationError):
            self.make_event(ctmc_ground_truth_state=state)

    def test_timestamp_beyond_shift_end_is_rejected(self):
        with pytest.raises(ValidationError):
            self.make_event(timestamp_mins=481.0)

    def test_timestamp_bound_follows_a_configured_shift_length(self):
        assert (
            self.make_event(
                shift_duration_mins=120, timestamp_mins=119.0
            ).timestamp_mins
            == 119.0
        )
        with pytest.raises(ValidationError):
            self.make_event(shift_duration_mins=120, timestamp_mins=121.0)

    def test_negative_time_since_last_visit_is_rejected(self):
        with pytest.raises(ValidationError):
            self.make_event(time_since_last_visit_mins=-1.0)


class TestSimulationResult:
    def test_valid_result_is_accepted(self):
        assert make_result().scenario == "ai_integrated"

    @pytest.mark.parametrize("utilisation", [-0.1, 1.1])
    def test_utilisation_outside_unit_interval_is_rejected(self, utilisation):
        with pytest.raises(ValidationError):
            make_result(workforce_utilisation=utilisation)

    def test_negative_response_time_is_rejected(self):
        with pytest.raises(ValidationError):
            make_result(mean_response_time_high_risk=-1.0)

    def test_unvisited_exceeding_cohort_size_is_rejected(self):
        with pytest.raises(ValidationError):
            make_result(n_patients=200, unvisited_patients=201)

    def test_unvisited_equal_to_cohort_size_is_accepted(self):
        result = make_result(n_patients=200, unvisited_patients=200)
        assert result.unvisited_patients == 200

    def test_unreached_high_risk_exceeding_cohort_size_is_rejected(self):
        with pytest.raises(ValidationError):
            make_result(n_patients=200, unreached_high_risk_patients=201)

    def test_cohort_check_follows_this_results_own_n_patients(self):
        """The unvisited-patient bound must use this result's own n_patients, not
        a freshly constructed SimulationConfig()."""
        assert (
            make_result(n_patients=50, unvisited_patients=50).unvisited_patients == 50
        )
        with pytest.raises(ValidationError):
            make_result(n_patients=50, unvisited_patients=51)

    def test_unreached_high_risk_may_exceed_unvisited(self):
        """They cover different populations: a patient visited early who later
        deteriorates without a follow-up was visited but not reached."""
        result = make_result(unvisited_patients=0, unreached_high_risk_patients=6)
        assert result.unreached_high_risk_patients == 6

    def test_to_csv_row_returns_every_field(self):
        row = make_result().to_csv_row()
        assert set(row) == set(SimulationResult.model_fields)
        assert row["scenario"] == "ai_integrated"
        assert row["solver_feasible"] is True
