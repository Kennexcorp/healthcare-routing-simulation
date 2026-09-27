"""Tests for SimulationConfig and the clinical design constants."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config import (
    BASELINE_DISTRIBUTIONS,
    BIOMETRIC_BOUNDS,
    BIOMETRIC_MEANS,
    BIOMETRIC_STDS,
    DEPOT_COORDS,
    DIAGNOSIS_CATEGORIES,
    INITIAL_TIME_WINDOWS,
    PRIORITY_WEIGHTS,
    Q_MATRIX,
    STATES,
    SimulationConfig,
)

SPO2, HR, SBP = 0, 1, 2


class TestSimulationConfig:
    def test_defaults_match_the_specification(self):
        config = SimulationConfig()
        assert config.n_patients == 160
        assert config.n_workers == 10
        assert config.shift_duration_mins == 480
        assert config.timestep_mins == 5
        assert config.default_theta == 0.70
        assert config.default_tau == 60

    def test_shift_divides_evenly_into_timesteps(self):
        """The simulation loop assumes a whole number of timesteps per shift."""
        config = SimulationConfig()
        assert config.shift_duration_mins % config.timestep_mins == 0

    def test_environment_variable_overrides_a_field(self, monkeypatch):
        monkeypatch.setenv("SIM_DEFAULT_THETA", "0.60")
        assert SimulationConfig().default_theta == 0.60

    def test_smoke_profile_overrides_apply(self, monkeypatch):
        monkeypatch.setenv("SIM_N_PATIENTS", "20")
        monkeypatch.setenv("SIM_N_WORKERS", "2")
        monkeypatch.setenv("SIM_SEARCH_N_ITER", "5")
        config = SimulationConfig()
        assert (config.n_patients, config.n_workers, config.search_n_iter) == (20, 2, 5)

    @pytest.mark.parametrize(
        ("variable", "value"),
        [
            ("SIM_DEFAULT_THETA", "1.5"),
            ("SIM_DEFAULT_THETA", "-0.1"),
            ("SIM_N_PATIENTS", "0"),
            ("SIM_N_WORKERS", "0"),
            ("SIM_TIMESTEP_MINS", "0"),
            ("SIM_CITY_RADIUS_KM", "0"),
            ("SIM_TRAVEL_SPEED_KMH", "0"),
            ("SIM_CONFIDENCE_LEVEL", "1.5"),
            ("SIM_N_TRAINING_SHIFTS", "0"),
            ("SIM_SEARCH_N_ITER", "0"),
            ("SIM_SENSITIVITY_REPS", "1"),
        ],
    )
    def test_out_of_range_override_is_rejected(self, monkeypatch, variable, value):
        monkeypatch.setenv(variable, value)
        with pytest.raises(ValidationError):
            SimulationConfig()

    def test_theta_sensitivity_range_matches_the_specification(self):
        config = SimulationConfig()
        assert config.theta_values == [0.5, 0.6, 0.7, 0.8, 0.9]
        assert config.sensitivity_reps == 10

    def test_tau_sensitivity_range_matches_the_specification(self):
        """RCP (2020) hospital assessment times for NEWS2 7+ and 5 to 6, and
        the NHS England two-hour standard."""
        config = SimulationConfig()
        assert config.tau_values == [30, 60, 120]

    def test_sensitivity_reps_is_independent_of_the_main_replication_count(
        self, monkeypatch
    ):
        """Seven settings at the main count would multiply the whole run by seven."""
        monkeypatch.setenv("SIM_INITIAL_BATCH_REPS", "40")
        assert SimulationConfig().sensitivity_reps == 10

    def test_training_seeds_do_not_collide_with_experiment_seeds(self):
        """Training seeds must stay clear of the replication seed range, so no
        patient trajectory is both trained on and evaluated in the simulation."""
        config = SimulationConfig()
        assert config.training_seed_base > config.initial_batch_reps * 100


class TestGeneratorMatrix:
    def test_every_row_sums_to_zero(self):
        """Defining property of a CTMC generator matrix."""
        for state in STATES:
            assert sum(Q_MATRIX[state].values()) == pytest.approx(0.0, abs=1e-12)

    def test_diagonal_entries_are_negative(self):
        for state in STATES:
            assert Q_MATRIX[state][state] < 0

    def test_off_diagonal_entries_are_non_negative(self):
        for from_state in STATES:
            for to_state in STATES:
                if from_state != to_state:
                    assert Q_MATRIX[from_state][to_state] >= 0

    def test_no_direct_transition_between_low_and_high(self):
        """Deterioration and recovery both pass through the moderate state."""
        assert Q_MATRIX["low"]["high"] == 0.0
        assert Q_MATRIX["high"]["low"] == 0.0

    def test_matrix_covers_exactly_the_declared_states(self):
        assert set(Q_MATRIX) == set(STATES)
        for state in STATES:
            assert set(Q_MATRIX[state]) == set(STATES)


class TestBiometricConstants:
    def test_blood_pressure_is_biphasic(self):
        """SBP rises then falls as risk worsens, not a monotonic decline
        (RCP, 2017)."""
        assert (
            BIOMETRIC_MEANS["moderate"][SBP]
            > BIOMETRIC_MEANS["low"][SBP]
            > BIOMETRIC_MEANS["high"][SBP]
        )

    def test_oxygen_saturation_falls_monotonically_with_risk(self):
        assert (
            BIOMETRIC_MEANS["low"][SPO2]
            > BIOMETRIC_MEANS["moderate"][SPO2]
            > BIOMETRIC_MEANS["high"][SPO2]
        )

    def test_heart_rate_rises_monotonically_with_risk(self):
        assert (
            BIOMETRIC_MEANS["low"][HR]
            < BIOMETRIC_MEANS["moderate"][HR]
            < BIOMETRIC_MEANS["high"][HR]
        )

    def test_every_state_has_three_means_and_three_deviations(self):
        for state in STATES:
            assert len(BIOMETRIC_MEANS[state]) == 3
            assert len(BIOMETRIC_STDS[state]) == 3

    def test_standard_deviations_are_positive(self):
        for state in STATES:
            assert all(deviation > 0 for deviation in BIOMETRIC_STDS[state])

    def test_state_means_lie_within_physiological_bounds(self):
        bounds = [
            BIOMETRIC_BOUNDS["spo2"],
            BIOMETRIC_BOUNDS["hr"],
            BIOMETRIC_BOUNDS["sbp"],
        ]
        for state in STATES:
            for mean, (lower, upper) in zip(
                BIOMETRIC_MEANS[state], bounds, strict=True
            ):
                assert lower <= mean <= upper

    def test_baseline_distributions_cover_all_three_parameters(self):
        assert set(BASELINE_DISTRIBUTIONS) == {"spo2", "hr", "sbp"}
        for mean, deviation in BASELINE_DISTRIBUTIONS.values():
            assert mean > 0
            assert deviation > 0


class TestRoutingConstants:
    def test_priority_weights_rank_by_clinical_urgency(self):
        assert (
            PRIORITY_WEIGHTS["high"]
            > PRIORITY_WEIGHTS["moderate"]
            > PRIORITY_WEIGHTS["low"]
        )

    def test_time_windows_tighten_as_risk_rises(self):
        assert (
            INITIAL_TIME_WINDOWS["high"][1]
            < INITIAL_TIME_WINDOWS["moderate"][1]
            < INITIAL_TIME_WINDOWS["low"][1]
        )

    def test_time_windows_start_at_shift_start_and_end_within_the_shift(self):
        config = SimulationConfig()
        for earliest, latest in INITIAL_TIME_WINDOWS.values():
            assert earliest == 0
            assert latest <= config.shift_duration_mins

    def test_depot_sits_at_the_centre_of_the_city_square(self):
        config = SimulationConfig()
        centre = config.city_radius_km / 2
        assert DEPOT_COORDS == (centre, centre)

    def test_diagnosis_categories_match_the_study_scope(self):
        assert DIAGNOSIS_CATEGORIES == [
            "COPD",
            "HF",
            "Diabetes",
            "Hypertension",
            "Other",
        ]
