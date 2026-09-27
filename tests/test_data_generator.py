"""Tests for the synthetic data generator."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import (
    BIOMETRIC_BOUNDS,
    BIOMETRIC_MEANS,
    Q_MATRIX,
    STATES,
    TARGET_HIGH_RISK_PREVALENCE,
    SimulationConfig,
)
from src.data_generator import (
    PARAMETER_NAMES,
    BiometricGenerator,
    CTMCModel,
    NEWS2Scorer,
    SyntheticCohort,
)
from src.models import Patient


def make_patient(**overrides) -> Patient:
    """Build a patient sitting exactly on the population baseline, so tests
    isolate state effects from personal baseline offsets."""
    fields = {
        "id": 0,
        "age": 70,
        "diagnosis": "COPD",
        "coords": (1.0, 2.0),
        "personal_baseline_mean": {"spo2": 97.0, "hr": 72.0, "sbp": 125.0},
    }
    fields.update(overrides)
    return Patient(**fields)


def make_cohort(config: SimulationConfig | None = None) -> SyntheticCohort:
    """Assemble a cohort with the production components."""
    return SyntheticCohort(
        config=config if config is not None else SimulationConfig(),
        ctmc_model=CTMCModel(),
        biometric_generator=BiometricGenerator(),
        news2_scorer=NEWS2Scorer(),
    )


class TestCTMCModel:
    def test_exit_rates_match_the_generator_matrix(self):
        model = CTMCModel()
        for state in STATES:
            assert model.total_exit_rate(state) == pytest.approx(
                -Q_MATRIX[state][state]
            )

    @pytest.mark.parametrize("state", STATES)
    def test_sojourn_mean_matches_the_exponential_rate(self, state):
        """Mean dwell time is the reciprocal of the total outgoing rate."""
        model = CTMCModel()
        rng = np.random.default_rng(0)
        samples = [model.sample_sojourn_time_hours(state, rng) for _ in range(20_000)]
        assert np.mean(samples) == pytest.approx(
            1 / model.total_exit_rate(state), rel=0.05
        )

    @pytest.mark.parametrize("state", STATES)
    def test_transition_never_re_enters_the_same_state(self, state):
        model = CTMCModel()
        rng = np.random.default_rng(1)
        assert all(model.sample_next_state(state, rng) != state for _ in range(500))

    def test_no_direct_transition_between_low_and_high(self):
        """Deterioration and recovery must both pass through moderate."""
        model = CTMCModel()
        rng = np.random.default_rng(2)
        assert all(
            model.sample_next_state("low", rng) == "moderate" for _ in range(500)
        )
        assert all(
            model.sample_next_state("high", rng) == "moderate" for _ in range(500)
        )

    def test_state_holds_while_elapsed_is_below_sojourn(self):
        model = CTMCModel()
        patient = make_patient(sojourn_time_hours=10.0)
        transitioned = model.advance_patient_state(
            patient, 5.0, np.random.default_rng(3)
        )
        assert transitioned is False
        assert patient.current_clinical_state == "low"
        assert patient.elapsed_time_in_state_hours == pytest.approx(5 / 60)

    def test_state_transitions_once_elapsed_reaches_sojourn(self):
        model = CTMCModel()
        patient = make_patient(sojourn_time_hours=0.05)  # under one 5-minute step
        transitioned = model.advance_patient_state(
            patient, 5.0, np.random.default_rng(3)
        )
        assert transitioned is True
        assert patient.current_clinical_state == "moderate"
        assert patient.elapsed_time_in_state_hours == 0.0
        assert patient.sojourn_time_hours > 0.0

    def test_same_seed_reproduces_the_same_trajectory(self):
        model = CTMCModel()

        def trajectory(seed: int) -> list[str]:
            rng = np.random.default_rng(seed)
            patient = make_patient(sojourn_time_hours=0.1)
            states = []
            for _ in range(200):
                model.advance_patient_state(patient, 5.0, rng)
                states.append(patient.current_clinical_state)
            return states

        assert trajectory(7) == trajectory(7)
        assert trajectory(7) != trajectory(8)

    def test_stationary_distribution_is_a_valid_distribution(self):
        stationary = CTMCModel().stationary_distribution()
        assert sum(stationary.values()) == pytest.approx(1.0)
        assert all(probability >= 0 for probability in stationary.values())


class TestBiometricGenerator:
    def test_observations_stay_within_physiological_bounds(self):
        """Truncation must hold even for a patient with an extreme baseline."""
        generator = BiometricGenerator()
        rng = np.random.default_rng(4)
        patient = make_patient(
            personal_baseline_mean={"spo2": 130.0, "hr": 250.0, "sbp": 400.0},
            current_clinical_state="high",
        )
        for _ in range(2_000):
            observation = generator.generate_observation(patient, rng)
            for value, parameter in zip(observation, PARAMETER_NAMES, strict=True):
                lower, upper = BIOMETRIC_BOUNDS[parameter]
                assert lower <= value <= upper

    @pytest.mark.parametrize("state", STATES)
    def test_recovered_means_approximate_the_state_means(self, state):
        generator = BiometricGenerator()
        rng = np.random.default_rng(5)
        patient = make_patient(current_clinical_state=state)
        samples = np.array(
            [generator.generate_observation(patient, rng) for _ in range(20_000)]
        )
        assert samples.mean(axis=0) == pytest.approx(BIOMETRIC_MEANS[state], rel=0.02)

    def test_personal_baseline_shifts_observations(self):
        """A patient whose baseline heart rate is high should read higher than
        one on the population mean, in the same clinical state."""
        generator = BiometricGenerator()
        rng = np.random.default_rng(6)
        typical = make_patient(
            personal_baseline_mean={"spo2": 97.0, "hr": 72.0, "sbp": 125.0}
        )
        tachycardic = make_patient(
            personal_baseline_mean={"spo2": 97.0, "hr": 92.0, "sbp": 125.0}
        )
        typical_hr = np.mean(
            [generator.generate_observation(typical, rng)[1] for _ in range(5_000)]
        )
        raised_hr = np.mean(
            [generator.generate_observation(tachycardic, rng)[1] for _ in range(5_000)]
        )
        assert raised_hr - typical_hr == pytest.approx(20.0, abs=1.0)

    def test_sampled_baselines_follow_the_population_distribution(self):
        generator = BiometricGenerator()
        rng = np.random.default_rng(8)
        baselines = [
            generator.sample_personal_baseline_mean(rng) for _ in range(10_000)
        ]
        assert np.mean([b["spo2"] for b in baselines]) == pytest.approx(97.0, abs=0.1)
        assert np.mean([b["hr"] for b in baselines]) == pytest.approx(72.0, abs=0.3)
        assert np.mean([b["sbp"] for b in baselines]) == pytest.approx(125.0, abs=0.4)

    def test_same_seed_reproduces_the_same_observations(self):
        generator = BiometricGenerator()
        patient = make_patient()
        first = generator.generate_observation(patient, np.random.default_rng(9))
        second = generator.generate_observation(patient, np.random.default_rng(9))
        assert first == second


class TestNEWS2Scorer:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (100.0, 0),
            (96.0, 0),
            (95.9, 1),
            (95.0, 1),
            (94.0, 1),
            (93.9, 2),
            (92.0, 2),
            (91.9, 3),
            (70.0, 3),
        ],
    )
    def test_spo2_bands(self, value, expected):
        assert NEWS2Scorer().score_parameter("spo2", value) == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (40.0, 3),
            (40.9, 3),
            (41.0, 1),  # RCP (2017) NEWS2 band
            (45.0, 1),
            (50.9, 1),
            (51.0, 0),
            (90.0, 0),
            (91.0, 1),
            (110.0, 1),
            (111.0, 2),
            (130.0, 2),
            (131.0, 3),
            (200.0, 3),
        ],
    )
    def test_heart_rate_bands(self, value, expected):
        assert NEWS2Scorer().score_parameter("hr", value) == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (60.0, 3),
            (90.9, 3),
            (91.0, 2),
            (100.9, 2),
            (101.0, 1),
            (110.9, 1),
            (111.0, 0),
            (219.0, 0),
            (220.0, 3),
        ],
    )
    def test_systolic_bp_bands(self, value, expected):
        assert NEWS2Scorer().score_parameter("sbp", value) == expected

    def test_non_integer_values_score_without_falling_through(self):
        """Observations are continuous, so values between the published integer
        band edges must still score, not drop to the maximum."""
        scorer = NEWS2Scorer()
        assert scorer.score_parameter("spo2", 95.5) == 1
        assert scorer.score_parameter("hr", 90.5) == 0
        assert scorer.score_parameter("sbp", 110.5) == 1

    def test_aggregate_sums_the_parameter_scores(self):
        scorer = NEWS2Scorer()
        # SpO2 95 scores 1, HR 105 scores 1, SBP 105 scores 1
        assert scorer.aggregate_score([95.0, 105.0, 105.0]) == 3

    @pytest.mark.parametrize(
        ("observation", "aggregate", "expected"),
        [
            ([97.0, 72.0, 125.0], 0, "low"),
            ([95.0, 105.0, 125.0], 2, "low"),
            ([95.0, 105.0, 105.0], 3, "moderate"),
            ([93.0, 105.0, 105.0], 4, "moderate"),
            ([93.0, 120.0, 95.0], 6, "high"),
        ],
    )
    def test_rescaled_aggregate_thresholds(self, observation, aggregate, expected):
        """Rescaled bands: 0-2 low, 3-4 moderate, 5+ high; none of these
        observations scores a red flag, so that rule cannot confound the
        boundaries being tested."""
        scorer = NEWS2Scorer()
        assert scorer.aggregate_score(observation) == aggregate
        assert max(scorer.parameter_scores(observation)) < 3
        assert scorer.assign_label(observation) == expected

    def test_red_flag_forces_high_regardless_of_aggregate(self):
        """A single extreme derangement escalates the label even when the
        composite is unremarkable (RCP, 2017)."""
        scorer = NEWS2Scorer()
        observation = [91.0, 72.0, 125.0]  # SpO2 scores 3, aggregate only 3
        assert scorer.aggregate_score(observation) == 3
        assert scorer.assign_label(observation) == "high"

    def test_state_mean_vitals_label_as_expected(self):
        """Sanity check against the clinical design."""
        scorer = NEWS2Scorer()
        assert scorer.assign_label(BIOMETRIC_MEANS["high"]) == "high"
        assert scorer.assign_label(BIOMETRIC_MEANS["low"]) == "low"


class TestSyntheticCohort:
    def test_cohort_is_initialised_to_specification(self):
        config = SimulationConfig()
        patients = make_cohort(config).initialise_patients(np.random.default_rng(10))
        assert len(patients) == config.n_patients
        assert {patient.id for patient in patients} == set(range(config.n_patients))
        assert all(45 <= patient.age <= 85 for patient in patients)
        assert all(patient.current_clinical_state == "low" for patient in patients)
        assert all(patient.sojourn_time_hours > 0 for patient in patients)
        assert all(
            0 <= coordinate <= config.city_radius_km
            for patient in patients
            for coordinate in patient.coords
        )

    def test_shift_has_one_row_per_patient_per_timestep(self, monkeypatch):
        monkeypatch.setenv("SIM_N_PATIENTS", "20")
        config = SimulationConfig()
        observations = make_cohort(config).generate_shift(seed=11)
        expected_timesteps = config.shift_duration_mins // config.timestep_mins
        assert len(observations) == config.n_patients * expected_timesteps
        assert observations["timestep"].nunique() == expected_timesteps

    def test_same_seed_reproduces_the_shift(self, monkeypatch):
        monkeypatch.setenv("SIM_N_PATIENTS", "20")
        cohort = make_cohort(SimulationConfig())
        first = cohort.generate_shift(seed=12)
        second = cohort.generate_shift(seed=12)
        different = cohort.generate_shift(seed=13)
        assert first.equals(second)
        assert not first["spo2"].equals(different["spo2"])

    def test_training_corpus_spans_all_shifts_with_unique_patient_ids(
        self, monkeypatch
    ):
        monkeypatch.setenv("SIM_N_PATIENTS", "10")
        monkeypatch.setenv("SIM_N_TRAINING_SHIFTS", "3")
        config = SimulationConfig()
        corpus = make_cohort(config).generate_training_corpus()
        assert corpus["seed"].nunique() == 3
        # Patient ids must not repeat across shifts: the split is at patient
        # level, and reused ids would leak the same person across folds.
        assert corpus["patient_id"].nunique() == 30

    def test_high_risk_prevalence_meets_the_design_target(self, monkeypatch):
        """The clinical validation gate: ground truth for the whole study
        depends on this landing inside the target range."""
        monkeypatch.setenv("SIM_N_PATIENTS", "200")
        monkeypatch.setenv("SIM_N_TRAINING_SHIFTS", "2")
        cohort = make_cohort(SimulationConfig())
        prevalence = cohort.validate_prevalence(cohort.generate_training_corpus())
        target_low, target_high = TARGET_HIGH_RISK_PREVALENCE
        assert target_low <= prevalence <= target_high

    def test_moderate_class_is_not_degenerate(self, monkeypatch):
        """The reason the NEWS2 bands were rescaled: macro-F1 across three
        classes is meaningless if one class holds almost nothing."""
        monkeypatch.setenv("SIM_N_PATIENTS", "200")
        monkeypatch.setenv("SIM_N_TRAINING_SHIFTS", "2")
        cohort = make_cohort(SimulationConfig())
        corpus = cohort.generate_training_corpus()
        assert (corpus["risk_label"] == "moderate").mean() > 0.05

    def test_blood_pressure_remains_biphasic_in_generated_data(self, monkeypatch):
        """Biphasic BP must survive generation, not just live in the constants."""
        monkeypatch.setenv("SIM_N_PATIENTS", "100")
        cohort = make_cohort(SimulationConfig())
        observations = cohort.generate_shift(seed=14)
        mean_sbp = observations.groupby("ctmc_state")["sbp"].mean()
        assert mean_sbp["moderate"] > mean_sbp["low"] > mean_sbp["high"]

    def test_future_label_is_the_class_one_lead_time_ahead(self, monkeypatch):
        """The classifier's target: training on the current label would
        reduce the task to recomputing NEWS2 from vitals already in the
        features."""
        monkeypatch.setenv("SIM_N_PATIENTS", "5")
        config = SimulationConfig()
        observations = make_cohort(config).generate_shift(seed=20)
        lead_time_steps = config.prediction_lead_time_mins // config.timestep_mins

        one_patient = observations[observations["patient_id"] == 0].sort_values(
            "timestep"
        )
        expected = one_patient["risk_label"].shift(-lead_time_steps)
        assert one_patient["future_risk_label"].equals(expected)

    def test_observations_without_a_future_label_are_dropped_from_the_corpus(
        self, monkeypatch
    ):
        monkeypatch.setenv("SIM_N_PATIENTS", "5")
        monkeypatch.setenv("SIM_N_TRAINING_SHIFTS", "1")
        config = SimulationConfig()
        corpus = make_cohort(config).generate_training_corpus()

        lead_time_steps = config.prediction_lead_time_mins // config.timestep_mins
        timesteps_per_shift = config.shift_duration_mins // config.timestep_mins
        expected_rows = config.n_patients * (timesteps_per_shift - lead_time_steps)
        assert len(corpus) == expected_rows
        assert corpus["future_risk_label"].notna().all()

    def test_future_label_differs_from_the_current_label_often_enough(
        self, monkeypatch
    ):
        """If the two labels agreed almost always, the forward-looking target
        would be no harder than the current one, and the leak would remain."""
        monkeypatch.setenv("SIM_N_PATIENTS", "200")
        monkeypatch.setenv("SIM_N_TRAINING_SHIFTS", "1")
        corpus = make_cohort(SimulationConfig()).generate_training_corpus()
        changed = (corpus["risk_label"] != corpus["future_risk_label"]).mean()
        assert changed > 0.10

    def test_descriptive_statistics_cover_every_class(self, monkeypatch):
        monkeypatch.setenv("SIM_N_PATIENTS", "100")
        cohort = make_cohort(SimulationConfig())
        statistics = cohort.compute_descriptive_statistics(
            cohort.generate_shift(seed=15)
        )
        assert list(statistics["risk_label"]) == STATES
        for parameter in PARAMETER_NAMES:
            assert f"{parameter}_mean" in statistics.columns
            assert f"{parameter}_std" in statistics.columns
        assert statistics["proportion"].sum() == pytest.approx(1.0)

    def test_prevalence_outside_the_acceptable_range_is_logged(self, caplog):
        """An out-of-range prevalence warns for review rather than silently
        retuning the generator matrix."""
        import pandas as pd

        cohort = make_cohort(SimulationConfig())
        all_high = pd.DataFrame({"risk_label": ["high"] * 100})
        with caplog.at_level("WARNING"):
            prevalence = cohort.validate_prevalence(all_high)
        assert prevalence == 1.0
        assert "outside the acceptable range" in caplog.text
