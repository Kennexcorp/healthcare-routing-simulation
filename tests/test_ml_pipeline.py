"""Tests for the machine learning pipeline: feature engineering, patient-level
splits, risk classification, and evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.config import (
    BASELINE_DISTRIBUTIONS,
    DIAGNOSIS_CATEGORIES,
    HIGH_RISK_CLASS_INDEX,
    RISK_CLASS_ORDER,
    SimulationConfig,
)
from src.ml_pipeline import (
    LABEL_COLUMN,
    FeatureEngineer,
    ModelEvaluator,
    ModelProfileMismatchError,
    RiskClassifier,
    create_patient_level_splits,
    encode_labels,
)

EXPECTED_FEATURE_COUNT = 28


def make_observations(n_patients: int = 4, n_timesteps: int = 8) -> pd.DataFrame:
    """Build a small deterministic observation frame in the generator's shape."""
    rng = np.random.default_rng(0)
    records = []
    for patient_id in range(n_patients):
        for timestep in range(n_timesteps):
            records.append(
                {
                    "patient_id": patient_id,
                    "timestep": timestep,
                    "spo2": 97.0 - timestep * 0.5,
                    "hr": 70.0 + timestep * 2.0,
                    "sbp": 125.0 + rng.normal(0, 1),
                    "baseline_spo2": 97.0,
                    "baseline_hr": 72.0,
                    "baseline_sbp": 125.0,
                    "time_since_last_visit_mins": timestep * 5.0,
                    "age": 60 + patient_id,
                    "diagnosis": DIAGNOSIS_CATEGORIES[
                        patient_id % len(DIAGNOSIS_CATEGORIES)
                    ],
                    "risk_label": RISK_CLASS_ORDER[timestep % len(RISK_CLASS_ORDER)],
                    "future_risk_label": RISK_CLASS_ORDER[
                        (timestep + 1) % len(RISK_CLASS_ORDER)
                    ],
                }
            )
    return pd.DataFrame.from_records(records)


class TestFeatureEngineer:
    def test_produces_the_specified_feature_count(self):
        features = FeatureEngineer().fit_transform(make_observations())
        assert features.shape[1] == EXPECTED_FEATURE_COUNT
        assert len(FeatureEngineer().get_feature_names_out()) == EXPECTED_FEATURE_COUNT

    def test_one_row_per_observation(self):
        observations = make_observations(n_patients=3, n_timesteps=5)
        assert FeatureEngineer().fit_transform(observations).shape[0] == len(
            observations
        )

    def test_missing_columns_raise(self):
        observations = make_observations().drop(columns=["baseline_hr"])
        with pytest.raises(ValueError, match="missing columns"):
            FeatureEngineer().fit_transform(observations)

    def test_cold_start_defaults_at_first_observation(self):
        """At a patient's first observation there is no history: deltas and
        rolling deviation are zero, and the rolling mean, min and max equal the
        current value."""
        observations = make_observations(n_patients=1, n_timesteps=4)
        features = pd.DataFrame(
            FeatureEngineer().fit_transform(observations),
            columns=FeatureEngineer().get_feature_names_out(),
        )
        first = features.iloc[0]
        for parameter in ("spo2", "hr", "sbp"):
            assert first[f"delta_{parameter}"] == 0.0
            assert first[f"rolling_std_{parameter}"] == 0.0
            assert first[f"rolling_mean_{parameter}"] == first[parameter]
            assert first[f"rolling_min_{parameter}"] == first[parameter]
            assert first[f"rolling_max_{parameter}"] == first[parameter]

    def test_delta_is_the_change_since_the_previous_observation(self):
        observations = make_observations(n_patients=1, n_timesteps=4)
        features = pd.DataFrame(
            FeatureEngineer().fit_transform(observations),
            columns=FeatureEngineer().get_feature_names_out(),
        )
        # spo2 falls by 0.5 and hr rises by 2.0 at every timestep by construction
        assert features["delta_spo2"].iloc[1] == pytest.approx(-0.5)
        assert features["delta_hr"].iloc[1] == pytest.approx(2.0)

    def test_zscore_measures_deviation_from_the_personal_baseline(self):
        observations = make_observations(n_patients=1, n_timesteps=1)
        observations.loc[0, "hr"] = 88.0
        observations.loc[0, "baseline_hr"] = 72.0
        features = pd.DataFrame(
            FeatureEngineer().fit_transform(observations),
            columns=FeatureEngineer().get_feature_names_out(),
        )
        expected = (88.0 - 72.0) / BASELINE_DISTRIBUTIONS["hr"][1]
        assert features["zscore_hr"].iloc[0] == pytest.approx(expected)

    def test_diagnosis_is_one_hot_encoded(self):
        observations = make_observations(n_patients=5, n_timesteps=1)
        features = pd.DataFrame(
            FeatureEngineer().fit_transform(observations),
            columns=FeatureEngineer().get_feature_names_out(),
        )
        dummies = features[
            [f"diagnosis_{category}" for category in DIAGNOSIS_CATEGORIES]
        ]
        assert (dummies.sum(axis=1) == 1.0).all()

    def test_features_do_not_leak_between_patients(self):
        """A patient's first observation must show cold-start values even when
        another patient's rows precede it, or history would bleed across
        patients."""
        observations = make_observations(n_patients=2, n_timesteps=3)
        features = pd.DataFrame(
            FeatureEngineer().fit_transform(observations),
            columns=FeatureEngineer().get_feature_names_out(),
        )
        first_row_of_second_patient = features.iloc[3]
        assert first_row_of_second_patient["delta_spo2"] == 0.0

    def test_row_order_is_preserved_when_input_is_shuffled(self):
        """Cross-validation hands the transformer arbitrary row subsets, so
        output rows must align with input rows, not with sorted order."""
        observations = make_observations(n_patients=3, n_timesteps=4)
        shuffled = observations.sample(frac=1.0, random_state=1)
        ordered_features = FeatureEngineer().fit_transform(observations)
        shuffled_features = FeatureEngineer().fit_transform(shuffled)
        realigned = shuffled_features[np.argsort(shuffled.index.to_numpy())]
        assert np.allclose(ordered_features, realigned)


class TestLabelEncoding:
    def test_labels_encode_in_clinical_severity_order(self):
        encoded = encode_labels(pd.Series(["low", "moderate", "high"]))
        assert list(encoded) == [0, 1, 2]

    def test_high_risk_index_matches_the_encoding(self):
        """Inference reads the high-risk probability by column index, so this
        constant must track the label order."""
        assert RISK_CLASS_ORDER[HIGH_RISK_CLASS_INDEX] == "high"


class TestPatientLevelSplits:
    def test_partitions_cover_every_observation_exactly_once(self):
        observations = make_observations(n_patients=20)
        masks = create_patient_level_splits(observations)
        total = (
            masks["train"].astype(int)
            + masks["calibration"].astype(int)
            + masks["test"].astype(int)
        )
        assert (total == 1).all()

    def test_no_patient_appears_in_more_than_one_partition(self):
        """The defining requirement: an observation-level split would let the
        model recognise individuals and inflate every reported metric."""
        observations = make_observations(n_patients=20)
        masks = create_patient_level_splits(observations)
        patient_sets = {
            name: set(observations.loc[mask, "patient_id"])
            for name, mask in masks.items()
        }
        assert not patient_sets["train"] & patient_sets["test"]
        assert not patient_sets["train"] & patient_sets["calibration"]
        assert not patient_sets["calibration"] & patient_sets["test"]

    def test_test_set_holds_the_requested_patient_proportion(self):
        observations = make_observations(n_patients=20)
        masks = create_patient_level_splits(observations, test_size=0.2)
        assert observations.loc[masks["test"], "patient_id"].nunique() == 4

    def test_split_is_reproducible(self):
        observations = make_observations(n_patients=20)
        first = create_patient_level_splits(observations, seed=7)
        second = create_patient_level_splits(observations, seed=7)
        different = create_patient_level_splits(observations, seed=8)
        assert np.array_equal(first["test"], second["test"])
        assert not np.array_equal(first["test"], different["test"])


class TestRiskClassifier:
    def test_all_three_specified_classifiers_are_candidates(self):
        candidates = RiskClassifier(SimulationConfig()).build_candidates()
        assert set(candidates) == {"logistic_regression", "random_forest", "xgboost"}
        for pipeline, _ in candidates.values():
            assert isinstance(pipeline, Pipeline)
            assert isinstance(pipeline.named_steps["features"], FeatureEngineer)

    def test_inference_without_a_model_raises(self):
        with pytest.raises(ValueError, match="No calibrated model"):
            RiskClassifier(SimulationConfig()).run_inference(make_observations())

    def test_saving_without_a_model_raises(self, tmp_path):
        with pytest.raises(ValueError, match="nothing to save"):
            RiskClassifier(SimulationConfig()).save(tmp_path / "model.joblib")

    def test_loading_a_missing_model_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Run the train stage"):
            RiskClassifier(SimulationConfig()).load(tmp_path / "absent.joblib")

    def test_inference_applies_the_high_risk_threshold(self):
        """The AND-gate trades sensitivity against false triggers by moving
        theta, so the threshold must govern the label, not argmax."""

        class StubModel:
            def predict_proba(self, features):
                # third column is the high-risk probability
                return np.array([[0.1, 0.1, 0.8], [0.3, 0.1, 0.6]])[: len(features)]

        classifier = RiskClassifier(SimulationConfig())
        classifier.set_calibrated_model("stub", StubModel())
        observations = make_observations(n_patients=1, n_timesteps=2)

        labels_strict, probabilities = classifier.run_inference(
            observations, threshold=0.7
        )
        assert list(labels_strict) == ["high", "low"]
        assert list(probabilities) == [0.8, 0.6]

        labels_lenient, _ = classifier.run_inference(observations, threshold=0.5)
        assert list(labels_lenient) == ["high", "high"]

    def test_save_and_load_round_trip(self, tmp_path):
        from sklearn.dummy import DummyClassifier

        model = DummyClassifier(strategy="prior").fit(
            np.zeros((6, 2)), [0, 1, 2, 0, 1, 2]
        )
        classifier = RiskClassifier(SimulationConfig())
        classifier.set_calibrated_model("dummy", model)
        destination = classifier.save(tmp_path / "model.joblib")

        reloaded = RiskClassifier(SimulationConfig())
        reloaded.load(destination)
        assert reloaded.selected_name == "dummy"


class TestModelEvaluator:
    def test_reports_every_specified_metric(self):
        class PerfectModel:
            def predict(self, features):
                return np.array([0, 1, 2, 0])[: len(features)]

            def predict_proba(self, features):
                return np.array(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
                )[: len(features)]

        labels = np.array([0, 1, 2, 0])
        metrics = ModelEvaluator().evaluate(
            PerfectModel(), np.zeros((4, 1)), labels, "perfect"
        )

        assert metrics["classifier"] == "perfect"
        assert metrics["macro_f1_score"] == pytest.approx(1.0)
        assert metrics["brier_score"] == pytest.approx(0.0)
        assert metrics["high_risk_recall"] == pytest.approx(1.0)
        assert metrics["high_risk_precision"] == pytest.approx(1.0)
        for risk_class in RISK_CLASS_ORDER:
            assert f"recall_{risk_class}" in metrics
            assert f"precision_{risk_class}" in metrics

    def test_high_risk_precision_penalises_false_positives(self):
        """A model that over-predicts high risk should score low precision even
        with perfect recall on that class."""

        class OverTriggersHigh:
            def predict(self, features):
                return np.full(len(features), HIGH_RISK_CLASS_INDEX)

            def predict_proba(self, features):
                return np.tile([0.0, 0.0, 1.0], (len(features), 1))

        metrics = ModelEvaluator().evaluate(
            OverTriggersHigh(), np.zeros((4, 1)), np.array([0, 0, 1, 2]), "over"
        )
        assert metrics["high_risk_recall"] == pytest.approx(1.0)
        assert metrics["high_risk_precision"] == pytest.approx(0.25)

    def test_accuracy_is_not_reported(self):
        """Accuracy is misleading at roughly 12% prevalence: predicting 'low'
        everywhere scores about 81%."""

        class Model:
            def predict(self, features):
                return np.zeros(len(features), dtype=int)

            def predict_proba(self, features):
                return np.tile([0.9, 0.05, 0.05], (len(features), 1))

        metrics = ModelEvaluator().evaluate(
            Model(), np.zeros((4, 1)), np.array([0, 0, 1, 2]), "majority"
        )
        assert "accuracy" not in metrics
        assert metrics["high_risk_recall"] == 0.0


class TestModelProfileGuard:
    """Loading a model trained under a different profile must fail loudly, not
    produce results that look valid and mean nothing."""

    def make_saved_model(self, tmp_path, config):
        classifier = RiskClassifier(config)
        observations = make_observations(n_patients=6, n_timesteps=4)
        labels = encode_labels(observations[LABEL_COLUMN])
        pipeline, _ = classifier.build_candidates()["random_forest"]
        pipeline.fit(observations, labels)
        classifier.set_calibrated_model(
            "random_forest", classifier.calibrate(pipeline, observations, labels)
        )
        return classifier.save(tmp_path / "model.joblib")

    def test_a_model_trained_under_the_same_profile_loads(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SIM_N_PATIENTS", "20")
        destination = self.make_saved_model(tmp_path, SimulationConfig())
        RiskClassifier(SimulationConfig()).load(destination)

    def test_a_model_trained_under_a_smaller_profile_is_refused(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("SIM_N_PATIENTS", "20")
        destination = self.make_saved_model(tmp_path, SimulationConfig())

        monkeypatch.setenv("SIM_N_PATIENTS", "200")
        with pytest.raises(ModelProfileMismatchError, match="trained under"):
            RiskClassifier(SimulationConfig()).load(destination)

    def test_a_missing_model_names_the_stage_that_creates_it(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="train stage"):
            RiskClassifier(SimulationConfig()).load(tmp_path / "absent.joblib")
