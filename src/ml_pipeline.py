"""Supervised risk classification pipeline: feature construction, model selection, calibration and evaluation, trained once and reused across every replication and scenario, with splits kept at patient level throughout."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.config import (
    BASELINE_DISTRIBUTIONS,
    CALIBRATION_SET_PROPORTION,
    DIAGNOSIS_CATEGORIES,
    HIGH_RISK_CLASS_INDEX,
    N_CROSS_VALIDATION_FOLDS,
    RISK_CLASS_ORDER,
    ROLLING_WINDOW_SIZE,
    SPLIT_SEED,
    TEST_SET_PROPORTION,
    SimulationConfig,
)
from src.data_generator import PARAMETER_NAMES

logger = logging.getLogger(__name__)

PATIENT_ID_COLUMN = "patient_id"
TIMESTEP_COLUMN = "timestep"

# Trained on `future_risk_label`, the class one lead time ahead; `risk_label`
# stays the ground truth for response time and trigger validation.
LABEL_COLUMN = "future_risk_label"
CURRENT_LABEL_COLUMN = "risk_label"

REQUIRED_COLUMNS = (
    PATIENT_ID_COLUMN,
    TIMESTEP_COLUMN,
    *PARAMETER_NAMES,
    *(f"baseline_{parameter}" for parameter in PARAMETER_NAMES),
    "time_since_last_visit_mins",
    "age",
    "diagnosis",
)


class ModelProfileMismatchError(Exception):
    """Raised when a persisted model was trained under a different configuration."""


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Builds the 28-feature vector from raw observations, stateless per patient so it can sit inside a cross-validated pipeline without leaking across folds."""

    def __init__(self, rolling_window_size: int = ROLLING_WINDOW_SIZE) -> None:
        """Args:
        rolling_window_size: Observations in the trailing window used for the
            rolling statistics.
        """
        self.rolling_window_size = rolling_window_size

    def fit(self, X: pd.DataFrame, y=None) -> FeatureEngineer:
        """No fitting required; present to satisfy the transformer interface.

        Args:
            X: Observation frame.
            y: Ignored.

        Returns:
            Self.
        """
        return self

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Return the ordered feature names.

        Args:
            input_features: Ignored.

        Returns:
            Array of 28 feature names.
        """
        names = list(PARAMETER_NAMES)
        for prefix in (
            "delta",
            "rolling_mean",
            "rolling_min",
            "rolling_max",
            "rolling_std",
            "zscore",
        ):
            names.extend(f"{prefix}_{parameter}" for parameter in PARAMETER_NAMES)
        names.append("time_since_last_visit_mins")
        names.append("age")
        names.extend(f"diagnosis_{category}" for category in DIAGNOSIS_CATEGORIES)
        return np.array(names)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Construct the feature matrix, with cold-start values at a patient's first observation (zero deltas and rolling SD, rolling mean/min/max equal to the current value) produced by a trailing window with a minimum of one period.

        Args:
            X: Observation frame carrying `REQUIRED_COLUMNS`.

        Returns:
            Feature matrix of shape (n_observations, 28), rows in input order.

        Raises:
            ValueError: If required columns are missing.
        """
        missing = set(REQUIRED_COLUMNS) - set(X.columns)
        if missing:
            raise ValueError(f"Observation frame is missing columns: {sorted(missing)}")

        parameters = list(PARAMETER_NAMES)
        frame = X.reset_index(drop=True).copy()
        frame["_input_position"] = np.arange(len(frame))
        frame = frame.sort_values([PATIENT_ID_COLUMN, TIMESTEP_COLUMN]).reset_index(
            drop=True
        )

        grouped = frame.groupby(PATIENT_ID_COLUMN)[parameters]
        rolling = grouped.rolling(self.rolling_window_size, min_periods=1)

        blocks = [frame[parameters]]
        blocks.append(grouped.diff().fillna(0.0))
        for statistic in ("mean", "min", "max", "std"):
            block = getattr(rolling, statistic)().reset_index(level=0, drop=True)
            # A single-observation window has undefined standard deviation.
            blocks.append(block.fillna(0.0))

        # Z-score against this patient's own baseline, not the population's.
        zscores = pd.DataFrame(
            {
                parameter: (frame[parameter] - frame[f"baseline_{parameter}"])
                / BASELINE_DISTRIBUTIONS[parameter][1]
                for parameter in parameters
            }
        )
        blocks.append(zscores)

        blocks.append(frame[["time_since_last_visit_mins", "age"]])
        diagnosis = pd.DataFrame(
            {
                f"diagnosis_{category}": (frame["diagnosis"] == category).astype(float)
                for category in DIAGNOSIS_CATEGORIES
            }
        )
        blocks.append(diagnosis)

        features = pd.concat([block.reset_index(drop=True) for block in blocks], axis=1)
        restored_order = np.argsort(frame["_input_position"].to_numpy())
        return features.to_numpy(dtype=float)[restored_order]


class BalancedXGBClassifier(XGBClassifier):
    """XGBoost with cost-sensitive weights recomputed on every fit: `scale_pos_weight` is binary-only, so balanced per-sample weights are the multiclass equivalent, computed inside `fit` so each cross-validation fold is weighted by its own training distribution."""

    def fit(self, X, y, **kwargs) -> BalancedXGBClassifier:
        """Fit with balanced sample weights derived from this fold's labels.

        Args:
            X: Feature matrix.
            y: Encoded labels.
            **kwargs: Passed through to XGBoost.

        Returns:
            Self.
        """
        sample_weights = compute_sample_weight("balanced", y)
        return super().fit(X, y, sample_weight=sample_weights, **kwargs)


def encode_labels(labels: pd.Series) -> np.ndarray:
    """Map risk labels to integers in clinical severity order.

    Order matters: inference reads the high-risk probability from column index
    2 of `predict_proba`, so 'high' must encode to 2.

    Args:
        labels: Risk label strings.

    Returns:
        Integer-encoded labels.
    """
    encoding = {label: index for index, label in enumerate(RISK_CLASS_ORDER)}
    return labels.map(encoding).to_numpy(dtype=int)


def create_patient_level_splits(
    observations: pd.DataFrame,
    test_size: float = TEST_SET_PROPORTION,
    calibration_size: float = CALIBRATION_SET_PROPORTION,
    seed: int = SPLIT_SEED,
) -> dict[str, np.ndarray]:
    """Partition observations into train, calibration and test sets by patient; calibration data must be unseen, or Platt scaling reports optimistic reliability from its own training predictions.

    Args:
        observations: Full corpus.
        test_size: Proportion of patients held out for final reporting.
        calibration_size: Proportion of the remaining patients used to fit the
            probability calibrator.
        seed: Seed controlling the patient partition.

    Returns:
        Boolean masks keyed 'train', 'calibration' and 'test'.
    """
    rng = np.random.default_rng(seed)
    patients = np.unique(observations[PATIENT_ID_COLUMN])
    shuffled = rng.permutation(patients)

    n_test = int(len(patients) * test_size)
    n_calibration = int((len(patients) - n_test) * calibration_size)
    test_patients = shuffled[:n_test]
    calibration_patients = shuffled[n_test : n_test + n_calibration]

    groups = observations[PATIENT_ID_COLUMN].to_numpy()
    test_mask = np.isin(groups, test_patients)
    calibration_mask = np.isin(groups, calibration_patients)
    return {
        "train": ~test_mask & ~calibration_mask,
        "calibration": calibration_mask,
        "test": test_mask,
    }


class RiskClassifier:
    """Model selection, calibration and inference for patient risk prediction."""

    def __init__(self, config: SimulationConfig) -> None:
        """Args:
        config: Supplies the randomised search budget.
        """
        self._config = config
        self._calibrated_model: CalibratedClassifierCV | None = None
        self._selected_name: str | None = None

    @property
    def selected_name(self) -> str:
        """Name of the classifier selected for the simulation.

        Raises:
            ValueError: If no model has been selected yet.
        """
        if self._selected_name is None:
            raise ValueError("No classifier has been selected; call select_best first.")
        return self._selected_name

    def build_candidates(self) -> dict[str, tuple[Pipeline, dict]]:
        """Assemble the three candidate pipelines with their search spaces; logistic regression is scaled since its penalty is scale-sensitive, unlike the tree ensembles.

        Returns:
            Classifier name mapped to (pipeline, parameter distribution).
        """
        return {
            "logistic_regression": (
                Pipeline(
                    [
                        ("features", FeatureEngineer()),
                        ("scaler", StandardScaler()),
                        (
                            "classifier",
                            # max_iter raised: saga needs more than 1000 iterations
                            # to converge on 28 standardised features.
                            # random_state pins saga's shuffling so coefficients
                            # don't drift between runs.
                            LogisticRegression(
                                class_weight="balanced",
                                max_iter=5000,
                                random_state=SPLIT_SEED,
                            ),
                        ),
                    ]
                ),
                {
                    "classifier__C": [0.001, 0.01, 0.1, 1, 10, 100],
                    # l1/l2 penalties expressed through l1_ratio (0.0 ridge,
                    # 1.0 lasso); the deprecated `penalty` argument is ignored
                    # whenever l1_ratio is set.
                    "classifier__l1_ratio": [0.0, 1.0],
                    # scikit-learn rejects liblinear for three-class problems,
                    # and only saga supports an l1_ratio.
                    "classifier__solver": ["saga"],
                },
            ),
            "random_forest": (
                Pipeline(
                    [
                        ("features", FeatureEngineer()),
                        (
                            "classifier",
                            RandomForestClassifier(
                                class_weight="balanced", random_state=SPLIT_SEED
                            ),
                        ),
                    ]
                ),
                {
                    "classifier__n_estimators": [100, 300, 500],
                    "classifier__max_depth": [5, 10, 20, None],
                    "classifier__min_samples_split": [2, 5, 10],
                    "classifier__min_samples_leaf": [1, 2, 4],
                    "classifier__max_features": ["sqrt", "log2"],
                },
            ),
            "xgboost": (
                Pipeline(
                    [
                        ("features", FeatureEngineer()),
                        (
                            "classifier",
                            BalancedXGBClassifier(
                                objective="multi:softprob",
                                num_class=len(RISK_CLASS_ORDER),
                                random_state=SPLIT_SEED,
                                tree_method="hist",
                            ),
                        ),
                    ]
                ),
                {
                    "classifier__n_estimators": [100, 300, 500, 1000],
                    "classifier__max_depth": [3, 4, 5, 6, 8],
                    "classifier__learning_rate": [0.01, 0.05, 0.1, 0.3],
                    "classifier__subsample": [0.6, 0.8, 1.0],
                    "classifier__colsample_bytree": [0.6, 0.8, 1.0],
                },
            ),
        }

    def search(
        self,
        name: str,
        pipeline: Pipeline,
        parameter_distribution: dict,
        features: pd.DataFrame,
        labels: np.ndarray,
        groups: np.ndarray,
    ) -> RandomizedSearchCV:
        """Run randomised hyperparameter search for one classifier, with folds stratified by risk class and grouped by patient so no patient splits across folds.

        Args:
            name: Classifier name, for logging.
            pipeline: Candidate pipeline.
            parameter_distribution: Search space.
            features: Training observations.
            labels: Encoded training labels.
            groups: Patient ids aligned to `features`.

        Returns:
            The fitted search object.
        """
        folds = StratifiedGroupKFold(n_splits=N_CROSS_VALIDATION_FOLDS)
        search = RandomizedSearchCV(
            pipeline,
            parameter_distribution,
            n_iter=self._config.search_n_iter,
            scoring="f1_macro",
            cv=folds,
            random_state=SPLIT_SEED,
            n_jobs=-1,
            refit=True,
        )
        logger.info("Searching %s over %d iterations", name, self._config.search_n_iter)
        search.fit(features, labels, groups=groups)
        logger.info("%s best cross-validated macro-F1 %.4f", name, search.best_score_)
        return search

    def calibrate(
        self, fitted_pipeline: Pipeline, features: pd.DataFrame, labels: np.ndarray
    ) -> CalibratedClassifierCV:
        """Apply Platt scaling to an already fitted pipeline: cost-sensitive weighting distorts probabilities that the AND-gate thresholds, so calibration matters here (van den Goorbergh et al., 2022).

        The estimator is frozen rather than passed with cv='prefit', which current scikit-learn no longer accepts.

        Args:
            fitted_pipeline: Pipeline already fitted on the training split.
            features: Calibration observations, disjoint from training.
            labels: Encoded calibration labels.

        Returns:
            The fitted calibrated classifier.
        """
        calibrated = CalibratedClassifierCV(
            FrozenEstimator(fitted_pipeline), method="sigmoid"
        )
        calibrated.fit(features, labels)
        return calibrated

    def set_calibrated_model(
        self, name: str, calibrated_model: CalibratedClassifierCV
    ) -> None:
        """Record the selected model used for all simulation inference.

        Args:
            name: Classifier name.
            calibrated_model: Fitted calibrated classifier.
        """
        self._selected_name = name
        self._calibrated_model = calibrated_model

    def run_inference(
        self, features: pd.DataFrame, threshold: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict risk labels and high-risk probabilities using `predict_proba` with a configurable threshold, so the AND-gate can trade sensitivity against false triggers by moving theta without retraining.

        Args:
            features: Observations to score.
            threshold: High-risk probability threshold. Defaults to the
                configured theta.

        Returns:
            Predicted labels as strings, and high-risk probabilities.

        Raises:
            ValueError: If no calibrated model has been set.
        """
        if self._calibrated_model is None:
            raise ValueError("No calibrated model set; train or load one first.")
        high_risk_threshold = (
            self._config.default_theta if threshold is None else threshold
        )

        probabilities = self._calibrated_model.predict_proba(features)
        high_risk_probability = probabilities[:, HIGH_RISK_CLASS_INDEX]
        moderate_probability = probabilities[:, RISK_CLASS_ORDER.index("moderate")]
        predicted_risk_label = np.where(
            high_risk_probability >= high_risk_threshold,
            "high",
            np.where(moderate_probability >= 0.5, "moderate", "low"),
        )
        return predicted_risk_label, high_risk_probability

    def save(self, destination: Path) -> Path:
        """Persist the calibrated model for use by the simulation stage.

        Args:
            destination: File path to write.

        Returns:
            The path written.

        Raises:
            ValueError: If no calibrated model has been set.
        """
        if self._calibrated_model is None:
            raise ValueError("No calibrated model set; nothing to save.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "name": self._selected_name,
                "model": self._calibrated_model,
                "training_profile": self.training_profile(self._config),
            },
            destination,
        )
        logger.info("Saved calibrated %s to %s", self._selected_name, destination)
        return destination

    @staticmethod
    def training_profile(config: SimulationConfig) -> dict[str, int]:
        """Summarise the configuration a model was trained under.

        Args:
            config: Configuration in force.

        Returns:
            The parameters that determine how much data and search a model saw.
        """
        return {
            "n_patients": config.n_patients,
            "n_training_shifts": config.n_training_shifts,
            "search_n_iter": config.search_n_iter,
            "prediction_lead_time_mins": config.prediction_lead_time_mins,
        }

    def load(self, source: Path) -> None:
        """Load a previously persisted calibrated model, refusing one trained under a different profile so a smoke-scale model can't silently produce full-scale results that mean nothing.

        Args:
            source: File path written by `save`.

        Raises:
            FileNotFoundError: If the model file does not exist.
            ModelProfileMismatchError: If it was trained under a different
                configuration.
        """
        if not source.exists():
            raise FileNotFoundError(
                f"No trained model at {source}. Run the train stage first."
            )
        payload = joblib.load(source)
        expected = self.training_profile(self._config)
        actual = payload.get("training_profile")
        if actual != expected:
            raise ModelProfileMismatchError(
                f"The model at {source} was trained under {actual}, but this run "
                f"is configured for {expected}. Re-run the train stage under the "
                f"current configuration."
            )
        self._selected_name = payload["name"]
        self._calibrated_model = payload["model"]
        logger.info("Loaded calibrated %s from %s", self._selected_name, source)


class ModelEvaluator:
    """Computes the held-out performance metrics reported for each classifier."""

    def evaluate(
        self, model, features: pd.DataFrame, labels: np.ndarray, name: str
    ) -> dict[str, float | str]:
        """Score one fitted model on the held-out test set, deliberately omitting accuracy since predicting 'low' everywhere would still score well given the class's low prevalence.

        Args:
            model: Fitted classifier exposing predict and predict_proba.
            features: Test observations.
            labels: Encoded test labels.
            name: Classifier name.

        Returns:
            Metric names mapped to values, including per-class precision and
            recall and their high-risk-class aliases.
        """
        predictions = model.predict(features)
        probabilities = model.predict_proba(features)

        per_class_recall = recall_score(
            labels,
            predictions,
            labels=range(len(RISK_CLASS_ORDER)),
            average=None,
            zero_division=0,
        )
        per_class_precision = precision_score(
            labels,
            predictions,
            labels=range(len(RISK_CLASS_ORDER)),
            average=None,
            zero_division=0,
        )
        high_risk_actual = (labels == HIGH_RISK_CLASS_INDEX).astype(int)

        metrics: dict[str, float | str] = {
            "classifier": name,
            "macro_f1_score": float(
                f1_score(labels, predictions, average="macro", zero_division=0)
            ),
            "roc_auc_macro_ovr": float(
                roc_auc_score(labels, probabilities, multi_class="ovr", average="macro")
            ),
            # Brier score is defined for a binary outcome, so it is reported
            # one-vs-rest for the high-risk class, the clinically decisive one.
            "brier_score": float(
                brier_score_loss(
                    high_risk_actual, probabilities[:, HIGH_RISK_CLASS_INDEX]
                )
            ),
        }
        for risk_class, recall in zip(RISK_CLASS_ORDER, per_class_recall, strict=True):
            metrics[f"recall_{risk_class}"] = float(recall)
        for risk_class, precision in zip(
            RISK_CLASS_ORDER, per_class_precision, strict=True
        ):
            metrics[f"precision_{risk_class}"] = float(precision)
        metrics["high_risk_recall"] = metrics["recall_high"]
        metrics["high_risk_precision"] = metrics["precision_high"]
        return metrics
