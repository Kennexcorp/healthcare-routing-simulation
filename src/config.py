"""Simulation configuration: environment-overridable runtime parameters in `SimulationConfig`, plus fixed clinical and structural design constants below."""

from __future__ import annotations

from math import inf

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SimulationConfig(BaseSettings):
    """Runtime simulation parameters, validated and environment-overridable.

    All fields are read from SIM_-prefixed environment variables or a .env file.

    Example:
        SIM_DEFAULT_THETA=0.60 uv run python main.py --stage simulate
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SIM_",
        case_sensitive=False,
    )

    # Cohort size calibrated against fleet capacity, so prioritisation changes
    # who is reached in time rather than only who is abandoned outright.
    n_patients: int = Field(default=160, ge=1)
    n_workers: int = Field(default=10, ge=1)
    shift_duration_mins: int = Field(default=480, ge=60)
    timestep_mins: int = Field(default=5, ge=1)
    city_radius_km: float = Field(default=10.0, gt=0)
    travel_speed_kmh: float = Field(default=30.0, gt=0)
    service_duration_mins: int = Field(default=30, ge=5)

    default_theta: float = Field(default=0.70, ge=0.0, le=1.0)
    default_tau: int = Field(default=60, ge=0)

    initial_solve_limit: int = Field(default=30, ge=5)
    reroute_solve_limit: int = Field(default=10, ge=1)

    initial_batch_reps: int = Field(default=10, ge=5)
    target_precision: float = Field(default=0.10, gt=0, le=1.0)
    confidence_level: float = Field(default=0.95, gt=0, le=1.0)

    theta_values: list[float] = Field(default=[0.5, 0.6, 0.7, 0.8, 0.9])
    # 30/60 minutes are RCP (2020) hospital assessment times for NEWS2 7+ and
    # 5-6; 120 is the NHS England two-hour urgent response standard. These are
    # reference points, not validated community-setting thresholds.
    tau_values: list[int] = Field(default=[30, 60, 120])
    # Kept separate from `initial_batch_reps`: tying it to the main replication
    # count would multiply the seven-setting sweep sevenfold.
    sensitivity_reps: int = Field(default=10, ge=2)

    # Drawn from `training_seed_base` upwards, disjoint from the experiment
    # replication seeds, so no patient trajectory is both trained on and evaluated.
    n_training_shifts: int = Field(default=5, ge=1)
    training_seed_base: int = Field(default=90_000, ge=0)
    search_n_iter: int = Field(default=50, ge=1)

    # Lead time for the classifier's target: predicts the class this many
    # minutes ahead rather than recomputing the current NEWS2 score.
    prediction_lead_time_mins: int = Field(default=30, ge=5)


# Depot sits at the centre of the 10km square and is node index 0 in every
# OR-Tools model.
DEPOT_COORDS = (5.0, 5.0)

# CTMC generator matrix, rates per hour; direct low-to-high and high-to-low
# transitions are zero since deterioration and recovery both pass through moderate.
Q_MATRIX = {
    "low": {"low": -0.10, "moderate": 0.10, "high": 0.00},
    "moderate": {"low": 0.20, "moderate": -0.35, "high": 0.15},
    "high": {"low": 0.00, "moderate": 0.15, "high": -0.15},
}
STATES = ["low", "moderate", "high"]

# Biometric mean vectors per state [SpO2, HR, SBP].
# Biphasic BP: systolic pressure rises in moderate then falls in high (RCP,
# 2017), not a monotonic decline.
BIOMETRIC_MEANS = {
    "low": [97.5, 72.0, 125.0],
    "moderate": [93.5, 105.0, 148.0],
    "high": [89.0, 135.0, 94.0],
}

BIOMETRIC_STDS = {
    "low": [1.5, 8.0, 10.0],
    "moderate": [2.0, 10.0, 12.0],
    "high": [2.5, 12.0, 8.0],
}

# Physiologically plausible bounds, used to truncate generated observations
BIOMETRIC_BOUNDS = {
    "spo2": (70.0, 100.0),
    "hr": (30.0, 200.0),
    "sbp": (60.0, 240.0),
}

# Population distributions [mean, std] from which each patient's personal
# baseline is drawn at initialisation
BASELINE_DISTRIBUTIONS = {
    "spo2": (97.0, 1.5),
    "hr": (72.0, 8.0),
    "sbp": (125.0, 10.0),
}

PRIORITY_WEIGHTS = {"high": 10, "moderate": 3, "low": 1}

# OR-Tools takes integer penalties only. Drop penalties are the priority weight
# times this scale, large enough that dropping any patient costs more than any
# achievable travel saving.
PRIORITY_SCALE = 100_000

# Time windows assigned at shift start, minutes [earliest, latest]. High risk is
# a hard window; moderate and low are soft and penalised on violation.
INITIAL_TIME_WINDOWS = {
    "high": (0, 120),
    "moderate": (0, 300),
    "low": (0, 480),
}

DIAGNOSIS_CATEGORIES = ["COPD", "HF", "Diabetes", "Hypertension", "Other"]

# NEWS2 single-parameter scoring bands (RCP, 2017), as ascending
# (upper_bound_exclusive, score) pairs; half-open since observations are
# continuous, unlike the published inclusive integer ranges.
NEWS2_SPO2_BANDS = ((92.0, 3), (94.0, 2), (96.0, 1), (inf, 0))

# The 41-50 band (scoring 1) is included per NEWS2 (RCP, 2017); omitting it
# would misclassify mild bradycardia as a red flag.
NEWS2_HR_BANDS = ((41.0, 3), (51.0, 1), (91.0, 0), (111.0, 1), (131.0, 2), (inf, 3))

NEWS2_SBP_BANDS = ((91.0, 3), (101.0, 2), (111.0, 1), (220.0, 0), (inf, 3))

# Any single parameter reaching this score triggers the NEWS2 red-flag rule and
# forces a high-risk label regardless of the aggregate (RCP, 2017).
NEWS2_MAX_PARAMETER_SCORE = 3

# Aggregate thresholds rescaled for the three-parameter subset: NEWS2's 5/7
# assume six parameters, so applied unchanged they would leave almost no
# observations in the moderate band, making macro-F1 uninformative.
NEWS2_MODERATE_THRESHOLD = 3
NEWS2_HIGH_THRESHOLD = 5

# Generated cohort validation gate. Prevalence outside the acceptable range is
# logged as a warning; outside the target range it is reported for review.
TARGET_HIGH_RISK_PREVALENCE = (0.10, 0.15)
ACCEPTABLE_HIGH_RISK_PREVALENCE = (0.08, 0.18)

# Rolling window for the trend features: three observations at a 5-minute
# timestep covers the 15 minutes the AND-gate's second condition evaluates.
ROLLING_WINDOW_SIZE = 3

# Condition two of the AND-gate: a magnitude test against the patient's own
# baseline rather than a slope test, since within a CTMC state deterioration
# is a step change, not a gradual drift.
BASELINE_DEVIATION_THRESHOLD_SD = 2.0

# Ordered risk classes; load-bearing, since inference reads the high-risk
# probability as predict_proba column index 2, so 'high' must stay last.
RISK_CLASS_ORDER = ["low", "moderate", "high"]
HIGH_RISK_CLASS_INDEX = 2

# Splits are at patient level throughout; an observation-level split would
# place the same patient in both train and test and inflate performance.
TEST_SET_PROPORTION = 0.2
CALIBRATION_SET_PROPORTION = 0.2
N_CROSS_VALIDATION_FOLDS = 5
SPLIT_SEED = 42
