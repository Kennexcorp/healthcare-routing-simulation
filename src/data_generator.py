"""Synthetic patient cohort generation: a CTMC drives the latent clinical state, a Gaussian emission model produces biometric observations, and NEWS2-inspired scoring assigns the ground-truth risk label.

All randomness flows through an explicit `numpy.random.Generator`, so a cohort reproduces exactly from its seed.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import (
    ACCEPTABLE_HIGH_RISK_PREVALENCE,
    BASELINE_DISTRIBUTIONS,
    BIOMETRIC_BOUNDS,
    BIOMETRIC_MEANS,
    BIOMETRIC_STDS,
    DIAGNOSIS_CATEGORIES,
    NEWS2_HIGH_THRESHOLD,
    NEWS2_HR_BANDS,
    NEWS2_MAX_PARAMETER_SCORE,
    NEWS2_MODERATE_THRESHOLD,
    NEWS2_SBP_BANDS,
    NEWS2_SPO2_BANDS,
    Q_MATRIX,
    STATES,
    SimulationConfig,
)
from src.models import Patient

logger = logging.getLogger(__name__)

# Column order for a biometric observation, used consistently everywhere a raw
# three-element vector is passed around.
PARAMETER_NAMES = ("spo2", "hr", "sbp")

MINUTES_PER_HOUR = 60.0


class CTMCModel:
    """Latent clinical state process: a three-state continuous time Markov chain."""

    def __init__(self) -> None:
        """Uses the clinical design constants `Q_MATRIX` and `STATES`."""
        self._states = list(STATES)
        self._generator_matrix = Q_MATRIX

    @property
    def states(self) -> list[str]:
        """Ordered state labels."""
        return list(self._states)

    def total_exit_rate(self, current_clinical_state: str) -> float:
        """Total rate of leaving a state, the negated diagonal entry of Q.

        Args:
            current_clinical_state: State to leave.

        Returns:
            Sum of outgoing transition rates per hour.
        """
        return -self._generator_matrix[current_clinical_state][current_clinical_state]

    def sample_sojourn_time_hours(
        self, current_clinical_state: str, rng: np.random.Generator
    ) -> float:
        """Draw how long the patient remains in a state before transitioning.

        Sojourn time in a CTMC state is exponentially distributed with rate
        equal to the total outgoing rate from that state.

        Args:
            current_clinical_state: State being entered.
            rng: Seeded random generator.

        Returns:
            Sojourn time in hours.
        """
        return float(
            rng.exponential(1.0 / self.total_exit_rate(current_clinical_state))
        )

    def sample_next_state(
        self, current_clinical_state: str, rng: np.random.Generator
    ) -> str:
        """Draw the state entered at the next transition.

        Jump probabilities are the outgoing rates normalised by their total, so
        a state is never re-entered directly and zero-rate transitions such as
        low-to-high cannot occur.

        Args:
            current_clinical_state: State being left.
            rng: Seeded random generator.

        Returns:
            The next clinical state.
        """
        outgoing_rates = np.array(
            [
                0.0
                if state == current_clinical_state
                else self._generator_matrix[current_clinical_state][state]
                for state in self._states
            ]
        )
        jump_probabilities = outgoing_rates / outgoing_rates.sum()
        return str(rng.choice(self._states, p=jump_probabilities))

    def advance_patient_state(
        self, patient: Patient, timestep_mins: float, rng: np.random.Generator
    ) -> bool:
        """Advance one patient's latent state by a single timestep.

        Time accrues in the current state until it reaches the sojourn time
        drawn on entry, at which point the patient transitions and a fresh
        sojourn time is drawn for the new state.

        Args:
            patient: Patient to advance. Mutated in place.
            timestep_mins: Length of the timestep in minutes.
            rng: Seeded random generator.

        Returns:
            True if the patient transitioned at this timestep.
        """
        patient.elapsed_time_in_state_hours += timestep_mins / MINUTES_PER_HOUR
        if patient.elapsed_time_in_state_hours < patient.sojourn_time_hours:
            return False

        next_state = self.sample_next_state(patient.current_clinical_state, rng)
        patient.current_clinical_state = next_state
        patient.sojourn_time_hours = self.sample_sojourn_time_hours(next_state, rng)
        patient.elapsed_time_in_state_hours = 0.0
        return True

    def stationary_distribution(self) -> dict[str, float]:
        """Long-run proportion of time spent in each state, reported for validation only since a shift never approaches this distribution.

        Returns:
            State labels mapped to their stationary probabilities.
        """
        rate_matrix = np.array(
            [
                [self._generator_matrix[row][column] for column in self._states]
                for row in self._states
            ]
        )
        # Solve pi @ Q = 0 subject to sum(pi) = 1, as a least-squares system.
        coefficients = np.vstack([rate_matrix.T, np.ones(len(self._states))])
        targets = np.append(np.zeros(len(self._states)), 1.0)
        probabilities, *_ = np.linalg.lstsq(coefficients, targets, rcond=None)
        return dict(zip(self._states, probabilities.tolist(), strict=True))


class BiometricGenerator:
    """State-dependent Gaussian emission model for SpO2, heart rate and systolic BP."""

    def __init__(self) -> None:
        """Uses the clinical design constants for means, spreads and bounds."""
        self._means = BIOMETRIC_MEANS
        self._standard_deviations = BIOMETRIC_STDS
        self._bounds = BIOMETRIC_BOUNDS
        self._baseline_distributions = BASELINE_DISTRIBUTIONS

    def sample_personal_baseline_mean(
        self, rng: np.random.Generator
    ) -> dict[str, float]:
        """Draw one patient's individual physiological baseline.

        Args:
            rng: Seeded random generator.

        Returns:
            Parameter names mapped to that patient's baseline value.
        """
        return {
            parameter: float(rng.normal(*self._baseline_distributions[parameter]))
            for parameter in PARAMETER_NAMES
        }

    def generate_observation(
        self, patient: Patient, rng: np.random.Generator
    ) -> list[float]:
        """Draw one biometric observation for the patient's current state: the state mean shifted by their personal baseline offset, plus Gaussian noise, with diagonal covariance since inter-parameter correlation is out of scope.

        Args:
            patient: Patient whose current state drives the distribution.
            rng: Seeded random generator.

        Returns:
            Observation as [SpO2, HR, SBP], truncated to physiological bounds.
        """
        state = patient.current_clinical_state
        observation = []
        for index, parameter in enumerate(PARAMETER_NAMES):
            population_baseline_mean = self._baseline_distributions[parameter][0]
            personal_offset = (
                patient.personal_baseline_mean[parameter] - population_baseline_mean
            )
            value = rng.normal(
                self._means[state][index] + personal_offset,
                self._standard_deviations[state][index],
            )
            lower_bound, upper_bound = self._bounds[parameter]
            observation.append(float(np.clip(value, lower_bound, upper_bound)))
        return observation


class NEWS2Scorer:
    """NEWS2-inspired three-parameter scoring and risk label assignment."""

    def __init__(self) -> None:
        """Uses the clinical design constants for bands and thresholds."""
        self._bands = {
            "spo2": NEWS2_SPO2_BANDS,
            "hr": NEWS2_HR_BANDS,
            "sbp": NEWS2_SBP_BANDS,
        }
        self._moderate_threshold = NEWS2_MODERATE_THRESHOLD
        self._high_threshold = NEWS2_HIGH_THRESHOLD

    def score_parameter(self, parameter: str, value: float) -> int:
        """Score a single vital sign against its NEWS2 bands.

        Args:
            parameter: One of 'spo2', 'hr', 'sbp'.
            value: Observed value.

        Returns:
            Band score from 0 to 3.
        """
        for upper_bound_exclusive, score in self._bands[parameter]:
            if value < upper_bound_exclusive:
                return score
        # Unreachable: every band list terminates at infinity.
        raise ValueError(f"No NEWS2 band matched {parameter}={value}")

    def parameter_scores(self, observation: list[float]) -> list[int]:
        """Score all three parameters of one observation.

        Args:
            observation: [SpO2, HR, SBP].

        Returns:
            Per-parameter scores in the same order.
        """
        return [
            self.score_parameter(parameter, value)
            for parameter, value in zip(PARAMETER_NAMES, observation, strict=True)
        ]

    def aggregate_score(self, observation: list[float]) -> int:
        """Sum the three parameter scores.

        Args:
            observation: [SpO2, HR, SBP].

        Returns:
            Composite aggregate, 0 to 9.
        """
        return sum(self.parameter_scores(observation))

    def assign_label(self, observation: list[float]) -> str:
        """Assign the ground-truth risk label, applying the single-parameter red-flag rule first so one extreme derangement escalates regardless of the composite score (RCP, 2017).

        Args:
            observation: [SpO2, HR, SBP].

        Returns:
            One of 'low', 'moderate', 'high'.
        """
        scores = self.parameter_scores(observation)
        if max(scores) >= NEWS2_MAX_PARAMETER_SCORE:
            return "high"
        aggregate = sum(scores)
        if aggregate >= self._high_threshold:
            return "high"
        if aggregate >= self._moderate_threshold:
            return "moderate"
        return "low"


class SyntheticCohort:
    """Orchestrates cohort initialisation and shift-length data generation."""

    def __init__(
        self,
        config: SimulationConfig,
        ctmc_model: CTMCModel,
        biometric_generator: BiometricGenerator,
        news2_scorer: NEWS2Scorer,
    ) -> None:
        """Args:
        config: Validated simulation configuration.
        ctmc_model: Latent state process.
        biometric_generator: Emission model.
        news2_scorer: Ground-truth labelling function.
        """
        self._config = config
        self._ctmc_model = ctmc_model
        self._biometric_generator = biometric_generator
        self._news2_scorer = news2_scorer

    def initialise_patients(self, rng: np.random.Generator) -> list[Patient]:
        """Create the patient cohort at shift start, every patient beginning in the low state with a pre-sampled sojourn time and uniform coordinates across the city square.

        Args:
            rng: Seeded random generator.

        Returns:
            The full cohort, ordered by id.
        """
        patients = []
        for patient_id in range(self._config.n_patients):
            patients.append(
                Patient(
                    id=patient_id,
                    age=int(rng.integers(45, 86)),
                    diagnosis=str(rng.choice(DIAGNOSIS_CATEGORIES)),
                    coords=(
                        float(rng.uniform(0, self._config.city_radius_km)),
                        float(rng.uniform(0, self._config.city_radius_km)),
                    ),
                    personal_baseline_mean=self._biometric_generator.sample_personal_baseline_mean(
                        rng
                    ),
                    current_clinical_state="low",
                    sojourn_time_hours=self._ctmc_model.sample_sojourn_time_hours(
                        "low", rng
                    ),
                )
            )
        return patients

    def generate_shift(self, seed: int) -> pd.DataFrame:
        """Generate one complete shift of observations for a fresh cohort.

        Args:
            seed: Seed for this shift. Fully determines the output.

        Returns:
            One row per patient per timestep, carrying the observation, the
            latent CTMC state, and the assigned NEWS2 label.
        """
        rng = np.random.default_rng(seed)
        patients = self.initialise_patients(rng)
        n_timesteps = self._config.shift_duration_mins // self._config.timestep_mins

        records = []
        for timestep_index in range(n_timesteps):
            simulation_time_mins = timestep_index * self._config.timestep_mins
            for patient in patients:
                if timestep_index > 0:
                    self.advance_patient(patient, rng)
                    # No worker visits occur during corpus generation, so time
                    # since last visit accrues from shift start (a completed
                    # visit resets it to zero during the simulation).
                    patient.time_since_last_visit_mins += self._config.timestep_mins
                observation = self.observe_patient(patient, rng)
                records.append(
                    {
                        "seed": seed,
                        "patient_id": patient.id,
                        "timestep": timestep_index,
                        "simulation_time_mins": simulation_time_mins,
                        "age": patient.age,
                        "diagnosis": patient.diagnosis,
                        "spo2": observation[0],
                        "hr": observation[1],
                        "sbp": observation[2],
                        "baseline_spo2": patient.personal_baseline_mean["spo2"],
                        "baseline_hr": patient.personal_baseline_mean["hr"],
                        "baseline_sbp": patient.personal_baseline_mean["sbp"],
                        "time_since_last_visit_mins": patient.time_since_last_visit_mins,
                        "ctmc_state": patient.current_clinical_state,
                        "risk_label": self.label_observation(observation),
                    }
                )
        return self._attach_future_label(pd.DataFrame.from_records(records))

    def advance_patient(self, patient: Patient, rng: np.random.Generator) -> bool:
        """Advance one patient's latent state by a single timestep.

        Exposed alongside `generate_shift` because the simulation drives the
        same three stages one timestep at a time, interleaved with routing.

        Args:
            patient: Patient to advance. Mutated in place.
            rng: Seeded random generator.

        Returns:
            True if the patient transitioned.
        """
        return self._ctmc_model.advance_patient_state(
            patient, self._config.timestep_mins, rng
        )

    def observe_patient(
        self, patient: Patient, rng: np.random.Generator
    ) -> list[float]:
        """Draw one observation and append it to the patient's history.

        Args:
            patient: Patient to observe. Mutated in place.
            rng: Seeded random generator.

        Returns:
            The observation as [SpO2, HR, SBP].
        """
        observation = self._biometric_generator.generate_observation(patient, rng)
        patient.biometric_history.append(observation)
        return observation

    def label_observation(self, observation: list[float]) -> str:
        """Assign the ground-truth risk label for one observation.

        Args:
            observation: [SpO2, HR, SBP].

        Returns:
            One of 'low', 'moderate', 'high'.
        """
        return self._news2_scorer.assign_label(observation)

    def _attach_future_label(self, observations: pd.DataFrame) -> pd.DataFrame:
        """Add `future_risk_label`, the patient's class one lead time ahead, which is what the classifier is trained to predict rather than recomputing `risk_label` from the same vitals.

        Args:
            observations: One shift of generated observations.

        Returns:
            The frame with `future_risk_label` added. Observations within
            one lead time of the shift end have no future to look at and carry a
            null, which `generate_training_corpus` drops.
        """
        lead_time_steps = (
            self._config.prediction_lead_time_mins // self._config.timestep_mins
        )
        ordered = observations.sort_values(["patient_id", "timestep"])
        ordered["future_risk_label"] = ordered.groupby("patient_id")[
            "risk_label"
        ].shift(-lead_time_steps)
        return ordered.sort_index()

    def generate_training_corpus(self) -> pd.DataFrame:
        """Generate the multi-shift corpus the ML pipeline trains on, using seeds disjoint from the experiment, unique patient ids across shifts, and dropping rows within one lead time of shift end that have no future label.

        Returns:
            All observations across all training shifts.
        """
        shifts = []
        for shift_index in range(self._config.n_training_shifts):
            seed = self._config.training_seed_base + shift_index
            shift = self.generate_shift(seed)
            shift["patient_id"] = (
                shift_index * self._config.n_patients + shift["patient_id"]
            )
            shifts.append(shift)
            logger.info(
                "Generated training shift %d of %d (seed %d, %d observations)",
                shift_index + 1,
                self._config.n_training_shifts,
                seed,
                len(shift),
            )
        corpus = pd.concat(shifts, ignore_index=True)
        return corpus.dropna(subset=["future_risk_label"]).reset_index(drop=True)

    def compute_descriptive_statistics(
        self, observations: pd.DataFrame
    ) -> pd.DataFrame:
        """Summarise each biometric parameter per risk class, for clinical plausibility validation following Tucker et al. (2020).

        Args:
            observations: Output of `generate_shift` or `generate_training_corpus`.

        Returns:
            One row per risk class, with mean and standard deviation of each
            parameter, observation count, and class proportion.
        """
        grouped = observations.groupby("risk_label")[list(PARAMETER_NAMES)]
        statistics = grouped.agg(["mean", "std"])
        statistics.columns = [
            f"{parameter}_{measure}" for parameter, measure in statistics.columns
        ]
        statistics["n_observations"] = observations.groupby("risk_label").size()
        statistics["proportion"] = statistics["n_observations"] / len(observations)
        return statistics.reindex(STATES).reset_index()

    def validate_prevalence(self, observations: pd.DataFrame) -> float:
        """Check the high-risk class proportion against the design target, logging a warning rather than raising since the remedy is to retune the generator matrix, a design decision not a runtime one.

        Args:
            observations: Generated observations carrying a `risk_label` column.

        Returns:
            Observed high-risk prevalence.
        """
        prevalence = float((observations["risk_label"] == "high").mean())
        acceptable_low, acceptable_high = ACCEPTABLE_HIGH_RISK_PREVALENCE
        if not acceptable_low <= prevalence <= acceptable_high:
            logger.warning(
                "High-risk prevalence %.2f%% is outside the acceptable range "
                "%.0f-%.0f%%. Retune lambda(moderate->high) and recheck.",
                prevalence * 100,
                acceptable_low * 100,
                acceptable_high * 100,
            )
        else:
            logger.info("High-risk prevalence %.2f%%", prevalence * 100)
        return prevalence
