"""The AND-gate re-routing trigger: fires only when predicted high risk, a baseline deviation, and time since last visit all hold at once, trading sensitivity for precision to avoid the alert fatigue that plagues clinical warning systems (van der Sijs et al., 2006)."""

from __future__ import annotations

import logging

import numpy as np

from src.config import (
    BASELINE_DEVIATION_THRESHOLD_SD,
    BASELINE_DISTRIBUTIONS,
    ROLLING_WINDOW_SIZE,
    SimulationConfig,
)
from src.data_generator import PARAMETER_NAMES
from src.models import Patient, TriggerEvent

logger = logging.getLogger(__name__)

SPO2_INDEX = PARAMETER_NAMES.index("spo2")
HEART_RATE_INDEX = PARAMETER_NAMES.index("hr")
SYSTOLIC_BP_INDEX = PARAMETER_NAMES.index("sbp")


class ANDGateTrigger:
    """Evaluates the three-condition re-routing trigger and records its quality."""

    def __init__(
        self,
        config: SimulationConfig,
        theta: float | None = None,
        tau: int | None = None,
    ) -> None:
        """Thresholds are injected rather than read from configuration directly,
        so the sensitivity analysis can sweep them without mutating global state.

        Args:
            config: Validated simulation configuration.
            theta: High-risk probability threshold. Defaults to the configured
                value.
            tau: Minutes since last visit required. Defaults to the configured
                value.
        """
        self._config = config
        self._theta = config.default_theta if theta is None else theta
        self._tau = config.default_tau if tau is None else tau
        self._events: list[TriggerEvent] = []
        # Patient id mapped to the time their last trigger fired, used to
        # suppress repeat triggers while the resulting deadline still stands.
        self._triggered_at_mins: dict[int, float] = {}

    @property
    def theta(self) -> float:
        """High-risk probability threshold."""
        return self._theta

    @property
    def tau(self) -> int:
        """Minutes since last visit required for the trigger to fire."""
        return self._tau

    @property
    def events(self) -> list[TriggerEvent]:
        """Every trigger event recorded so far."""
        return list(self._events)

    def baseline_deviation_zscore(self, patient: Patient) -> np.ndarray:
        """Standardised deviation of each vital sign from the patient's own normal.

        Averaged over the trailing window so that a single noisy reading cannot
        fire the condition, and scaled by the population spread of each
        parameter so the three signals are comparable.

        Args:
            patient: Patient whose biometric history is examined.

        Returns:
            Z-score per parameter as [SpO2, HR, SBP], or an empty array when the
            patient has no observations yet.
        """
        window = patient.biometric_history[-ROLLING_WINDOW_SIZE:]
        if not window:
            return np.array([])
        window_mean = np.mean(window, axis=0)
        personal_baseline = np.array(
            [patient.personal_baseline_mean[parameter] for parameter in PARAMETER_NAMES]
        )
        population_spread = np.array(
            [BASELINE_DISTRIBUTIONS[parameter][1] for parameter in PARAMETER_NAMES]
        )
        return (window_mean - personal_baseline) / population_spread

    def is_deviating_from_baseline(self, patient: Patient) -> bool:
        """Condition two: has the patient departed from their own normal range?

        SpO2 or SBP falling below baseline, or HR rising above it, count as deterioration; a BP rise is the compensatory biphasic phase, not decompensation.

        Args:
            patient: Patient whose biometric history is examined.

        Returns:
            True if any monitored signal has departed adversely from baseline.
        """
        deviation = self.baseline_deviation_zscore(patient)
        if deviation.size == 0:
            return False
        threshold = BASELINE_DEVIATION_THRESHOLD_SD
        return bool(
            deviation[SPO2_INDEX] <= -threshold
            or deviation[HEART_RATE_INDEX] >= threshold
            or deviation[SYSTOLIC_BP_INDEX] <= -threshold
        )

    def is_suppressed(self, patient: Patient, simulation_time_mins: float) -> bool:
        """Whether a repeat trigger for this patient should be ignored, since re-triggering before its existing tau deadline would churn routes for nothing.

        Args:
            patient: Patient being evaluated.
            simulation_time_mins: Current simulation clock.

        Returns:
            True if this patient's trigger should be suppressed.
        """
        last_triggered = self._triggered_at_mins.get(patient.id)
        if last_triggered is None:
            return False
        return simulation_time_mins < last_triggered + self._tau

    def evaluate(
        self,
        patient: Patient,
        predicted_risk_label: str,
        high_risk_probability: float,
        simulation_time_mins: float,
    ) -> bool:
        """Evaluate all three conditions for one patient.

        Args:
            patient: Patient being evaluated.
            predicted_risk_label: Classifier output for this patient.
            high_risk_probability: Predicted probability of the high-risk class.
            simulation_time_mins: Current simulation clock.

        Returns:
            True only if all three conditions hold and the patient is eligible.
        """
        if patient.visited or self.is_suppressed(patient, simulation_time_mins):
            return False

        predicted_high_risk = (
            predicted_risk_label == "high" and high_risk_probability >= self._theta
        )
        overdue_for_a_visit = patient.time_since_last_visit_mins >= self._tau

        # Ordered cheapest first: the costly deviation test is skipped whenever a cheaper condition already rules the patient out.
        return bool(
            predicted_high_risk
            and overdue_for_a_visit
            and self.is_deviating_from_baseline(patient)
        )

    def evaluate_cohort(
        self,
        patients: list[Patient],
        predicted_risk_labels: dict[int, str],
        high_risk_probabilities: dict[int, float],
        simulation_time_mins: float,
    ) -> list[Patient]:
        """Evaluate the whole cohort at one timestep.

        Returns every patient that fired, so the caller can batch them into a single re-solve rather than paying the solver cost per patient.

        Args:
            patients: The full cohort.
            predicted_risk_labels: Patient id mapped to predicted label.
            high_risk_probabilities: Patient id mapped to high-risk probability.
            simulation_time_mins: Current simulation clock.

        Returns:
            Patients whose AND-gate fired at this timestep.
        """
        triggered_patients = [
            patient
            for patient in patients
            if self.evaluate(
                patient,
                predicted_risk_labels[patient.id],
                high_risk_probabilities[patient.id],
                simulation_time_mins,
            )
        ]
        if triggered_patients:
            logger.info(
                "AND-gate fired for %d patient(s) at %.0f mins: %s",
                len(triggered_patients),
                simulation_time_mins,
                [patient.id for patient in triggered_patients],
            )
        return triggered_patients

    def record_events(
        self,
        triggered_patients: list[Patient],
        high_risk_probabilities: dict[int, float],
        simulation_time_mins: float,
        rerouting_churn_proportion: float,
        additional_travel_mins: float = 0.0,
    ) -> list[TriggerEvent]:
        """Record one event per triggered patient after the re-solve.

        Churn is a property of the re-solve, so a batch of triggers at the same timestep all carry the same value.

        Args:
            triggered_patients: Patients whose gate fired.
            high_risk_probabilities: Patient id mapped to high-risk probability.
            simulation_time_mins: Time the triggers fired.
            rerouting_churn_proportion: Churn caused by the resulting re-solve.
            additional_travel_mins: Extra travel imposed on other patients.

        Returns:
            The newly recorded events.
        """
        recorded = []
        for patient in triggered_patients:
            event = TriggerEvent(
                patient_id=patient.id,
                timestamp_mins=simulation_time_mins,
                ml_risk_label="high",
                high_risk_probability=high_risk_probabilities[patient.id],
                biometric_deviation_detected=True,
                time_since_last_visit_mins=patient.time_since_last_visit_mins,
                ctmc_ground_truth_state=patient.current_clinical_state,
                rerouting_churn_proportion=rerouting_churn_proportion,
                additional_travel_mins=additional_travel_mins,
                shift_duration_mins=self._config.shift_duration_mins,
            )
            self._events.append(event)
            self._triggered_at_mins[patient.id] = simulation_time_mins
            recorded.append(event)
        return recorded

    def resolve_due_events(
        self, patients: list[Patient], simulation_time_mins: float
    ) -> list[TriggerEvent]:
        """Resolve every event whose lead time has now elapsed.

        The patient's current state is exactly what the false trigger rate gets validated against.

        Args:
            patients: The full cohort, carrying current clinical state.
            simulation_time_mins: Current simulation clock.

        Returns:
            The events resolved at this timestep.
        """
        state_of_patient = {
            patient.id: patient.current_clinical_state for patient in patients
        }
        lead_time_mins = self._config.prediction_lead_time_mins

        resolved = []
        for event in self._events:
            if event.is_resolved:
                continue
            if simulation_time_mins >= event.timestamp_mins + lead_time_mins:
                event.resolve(state_of_patient[event.patient_id])
                resolved.append(event)
        return resolved

    @property
    def resolved_events(self) -> list[TriggerEvent]:
        """Events whose outcome is known."""
        return [event for event in self._events if event.is_resolved]

    @property
    def unresolved_event_count(self) -> int:
        """Events still awaiting their outcome.

        At shift end these are triggers that fired too late to be validated, reported separately rather than counted in the rate.
        """
        return len(self._events) - len(self.resolved_events)

    @property
    def false_trigger_rate(self) -> float:
        """Proportion of resolved triggers that fired on a patient who did not
        reach the high-risk state.

        Returns:
            The rate, or 0.0 when no event has been resolved.
        """
        resolved = self.resolved_events
        if not resolved:
            return 0.0
        return sum(event.is_false_trigger for event in resolved) / len(resolved)

    @property
    def mean_rerouting_churn(self) -> float:
        """Mean churn across all recorded triggers.

        Returns:
            The mean, or 0.0 when nothing has fired.
        """
        if not self._events:
            return 0.0
        return float(
            np.mean([event.rerouting_churn_proportion for event in self._events])
        )

    @property
    def mean_additional_travel_per_false_trigger(self) -> float:
        """Operational cost of the triggers that turned out to be unnecessary.

        Returns:
            Mean additional travel minutes per false trigger, or 0.0 when there
            were none.
        """
        false_triggers = [
            event for event in self.resolved_events if event.is_false_trigger
        ]
        if not false_triggers:
            return 0.0
        return float(
            np.mean([event.additional_travel_mins for event in false_triggers])
        )
