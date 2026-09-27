"""Per-condition diagnostics for the AND-gate trigger.

The main experiment only records whether the gate fired; this counts each
condition separately, along the path the full gate actually took, to show
which one binds most often.

Usage:
    uv run python scripts/gate_diagnostics.py
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import pandas as pd

from src.config import SimulationConfig
from src.data_generator import (
    BiometricGenerator,
    CTMCModel,
    NEWS2Scorer,
    SyntheticCohort,
)
from src.ml_pipeline import RiskClassifier
from src.models import Patient
from src.routing import DistanceMatrix, DynamicRouter, RoutingModel
from src.simulation import AI_INTEGRATED, ReplicationManager, Simulation
from src.trigger import ANDGateTrigger

ARTEFACT_DIRECTORY = Path(__file__).resolve().parent.parent
RESULTS_DIRECTORY = ARTEFACT_DIRECTORY / "results"
MODEL_PATH = ARTEFACT_DIRECTORY / "models" / "risk_classifier.joblib"

COUNT_KEYS = (
    "eligible",
    "ml_prediction",
    "overdue",
    "deviating",
    "fired",
    "ml_blocked_by_overdue_only",
    "ml_blocked_by_deviation_only",
    "ml_blocked_by_both",
)

logger = logging.getLogger(__name__)


class CountingANDGateTrigger(ANDGateTrigger):
    """The AND-gate, unchanged in behaviour, that tallies each condition."""

    def __init__(
        self,
        config: SimulationConfig,
        theta: float | None = None,
        tau: int | None = None,
    ) -> None:
        super().__init__(config, theta, tau)
        # Every key starts at zero so a condition that never binds still appears
        # as a column rather than silently going missing from the output.
        self.counts: Counter[str] = Counter(dict.fromkeys(COUNT_KEYS, 0))

    def evaluate(
        self,
        patient: Patient,
        predicted_risk_label: str,
        high_risk_probability: float,
        simulation_time_mins: float,
    ) -> bool:
        """Return the parent's decision, and recompute every condition without
        short-circuiting so each one's pass rate can be tallied."""
        fired = super().evaluate(
            patient, predicted_risk_label, high_risk_probability, simulation_time_mins
        )
        if patient.visited or self.is_suppressed(patient, simulation_time_mins):
            return fired

        ml_prediction = (
            predicted_risk_label == "high" and high_risk_probability >= self.theta
        )
        overdue = patient.time_since_last_visit_mins >= self.tau
        deviating = self.is_deviating_from_baseline(patient)

        self.counts["eligible"] += 1
        self.counts["ml_prediction"] += ml_prediction
        self.counts["overdue"] += overdue
        self.counts["deviating"] += deviating
        self.counts["fired"] += fired
        if ml_prediction:
            # Among patients the classifier flags, which other condition
            # stopped the gate firing?
            self.counts["ml_blocked_by_overdue_only"] += deviating and not overdue
            self.counts["ml_blocked_by_deviation_only"] += overdue and not deviating
            self.counts["ml_blocked_by_both"] += not overdue and not deviating
        return fired


def build_cohort(config: SimulationConfig) -> SyntheticCohort:
    """Assemble the synthetic cohort generator from its components."""
    return SyntheticCohort(
        config=config,
        ctmc_model=CTMCModel(),
        biometric_generator=BiometricGenerator(),
        news2_scorer=NEWS2Scorer(),
    )


def run_diagnostics() -> pd.DataFrame:
    """Run the AI-integrated scenario on the experiment seeds with counting,
    returning one row per replication with the raw counts and matching
    outcome metrics."""
    config = SimulationConfig()
    distance_matrix = DistanceMatrix(config.travel_speed_kmh)
    risk_classifier = RiskClassifier(config)
    risk_classifier.load(MODEL_PATH)

    rows = []
    for replication_id, seed in enumerate(
        ReplicationManager(config).base_seeds(config.initial_batch_reps)
    ):
        trigger = CountingANDGateTrigger(config)
        simulation = Simulation(
            config=config,
            cohort=build_cohort(config),
            router=DynamicRouter(
                config, distance_matrix, RoutingModel(config, distance_matrix)
            ),
            distance_matrix=distance_matrix,
            scenario=AI_INTEGRATED,
            risk_classifier=risk_classifier,
            trigger=trigger,
        )
        result = simulation.run(replication_id, seed)
        logger.info(
            "Replication %d (seed %d): %d fired of %d eligible evaluations",
            replication_id,
            seed,
            trigger.counts["fired"],
            trigger.counts["eligible"],
        )
        rows.append(
            {
                "replication_id": replication_id,
                "seed": seed,
                **trigger.counts,
                "mean_response_time_high_risk": result.mean_response_time_high_risk,
                "rerouting_events_per_shift": result.rerouting_events_per_shift,
            }
        )
    return pd.DataFrame(rows)


def pooled_rates(frame: pd.DataFrame) -> dict[str, float]:
    """Pool the replications into each condition's pass rate, the
    conjunction's rate, and how classifier-flagged evaluations split by what
    blocked them."""
    totals = frame.sum(numeric_only=True)
    eligible = totals["eligible"]
    flagged = totals["ml_prediction"]
    return {
        "ml_prediction_rate": totals["ml_prediction"] / eligible,
        "overdue_rate": totals["overdue"] / eligible,
        "deviating_rate": totals["deviating"] / eligible,
        "fired_rate": totals["fired"] / eligible,
        "fired_share_of_ml_flagged": totals["fired"] / flagged,
        "ml_flagged_blocked_by_overdue_only": totals["ml_blocked_by_overdue_only"]
        / flagged,
        "ml_flagged_blocked_by_deviation_only": totals["ml_blocked_by_deviation_only"]
        / flagged,
        "ml_flagged_blocked_by_both": totals["ml_blocked_by_both"] / flagged,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    frame = run_diagnostics()
    output_path = RESULTS_DIRECTORY / "gate_diagnostics.csv"
    frame.to_csv(output_path, index=False)
    logger.info("Wrote %s", output_path)
    for name, value in pooled_rates(frame).items():
        logger.info("%s: %.4f", name, value)


if __name__ == "__main__":
    main()
