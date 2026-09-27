"""Command-line entry point for the healthcare routing simulation; running a file inside src/ directly would break its own `from src.x import y` imports."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

RESULTS_DIRECTORY = Path(__file__).parent / "results"
LOG_FILE_PATH = RESULTS_DIRECTORY / "simulation.log"

# A build artefact, not a reportable result, so it lives outside results/.
MODEL_PATH = Path(__file__).parent / "models" / "risk_classifier.joblib"


def configure_logging() -> None:
    """Configure root logging to write to both the results log file and stderr.

    The log records solver status, trigger events, and CTMC validation warnings for every run.
    """
    RESULTS_DIRECTORY.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE_PATH),
            logging.StreamHandler(),
        ],
    )


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments and return the requested pipeline stage."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Run one stage of the community healthcare routing simulation. "
            "Simulation parameters are set through SIM_-prefixed environment "
            "variables or a .env file; see .env.example."
        ),
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=tuple(STAGE_HANDLERS),
        help="Pipeline stage to run, in order: generate, train, simulate, analyse.",
    )
    return parser.parse_args(argv)


def run_generate() -> None:
    """Generate the training corpus and validate its clinical plausibility.

    Not written to disk: fully determined by training_seed_base, so later stages just regenerate it.
    """
    from src.analysis import ResultsExporter
    from src.config import SimulationConfig
    from src.data_generator import CTMCModel

    logger = logging.getLogger(__name__)
    config = SimulationConfig()
    ctmc_model = CTMCModel()
    cohort = build_cohort(config)

    logger.info(
        "Generating %d training shifts of %d patients from seed %d",
        config.n_training_shifts,
        config.n_patients,
        config.training_seed_base,
    )
    observations = cohort.generate_training_corpus()

    stationary = ctmc_model.stationary_distribution()
    logger.info(
        "CTMC stationary distribution: %s",
        ", ".join(
            f"{state} {probability:.1%}" for state, probability in stationary.items()
        ),
    )

    cohort.validate_prevalence(observations)
    label_counts = observations["risk_label"].value_counts(normalize=True)
    logger.info(
        "Label distribution: %s",
        ", ".join(f"{label} {share:.2%}" for label, share in label_counts.items()),
    )

    exporter = ResultsExporter(RESULTS_DIRECTORY)
    exporter.export_dataframe(
        cohort.compute_descriptive_statistics(observations), "descriptive_stats.csv"
    )
    exporter.plot_biometric_distributions(observations)


def build_cohort(config):
    """Assemble the synthetic cohort generator from its components."""
    from src.data_generator import (
        BiometricGenerator,
        CTMCModel,
        NEWS2Scorer,
        SyntheticCohort,
    )

    return SyntheticCohort(
        config=config,
        ctmc_model=CTMCModel(),
        biometric_generator=BiometricGenerator(),
        news2_scorer=NEWS2Scorer(),
    )


def _prepare_training_data(config):
    """Generate the training corpus and split it at the patient level."""
    from src.ml_pipeline import LABEL_COLUMN, create_patient_level_splits, encode_labels

    logger = logging.getLogger(__name__)
    observations = build_cohort(config).generate_training_corpus()
    masks = create_patient_level_splits(observations)
    labels = encode_labels(observations[LABEL_COLUMN])
    logger.info(
        "Target is the NEWS2 class %d minutes ahead; %.2f%% of the corpus is "
        "high risk that far ahead",
        config.prediction_lead_time_mins,
        (observations[LABEL_COLUMN] == "high").mean() * 100,
    )
    logger.info(
        "Split by patient: %d train, %d calibration, %d test observations",
        masks["train"].sum(),
        masks["calibration"].sum(),
        masks["test"].sum(),
    )
    return observations, masks, labels


def _search_candidates(
    classifier, evaluator, train_features, train_labels, test_features, test_labels
):
    """Search every candidate classifier and evaluate each on the test set."""
    import pandas as pd

    from src.config import HIGH_RISK_CLASS_INDEX

    logger = logging.getLogger(__name__)
    metrics_rows = []
    roc_curves = {}
    feature_importances = {}
    searches = {}
    for name, (pipeline, distribution) in classifier.build_candidates().items():
        search = classifier.search(
            name,
            pipeline,
            distribution,
            train_features,
            train_labels,
            train_features["patient_id"].to_numpy(),
        )
        searches[name] = search

        metrics = evaluator.evaluate(
            search.best_estimator_, test_features, test_labels, name
        )
        metrics["cross_validated_macro_f1"] = float(search.best_score_)
        # Refit time only; search_n_iter would dominate a per-classifier comparison.
        metrics["train_time_secs"] = float(search.refit_time_)
        metrics_rows.append(metrics)
        logger.info(
            "%s test macro-F1 %.4f, high-risk recall %.4f",
            name,
            metrics["macro_f1_score"],
            metrics["high_risk_recall"],
        )

        probabilities = search.best_estimator_.predict_proba(test_features)
        roc_curves[name] = (
            (test_labels == HIGH_RISK_CLASS_INDEX).astype(int),
            probabilities[:, HIGH_RISK_CLASS_INDEX],
        )
        estimator = search.best_estimator_.named_steps["classifier"]
        if hasattr(estimator, "feature_importances_"):
            feature_names = search.best_estimator_.named_steps[
                "features"
            ].get_feature_names_out()
            feature_importances[name] = pd.Series(
                estimator.feature_importances_, index=feature_names
            )

    return metrics_rows, roc_curves, feature_importances, searches


def _select_and_calibrate(classifier, searches, observations, labels, masks):
    """Select the best candidate on cross-validated score and calibrate it."""
    import time

    logger = logging.getLogger(__name__)
    # Selected on cross-validated score so the test set stays unbiased for reporting.
    selected_name = max(searches, key=lambda name: searches[name].best_score_)
    logger.info("Selected %s for simulation inference", selected_name)

    calibration_started = time.perf_counter()
    calibrated = classifier.calibrate(
        searches[selected_name].best_estimator_,
        observations[masks["calibration"]],
        labels[masks["calibration"]],
    )
    calibration_elapsed = time.perf_counter() - calibration_started
    classifier.set_calibrated_model(selected_name, calibrated)
    return selected_name, calibrated, calibration_elapsed


def _evaluate_calibrated_model(
    evaluator,
    calibrated,
    test_features,
    test_labels,
    selected_name,
    searches,
    calibration_elapsed,
    metrics_rows,
):
    """Evaluate the calibrated model and append its row to metrics_rows."""
    logger = logging.getLogger(__name__)
    calibrated_metrics = evaluator.evaluate(
        calibrated, test_features, test_labels, f"{selected_name}_calibrated"
    )
    calibrated_metrics["cross_validated_macro_f1"] = float(
        searches[selected_name].best_score_
    )
    # Training cost includes both the underlying refit and the calibration fit, since deploying requires both.
    calibrated_metrics["train_time_secs"] = (
        searches[selected_name].refit_time_ + calibration_elapsed
    )
    metrics_rows.append(calibrated_metrics)
    logger.info(
        "Calibration changed the Brier score from %.4f to %.4f",
        next(
            row["brier_score"]
            for row in metrics_rows
            if row["classifier"] == selected_name
        ),
        calibrated_metrics["brier_score"],
    )
    return calibrated_metrics


def _export_training_artefacts(
    exporter,
    metrics_rows,
    roc_curves,
    feature_importances,
    calibrated,
    test_features,
    test_labels,
    selected_name,
    classifier,
):
    """Write every training-stage CSV and figure, and persist the model."""
    import pandas as pd

    from src.config import HIGH_RISK_CLASS_INDEX

    exporter.export_dataframe(pd.DataFrame(metrics_rows), "ml_results.csv")
    exporter.plot_roc_curves(roc_curves)
    exporter.plot_calibration_curve(
        (test_labels == HIGH_RISK_CLASS_INDEX).astype(int),
        calibrated.predict_proba(test_features)[:, HIGH_RISK_CLASS_INDEX],
        f"{selected_name} (calibrated)",
    )
    if feature_importances:
        importance_frame = pd.DataFrame(feature_importances)
        if selected_name in importance_frame.columns:
            importance_frame = importance_frame.sort_values(
                selected_name, ascending=False
            )
        exporter.export_dataframe(
            importance_frame.reset_index(names="feature"), "feature_importance.csv"
        )
        exporter.plot_feature_importance(feature_importances)
    classifier.save(MODEL_PATH)


def run_train() -> None:
    """Train, calibrate and persist the risk classifier once, so every replication and scenario shares one fixed model instead of mixing routing effects with model differences."""
    from src.analysis import ResultsExporter
    from src.config import SimulationConfig
    from src.ml_pipeline import ModelEvaluator, RiskClassifier

    config = SimulationConfig()
    observations, masks, labels = _prepare_training_data(config)

    classifier = RiskClassifier(config)
    evaluator = ModelEvaluator()
    train_features = observations[masks["train"]]
    train_labels = labels[masks["train"]]
    test_features = observations[masks["test"]]
    test_labels = labels[masks["test"]]

    metrics_rows, roc_curves, feature_importances, searches = _search_candidates(
        classifier, evaluator, train_features, train_labels, test_features, test_labels
    )
    selected_name, calibrated, calibration_elapsed = _select_and_calibrate(
        classifier, searches, observations, labels, masks
    )
    _evaluate_calibrated_model(
        evaluator,
        calibrated,
        test_features,
        test_labels,
        selected_name,
        searches,
        calibration_elapsed,
        metrics_rows,
    )

    exporter = ResultsExporter(RESULTS_DIRECTORY)
    _export_training_artefacts(
        exporter,
        metrics_rows,
        roc_curves,
        feature_importances,
        calibrated,
        test_features,
        test_labels,
        selected_name,
        classifier,
    )


def build_router(scenario, config, distance_matrix, tau=None):
    """Select the routing strategy for a scenario; tau overrides the re-solve deadline for ai_integrated only."""
    from src.routing import (
        BaselineRouter,
        DynamicRouter,
        RoutingModel,
        StaticPriorityRouter,
    )

    if scenario == "unprioritised":
        return BaselineRouter(config, distance_matrix)
    routing_model = RoutingModel(config, distance_matrix)
    if scenario == "static_priority":
        return StaticPriorityRouter(config, distance_matrix, routing_model)
    return DynamicRouter(config, distance_matrix, routing_model, tau=tau)


def run_simulate() -> None:
    """Run every scenario across replications under common random numbers."""
    import pandas as pd

    from src.analysis import ResultsExporter
    from src.config import SimulationConfig
    from src.ml_pipeline import RiskClassifier
    from src.routing import DistanceMatrix
    from src.simulation import SCENARIOS, ReplicationManager, Simulation
    from src.trigger import ANDGateTrigger

    logger = logging.getLogger(__name__)
    config = SimulationConfig()
    distance_matrix = DistanceMatrix(config.travel_speed_kmh)
    manager = ReplicationManager(config)

    risk_classifier = RiskClassifier(config)
    risk_classifier.load(MODEL_PATH)

    seeds = manager.base_seeds(config.initial_batch_reps)
    logger.info(
        "Running %d replications of %d scenarios on seeds %d-%d",
        len(seeds),
        len(SCENARIOS),
        seeds[0],
        seeds[-1],
    )

    results = []
    for replication_id, seed in enumerate(seeds):
        for scenario in SCENARIOS:
            # A fresh trigger per run: its recorded events are per-replication
            # metrics, not cumulative across the experiment.
            trigger = ANDGateTrigger(config) if scenario == "ai_integrated" else None
            simulation = Simulation(
                config=config,
                cohort=build_cohort(config),
                router=build_router(scenario, config, distance_matrix),
                distance_matrix=distance_matrix,
                scenario=scenario,
                risk_classifier=risk_classifier
                if scenario == "ai_integrated"
                else None,
                trigger=trigger,
            )
            result = simulation.run(replication_id, seed)
            results.append(result)
            logger.info(
                "Replication %d %s: response %.1f mins, %d unvisited, %d triggers",
                replication_id,
                scenario,
                result.mean_response_time_high_risk,
                result.unvisited_patients,
                result.rerouting_events_per_shift,
            )

    frame = pd.DataFrame([result.to_csv_row() for result in results])
    primary = [
        result.mean_response_time_high_risk
        for result in results
        if result.scenario == "ai_integrated"
    ]
    half_width = manager.confidence_interval_half_width(primary)
    frame["confidence_interval_half_width"] = half_width
    required = manager.required_replications(primary)
    logger.info(
        "Achieved half-width %.2f mins on %d replications; %d required for %.0f%% "
        "relative precision",
        half_width,
        len(primary),
        required,
        config.target_precision * 100,
    )

    ResultsExporter(RESULTS_DIRECTORY).export_dataframe(frame, "simulation_results.csv")


def run_sensitivity_sweep() -> pd.DataFrame:
    """Re-run the AI-integrated scenario at every theta/tau setting on the main experiment's seeds.

    Returns:
        The summarised sensitivity results, one row per setting.
    """
    import pandas as pd

    from src.analysis import SensitivityAnalysis
    from src.config import SimulationConfig
    from src.ml_pipeline import RiskClassifier
    from src.routing import DistanceMatrix
    from src.simulation import ReplicationManager, Simulation
    from src.trigger import ANDGateTrigger

    logger = logging.getLogger(__name__)
    config = SimulationConfig()
    sensitivity_analysis = SensitivityAnalysis(config)
    settings = sensitivity_analysis.parameter_settings()
    seeds = ReplicationManager(config).base_seeds(config.sensitivity_reps)
    logger.info(
        "Sweeping %d threshold settings across %d replications each",
        len(settings),
        len(seeds),
    )

    risk_classifier = RiskClassifier(config)
    risk_classifier.load(MODEL_PATH)
    distance_matrix = DistanceMatrix(config.travel_speed_kmh)

    sensitivity_rows = []
    for setting in settings:
        for replication_id, seed in enumerate(seeds):
            simulation = Simulation(
                config=config,
                cohort=build_cohort(config),
                router=build_router(
                    "ai_integrated", config, distance_matrix, tau=setting["tau"]
                ),
                distance_matrix=distance_matrix,
                scenario="ai_integrated",
                risk_classifier=risk_classifier,
                trigger=ANDGateTrigger(
                    config, theta=setting["theta"], tau=setting["tau"]
                ),
            )
            row = simulation.run(replication_id, seed).to_csv_row()
            row.update(setting)
            sensitivity_rows.append(row)
        logger.info("Completed theta=%.2f tau=%d", setting["theta"], setting["tau"])

    return sensitivity_analysis.summarise(pd.DataFrame(sensitivity_rows))


def run_analyse() -> None:
    """Compare the scenarios statistically and sweep the AND-gate thresholds, reading simulate-stage results rather than re-running them."""
    import pandas as pd

    from src.analysis import (
        PRIMARY_METRIC,
        SECONDARY_METRICS,
        ResultsExporter,
        StatisticalAnalyser,
    )

    logger = logging.getLogger(__name__)
    exporter = ResultsExporter(RESULTS_DIRECTORY)

    results_path = RESULTS_DIRECTORY / "simulation_results.csv"
    if not results_path.exists():
        raise FileNotFoundError(
            f"No replication results at {results_path}. Run the simulate stage first."
        )
    with results_path.open() as handle:
        results = pd.read_csv(handle)

    analyser = StatisticalAnalyser()
    comparisons = analyser.compare_metrics(
        results, [PRIMARY_METRIC, *SECONDARY_METRICS]
    )
    exporter.export_dataframe(comparisons, "statistical_tests.csv")
    exporter.plot_response_time_comparison(results)
    exporter.plot_paired_response_times(results)

    # The sweep takes hours, so an existing sweep is reused; its figures are
    # still redrawn so a rerun of this stage always refreshes every figure.
    sensitivity_path = RESULTS_DIRECTORY / "sensitivity_analysis.csv"
    if sensitivity_path.exists():
        logger.warning(
            "Reusing existing %s; delete it to force a fresh sweep",
            sensitivity_path.name,
        )
        with sensitivity_path.open() as handle:
            sensitivity = pd.read_csv(handle)
    else:
        sensitivity = run_sensitivity_sweep()
        exporter.export_dataframe(sensitivity, "sensitivity_analysis.csv")
    exporter.plot_sensitivity(sensitivity, "theta")
    exporter.plot_sensitivity(sensitivity, "tau")
    exporter.plot_theta_tradeoff(sensitivity)


STAGE_HANDLERS = {
    "generate": run_generate,
    "train": run_train,
    "simulate": run_simulate,
    "analyse": run_analyse,
}


def run_stage(stage: str) -> None:
    """Dispatch to the requested pipeline stage.

    Args:
        stage: Name of the stage to run.
    """
    STAGE_HANDLERS[stage]()


def main() -> None:
    """Configure logging, then run the requested stage."""
    arguments = parse_arguments()
    configure_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting stage: %s", arguments.stage)
    try:
        run_stage(arguments.stage)
    except NotImplementedError as error:
        logger.error("%s", error)
        sys.exit(1)
    logger.info("Completed stage: %s", arguments.stage)


if __name__ == "__main__":
    main()
