"""Statistical analysis and results output: scenarios share a seed per replication, so comparisons use a paired Wilcoxon test with Holm-Bonferroni correction."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

# Select a non-interactive backend before pyplot is imported. Figures are
# written to disk from batch runs that have no display attached.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import t, wilcoxon
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve
from statsmodels.stats.multitest import multipletests

from src.config import STATES
from src.data_generator import PARAMETER_NAMES

logger = logging.getLogger(__name__)

# Figures are reproduced in print, where anything below 300 DPI shows visible artefacts.
FIGURE_DPI = 300

# Family-wise error rate across the three pairwise scenario comparisons.
FAMILY_WISE_ALPHA = 0.05

# Metrics compared between scenarios; each secondary metric gets its own Holm
# correction since it tests operational cost, not the primary hypothesis.
PRIMARY_METRIC = "mean_response_time_high_risk"
SECONDARY_METRICS = ("total_travel_distance_km", "workforce_utilisation")

# Rank-biserial correlation magnitude bands (Kerby, 2014); the test decides
# significance, not these.
EFFECT_SIZE_BANDS = ((0.1, "negligible"), (0.3, "small"), (0.5, "medium"))

PARAMETER_AXIS_LABELS = {
    "spo2": "SpO$_2$ (%)",
    "hr": "Heart rate (bpm)",
    "sbp": "Systolic BP (mmHg)",
}

# Subscripts use mathtext since the seaborn theme font has no subscript glyph.
# Figures use print names; result files keep the stable code names.
CLASSIFIER_LABELS = {
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
}
STATE_LABELS = {"low": "Low", "moderate": "Moderate", "high": "High"}
SCENARIO_LABELS = {
    "unprioritised": "Unprioritised",
    "static_priority": "Static priority",
    "ai_integrated": "AI-integrated",
}
THRESHOLD_SYMBOLS = {"theta": "\u03b8", "tau": "\u03c4"}
FEATURE_PARAMETER_LABELS = {
    "spo2": "SpO$_2$",
    "hr": "heart rate",
    "sbp": "systolic BP",
}
FEATURE_PREFIX_TEMPLATES = {
    "delta": "Change in {}",
    "rolling_mean": "Rolling mean {}",
    "rolling_min": "Rolling min {}",
    "rolling_max": "Rolling max {}",
    "rolling_std": "Rolling SD {}",
    "zscore": "{} z-score",
}
FEATURE_OTHER_LABELS = {
    "time_since_last_visit_mins": "Time since last visit",
    "age": "Age",
}


def classifier_label(name: str) -> str:
    """Print name for a classifier key, keeping any suffix such as "(calibrated)".

    Args:
        name: Classifier key, optionally followed by text such as "(calibrated)".

    Returns:
        The name with each classifier key replaced by its print name.
    """
    for key, label in CLASSIFIER_LABELS.items():
        name = name.replace(key, label)
    return name


def feature_label(name: str) -> str:
    """Print name for an engineered feature, for figure axes.

    Args:
        name: Feature name as produced by `FeatureEngineer`.

    Returns:
        A readable name such as "Heart rate z-score".
    """
    if name in FEATURE_PARAMETER_LABELS:
        return f"Current {FEATURE_PARAMETER_LABELS[name]}"
    if name in FEATURE_OTHER_LABELS:
        return FEATURE_OTHER_LABELS[name]
    if name.startswith("diagnosis_"):
        return f"Diagnosis: {name.removeprefix('diagnosis_')}"
    for prefix, template in FEATURE_PREFIX_TEMPLATES.items():
        parameter = name.removeprefix(f"{prefix}_")
        if parameter in FEATURE_PARAMETER_LABELS:
            text = template.format(FEATURE_PARAMETER_LABELS[parameter])
            return text[:1].upper() + text[1:]
    return name


class ResultsExporter:
    """Writes every CSV and figure the study produces."""

    def __init__(self, results_directory: Path) -> None:
        """Args:
        results_directory: Destination for CSV output. Figures are written to
            a `figures` subdirectory beneath it. Both are created if absent.
        """
        self._results_directory = Path(results_directory)
        self._figures_directory = self._results_directory / "figures"
        self._results_directory.mkdir(parents=True, exist_ok=True)
        self._figures_directory.mkdir(parents=True, exist_ok=True)

    @property
    def figures_directory(self) -> Path:
        """Directory figures are written to."""
        return self._figures_directory

    def export_dataframe(self, frame: pd.DataFrame, filename: str) -> Path:
        """Write a results table to CSV.

        Args:
            frame: Table to write.
            filename: File name including the .csv extension.

        Returns:
            Path written.
        """
        destination = self._results_directory / filename
        frame.to_csv(destination, index=False)
        logger.info("Wrote %s (%d rows)", destination, len(frame))
        return destination

    def export_figure(self, figure: plt.Figure, filename: str) -> Path:
        """Write a figure at print resolution and close it.

        Args:
            figure: Figure to write.
            filename: File name including the .png extension.

        Returns:
            Path written.
        """
        destination = self._figures_directory / filename
        figure.savefig(destination, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)
        logger.info("Wrote %s", destination)
        return destination

    def plot_biometric_distributions(self, observations: pd.DataFrame) -> Path:
        """Plot each biometric parameter's distribution per risk class, to check visually that classes separate and SBP shows the biphasic pattern.

        Args:
            observations: Generated observations with a `risk_label` column.

        Returns:
            Path written.
        """
        sns.set_theme(style="whitegrid")
        figure, axes = plt.subplots(1, len(PARAMETER_NAMES), figsize=(15, 4.5))
        present_states = [
            STATE_LABELS[state]
            for state in STATES
            if state in set(observations["risk_label"])
        ]
        labelled = observations.assign(
            **{"Risk class": observations["risk_label"].map(STATE_LABELS)}
        )
        for axis, parameter in zip(axes, PARAMETER_NAMES, strict=True):
            sns.kdeplot(
                data=labelled,
                x=parameter,
                hue="Risk class",
                hue_order=present_states,
                fill=True,
                common_norm=False,
                alpha=0.4,
                ax=axis,
            )
            axis.set_xlabel(PARAMETER_AXIS_LABELS[parameter])
            axis.set_ylabel("Density")
        figure.suptitle("Biometric distributions by NEWS2-inspired risk class")
        figure.tight_layout()
        return self.export_figure(figure, "biometric_distributions.png")

    def plot_calibration_curve(
        self,
        high_risk_actual: np.ndarray,
        high_risk_probability: np.ndarray,
        classifier_name: str,
    ) -> Path:
        """Plot a reliability diagram for the high-risk class, since the AND-gate thresholds a predicted probability and needs it not to be over-confident.

        Args:
            high_risk_actual: Binary outcome, 1 where the true label is high.
            high_risk_probability: Predicted high-risk probability.
            classifier_name: Name shown in the legend.

        Returns:
            Path written.
        """
        observed_fraction, mean_predicted = calibration_curve(
            high_risk_actual, high_risk_probability, n_bins=10, strategy="quantile"
        )
        figure, axis = plt.subplots(figsize=(6, 6))
        axis.plot(
            [0, 1], [0, 1], linestyle="--", color="grey", label="Perfect calibration"
        )
        axis.plot(
            mean_predicted,
            observed_fraction,
            marker="o",
            label=classifier_label(classifier_name),
        )
        axis.set_xlabel("Mean predicted probability")
        axis.set_ylabel("Observed fraction high risk")
        axis.set_title("Calibration of high-risk probability")
        axis.legend(loc="upper left")
        figure.tight_layout()
        return self.export_figure(figure, "calibration_curve.png")

    def plot_roc_curves(self, curves: dict[str, tuple[np.ndarray, np.ndarray]]) -> Path:
        """Plot one-vs-rest ROC curves for the high-risk class.

        Args:
            curves: Classifier name mapped to (binary actual, predicted
                high-risk probability).

        Returns:
            Path written.
        """
        figure, axis = plt.subplots(figsize=(6.5, 6))
        for classifier_name, (actual, probability) in curves.items():
            false_positive_rate, true_positive_rate, _ = roc_curve(actual, probability)
            area = roc_auc_score(actual, probability)
            axis.plot(
                false_positive_rate,
                true_positive_rate,
                label=f"{classifier_label(classifier_name)} (AUC {area:.3f})",
            )
        axis.plot([0, 1], [0, 1], linestyle="--", color="grey")
        axis.set_xlabel("False positive rate")
        axis.set_ylabel("True positive rate")
        axis.set_title("ROC curves, high risk versus rest")
        axis.legend(loc="lower right")
        figure.tight_layout()
        return self.export_figure(figure, "roc_curves.png")

    def plot_response_time_comparison(
        self, results: pd.DataFrame, metric: str = "mean_response_time_high_risk"
    ) -> Path:
        """Plot the primary metric across the three scenarios, with individual points overlaid since replications are paired by seed.

        Args:
            results: One row per replication per scenario.
            metric: Column to plot.

        Returns:
            Path written.
        """
        sns.set_theme(style="whitegrid")
        order = [
            SCENARIO_LABELS[scenario]
            for scenario in SCENARIO_LABELS
            if scenario in set(results["scenario"])
        ]
        results = results.assign(scenario=results["scenario"].map(SCENARIO_LABELS))
        figure, axis = plt.subplots(figsize=(8, 6))
        sns.boxplot(
            data=results, x="scenario", y=metric, order=order, showfliers=False, ax=axis
        )
        sns.stripplot(
            data=results,
            x="scenario",
            y=metric,
            order=order,
            color="black",
            alpha=0.5,
            size=4,
            ax=axis,
        )
        axis.set_xlabel("Routing scenario")
        axis.set_ylabel("Mean response time to high-risk patients (minutes)")
        axis.set_title("Response time by routing scenario")
        figure.tight_layout()
        return self.export_figure(figure, "response_time_comparison.png")

    def plot_paired_response_times(
        self,
        results: pd.DataFrame,
        metric: str = "mean_response_time_high_risk",
        treatment: str = "ai_integrated",
        baselines: tuple[str, ...] = ("static_priority", "unprioritised"),
    ) -> Path:
        """Plot each replication as a line from a baseline to the treatment scenario, since the comparison is paired by seed and a box plot hides which values belong together.

        Args:
            results: One row per replication per scenario, carrying `seed`.
            metric: Column to plot.
            treatment: Scenario every baseline is compared against.
            baselines: Scenarios shown, one panel each.

        Returns:
            Path written.

        Raises:
            ValueError: If a seed is missing from any scenario plotted, since
                an unmatched replication has no pair to draw.
        """
        scenarios = [*baselines, treatment]
        paired = results[results["scenario"].isin(scenarios)].pivot(
            index="seed", columns="scenario", values=metric
        )
        missing = [scenario for scenario in scenarios if scenario not in paired]
        if missing or paired.isna().to_numpy().any():
            raise ValueError(
                f"Every seed needs a {metric} value in each of {scenarios}; "
                f"missing scenarios: {missing or 'none'}"
            )
        sns.set_theme(style="whitegrid")
        figure, axes = plt.subplots(
            1, len(baselines), figsize=(6 * len(baselines), 5), sharey=True
        )
        axes = np.atleast_1d(axes)
        for axis, baseline in zip(axes, baselines, strict=True):
            favoured = 0
            for baseline_value, treatment_value in zip(
                paired[baseline], paired[treatment], strict=True
            ):
                improved = treatment_value < baseline_value
                favoured += improved
                axis.plot(
                    [0, 1],
                    [baseline_value, treatment_value],
                    marker="o",
                    color="tab:blue" if improved else "tab:red",
                    alpha=0.8,
                )
            axis.set_xticks(
                [0, 1], [SCENARIO_LABELS[baseline], SCENARIO_LABELS[treatment]]
            )
            axis.set_xlim(-0.3, 1.3)
            axis.set_title(
                f"{SCENARIO_LABELS[treatment]} lower in {favoured} of {len(paired)}"
            )
        axes[0].set_ylabel("Mean response time to high-risk patients (minutes)")
        figure.suptitle("Paired replications (same seed joined)")
        figure.tight_layout()
        return self.export_figure(figure, "paired_response_times.png")

    @staticmethod
    def sensitivity_curve(sensitivity: pd.DataFrame, parameter: str) -> pd.DataFrame:
        """Select the settings that form one sensitivity curve, adding back the default setting so the curve includes the main experiment's operating point.

        Args:
            sensitivity: Summarised sensitivity results carrying `swept`.
            parameter: Either 'theta' or 'tau'.

        Returns:
            The settings for this curve, sorted by the swept parameter.
        """
        other = "tau" if parameter == "theta" else "theta"
        default_other = sensitivity.loc[sensitivity["swept"] == parameter, other].iloc[
            0
        ]
        curve = sensitivity[
            (sensitivity["swept"] == parameter)
            | ((sensitivity["swept"] == other) & (sensitivity[other] == default_other))
        ]
        return curve.sort_values(parameter)

    def plot_sensitivity(
        self,
        sensitivity: pd.DataFrame,
        parameter: str,
        metric: str = "mean_response_time_high_risk",
    ) -> Path:
        """Plot the primary metric against one swept threshold, with 95 percent t interval error bars so a difference can be read against its spread.

        Args:
            sensitivity: Summarised sensitivity results carrying `swept`, the
                standard deviation of the primary metric and the replication
                count.
            parameter: Either 'theta' or 'tau'.
            metric: Column to plot.

        Returns:
            Path written.
        """
        sns.set_theme(style="whitegrid")
        curve = self.sensitivity_curve(sensitivity, parameter)
        n_replications = curve["n_replications"]
        half_width = (
            t.ppf(0.975, n_replications - 1)
            * curve["sd_response_time_high_risk"]
            / np.sqrt(n_replications)
        )
        figure, axis = plt.subplots(figsize=(7, 5))
        axis.errorbar(
            curve[parameter], curve[metric], yerr=half_width, marker="o", capsize=4
        )
        symbol = THRESHOLD_SYMBOLS[parameter]
        axis.set_xlabel(
            f"High-risk probability threshold ({symbol})"
            if parameter == "theta"
            else f"Time since last visit threshold ({symbol}, minutes)"
        )
        axis.set_ylabel("Mean response time to high-risk patients (minutes)")
        axis.set_title(f"Sensitivity to {symbol}")
        figure.tight_layout()
        return self.export_figure(figure, f"sensitivity_{parameter}.png")

    def plot_theta_tradeoff(self, sensitivity: pd.DataFrame) -> Path:
        """Plot re-solves per shift and the false trigger rate against theta, the gate costs that change with the threshold while response time barely moves.

        Args:
            sensitivity: Summarised sensitivity results carrying `swept`,
                `rerouting_events_per_shift` and `false_trigger_rate`.

        Returns:
            Path written.
        """
        sns.set_theme(style="whitegrid")
        curve = self.sensitivity_curve(sensitivity, "theta")
        symbol = THRESHOLD_SYMBOLS["theta"]
        figure, (events_axis, false_axis) = plt.subplots(1, 2, figsize=(12, 4.5))
        events_axis.plot(
            curve["theta"], curve["rerouting_events_per_shift"], marker="o"
        )
        events_axis.set_ylabel("Re-routing events per shift")
        events_axis.set_title("Re-solves")
        false_axis.plot(
            curve["theta"], curve["false_trigger_rate"], marker="o", color="tab:red"
        )
        false_axis.set_ylabel("False trigger rate")
        false_axis.set_title("False triggers")
        for axis in (events_axis, false_axis):
            axis.set_xlabel(f"High-risk probability threshold ({symbol})")
            axis.set_ylim(bottom=0)
        held_tau = curve["tau"].iloc[0]
        figure.suptitle(
            f"Gate activity against {symbol} "
            f"({THRESHOLD_SYMBOLS['tau']} = {held_tau:g} minutes)"
        )
        figure.tight_layout()
        return self.export_figure(figure, "sensitivity_theta_tradeoff.png")

    def plot_feature_importance(
        self, importances: dict[str, pd.Series], top_n: int = 15
    ) -> Path:
        """Plot the highest-ranked features for the tree ensembles.

        Args:
            importances: Classifier name mapped to a Series of importance
                values indexed by feature name.
            top_n: Number of features to show per classifier.

        Returns:
            Path written.
        """
        figure, axes = plt.subplots(
            1, len(importances), figsize=(7 * len(importances), 6)
        )
        axes = np.atleast_1d(axes)
        for axis, (classifier_name, importance) in zip(
            axes, importances.items(), strict=True
        ):
            ranked = importance.sort_values(ascending=False).head(top_n).iloc[::-1]
            axis.barh([feature_label(name) for name in ranked.index], ranked.to_numpy())
            axis.set_xlabel("Importance")
            axis.set_title(classifier_label(classifier_name))
        figure.suptitle(f"Top {top_n} features by importance")
        figure.tight_layout()
        return self.export_figure(figure, "feature_importance.png")


class StatisticalAnalyser:
    """Paired scenario comparisons with multiplicity control and effect sizes."""

    def __init__(self, alpha: float = FAMILY_WISE_ALPHA) -> None:
        """Args:
        alpha: Family-wise error rate across the pairwise comparisons.
        """
        self._alpha = alpha

    def compare_scenarios(
        self, results_a: list[float], results_b: list[float]
    ) -> dict[str, float]:
        """Compare one pair of scenarios across matched replications with a paired Wilcoxon test, since response times are bounded and censored so normality cannot be assumed.

        Args:
            results_a: Metric per replication under the first scenario.
            results_b: The same metric under the second, in the same order.

        Returns:
            Test statistic, p-value, effect size, and the mean difference,
            which carries the direction the effect size alone cannot.

        Raises:
            ValueError: If the two samples are not the same length.
        """
        if len(results_a) != len(results_b):
            raise ValueError(
                "Paired comparison requires equal-length samples: "
                f"{len(results_a)} and {len(results_b)}"
            )
        differences = np.asarray(results_a, dtype=float) - np.asarray(
            results_b, dtype=float
        )
        non_zero = differences[differences != 0]
        n_pairs = len(non_zero)
        mean_difference = float(np.mean(differences)) if len(differences) else 0.0

        # Identical replications leave nothing to test; scipy raises rather
        # than returning p=1.
        if n_pairs == 0:
            return {
                "statistic": 0.0,
                "p_value": 1.0,
                "rank_biserial_correlation": 0.0,
                "mean_difference": mean_difference,
                "n_pairs": 0,
            }

        statistic, p_value = wilcoxon(results_a, results_b)
        # scipy's two-sided statistic is already min(W+, W-), the smaller
        # signed-rank sum this formula needs.
        smaller_rank_sum = statistic
        rank_biserial = 1 - (4 * smaller_rank_sum) / (n_pairs * (n_pairs + 1))
        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "rank_biserial_correlation": float(rank_biserial),
            "mean_difference": mean_difference,
            "n_pairs": n_pairs,
        }

    def holm_bonferroni(self, p_values: list[float]) -> list[bool]:
        """Decide which hypotheses survive correction for multiple testing, using Holm's step-down procedure via statsmodels.

        Args:
            p_values: Unadjusted p-values, in any order.

        Returns:
            Rejection decision per p-value, in the input order.
        """
        reject, _, _, _ = multipletests(p_values, alpha=self._alpha, method="holm")
        return reject.tolist()

    def holm_adjusted_p_values(self, p_values: list[float]) -> list[float]:
        """Adjust p-values for multiple testing with Holm's step-down method, so reported values and `holm_bonferroni` decisions cannot disagree.

        Args:
            p_values: Unadjusted p-values, in any order.

        Returns:
            Holm-adjusted p-value per input, in the input order.
        """
        _, adjusted, _, _ = multipletests(p_values, alpha=self._alpha, method="holm")
        return adjusted.tolist()

    def interpret_effect_size(self, rank_biserial_correlation: float) -> str:
        """Label an effect size magnitude.

        Args:
            rank_biserial_correlation: Effect size between 0 and 1.

        Returns:
            One of 'negligible', 'small', 'medium' or 'large'.
        """
        magnitude = abs(rank_biserial_correlation)
        for threshold, label in EFFECT_SIZE_BANDS:
            if magnitude < threshold:
                return label
        return "large"

    def compare_all_scenarios(
        self, results: pd.DataFrame, metric: str = "mean_response_time_high_risk"
    ) -> pd.DataFrame:
        """Run every pairwise comparison and correct for multiplicity.

        Args:
            results: One row per replication per scenario.
            metric: Column to compare.

        Returns:
            One row per comparison, with corrected significance decisions.
        """
        by_scenario = {
            scenario: frame.sort_values("replication_id")[metric].tolist()
            for scenario, frame in results.groupby("scenario")
        }
        scenarios = sorted(by_scenario)

        rows = []
        for first_index, scenario_a in enumerate(scenarios):
            for scenario_b in scenarios[first_index + 1 :]:
                comparison = self.compare_scenarios(
                    by_scenario[scenario_a], by_scenario[scenario_b]
                )
                rows.append(
                    {
                        "metric": metric,
                        "scenario_a": scenario_a,
                        "scenario_b": scenario_b,
                        "mean_a": float(np.mean(by_scenario[scenario_a])),
                        "mean_b": float(np.mean(by_scenario[scenario_b])),
                        **comparison,
                        "effect_size_label": self.interpret_effect_size(
                            comparison["rank_biserial_correlation"]
                        ),
                    }
                )

        frame = pd.DataFrame(rows)
        frame["p_value_holm"] = self.holm_adjusted_p_values(frame["p_value"].tolist())
        frame["significant_after_holm"] = self.holm_bonferroni(
            frame["p_value"].tolist()
        )
        for row in frame.itertuples():
            logger.info(
                "%s vs %s: mean %.1f vs %.1f, p=%.4f, r=%.3f (%s), significant=%s",
                row.scenario_a,
                row.scenario_b,
                row.mean_a,
                row.mean_b,
                row.p_value,
                row.rank_biserial_correlation,
                row.effect_size_label,
                row.significant_after_holm,
            )
        return frame

    def compare_metrics(
        self, results: pd.DataFrame, metrics: list[str]
    ) -> pd.DataFrame:
        """Compare the scenarios on several metrics, correcting each independently since the secondary outcomes test distinct questions rather than restating the primary hypothesis.

        Args:
            results: One row per replication per scenario.
            metrics: Metric columns to compare, in report order.

        Returns:
            The per-comparison rows for every metric, concatenated, each
            carrying its own `metric` value and Holm decision.
        """
        return pd.concat(
            [self.compare_all_scenarios(results, metric=metric) for metric in metrics],
            ignore_index=True,
        )


class SensitivityAnalysis:
    """Sweeps the two AND-gate thresholds one at a time."""

    def __init__(self, config) -> None:
        """Args:
        config: Supplies the sweep values and the defaults held fixed.
        """
        self._config = config

    def parameter_settings(self) -> list[dict[str, float]]:
        """Enumerate the configurations to run, varying each parameter against the other held at its default rather than as a full grid, to keep the sweep to ten runs rather than thirty.

        Returns:
            One mapping per configuration, carrying theta, tau, and the name of
            the parameter being varied.
        """
        settings = []
        for theta in self._config.theta_values:
            settings.append(
                {"theta": theta, "tau": self._config.default_tau, "swept": "theta"}
            )
        for tau in self._config.tau_values:
            if tau == self._config.default_tau:
                # Already covered by the theta sweep at its default point.
                continue
            settings.append(
                {"theta": self._config.default_theta, "tau": tau, "swept": "tau"}
            )
        return settings

    def summarise(self, results: pd.DataFrame) -> pd.DataFrame:
        """Aggregate sensitivity replications into one row per configuration.

        Args:
            results: One row per replication, carrying theta, tau and swept.

        Returns:
            Mean primary metric and trigger behaviour per configuration, with
            the standard deviation of the primary metric so the robustness
            claim can be read with its spread.
        """
        return (
            results.groupby(["swept", "theta", "tau"], as_index=False)
            .agg(
                mean_response_time_high_risk=(
                    "mean_response_time_high_risk",
                    "mean",
                ),
                sd_response_time_high_risk=(
                    "mean_response_time_high_risk",
                    "std",
                ),
                rerouting_events_per_shift=("rerouting_events_per_shift", "mean"),
                false_trigger_rate=("false_trigger_rate", "mean"),
                unreached_high_risk_patients=("unreached_high_risk_patients", "mean"),
                n_replications=("replication_id", "count"),
            )
            .sort_values(["swept", "theta", "tau"])
            .reset_index(drop=True)
        )
