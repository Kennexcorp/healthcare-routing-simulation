"""Tests for statistical analysis and results output.

Covers `ResultsExporter`, the paired statistical comparison with Holm-Bonferroni
correction, and the theta/tau sensitivity sweep.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from src.analysis import (
    FIGURE_DPI,
    ResultsExporter,
    SensitivityAnalysis,
    StatisticalAnalyser,
    classifier_label,
    feature_label,
)
from src.config import SimulationConfig
from src.ml_pipeline import FeatureEngineer


@pytest.fixture
def observations() -> pd.DataFrame:
    """A small frame spanning all three risk classes."""
    return pd.DataFrame(
        {
            "spo2": [97.0, 93.0, 89.0, 96.5, 94.0, 90.0],
            "hr": [72.0, 105.0, 135.0, 70.0, 100.0, 130.0],
            "sbp": [125.0, 148.0, 94.0, 120.0, 145.0, 96.0],
            "risk_label": ["low", "moderate", "high", "low", "moderate", "high"],
        }
    )


class TestResultsExporter:
    def test_missing_directories_are_created(self, tmp_path):
        exporter = ResultsExporter(tmp_path / "results")
        assert (tmp_path / "results").is_dir()
        assert exporter.figures_directory.is_dir()

    def test_dataframe_is_written_without_the_index(self, tmp_path):
        frame = pd.DataFrame({"scenario": ["ai_integrated"], "response_time": [35.5]})
        destination = ResultsExporter(tmp_path).export_dataframe(frame, "results.csv")
        assert destination.exists()
        assert pd.read_csv(destination).equals(frame)

    def test_figure_is_written_at_print_resolution(self, tmp_path):
        figure, axis = plt.subplots()
        axis.plot([0, 1], [0, 1])
        destination = ResultsExporter(tmp_path).export_figure(figure, "figure.png")
        assert destination.exists()
        assert destination.parent.name == "figures"
        assert FIGURE_DPI == 300

    def test_figure_is_closed_after_writing(self, tmp_path):
        """Batch runs produce many figures; leaving them open exhausts memory
        and triggers matplotlib's open-figure warning."""
        figure, _ = plt.subplots()
        ResultsExporter(tmp_path).export_figure(figure, "figure.png")
        assert not plt.fignum_exists(figure.number)

    def test_biometric_distribution_plot_is_written(self, tmp_path, observations):
        destination = ResultsExporter(tmp_path).plot_biometric_distributions(
            observations
        )
        assert destination.exists()
        assert destination.name == "biometric_distributions.png"
        assert destination.stat().st_size > 0

    def test_biometric_plot_handles_a_missing_class(self, tmp_path, observations):
        """Smoke-scale runs can produce no high-risk observations at all; the
        plot must still render rather than failing on an absent category."""
        low_only = observations[observations["risk_label"] == "low"]
        assert ResultsExporter(tmp_path).plot_biometric_distributions(low_only).exists()


def make_results(
    means: dict[str, list[float]], **extra_metrics: dict[str, list[float]]
) -> pd.DataFrame:
    """Build a replication results frame from per-scenario metric values."""
    records = []
    for scenario, values in means.items():
        for replication_id, value in enumerate(values):
            record = {
                "replication_id": replication_id,
                "seed": replication_id + 1,
                "scenario": scenario,
                "mean_response_time_high_risk": value,
                "rerouting_events_per_shift": 3,
                "false_trigger_rate": 0.3,
                "unreached_high_risk_patients": 2,
            }
            for metric, per_scenario in extra_metrics.items():
                record[metric] = per_scenario[scenario][replication_id]
            records.append(record)
    return pd.DataFrame.from_records(records)


class TestStatisticalAnalyser:
    def test_a_consistent_improvement_is_detected(self):
        """Every replication improves, so the test should reject and the effect
        size should be maximal."""
        analyser = StatisticalAnalyser()
        baseline = [100.0, 110.0, 120.0, 130.0, 140.0, 150.0]
        improved = [90.0, 95.0, 100.0, 115.0, 120.0, 130.0]
        comparison = analyser.compare_scenarios(baseline, improved)
        assert comparison["p_value"] < 0.05
        assert comparison["rank_biserial_correlation"] == pytest.approx(1.0)
        assert comparison["mean_difference"] > 0

    def test_effect_size_matches_a_hand_computed_case(self):
        """Hand-computed check: the rank-biserial correlation should come out to 0.8."""
        analyser = StatisticalAnalyser()
        comparison = analyser.compare_scenarios(
            [10.0, 12.0, 13.0, 14.0], [11.0, 10.0, 10.0, 10.0]
        )
        assert comparison["n_pairs"] == 4
        assert comparison["rank_biserial_correlation"] == pytest.approx(0.8)

    def test_mean_difference_carries_the_direction(self):
        """Rank-biserial correlation is a magnitude only, so direction must come
        from the mean difference instead."""
        analyser = StatisticalAnalyser()
        worse = analyser.compare_scenarios(
            [100.0] * 5, [120.0, 130.0, 125.0, 140.0, 135.0]
        )
        better = analyser.compare_scenarios(
            [120.0, 130.0, 125.0, 140.0, 135.0], [100.0] * 5
        )
        assert worse["mean_difference"] < 0 < better["mean_difference"]
        assert worse["rank_biserial_correlation"] == better["rank_biserial_correlation"]

    def test_identical_scenarios_are_not_significant(self):
        """scipy raises on an all-zero difference vector, so this path must be
        handled rather than crashing the analysis stage."""
        analyser = StatisticalAnalyser()
        comparison = analyser.compare_scenarios([100.0] * 5, [100.0] * 5)
        assert comparison["p_value"] == 1.0
        assert comparison["rank_biserial_correlation"] == 0.0
        assert comparison["n_pairs"] == 0

    def test_mismatched_sample_lengths_are_rejected(self):
        """A paired test on unpaired data would be silently wrong."""
        with pytest.raises(ValueError, match="equal-length"):
            StatisticalAnalyser().compare_scenarios([1.0, 2.0], [1.0])

    def test_zero_differences_are_excluded_from_the_pair_count(self):
        analyser = StatisticalAnalyser()
        comparison = analyser.compare_scenarios([10.0, 10.0, 12.0], [10.0, 10.0, 8.0])
        assert comparison["n_pairs"] == 1


class TestHolmBonferroni:
    def test_the_smallest_p_value_faces_the_strictest_threshold(self):
        """With three tests the thresholds are 0.0167, 0.025 and 0.05."""
        analyser = StatisticalAnalyser(alpha=0.05)
        assert analyser.holm_bonferroni([0.01, 0.02, 0.04]) == [True, True, True]
        assert analyser.holm_bonferroni([0.02, 0.03, 0.04]) == [False, False, False]

    def test_rejection_stops_at_the_first_failure(self):
        """Holm is a step-down procedure, so once one hypothesis survives, every larger p-value survives with it."""
        analyser = StatisticalAnalyser(alpha=0.05)
        assert analyser.holm_bonferroni([0.001, 0.30, 0.04]) == [True, False, False]

    def test_decisions_are_returned_in_input_order(self):
        analyser = StatisticalAnalyser(alpha=0.05)
        assert analyser.holm_bonferroni([0.30, 0.001]) == [False, True]

    def test_correction_is_stricter_than_no_correction(self):
        """A p-value below alpha but above its Holm threshold must not survive."""
        analyser = StatisticalAnalyser(alpha=0.05)
        assert analyser.holm_bonferroni([0.02, 0.02, 0.02]) == [False, False, False]

    def test_adjusted_p_values_scale_by_the_number_of_tests_still_open(self):
        """Each sorted p-value is scaled by its remaining rank, then made non-decreasing."""
        adjusted = StatisticalAnalyser().holm_adjusted_p_values([0.01, 0.02, 0.04])
        assert adjusted == pytest.approx([0.03, 0.04, 0.04])

    def test_adjusted_p_values_are_returned_in_input_order(self):
        adjusted = StatisticalAnalyser().holm_adjusted_p_values([0.30, 0.001])
        assert adjusted == pytest.approx([0.30, 0.002])

    @pytest.mark.parametrize(
        "p_values", [[0.01, 0.02, 0.04], [0.02, 0.03, 0.04], [0.001, 0.30, 0.04]]
    )
    def test_adjusted_p_values_agree_with_the_rejection_decisions(self, p_values):
        analyser = StatisticalAnalyser(alpha=0.05)
        decisions = analyser.holm_bonferroni(p_values)
        adjusted = analyser.holm_adjusted_p_values(p_values)
        assert [value < 0.05 for value in adjusted] == decisions

    @pytest.mark.parametrize(
        ("correlation", "label"),
        [(0.05, "negligible"), (0.2, "small"), (0.4, "medium"), (0.7, "large")],
    )
    def test_effect_sizes_are_labelled_by_magnitude(self, correlation, label):
        assert StatisticalAnalyser().interpret_effect_size(correlation) == label


class TestCompareAllScenarios:
    def test_every_pair_is_compared_once(self):
        results = make_results(
            {
                "unprioritised": [150.0, 160.0, 155.0, 170.0, 165.0],
                "static_priority": [140.0, 145.0, 150.0, 155.0, 160.0],
                "ai_integrated": [120.0, 125.0, 130.0, 135.0, 140.0],
            }
        )
        comparisons = StatisticalAnalyser().compare_all_scenarios(results)
        assert len(comparisons) == 3
        pairs = set(zip(comparisons["scenario_a"], comparisons["scenario_b"]))
        assert len(pairs) == 3

    def test_the_output_carries_both_means_so_direction_is_readable(self):
        results = make_results(
            {
                "ai_integrated": [120.0, 125.0, 130.0, 135.0, 140.0],
                "unprioritised": [150.0, 160.0, 155.0, 170.0, 165.0],
            }
        )
        row = StatisticalAnalyser().compare_all_scenarios(results).iloc[0]
        assert row["mean_a"] < row["mean_b"]
        assert "significant_after_holm" in row

    def test_the_output_carries_the_holm_adjusted_p_value(self):
        results = make_results(
            {
                "ai_integrated": [120.0, 125.0, 130.0, 135.0, 140.0],
                "static_priority": [140.0, 150.0, 145.0, 160.0, 155.0],
                "unprioritised": [150.0, 160.0, 155.0, 170.0, 165.0],
            }
        )
        comparisons = StatisticalAnalyser().compare_all_scenarios(results)
        assert (comparisons["p_value_holm"] >= comparisons["p_value"]).all()
        assert (comparisons["p_value_holm"] <= 1.0).all()

    def test_replications_are_paired_by_id_not_by_row_order(self):
        """Scenarios share seeds, so pairing must follow replication_id even
        when the rows arrive shuffled."""
        results = make_results(
            {
                "unprioritised": [150.0, 160.0, 155.0, 170.0, 165.0],
                "ai_integrated": [120.0, 125.0, 130.0, 135.0, 140.0],
            }
        )
        shuffled = results.sample(frac=1.0, random_state=3)
        ordered = StatisticalAnalyser().compare_all_scenarios(results)
        from_shuffled = StatisticalAnalyser().compare_all_scenarios(shuffled)
        assert ordered["p_value"].tolist() == from_shuffled["p_value"].tolist()


class TestCompareMetrics:
    def test_every_metric_gets_its_own_three_comparisons(self):
        results = make_results(
            {
                "ai_integrated": [120.0, 125.0, 130.0, 135.0, 140.0],
                "static_priority": [140.0, 145.0, 150.0, 155.0, 160.0],
                "unprioritised": [150.0, 160.0, 155.0, 170.0, 165.0],
            },
            total_travel_distance_km={
                "ai_integrated": [200.0, 205.0, 210.0, 215.0, 220.0],
                "static_priority": [150.0, 152.0, 154.0, 156.0, 158.0],
                "unprioritised": [180.0, 185.0, 175.0, 190.0, 182.0],
            },
        )
        comparisons = StatisticalAnalyser().compare_metrics(
            results, ["mean_response_time_high_risk", "total_travel_distance_km"]
        )
        assert set(comparisons["metric"]) == {
            "mean_response_time_high_risk",
            "total_travel_distance_km",
        }
        assert (comparisons.groupby("metric").size() == 3).all()

    def test_holm_is_applied_within_a_metric_not_across_the_pool(self):
        """Adding a second metric to the list must not change the first
        metric's Holm decisions: each metric is its own family."""
        results = make_results(
            {
                "ai_integrated": [120.0, 125.0, 130.0, 135.0, 140.0],
                "static_priority": [140.0, 145.0, 150.0, 155.0, 160.0],
                "unprioritised": [150.0, 160.0, 155.0, 170.0, 165.0],
            },
            total_travel_distance_km={
                "ai_integrated": [200.0, 205.0, 210.0, 215.0, 220.0],
                "static_priority": [150.0, 152.0, 154.0, 156.0, 158.0],
                "unprioritised": [180.0, 185.0, 175.0, 190.0, 182.0],
            },
        )
        analyser = StatisticalAnalyser()
        pooled = analyser.compare_metrics(
            results, ["mean_response_time_high_risk", "total_travel_distance_km"]
        )
        primary = pooled[pooled["metric"] == "mean_response_time_high_risk"]
        direct = analyser.compare_all_scenarios(results)
        assert primary["p_value"].tolist() == direct["p_value"].tolist()
        assert (
            primary["significant_after_holm"].tolist()
            == direct["significant_after_holm"].tolist()
        )


class TestSensitivityAnalysis:
    def test_each_parameter_is_swept_against_the_other_at_its_default(self):
        config = SimulationConfig()
        settings = SensitivityAnalysis(config).parameter_settings()
        theta_settings = [s for s in settings if s["swept"] == "theta"]
        tau_settings = [s for s in settings if s["swept"] == "tau"]
        assert [s["theta"] for s in theta_settings] == config.theta_values
        assert all(s["tau"] == config.default_tau for s in theta_settings)
        assert all(s["theta"] == config.default_theta for s in tau_settings)

    def test_the_shared_default_point_is_not_run_twice(self):
        """theta at its default and tau at its default are the same
        configuration; running it twice would waste a full replication set."""
        settings = SensitivityAnalysis(SimulationConfig()).parameter_settings()
        assert len(settings) == 7
        assert len({(s["theta"], s["tau"]) for s in settings}) == 7

    def test_summary_reports_one_row_per_configuration(self):
        config = SimulationConfig()
        rows = []
        for setting in SensitivityAnalysis(config).parameter_settings():
            for replication_id in range(3):
                rows.append(
                    {
                        "replication_id": replication_id,
                        "mean_response_time_high_risk": 100.0 + replication_id,
                        "rerouting_events_per_shift": 4,
                        "false_trigger_rate": 0.25,
                        "unreached_high_risk_patients": 1,
                        **setting,
                    }
                )
        summary = SensitivityAnalysis(config).summarise(pd.DataFrame(rows))
        assert len(summary) == 7
        assert (summary["n_replications"] == 3).all()
        assert summary["mean_response_time_high_risk"].iloc[0] == pytest.approx(101.0)

    def test_summary_reports_the_response_time_standard_deviation(self):
        config = SimulationConfig()
        rows = []
        for setting in SensitivityAnalysis(config).parameter_settings():
            for replication_id, value in enumerate([90.0, 100.0, 110.0]):
                rows.append(
                    {
                        "replication_id": replication_id,
                        "mean_response_time_high_risk": value,
                        "rerouting_events_per_shift": 4,
                        "false_trigger_rate": 0.25,
                        "unreached_high_risk_patients": 1,
                        **setting,
                    }
                )
        summary = SensitivityAnalysis(config).summarise(pd.DataFrame(rows))
        assert summary["sd_response_time_high_risk"].iloc[0] == pytest.approx(10.0)


class TestFigureLabels:
    @pytest.mark.parametrize(
        ("name", "label"),
        [
            ("logistic_regression", "Logistic regression"),
            ("random_forest", "Random forest"),
            ("xgboost", "XGBoost"),
            ("xgboost (calibrated)", "XGBoost (calibrated)"),
            ("unlisted_model", "unlisted_model"),
        ],
    )
    def test_classifier_keys_become_their_print_names(self, name, label):
        assert classifier_label(name) == label

    @pytest.mark.parametrize(
        ("name", "label"),
        [
            ("zscore_hr", "Heart rate z-score"),
            ("zscore_spo2", "SpO$_2$ z-score"),
            ("rolling_mean_spo2", "Rolling mean SpO$_2$"),
            ("rolling_std_sbp", "Rolling SD systolic BP"),
            ("delta_hr", "Change in heart rate"),
            ("sbp", "Current systolic BP"),
            ("diagnosis_COPD", "Diagnosis: COPD"),
            ("age", "Age"),
            ("time_since_last_visit_mins", "Time since last visit"),
        ],
    )
    def test_feature_names_become_readable_labels(self, name, label):
        assert feature_label(name) == label

    def test_every_engineered_feature_has_a_label_that_is_not_its_code_name(self):
        for name in FeatureEngineer().get_feature_names_out():
            assert feature_label(name) != name

    def test_an_unrecognised_feature_name_is_left_as_it_is(self):
        assert feature_label("not_a_feature") == "not_a_feature"


class TestComparisonFigures:
    def test_response_time_comparison_is_written(self, tmp_path):
        results = make_results(
            {
                "unprioritised": [150.0, 160.0],
                "static_priority": [140.0, 145.0],
                "ai_integrated": [120.0, 125.0],
            }
        )
        destination = ResultsExporter(tmp_path).plot_response_time_comparison(results)
        assert destination.name == "response_time_comparison.png"
        assert destination.stat().st_size > 0

    def test_paired_response_times_figure_is_written(self, tmp_path):
        results = make_results(
            {
                "unprioritised": [150.0, 160.0],
                "static_priority": [140.0, 145.0],
                "ai_integrated": [120.0, 150.0],
            }
        )
        destination = ResultsExporter(tmp_path).plot_paired_response_times(results)
        assert destination.name == "paired_response_times.png"
        assert destination.stat().st_size > 0

    def test_paired_figure_rejects_a_seed_missing_from_one_scenario(self, tmp_path):
        """An unmatched replication has no partner to join, so drawing it
        would silently misrepresent the paired design."""
        results = make_results(
            {
                "unprioritised": [150.0, 160.0],
                "static_priority": [140.0, 145.0],
                "ai_integrated": [120.0],
            }
        )
        with pytest.raises(ValueError, match="Every seed"):
            ResultsExporter(tmp_path).plot_paired_response_times(results)

    def test_paired_figure_rejects_a_missing_scenario(self, tmp_path):
        results = make_results(
            {"static_priority": [140.0, 145.0], "ai_integrated": [120.0, 125.0]}
        )
        with pytest.raises(ValueError, match="unprioritised"):
            ResultsExporter(tmp_path).plot_paired_response_times(results)

    def test_theta_tradeoff_figure_is_written(self, tmp_path):
        config = SimulationConfig()
        rows = [
            {
                "swept": setting["swept"],
                "theta": setting["theta"],
                "tau": setting["tau"],
                "rerouting_events_per_shift": 30.0 * (1 - setting["theta"]),
                "false_trigger_rate": 1 - setting["theta"],
            }
            for setting in SensitivityAnalysis(config).parameter_settings()
        ]
        destination = ResultsExporter(tmp_path).plot_theta_tradeoff(pd.DataFrame(rows))
        assert destination.name == "sensitivity_theta_tradeoff.png"
        assert destination.stat().st_size > 0

    @pytest.mark.parametrize("parameter", ["theta", "tau"])
    def test_a_sensitivity_figure_is_written_per_parameter(self, tmp_path, parameter):
        config = SimulationConfig()
        rows = [
            {
                "swept": setting["swept"],
                "theta": setting["theta"],
                "tau": setting["tau"],
                "mean_response_time_high_risk": 100.0,
                "sd_response_time_high_risk": 10.0,
                "n_replications": 10,
            }
            for setting in SensitivityAnalysis(config).parameter_settings()
        ]
        destination = ResultsExporter(tmp_path).plot_sensitivity(
            pd.DataFrame(rows), parameter
        )
        assert destination.name == f"sensitivity_{parameter}.png"

    @pytest.mark.parametrize(
        ("parameter", "expected_values"),
        [
            ("theta", [0.5, 0.6, 0.7, 0.8, 0.9]),
            ("tau", [30, 60, 120]),
        ],
    )
    def test_each_curve_includes_the_default_setting(self, parameter, expected_values):
        config = SimulationConfig()
        rows = [
            {
                "swept": setting["swept"],
                "theta": setting["theta"],
                "tau": setting["tau"],
            }
            for setting in SensitivityAnalysis(config).parameter_settings()
        ]
        curve = ResultsExporter.sensitivity_curve(pd.DataFrame(rows), parameter)
        assert list(curve[parameter]) == expected_values
