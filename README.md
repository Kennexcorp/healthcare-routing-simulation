# An AI-Integrated Framework for Predictive Patient Risk Assessment and Dynamic Routing Optimisation in Community Healthcare Services

MSc Applied Artificial Intelligence dissertation artefact (CO7047, University
of Chester). A simulation study on synthetic data: a three-state continuous-time
Markov chain drives patient deterioration, a classifier predicts risk from
biometric features, and Google OR-Tools solves a priority-constrained dynamic
vehicle routing problem over a community nursing shift. No real patient data
is involved.

**Research question:** to what extent can integrating predictive patient risk
modelling with dynamic routing optimisation improve response times for
high-risk patients in community healthcare services?

## Method

Each simulated shift runs 160 patients and 10 community nursing workers over
480 minutes. Three routing scenarios are compared under a common random number
strategy, so differences between them are attributable to the routing
strategy rather than to different random draws:

| Scenario            | Behaviour                                                         |
| ------------------- | ----------------------------------------------------------------- |
| `unprioritised`   | Nearest-neighbour heuristic, no priorities, no re-routing         |
| `static_priority` | NEWS2 thresholds at shift start, OR-Tools solved once, no updates |
| `ai_integrated`   | AND-gate trigger drives mid-shift re-solves                       |

The primary outcome metric is `mean_response_time_high_risk`.

The AND-gate trigger is the novel contribution: it fires a mid-shift re-solve
only when three conditions hold at once for a patient — an ML-predicted
high-risk probability above a threshold, an adverse deviation from that
patient's own biometric baseline, and enough time elapsed since their last
visit.

## Repository layout

| Path                          | Contents                                                                             |
| ----------------------------- | ------------------------------------------------------------------------------------ |
| `main.py`                   | Command-line entry point; dispatches to each pipeline stage                          |
| `src/config.py`             | `SimulationConfig` and the clinical design constants                               |
| `src/models.py`             | `Patient`, `Worker`, `TriggerEvent`, `RoutingSolution`, `SimulationResult` |
| `src/data_generator.py`     | `CTMCModel`, `BiometricGenerator`, `NEWS2Scorer`, `SyntheticCohort`          |
| `src/ml_pipeline.py`        | `FeatureEngineer`, `RiskClassifier`, `ModelEvaluator`                          |
| `src/routing.py`            | `DistanceMatrix`, `RoutingModel`, `AbstractRouter` and its three routers       |
| `src/trigger.py`            | `ANDGateTrigger`                                                                   |
| `src/simulation.py`         | `SimulationState`, `Simulation`, `ReplicationManager`                          |
| `src/analysis.py`           | `StatisticalAnalyser`, `SensitivityAnalysis`, `ResultsExporter`                |
| `scripts/capacity_sweep.py` | Standalone script measuring baseline coverage across cohort sizes                    |
| `tests/`                    | One `test_*.py` module per `src/` module                                          |
| `models/`                   | Fitted risk classifier used for the reported results                                 |
| `results/`                  | Generated CSVs, dissertation artefacts                                               |
| `results/figures/`          | Generated figures at 300 DPI, dissertation artefacts                                 |

## Requirements

- Python 3.13 or higher
- [uv](https://docs.astral.sh/uv/) for dependency management

## Installation

```bash
uv sync
```

This creates the virtual environment and installs every dependency pinned in
`uv.lock`.

## Configuration

All simulation parameters are `SimulationConfig` fields (`src/config.py`),
overridable through `SIM_`-prefixed environment variables or a `.env` file.
Clinical design constants (CTMC rates, biometric distributions, NEWS2 bands,
priority weights) are fixed module constants and are not configurable.

`.env.example` documents every override, under two profiles:

- **Full specification profile** — the `SimulationConfig` defaults; this is
  the profile used to produce the results reported in the dissertation.
- **Smoke profile** — a small cohort and short solver limits for local
  development and testing. Output produced under this profile is not a
  dissertation artefact.

Copy `.env.example` to `.env` to apply a profile, or set any variable inline
for a single run.

## Running the pipeline

Every stage is run through `main.py`, in this order:

```bash
uv run python main.py --stage generate   # synthetic cohort + descriptive stats
uv run python main.py --stage train      # classifier training and calibration
uv run python main.py --stage simulate   # the three-scenario experiment
uv run python main.py --stage analyse    # statistical comparison + sensitivity sweep
```

`simulate` requires a trained model produced by `train`. The fitted model used
for the reported results is included at `models/risk_classifier.joblib`, so
`simulate` and `analyse` can be run without retraining; running `train`
regenerates it. The file is a joblib (pickle) archive, so load it only from
this repository. `generate`, `train` and `analyse` each read and write their
own artefacts under `results/`.

For local development, override the full-specification profile with a small
cohort rather than iterating at full scale:

```bash
SIM_N_PATIENTS=20 SIM_N_WORKERS=2 SIM_INITIAL_BATCH_REPS=5 \
SIM_INITIAL_SOLVE_LIMIT=5 SIM_REROUTE_SOLVE_LIMIT=2 \
uv run python main.py --stage simulate
```

## Results

### CSV files (`results/`)

| File                         | Contents                                                                            |
| ---------------------------- | ----------------------------------------------------------------------------------- |
| `descriptive_stats.csv`    | Mean and SD of each biometric per risk class, class distribution                    |
| `ml_results.csv`           | Macro-F1, per-class recall, ROC-AUC, Brier score for every classifier candidate     |
| `feature_importance.csv`   | Feature importances for the tree-based classifiers                                  |
| `simulation_results.csv`   | Every replication result across all three scenarios                                 |
| `sensitivity_analysis.csv` | Mean response time across the theta and tau sweep                                   |
| `statistical_tests.csv`    | Wilcoxon statistics, Holm-Bonferroni adjusted p-values, effect sizes                |
| `capacity_sweep.csv`       | Unprioritised-baseline coverage across cohort sizes (`scripts/capacity_sweep.py`) |

### Figures (`results/figures/`)

| File                             | Contents                                                 |
| -------------------------------- | -------------------------------------------------------- |
| `biometric_distributions.png`  | SpO2/HR/SBP distributions per risk class                 |
| `calibration_curve.png`        | Reliability diagram for the selected classifier          |
| `roc_curves.png`               | ROC curves for every classifier candidate                |
| `feature_importance.png`       | Feature importance plot                                  |
| `response_time_comparison.png` | Response time across the three routing scenarios         |
| `sensitivity_theta.png`        | Response time against the ML risk threshold theta        |
| `sensitivity_tau.png`          | Response time against the time-since-visit threshold tau |

## Testing and linting

```bash
uv run pytest                              # full test suite
uv run pytest tests/test_trigger.py        # one module
uv run ruff check .                        # lint
uv run ruff format .                       # format
```

## Reproducibility

- **Random seeds.** All randomness is drawn from `numpy.random.default_rng`,
  seeded explicitly and never set globally.

  - The main experiment and the sensitivity sweep share one seed sequence,
    `1, 2, 3, ...`, one seed per replication, used identically across all
    three scenarios so scenario comparisons are paired. The sensitivity
    sweep reuses a prefix of this same sequence.
  - The ML training corpus is generated from a disjoint sequence starting at
    `training_seed_base` (default `90000`), one seed per training shift, so
    no patient trajectory is both trained on and evaluated in the simulation.
  - The patient-level train/calibration/test split uses a fixed seed
    (`SPLIT_SEED = 42` in `src/config.py`), not varied per replication.
- **Dependency versions.** `uv.lock`, committed at the repository root, pins
  the exact version of every dependency.
- **Replication count and confidence interval.** The achieved replication
  count and the `confidence_interval_half_width` column are recorded per
  scenario directly in `results/simulation_results.csv`; that file is the
  authoritative source for both figures.
