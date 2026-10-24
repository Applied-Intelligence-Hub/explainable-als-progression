# Explainable ALS Functional Progression

The study classifies rapid functional decline at three and six months using PRO-ACT clinical-trial records. Baseline predictors are separated in time from the follow-up observations used to estimate ALSFRS-R slopes. Model development also separates patients, estimates target cutoffs within training data, and selects a decision threshold from development predictions.

![Study I pipeline](assets/study1_pipeline.png)

## Study design

| Component | Definition |
|---|---|
| Outcome | Rapid versus slow ALSFRS-R decline at 90 or 180 days |
| Baseline | Earliest valid ALSFRS-R total score and assessment date |
| Target | Ordinary least-squares slope, expressed in points per 30 days |
| Eligibility | An assessment within the endpoint band of +/-7 days and at least two distinct assessment times |
| Cohorts | 2,407 patients at three months; 1,392 at six months |
| Predictors | 35 baseline clinical features in the final v2 dataset |
| Classifiers | KNN, Decision Tree, Random Forest, SVM, Logistic Regression, LightGBM, XGBoost |
| Imbalance conditions | Unbalanced and balanced modes where supported; 13 configurations at six months |
| Model selection | Optuna optimisation of cross-validated PR-AUC (average precision) |
| Primary validation | Patient-level 80/20 split and grouped five-fold cross-validation within development data |
| Complementary validation | Fixed-parameter stratified five-fold CV repeated three times, DEV only |
| Explanations | SHAP, LIME, and logistic-regression coefficients |

Later clinical records do not enter the baseline feature matrix. During cross-validation, the 30th-percentile slope cutoff is estimated from each training fold. Held-out labels use the development-partition cutoff. Global labels are used only to stratify the initial split, not as the modelling targets. Imputation and any feature scaling are fitted within the training data.

## Reported results

These values are the dissertation results, not results from a new run of this repository.

| Six-month XGBoost result | Value |
|---|---|
| Mean development CV PR-AUC | 0.434 |
| Held-out PR-AUC | 0.456, 95% CI [0.364, 0.568] |
| Held-out patients | 279, including 84 rapid progressors |
| Development-selected threshold | 0.21 |
| Recall at that threshold | 85.7% (72/84) |
| Precision at that threshold | 35.1% |
| False positives | 133 |

![Held-out precision-recall curves](assets/step6_pr_curves.png)

The threshold maximises development F1 and is an exploratory operating point. Its high recall comes with many false positives; no clinical cost matrix or intervention was specified. PR-AUC describes score ranking and should be interpreted alongside the confusion matrix and threshold-dependent metrics.

Baseline ALSFRS-R and FVC had the largest global SHAP contributions. FVC in litres was missing for 43.7% of the six-month cohort, so its attribution must be read alongside the missing-data analysis. The supplementary scripts examine feature ablation, calibration, subgroup errors, decision curves, learning curves, bootstrap intervals, partial dependence, and imputation sensitivity.

![Global XGBoost SHAP analysis](assets/shap_beeswarm.png)

## Repository layout

```text
01_data/
  raw/               Authorised PRO-ACT CSV files, stored locally
  interim/           Baseline table and follow-up targets, generated locally
  processed/         Feature tables, datasets and split manifests, generated locally
02_notebooks/        Target construction, baseline feature assembly and development notebooks
03_src/
  data/              Baseline construction, dataset assembly, cleaning and fixed splits
  tuning/            Unified Optuna search; archive_v1 contains earlier tuning scripts
  evaluation/        Held-out evaluation and development-only threshold selection
  analysis/          Explainability, figures and supplementary analyses
  training/          Full-cohort prototype model fitting
  report/            Prototype report and synthetic-profile generators
04_outputs/          Published selected parameters; other generated outputs stay local
models/              Locally generated fitted-model artefacts
assets/              Selected published figures used in this README
requirements.txt     Python dependencies
```

## Installation

Use Python 3.12 (verified locally with 3.12.10). From the repository root:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The original environment was not archived as a historical lock. Core dependencies in `requirements.txt` are now pinned to the Python 3.12.10 environment inspected on 23 October 2026. `requirements-verified.txt` records its full installed-package inventory, including transitive and development dependencies. This snapshot does not establish the environment at the time of every original run, or guarantee bitwise agreement across platforms. Exact partition reproduction also requires the original row order and verification against the split manifests described below.

The scripts do not require a notebook interface. For optional notebook execution/export, install `requirements-notebooks.txt` after the core requirements; those interface tools were not part of the verified runtime snapshot.

## Data preparation

Obtain authorised access through the [PRO-ACT platform](https://ncri1.partners.org/proact) and place the original `PROACT_*.csv` tables in `01_data/raw/`. The analysis uses the extract's original column names.

Run the following from the repository root. The two notebooks execute from `02_notebooks/`, because their paths are relative to that directory. Executed copies are written to the ignored output directory.

```bash
python 03_src/data/build_baseline.py
cd 02_notebooks
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output-dir ../04_outputs/executed_notebooks 01a_step3_build_targets.ipynb
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output-dir ../04_outputs/executed_notebooks 02_step4_build_features.ipynb
cd ..
python 03_src/data/build_datasets_v1.py
python 03_src/data/build_features_v2.py
python 03_src/data/fix_vitals_and_rebuild.py
python 03_src/data/create_holdout_split.py
```

`build_baseline.py` reconstructs the original 5,436-row baseline table. `build_datasets_v1.py` retains the feature-table ordering used for the original seeded split and prefers the original `baseline_with_targets_step3_nolabels.csv` when that authorised local artefact is available. Otherwise it uses the newly generated v2 target file. Final model development uses the v2 predictors.

For an exact reproduction of the dissertation's patient assignments, retain the original local split manifests. The split script preserves existing valid manifests instead of overwriting them. Recreating slopes can introduce floating-point differences around the 30th-percentile boundary: two near-tied three-month slopes differed in the available target exports, which changes their stratification labels and can change the seeded split. Row ordering also matters. The repository preserves the analysis algorithm but does not distribute the patient-level artefacts needed to guarantee identical historical assignments.

The source notebooks contain no saved outputs. Some additional notebooks and `tuning/archive_v1/` record earlier development experiments; they are not the execution path for the final held-out results. `02b_step4_build_features_v2.ipynb` provides the notebook counterpart of v2 feature assembly, while the sequence above uses the scripts.

## Primary analysis

Run scripts from the repository root:

```bash
python 03_src/tuning/optuna_tune_all.py --horizon 6m
python 03_src/tuning/optuna_tune_all.py --horizon 3m --models LR RF LGBM XGB
python 03_src/evaluation/evaluate_holdout.py
python 03_src/analysis/step8_explainability.py
```

The tuning script accepts `--help`, `--models`, `--trials` and `--unbalanced-only`. Six-month tuning covers all 13 supported configurations. The three-month command reproduces the four model families used in the cross-horizon comparison. Held-out evaluation and the main explainability script use the six-month cohort.

## Supplementary analyses

After preparing the data and obtaining the selected parameters:

```bash
python 03_src/data/analyze_fvc_missingness.py
python 03_src/analysis/ablation_v2.py
python 03_src/analysis/step7_statistical_comparison.py
python 03_src/analysis/step9_supplementary_analyses.py
python 03_src/analysis/missing_data_sensitivity.py
python 03_src/analysis/generate_study1_eda_figures.py
python 03_src/analysis/generate_step5_visuals.py
python 03_src/analysis/generate_3m_vs_6m_visuals.py
```

The `training/` and `report/` scripts support the dissertation's reporting prototype. Those training scripts fit the complete cohort and must not be used to produce the held-out evaluation. The primary evaluation and explainability scripts fit their models on development patients only.

Some figure scripts also create an `overleaf/` directory with duplicate figure exports. A LaTeX project is not required to run the analyses.

## Reviewer follow-up checks

After generating the original datasets, split manifests, tuning histories and
held-out predictions, run:

```bash
python 03_src/analysis/reviewer_followup.py
```

This supplementary script reads the locally retained predictions from Step 8. It
checks the test identifiers, labels and confusion matrix before computing baseline
and selection summaries, subgroup recall intervals, decision-curve intervals,
scaled Brier score, descriptive calibration diagnostics, encoding summaries and
the Study I trial-history audit. It writes aggregate outputs to
`04_outputs/reviewer_followup/` and never refits a predictive model. Bootstrap
intervals use 2,000 resamples by default (`--bootstrap`). `--project-root` and
`--output` can point to another authorised local study directory and output path.
The assertions intentionally check the dissertation's cohorts and original split;
they should not be weakened to make a different data extract appear identical.

The threshold plot now leaves precision undefined when no positive prediction is
made; the selected thresholds are unchanged. The LR coefficient plot has extra
space for negative value labels. These are display corrections, not model changes.

### Fixed-parameter validation and encoding sensitivity

The primary model, test partition and threshold remain unchanged. Additional
checks use the saved configurations without another Optuna search:

```bash
python 03_src/analysis/repeated_stratified_validation.py
python 03_src/analysis/race_encoding_sensitivity.py
python 03_src/analysis/review_lime_categorical.py
python 03_src/analysis/test_supplementary_revision.py
```

The first command uses the original DEV manifests, with five stratified folds
repeated three times (seed 42). Provisional DEV labels balance fold allocation;
each training fold still supplies its own target cutoff and fitted preprocessing.
It evaluates 13 six-month and eight three-month saved configurations, giving 315
fits, without accessing TEST. Balanced RF leads the six-month check at AP 0.422,
while the original selected XGBoost configuration gives 0.414. This assesses
partition sensitivity of selected parameters, not nested validation or a reason
to select a replacement after inspecting TEST. Overlapping folds are not used
as independent observations in new significance tests.

The 21 selected six-/three-month parameter JSONs are included under
`04_outputs/tables/step5_tuning/`; they contain configuration and aggregate scores,
not patient records. The fixed-parameter commands can reuse them without running
Optuna again. Reproducing the original fold assignments still requires authorised
data and the original local split manifests. Do not overwrite the published
reference configurations with a new search and then call it the same sensitivity.

Run the race command after the repeated-validation command. It adds 90 six-month
DEV fits for fixed XGBoost-unbalanced and LR-balanced configurations:

| Configuration | Numeric median AP | Recorded states AP | Race omitted AP |
| --- | ---: | ---: | ---: |
| XGBoost unbalanced | 0.414495 | 0.415943 | 0.417128 |
| LR balanced | 0.397072 | 0.397376 | 0.397381 |

Recorded states distinguish explicit zero, explicit one and an unrecorded field,
without forcing a mutually exclusive race category. The numeric reference must
reproduce the repeated-validation scores. No TEST evaluation or threshold
selection is performed. Small changes in ranking do not validate the original
median coding, establish fairness or make coefficients comparable across scales.

### Categorical-aware LIME

The current Step 8 and the focused `review_lime_categorical.py` script sample
one-hot groups as original categorical variables and retain observed binary or
discrete categories. The focused script checks the stored probabilities and
four case assignments before regenerating only LIME and grouped SHAP/LIME
comparisons. SHAP contributions are summed within original variables before
absolute ranking. The primary test AP remains 0.456462 and threshold 0.21.

Categorical validity is not local fidelity: weighted surrogate R2 ranges from
0.198 to 0.456, and the false-negative surrogate crosses the selected threshold.
Sampling variables separately can still produce implausible clinical combinations.
The focused script requires the original local Step 8 prediction/profile files;
these private artefacts are not included in this repository.

New aggregate scores and warnings are written to
`04_outputs/reviewer_followup/revised_analyses/`. Fold membership and local case
weights stay under ignored `01_data/interim/`; prediction/profile files also
remain ignored. The tests use synthetic fixtures and do not retrain the study
models. Selected aggregate results are in
[`assets/reviewer_followup/results.json`](assets/reviewer_followup/results.json).

Patient separation does not establish source-trial independence. No reliable
trial-membership variable was reconstructed, so no leave-trial-out validation is
claimed. The original primary race encoding is retained with its limitation
explicit; the sensitivity analyses do not remove trial-recording effects or
constitute a clean-environment reproduction.

## Data use and limitations

This repository distributes code and selected published figures. Source data, patient-level derivatives, predictions, profiles, explanation matrices and fitted models remain local and are ignored by Git. PRO-ACT access and reuse remain subject to its [terms and conditions](https://ncri1.partners.org/ProACT/Document/DisplayLatest/1); the code licence does not grant permission to redistribute those records.

The study is secondary analysis of de-identified trial records, with no new participant recruitment or clinical data collection. Its results provide internal validation in PRO-ACT. External and prospective evaluation, calibration and an explicit clinical decision objective are needed before clinical use.

## licence

The repository retains the GNU GPL v3 [licence](LICENSE) supplied by Applied Intelligence Hub.
