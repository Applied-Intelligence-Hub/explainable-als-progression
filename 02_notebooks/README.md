# Notebook guide

The primary preparation sequence uses `01a_step3_build_targets.ipynb` and `02_step4_build_features.ipynb`, followed by the scripts listed in the root README. Run notebook kernels with `02_notebooks/` as their working directory.

| Notebook | Role |
|---|---|
| `01a_step3_build_targets.ipynb` | Primary 90- and 180-day slope construction |
| `01b_step3_eda_slopes.ipynb` | Optional descriptive checks of slope coverage and quality |
| `02_step4_build_features.ipynb` | Primary vital-sign and FVC feature construction |
| `02b_step4_build_features_v2.ipynb` | Notebook counterpart of v2 feature assembly; the scripted route is recommended |
| `03_step5_train_models.ipynb` | Earlier development experiments, superseded by unified Optuna tuning and held-out evaluation |
| `04_step6_sanity_check.ipynb` | Earlier full-cohort prototype checks for Logistic Regression |
| `04_step6_sanity_check_xgb.ipynb` | Earlier full-cohort prototype checks for XGBoost |
| `06_eda_treatment_progression.ipynb` | Descriptive treatment and medication analyses |

Use `03_src/evaluation/evaluate_holdout.py` for the reported held-out evaluation and `03_src/analysis/step8_explainability.py` for development-only model fitting and the final SHAP/LIME analyses. Earlier prototype checks do not reproduce the final validation protocol.

Source notebooks are committed without outputs. The root README's `nbconvert` commands place executed copies in the ignored output directory.
