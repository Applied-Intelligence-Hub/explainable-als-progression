"""
Step 8 — Explainability: SHAP (global + local) + LIME (local) + LR coefficients.

Protocol
--------
1. Fit the selected XGBoost configuration on DEV only
2. SHAP global: mean-|SHAP| bar plot + beeswarm (summary) plot
3. SHAP local:  waterfall plots for 4 representative patients (TP, FP, FN, TN)
4. LIME local:  explanations for the same 4 patients (side-by-side with SHAP)
5. LR coefficients: bar chart from the balanced LR fitted on DEV only
6. SHAP vs LIME concordance table (top-N feature overlap)

Usage
-----
  cd g:\\TESE\\CodigoMain
  python 03_src/analysis/step8_explainability.py

Outputs
-------
  04_outputs/figures/step8_xai/shap_bar_global.pdf / .png
  04_outputs/figures/step8_xai/shap_beeswarm.pdf / .png
  04_outputs/figures/step8_xai/shap_waterfall_TP.pdf / .png
  04_outputs/figures/step8_xai/shap_waterfall_FP.pdf / .png
  04_outputs/figures/step8_xai/shap_waterfall_FN.pdf / .png
  04_outputs/figures/step8_xai/shap_waterfall_TN.pdf / .png
  04_outputs/figures/step8_xai/lime_local_TP.pdf / .png
  04_outputs/figures/step8_xai/lime_local_FP.pdf / .png
  04_outputs/figures/step8_xai/lime_local_FN.pdf / .png
  04_outputs/figures/step8_xai/lime_local_TN.pdf / .png
  04_outputs/figures/step8_xai/lr_coefficients.pdf / .png
  04_outputs/tables/step8_xai/shap_feature_importance.csv
  04_outputs/tables/step8_xai/shap_vs_lime_concordance.csv
  04_outputs/tables/step8_xai/lr_coefficients.csv
  04_outputs/tables/step8_xai/case_study_profiles.csv
"""

from __future__ import annotations

import json
import os
import time
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

import shap
import sklearn
from lime.lime_tabular import LimeTabularExplainer

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from study1_style import (
    FIGURE_INK,
    RISK_DECREASE,
    RISK_INCREASE,
    SHAP_CMAP,
    apply_study1_style,
    style_axis,
)

apply_study1_style()


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
PROCESSED    = os.path.join("01_data", "processed")
OUT_FIG      = os.path.join("04_outputs", "figures", "step8_xai")
OUT_TAB      = os.path.join("04_outputs", "tables", "step8_xai")
OVERLEAF_FIG = os.path.join("overleaf", "images", "figures")
OVERLEAF_TAB = os.path.join("overleaf", "images", "tables")
TUNING_DIR    = os.path.join("04_outputs", "tables", "step5_tuning")

HORIZON    = "6m"
SEED       = 42
RAPID_FRAC = 0.30
THRESHOLD  = 0.21          # F1-optimal threshold from Step 6

N_TOP_FEATURES  = 15       # for bar chart / concordance
N_LIME_SAMPLES  = 2000     # perturbation samples for LIME

DROP_COLS = {
    "subject_id",
    "t0_delta_days",
    "vitals_delta_days",
    "fvc_delta_days",
    "slope_90d_per_30d",
    "slope_180d_per_30d",
    "ALSFRS_Responded_By",
}

# Readable feature names for plots
FEATURE_LABELS = {
    "ALSFRS_R_t0":                     "ALSFRS-R total (baseline)",
    "Age":                             "Age",
    "Sex":                             "Sex",
    "Ethnicity":                       "Ethnicity",
    "Race_Caucasian":                  "Race: Caucasian",
    "Race_Black_African_American":     "Race: Black/African-American",
    "Race_Asian":                      "Race: Asian",
    "Race_Americ_Indian_Alaska_Native":"Race: Am. Indian/Alaska Native",
    "Race_Hawaiian_Pacific_Islander":  "Race: Hawaiian/Pacific Isl.",
    "Race_Other":                      "Race: Other",
    "Race_Unknown":                    "Race: Unknown",
    "R_1_Dyspnea":                     "ALSFRS-R item: Dyspnea",
    "R_2_Orthopnea":                   "ALSFRS-R item: Orthopnea",
    "R_3_Respiratory_Insufficiency":   "ALSFRS-R item: Resp. Insufficiency",
    "Mode_of_Administration":          "Mode of Administration",
    "Weight_kg_t0":                    "Weight (kg)",
    "Height_cm_t0":                    "Height (cm)",
    "BMI_t0":                          "BMI",
    "Pulse":                           "Pulse",
    "Respiratory_Rate":                "Respiratory Rate",
    "Temperature":                     "Temperature",
    "Blood_Pressure_Systolic":         "BP Systolic",
    "Blood_Pressure_Diastolic":        "BP Diastolic",
    "Baseline_Standing_BP_Systolic":   "Standing BP Systolic",
    "Baseline_Standing_BP_Diastolic":  "Standing BP Diastolic",
    "Baseline_Supine_BP_Systolic":     "Supine BP Systolic",
    "Baseline_Supine_BP_Diastolic":    "Supine BP Diastolic",
    "FVC_Liters_best_t0":             "FVC (litres)",
    "FVC_pctNormal_best_t0":          "FVC (% normal)",
    "subject_normal":                  "FVC subject normal flag",
    "riluzole_pre_t0":                 "Riluzole (pre-baseline)",
    "study_arm":                       "Study arm",
    "n_conmeds_pre_t0":                "Concomitant medications (n)",
    "alt_t0":                          "ALT (baseline)",
    "creatinine_t0":                   "Creatinine (baseline)",
}

ONE_HOT_PREFIXES = {
    "Sex_":                    "Sex: ",
    "Ethnicity_":              "Ethnicity: ",
    "Mode_of_Administration_": "Administration: ",
    "study_arm_":              "Study arm: ",
}


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def ensure_dirs():
    for d in (OUT_FIG, OUT_TAB, OVERLEAF_FIG, OVERLEAF_TAB):
        os.makedirs(d, exist_ok=True)


def save_fig(fig, name):
    """Save figure to outputs + overleaf, both PDF and PNG."""
    for d in (OUT_FIG, OVERLEAF_FIG):
        fig.savefig(os.path.join(d, f"{name}.pdf"), dpi=300, bbox_inches="tight")
        fig.savefig(os.path.join(d, f"{name}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {name}")


def readable(name):
    """Return a readable label for a feature name."""
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    for prefix, label in ONE_HOT_PREFIXES.items():
        if name.startswith(prefix):
            return f"{label}{name[len(prefix):].replace('_', ' ')}"
    return name.replace("_", " ")


def recolour_directional_bars(fig):
    """Apply Study I directional colours to horizontal explanation bars."""
    for ax in fig.axes:
        for patch in ax.patches:
            face = to_rgba(patch.get_facecolor())
            if hasattr(patch, "get_width"):
                colour = RISK_INCREASE if patch.get_width() >= 0 else RISK_DECREASE
            else:
                colour = RISK_INCREASE if face[0] > face[2] else RISK_DECREASE
            patch.set_facecolor(colour)
            patch.set_edgecolor(FIGURE_INK)
        for text in ax.texts:
            colour = to_rgba(text.get_color())
            if max(colour[:3]) - min(colour[:3]) > 0.35:
                text.set_color(
                    RISK_INCREASE if colour[0] > colour[2] else RISK_DECREASE
                )
        style_axis(ax, "x")


def load_data():
    """Load the fixed DEV and TEST partitions with features and labels."""
    ds_path = os.path.join(PROCESSED, f"dataset_{HORIZON}_v2.csv")
    hs_path = os.path.join(PROCESSED, f"holdout_split_{HORIZON}.csv")

    df = pd.read_csv(ds_path)
    hs = pd.read_csv(hs_path)

    slope_col = "slope_180d_per_30d"
    df[slope_col] = pd.to_numeric(df[slope_col], errors="coerce")
    df = df.dropna(subset=[slope_col])

    dev_ids  = set(hs.loc[hs["partition"] == "dev",  "subject_id"])
    test_ids = set(hs.loc[hs["partition"] == "test", "subject_id"])

    df_dev  = df[df["subject_id"].isin(dev_ids)].copy()
    df_test = df[df["subject_id"].isin(test_ids)].copy()

    # Label threshold from DEV only
    cutoff = float(df_dev[slope_col].quantile(RAPID_FRAC))

    feat_cols = [c for c in df.columns if c not in DROP_COLS]

    X_dev = df_dev[feat_cols].reset_index(drop=True)
    y_dev = (df_dev[slope_col] <= cutoff).astype(int).reset_index(drop=True)
    X_test = df_test[feat_cols].reset_index(drop=True)
    y_test = (df_test[slope_col] <= cutoff).astype(int).reset_index(drop=True)
    ids_test = df_test["subject_id"].reset_index(drop=True)

    print(f"  DEV n={len(X_dev)}, Rapid prevalence={y_dev.mean():.3f}")
    print(f"  TEST n={len(X_test)}, Rapid prevalence={y_test.mean():.3f}")
    print(f"  Slope cutoff (DEV P30): {cutoff:.4f}")
    return X_dev, y_dev, X_test, y_test, ids_test, feat_cols, cutoff


def load_params(model_key, mode):
    """Load the selected hyperparameters saved during Step 5."""
    path = os.path.join(TUNING_DIR, f"{model_key}_{mode}_{HORIZON}_best.json")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)["full_params"], path


def build_preprocessor(X, scale_numeric=False):
    """Build the same numeric/categorical preprocessing used for evaluation."""
    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    numeric_steps = [("imp", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("sc", StandardScaler()))

    numeric_pipe = Pipeline(numeric_steps)
    categorical_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])


def train_dev_models(X_dev, y_dev):
    """Fit the selected XGB and balanced LR configurations on DEV only."""
    xgb_params, xgb_path = load_params("XGB", "unbalanced")
    lr_params, lr_path = load_params("LR", "balanced")
    sklearn_version = tuple(
        int(part) for part in sklearn.__version__.split(".")[:2]
    )
    if sklearn_version >= (1, 8):
        penalty = lr_params.pop("penalty", None)
        if penalty == "l1":
            lr_params["l1_ratio"] = 1
        elif penalty == "l2":
            lr_params["l1_ratio"] = 0

    pipe_xgb = Pipeline([
        ("prep", build_preprocessor(X_dev)),
        ("clf", XGBClassifier(**xgb_params)),
    ])
    pipe_lr = Pipeline([
        ("prep", build_preprocessor(X_dev, scale_numeric=True)),
        ("clf", LogisticRegression(**lr_params, class_weight="balanced")),
    ])

    print("  Fitting selected XGBoost configuration on DEV...")
    pipe_xgb.fit(X_dev, y_dev)
    print("  Fitting balanced Logistic Regression configuration on DEV...")
    pipe_lr.fit(X_dev, y_dev)
    return pipe_xgb, pipe_lr, xgb_path, lr_path


def pick_case_studies(y_true, y_pred, proba):
    """
    Select one representative patient from each confusion-matrix quadrant.
    Pick the patient closest to the quadrant's median predicted probability,
    so the case study is 'typical' rather than extreme.
    """
    cases = {}
    for label, mask_fn in [
        ("TP", lambda yt, yp: (yt == 1) & (yp == 1)),
        ("FP", lambda yt, yp: (yt == 0) & (yp == 1)),
        ("FN", lambda yt, yp: (yt == 1) & (yp == 0)),
        ("TN", lambda yt, yp: (yt == 0) & (yp == 0)),
    ]:
        mask = mask_fn(y_true, y_pred)
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            print(f"  [WARN] No {label} cases found - skipping")
            continue
        # pick patient closest to quadrant median probability
        med = np.median(proba[idxs])
        best = idxs[np.argmin(np.abs(proba[idxs] - med))]
        cases[label] = int(best)
    return cases


# ─────────────────────────────────────────────────────────────
# SHAP analysis
# ─────────────────────────────────────────────────────────────
def run_shap(pipe, X_test, feat_cols, case_indices, subject_ids):
    """Compute SHAP values using TreeExplainer on the XGBoost model."""
    print("\n-- SHAP analysis --")

    # Extract the preprocessor and the XGBoost booster from the pipeline
    preproc = pipe.named_steps["prep"]
    clf     = pipe.named_steps["clf"]

    X_processed = preproc.transform(X_test)

    # Get transformed feature names
    transformed_names = []
    for name, trans, cols in preproc.transformers_:
        if name == "num":
            transformed_names.extend(cols)
        elif name == "cat":
            if hasattr(trans, "named_steps"):
                ohe = (trans.named_steps.get("onehot")
                       or trans.named_steps.get("oh"))
            else:
                ohe = trans[-1]
            transformed_names.extend(ohe.get_feature_names_out(cols).tolist())

    if isinstance(X_processed, np.ndarray):
        X_df = pd.DataFrame(X_processed, columns=transformed_names)
    else:
        X_df = pd.DataFrame(X_processed.toarray(), columns=transformed_names)

    # TreeSHAP (exact, fast)
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer(X_df)
    shap_values.feature_names = [readable(name) for name in transformed_names]

    # ── Global: mean |SHAP| bar chart ──
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": transformed_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    importance_df.to_csv(os.path.join(OUT_TAB, "shap_feature_importance.csv"), index=False)

    top_n = importance_df.head(N_TOP_FEATURES)

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(
        [readable(f) for f in top_n["feature"].values[::-1]],
        top_n["mean_abs_shap"].values[::-1],
        color=FIGURE_INK, edgecolor=FIGURE_INK, height=0.65,
    )
    # annotate values
    for bar, val in zip(bars, top_n["mean_abs_shap"].values[::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8, color=FIGURE_INK)
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title("Global Feature Importance — XGBoost (6-month, TEST set)",
                 fontsize=12, fontweight="bold")
    style_axis(ax, "x")
    fig.tight_layout()
    save_fig(fig, "shap_bar_global")

    # ── Global: beeswarm ──
    fig, ax = plt.subplots(figsize=(11, 8))
    shap.plots.beeswarm(
        shap_values,
        max_display=N_TOP_FEATURES,
        color=SHAP_CMAP,
        plot_size=None,
        show=False,
    )
    fig = plt.gcf()
    ax.set_title("SHAP Beeswarm — XGBoost (6-month, TEST set)",
                 fontsize=13, fontweight="bold")
    ax.tick_params(axis="both", labelsize=10)
    ax.xaxis.label.set_size(11)
    style_axis(ax, "x")
    for axis in fig.axes[1:]:
        axis.tick_params(labelsize=9)
        axis.grid(False)
    fig.tight_layout()
    save_fig(fig, "shap_beeswarm")

    # ── Local: waterfall plots for case studies ──
    for label, idx in case_indices.items():
        fig, ax = plt.subplots(figsize=(8, 6))
        shap.plots.waterfall(shap_values[idx], max_display=12, show=False)
        subject_id = int(subject_ids.iloc[idx])
        plt.title(f"SHAP Waterfall — {label} (Subject ID {subject_id})",
                  fontsize=11, fontweight="bold")
        fig = plt.gcf()
        recolour_directional_bars(fig)
        fig.tight_layout()
        save_fig(fig, f"shap_waterfall_{label}")

    return shap_values, importance_df, X_df, transformed_names


# ─────────────────────────────────────────────────────────────
# LIME analysis
# ─────────────────────────────────────────────────────────────
def run_lime(pipe, X_test, X_dev, case_indices, transformed_names, subject_ids):
    """Generate LIME explanations for the same 4 case-study patients.

    LIME requires numeric input, so we operate on the preprocessed
    (imputed + encoded) feature space and wrap only the classifier
    for predictions.
    """
    print("\n-- LIME analysis --")

    preproc = pipe.named_steps["prep"]
    clf     = pipe.named_steps["clf"]

    # Transform train/test to numeric arrays
    X_dev_proc  = preproc.transform(X_dev)
    X_test_proc = preproc.transform(X_test)

    if not isinstance(X_dev_proc, np.ndarray):
        X_dev_proc  = X_dev_proc.toarray()
    if not isinstance(X_test_proc, np.ndarray):
        X_test_proc = X_test_proc.toarray()

    readable_names = [readable(f) for f in transformed_names]

    # Predict wrapper — clf expects already-transformed data
    def predict_fn(X):
        return clf.predict_proba(X)

    explainer = LimeTabularExplainer(
        training_data=X_dev_proc,
        feature_names=readable_names,
        class_names=["Slow", "Rapid"],
        mode="classification",
        random_state=SEED,
        discretize_continuous=True,
    )

    lime_rankings = {}

    for label, idx in case_indices.items():
        instance = X_test_proc[idx]

        explanation = explainer.explain_instance(
            instance,
            predict_fn,
            num_features=N_TOP_FEATURES,
            num_samples=N_LIME_SAMPLES,
            labels=(1,),
        )

        # Get the explanation for the positive class (Rapid = 1)
        exp_map = explanation.as_map()[1]
        lime_rankings[label] = [
            transformed_names[feature_idx]
            for feature_idx, _ in exp_map[:N_TOP_FEATURES]
        ]

        # Generate LIME figure
        fig = explanation.as_pyplot_figure(label=1)
        recolour_directional_bars(fig)
        fig.set_size_inches(9, 6)
        subject_id = int(subject_ids.iloc[idx])
        fig.suptitle(f"LIME Explanation — {label} (Subject ID {subject_id})",
                     fontsize=11, fontweight="bold", y=1.02)
        fig.tight_layout()
        save_fig(fig, f"lime_local_{label}")

    return lime_rankings


# ─────────────────────────────────────────────────────────────
# Concordance: SHAP vs LIME
# ─────────────────────────────────────────────────────────────
def concordance_table(lime_rankings, case_indices, shap_values,
                      transformed_names, top_k=10):
    """Compare local top-K feature rankings from SHAP and LIME."""
    print("\n-- SHAP vs LIME concordance --")

    rows = []
    for label, idx in case_indices.items():
        if label not in lime_rankings:
            continue

        local_abs = np.abs(shap_values.values[idx])
        shap_indices = np.argsort(local_abs)[::-1][:top_k]
        shap_features = [transformed_names[i] for i in shap_indices]
        lime_features = lime_rankings[label][:top_k]
        overlap = set(shap_features) & set(lime_features)

        rows.append({
            "case": label,
            "shap_top_k": ", ".join(readable(f) for f in shap_features),
            "lime_top_k": ", ".join(readable(f) for f in lime_features),
            "overlap_count": len(overlap),
            "overlap_features": ", ".join(
                sorted(readable(f) for f in overlap)
            ),
        })

    conc_df = pd.DataFrame(rows)
    conc_df.to_csv(os.path.join(OUT_TAB, "shap_vs_lime_concordance.csv"), index=False)
    print(f"  [OK] Concordance table saved ({len(rows)} cases, top-{top_k})")
    return conc_df


# ─────────────────────────────────────────────────────────────
# LR coefficient analysis
# ─────────────────────────────────────────────────────────────
def run_lr_coefficients(pipe_lr):
    """Extract and plot coefficients from the balanced LR fitted on DEV."""
    print("\n-- Logistic Regression coefficients --")

    preproc = pipe_lr.named_steps["prep"]
    clf     = pipe_lr.named_steps["clf"]

    # Get transformed feature names
    transformed_names = []
    for name, trans, cols in preproc.transformers_:
        if name == "num":
            transformed_names.extend(cols)
        elif name == "cat":
            if hasattr(trans, "named_steps"):
                ohe = (trans.named_steps.get("onehot")
                       or trans.named_steps.get("oh"))
            else:
                ohe = trans[-1]
            transformed_names.extend(ohe.get_feature_names_out(cols).tolist())

    coefs = clf.coef_[0]
    coef_df = pd.DataFrame({
        "feature": transformed_names,
        "coefficient": coefs,
        "abs_coefficient": np.abs(coefs),
    }).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)

    coef_df.to_csv(os.path.join(OUT_TAB, "lr_coefficients.csv"), index=False)

    # Plot top coefficients
    top = coef_df.head(N_TOP_FEATURES).copy()
    top = top.sort_values("abs_coefficient", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = [RISK_INCREASE if c > 0 else RISK_DECREASE
              for c in top["coefficient"]]
    bars = ax.barh(
        [readable(f) for f in top["feature"]],
        top["coefficient"],
        color=colors, edgecolor=FIGURE_INK, height=0.65,
    )
    # annotate
    for bar, val in zip(bars, top["coefficient"]):
        offset = 0.01 if val >= 0 else -0.01
        ha = "left" if val >= 0 else "right"
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center", ha=ha, fontsize=8, color=FIGURE_INK)
    ax.axvline(0, color=FIGURE_INK, lw=0.8, ls="--")
    ax.set_xlabel("Model coefficient", fontsize=11)
    ax.set_title("Logistic Regression Coefficients — Top Features (6-month)",
                 fontsize=12, fontweight="bold")

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color=RISK_INCREASE, label="Increases rapid-progression risk"),
        Patch(color=RISK_DECREASE, label="Decreases rapid-progression risk"),
    ], loc="lower left", ncol=1, fontsize=7.5)

    style_axis(ax, "x")
    fig.tight_layout()
    save_fig(fig, "lr_coefficients")
    print(f"  [OK] LR coefficients: {len(coef_df)} features")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    t_start = time.time()
    ensure_dirs()

    # ── Load data and fit selected configurations on DEV only ──
    print("Loading data and fitting DEV-only models...")
    X_dev, y_dev, X_test, y_test, ids_test, feat_cols, cutoff = load_data()
    pipe_xgb, pipe_lr, xgb_params_path, lr_params_path = train_dev_models(
        X_dev, y_dev
    )

    # ── Predictions on TEST ──
    proba = pipe_xgb.predict_proba(X_test)[:, 1]
    y_pred = (proba >= THRESHOLD).astype(int)
    pr_auc = average_precision_score(y_test, proba)
    roc_auc = roc_auc_score(y_test, proba)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f"  Predictions: Rapid={y_pred.sum()}, Slow={len(y_pred) - y_pred.sum()}")
    print(f"  TEST PR-AUC={pr_auc:.6f}, ROC-AUC={roc_auc:.6f}")
    print(f"  Confusion matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    predictions_df = pd.DataFrame({
        "subject_id": ids_test,
        "y_true": y_test,
        "proba_rapid": proba,
        "y_pred": y_pred,
    })
    predictions_df.to_csv(
        os.path.join(OUT_TAB, "test_predictions.csv"), index=False
    )

    metadata = {
        "training_partition": "DEV only",
        "explanation_partition": "TEST",
        "n_dev": len(X_dev),
        "n_test": len(X_test),
        "rapid_progression_cutoff": cutoff,
        "classification_threshold": THRESHOLD,
        "xgb_parameter_file": xgb_params_path,
        "lr_parameter_file": lr_params_path,
        "test_pr_auc": pr_auc,
        "test_roc_auc": roc_auc,
        "confusion_matrix": {
            "true_positive": int(tp),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_negative": int(tn),
        },
    }
    with open(
        os.path.join(OUT_TAB, "xai_run_metadata.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(metadata, handle, indent=2)

    # ── Select case studies ──
    case_indices = pick_case_studies(y_test, y_pred, proba)
    print(f"  Case studies: {case_indices}")

    # Save case-study profiles
    profiles = []
    for label, idx in case_indices.items():
        row = X_test.iloc[idx].to_dict()
        row["case"] = label
        row["y_true"] = int(y_test.iloc[idx])
        row["y_pred"] = int(y_pred[idx])
        row["proba_rapid"] = float(proba[idx])
        row["subject_id"] = int(ids_test.iloc[idx])
        profiles.append(row)
    profiles_df = pd.DataFrame(profiles)
    profiles_df.to_csv(os.path.join(OUT_TAB, "case_study_profiles.csv"), index=False)

    # ── SHAP ──
    shap_values, shap_imp, X_proc, transformed_names = run_shap(
        pipe_xgb, X_test, feat_cols, case_indices, ids_test)

    # ── LIME ──
    lime_rankings = run_lime(
        pipe_xgb, X_test, X_dev, case_indices, transformed_names, ids_test
    )

    # ── Concordance ──
    concordance_table(
        lime_rankings, case_indices, shap_values, transformed_names
    )

    # ── LR coefficients ──
    run_lr_coefficients(pipe_lr)

    elapsed = time.time() - t_start
    print(f"\n{'='*50}")
    print(f"Step 8 complete in {elapsed:.1f}s")
    print(f"  Figures: {OUT_FIG}")
    print(f"  Tables:  {OUT_TAB}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
