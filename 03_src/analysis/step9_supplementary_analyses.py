"""
Step 9 — Supplementary Analyses: Error profiling, Subgroup evaluation,
         Decision Curve Analysis, Learning Curves, Bootstrap CIs, and PDPs.

Protocol
--------
1. Load final XGBoost model + TEST set (holdout_split_6m)
2. Error analysis:  systematic profiling of FP vs FN vs TP vs TN by clinical
   variables (age, sex, ALSFRS-R severity, FVC status)
3. Subgroup analysis: PR-AUC, recall, precision stratified by age group, sex,
   ALSFRS-R severity tertile, FVC tertile
4. Decision Curve Analysis: net-benefit curves (model vs treat-all vs treat-none)
5. Learning Curves: PR-AUC on TEST as a function of DEV training-set size
6. Bootstrap confidence intervals for TEST-set metrics (PR-AUC, ROC-AUC, recall,
   precision, F1, Brier)
7. Partial Dependence Plots for top-3 SHAP features

Usage
-----
  cd g:\\TESE\\CodigoMain
  python 03_src/analysis/step9_supplementary_analyses.py

Outputs
-------
  04_outputs/figures/step9_supplementary/error_profile_age.pdf / .png
  04_outputs/figures/step9_supplementary/error_profile_alsfrs.pdf / .png
  04_outputs/figures/step9_supplementary/error_profile_sex.pdf / .png
  04_outputs/figures/step9_supplementary/error_profile_fvc.pdf / .png
  04_outputs/figures/step9_supplementary/subgroup_pr_auc.pdf / .png
  04_outputs/figures/step9_supplementary/subgroup_recall.pdf / .png
  04_outputs/figures/step9_supplementary/decision_curve.pdf / .png
  04_outputs/figures/step9_supplementary/learning_curve.pdf / .png
  04_outputs/figures/step9_supplementary/bootstrap_ci.pdf / .png
  04_outputs/figures/step9_supplementary/pdp_top3.pdf / .png
  04_outputs/tables/step9_supplementary/error_profile_summary.csv
  04_outputs/tables/step9_supplementary/error_profile_counts.csv
  04_outputs/tables/step9_supplementary/subgroup_metrics.csv
  04_outputs/tables/step9_supplementary/decision_curve_data.csv
  04_outputs/tables/step9_supplementary/learning_curve_data.csv
  04_outputs/tables/step9_supplementary/learning_curve_repeats.csv
  04_outputs/tables/step9_supplementary/bootstrap_ci.csv
  04_outputs/tables/step9_supplementary/pdp_data.csv
  04_outputs/tables/step9_supplementary/bootstrap_distributions.csv
"""

from __future__ import annotations

import json
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBClassifier
from study1_style import (
    FIGURE_INK,
    PALETTE,
    apply_study1_style,
    style_axis,
)

apply_study1_style()


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
PROCESSED    = os.path.join("01_data", "processed")
TUNING_DIR   = os.path.join("04_outputs", "tables", "step5_tuning")
OUT_FIG      = os.path.join("04_outputs", "figures", "step9_supplementary")
OUT_TAB      = os.path.join("04_outputs", "tables", "step9_supplementary")
OVERLEAF_FIG = os.path.join("overleaf", "images", "figures")

HORIZON    = "6m"
SEED       = 42
RAPID_FRAC = 0.30
THRESHOLD  = 0.21          # F1-optimal threshold from Step 6
N_SPLITS   = 5

DROP_COLS = {
    "subject_id",
    "t0_delta_days",
    "vitals_delta_days",
    "fvc_delta_days",
    "slope_90d_per_30d",
    "slope_180d_per_30d",
    "ALSFRS_Responded_By",
}

# Learning-curve training fractions
LC_FRACTIONS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
LC_REPEATS = 20

# Bootstrap
N_BOOTSTRAP = 2000
CI_ALPHA     = 0.05     # 95% CI

# PDP top features (from SHAP global analysis)
PDP_FEATURES = ["ALSFRS_R_t0", "FVC_Liters_best_t0", "Age"]

# Readable feature names for plots
FEATURE_LABELS = {
    "ALSFRS_R_t0":                     "ALSFRS-R total (baseline)",
    "Age":                             "Age",
    "FVC_Liters_best_t0":              "FVC (litres)",
    "FVC_pctNormal_best_t0":           "FVC (% normal)",
    "Respiratory_Rate":                "Respiratory Rate",
    "Pulse":                           "Pulse",
    "alt_t0":                          "ALT (baseline)",
    "Temperature":                     "Temperature",
    "Weight_kg_t0":                    "Weight (kg)",
    "BMI_t0":                          "BMI",
    "creatinine_t0":                   "Creatinine (baseline)",
    "n_conmeds_pre_t0":                "Concomitant medications (n)",
    "Height_cm_t0":                    "Height (cm)",
}


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def ensure_dirs():
    for d in (OUT_FIG, OUT_TAB, OVERLEAF_FIG):
        os.makedirs(d, exist_ok=True)


def save_fig(fig, name):
    """Save figure to outputs + overleaf, both PDF and PNG."""
    for d in (OUT_FIG, OVERLEAF_FIG):
        fig.savefig(os.path.join(d, f"{name}.pdf"), dpi=300, bbox_inches="tight")
        fig.savefig(os.path.join(d, f"{name}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {name}")


def load_data():
    """Load DEV + TEST partitions with features and labels."""
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

    cutoff = float(df_dev[slope_col].quantile(RAPID_FRAC))

    feat_cols = [c for c in df.columns if c not in DROP_COLS]

    X_dev  = df_dev[feat_cols].reset_index(drop=True)
    X_test = df_test[feat_cols].reset_index(drop=True)
    y_dev  = (df_dev[slope_col] <= cutoff).astype(int).reset_index(drop=True)
    y_test = (df_test[slope_col] <= cutoff).astype(int).reset_index(drop=True)
    groups_dev = df_dev["subject_id"].reset_index(drop=True)

    # Keep raw clinical values for stratification (before pipeline transforms)
    raw_test = df_test[["subject_id", "Age", "Sex",
                        "ALSFRS_R_t0", "FVC_pctNormal_best_t0"]].reset_index(drop=True)

    print(f"  DEV n={len(X_dev)},  TEST n={len(X_test)}")
    print(f"  Slope cutoff (DEV P30): {cutoff:.4f}")
    print(f"  Rapid prevalence — DEV: {y_dev.mean():.3f}  TEST: {y_test.mean():.3f}")
    return X_dev, y_dev, groups_dev, X_test, y_test, feat_cols, cutoff, raw_test


def build_preprocessor(X: pd.DataFrame, scale: bool) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    if scale:
        num_pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler()),
        ])
    else:
        num_pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
        ])

    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )


def load_xgb_params():
    """Load best Optuna hyperparameters for XGBoost unbalanced."""
    path = os.path.join(TUNING_DIR, f"XGB_unbalanced_{HORIZON}_best.json")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["full_params"]


def train_xgb_pipeline(X_train, y_train, params):
    """Train XGBoost pipeline on given data and return fitted pipeline."""
    preproc = build_preprocessor(X_train, scale=False)
    p = dict(params)
    p.setdefault("random_state", SEED)
    p["use_label_encoder"] = False
    p["eval_metric"] = "logloss"
    clf = XGBClassifier(**p)
    pipe = Pipeline([("pre", preproc), ("clf", clf)])
    pipe.fit(X_train, y_train)
    return pipe


# ─────────────────────────────────────────────────────────────
# 1. Error Analysis
# ─────────────────────────────────────────────────────────────
def run_error_analysis(pipe, X_test, y_test, raw_test):
    """Profile TP/FP/FN/TN by clinical subgroups."""
    print("\n── Error Analysis ──")

    proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (proba >= THRESHOLD).astype(int)

    # Assign quadrant
    quads = pd.Series("TN", index=y_test.index)
    quads[(y_test == 1) & (y_pred == 1)] = "TP"
    quads[(y_test == 0) & (y_pred == 1)] = "FP"
    quads[(y_test == 1) & (y_pred == 0)] = "FN"

    prof = raw_test.copy()
    prof["quadrant"] = quads.values
    prof["proba_rapid"] = proba

    # Age groups
    prof["Age_group"] = pd.cut(
        prof["Age"],
        bins=[0, 49, 57, 65, np.inf],
        labels=["≤49", "50-57", "58-65", ">65"],
    ).cat.add_categories(["Missing"]).fillna("Missing")

    # ALSFRS-R severity
    prof["ALSFRS_severity"] = pd.cut(
        prof["ALSFRS_R_t0"],
        bins=[0, 35, 39, 42, np.inf],
        labels=["≤35 (severe)", "36-39", "40-42", ">42 (mild)"],
    ).cat.add_categories(["Missing"]).fillna("Missing")

    # FVC status
    prof["FVC_group"] = pd.cut(
        prof["FVC_pctNormal_best_t0"],
        bins=[0, 77, 87, 98, np.inf],
        labels=["≤77%", "78-87%", "88-98%", ">98%"],
    ).cat.add_categories(["Missing"]).fillna("Missing")

    # Save full profile table
    prof.to_csv(os.path.join(OUT_TAB, "error_profile_summary.csv"), index=False)
    _save_error_counts(prof)

    # --- Plot 1: Error distribution by Age ---
    _plot_error_bars(prof, "Age_group", "error_profile_age",
                     "Distribution of Prediction Outcomes by Age Group")

    # --- Plot 2: Error distribution by ALSFRS-R severity ---
    _plot_error_bars(prof, "ALSFRS_severity", "error_profile_alsfrs",
                     "Distribution of Prediction Outcomes by ALSFRS-R Severity")

    # --- Plot 3: Error distribution by Sex ---
    _plot_error_bars(prof, "Sex", "error_profile_sex",
                     "Distribution of Prediction Outcomes by Sex")

    # --- Plot 4: Error distribution by FVC ---
    _plot_error_bars(prof, "FVC_group", "error_profile_fvc",
                     "Distribution of Prediction Outcomes by FVC (% normal)")

    print(f"  Quadrant counts: {quads.value_counts().to_dict()}")
    return prof


def _save_error_counts(prof):
    """Save counts and within-group proportions for each stratification."""
    rows = []
    for group_col, label in [
        ("Age_group", "Age"),
        ("Sex", "Sex"),
        ("ALSFRS_severity", "ALSFRS-R severity"),
        ("FVC_group", "FVC (% normal)"),
    ]:
        ct = pd.crosstab(prof[group_col], prof["quadrant"], dropna=False)
        for quadrant in ["TP", "FP", "FN", "TN"]:
            if quadrant not in ct.columns:
                ct[quadrant] = 0
        ct = ct[["TP", "FP", "FN", "TN"]]

        for group, counts in ct.iterrows():
            total = int(counts.sum())
            if total == 0:
                continue
            row = {
                "stratification": label,
                "group": str(group),
                "n": total,
            }
            for quadrant in ["TP", "FP", "FN", "TN"]:
                count = int(counts[quadrant])
                row[f"{quadrant.lower()}_n"] = count
                row[f"{quadrant.lower()}_pct"] = (
                    100 * count / total if total else np.nan
                )
            rows.append(row)

    pd.DataFrame(rows).to_csv(
        os.path.join(OUT_TAB, "error_profile_counts.csv"), index=False
    )


def _plot_error_bars(df, group_col, fname, title):
    """Stacked bar chart: quadrant proportions per group."""
    ct = pd.crosstab(df[group_col], df["quadrant"])
    group_orders = {
        "ALSFRS_severity": ["≤35 (severe)", "36-39", "40-42", ">42 (mild)"],
        "FVC_group": ["≤77%", "78-87%", "88-98%", ">98%", "Missing"],
    }
    if group_col in group_orders:
        ct = ct.reindex(group_orders[group_col], fill_value=0)
    # Ensure column order
    for q in ["TP", "FP", "FN", "TN"]:
        if q not in ct.columns:
            ct[q] = 0
    ct = ct[["TP", "FP", "FN", "TN"]]

    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

    colors = {
        "TP": PALETTE["velvet_purple"],
        "FP": PALETTE["powder_blush"],
        "FN": PALETTE["deep_rose"],
        "TN": PALETTE["glaucous"],
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: absolute counts
    ct.plot(kind="bar", stacked=True, ax=axes[0],
            color=[colors[c] for c in ct.columns], edgecolor=FIGURE_INK, linewidth=0.5)
    axes[0].set_title("Absolute counts", fontsize=11)
    axes[0].set_ylabel("Subjects")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].legend(loc="upper right", fontsize=8)
    # Annotate totals
    for i, total in enumerate(ct.sum(axis=1)):
        axes[0].text(i, total + 0.5, f"n={total}", ha="center", va="bottom", fontsize=8)

    # Right: percentage
    ct_pct.plot(kind="bar", stacked=True, ax=axes[1],
                color=[colors[c] for c in ct_pct.columns], edgecolor=FIGURE_INK, linewidth=0.5)
    axes[1].set_title("Proportion (%)", fontsize=11)
    axes[1].set_ylabel("Percentage")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].set_ylim(0, 105)
    axes[1].legend(loc="upper right", fontsize=8)
    # Annotate FN% (most clinically relevant)
    for i, (idx, row) in enumerate(ct_pct.iterrows()):
        fn_pct = row["FN"]
        if fn_pct > 0:
            # Position at the bottom of the FN segment
            bottom = row["TP"] + row["FP"]
            axes[1].text(i, bottom + fn_pct / 2, f"{fn_pct:.0f}%",
                         ha="center", va="center", fontsize=8, fontweight="bold")

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    for ax in axes:
        style_axis(ax, "y")
    if fname in {"error_profile_alsfrs", "error_profile_fvc"}:
        axes[0].get_legend().remove()
        axes[1].legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            fontsize=8,
            borderaxespad=0,
        )
        fig.tight_layout(rect=(0, 0, 0.94, 1))
    else:
        fig.tight_layout()
    save_fig(fig, fname)


# ─────────────────────────────────────────────────────────────
# 2. Subgroup Analysis
# ─────────────────────────────────────────────────────────────
def run_subgroup_analysis(pipe, X_test, y_test, raw_test):
    """Compute PR-AUC, recall, precision per clinical subgroup."""
    print("\n── Subgroup Analysis ──")

    proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (proba >= THRESHOLD).astype(int)

    prof = raw_test.copy()
    prof["y_true"] = y_test.values
    prof["y_pred"] = y_pred
    prof["proba"] = proba

    # Define subgroups
    prof["Age_group"] = pd.cut(
        prof["Age"],
        bins=[0, 49, 57, 65, np.inf],
        labels=["≤49", "50-57", "58-65", ">65"],
    ).cat.add_categories(["Missing"]).fillna("Missing")
    prof["ALSFRS_severity"] = pd.cut(
        prof["ALSFRS_R_t0"],
        bins=[0, 35, 39, 42, np.inf],
        labels=["≤35", "36-39", "40-42", ">42"],
    ).cat.add_categories(["Missing"]).fillna("Missing")
    prof["FVC_group"] = pd.cut(
        prof["FVC_pctNormal_best_t0"],
        bins=[0, 77, 87, 98, np.inf],
        labels=["≤77%", "78-87%", "88-98%", ">98%"],
    ).cat.add_categories(["Missing"]).fillna("Missing")

    stratification_vars = [
        ("Age_group", "Age Group"),
        ("Sex", "Sex"),
        ("ALSFRS_severity", "ALSFRS-R Severity"),
        ("FVC_group", "FVC (% normal)"),
    ]

    rows = []
    for col, label in stratification_vars:
        for group_val, grp in prof.groupby(col, observed=True):
            n = len(grp)
            n_rapid = int(grp["y_true"].sum())
            if n_rapid < 3 or n < 10:
                continue
            metrics = {
                "Stratification": label,
                "Group": str(group_val),
                "N": n,
                "N_rapid": n_rapid,
                "Prevalence": n_rapid / n,
            }
            try:
                metrics["PR_AUC"] = average_precision_score(grp["y_true"], grp["proba"])
            except ValueError:
                metrics["PR_AUC"] = np.nan
            metrics["Recall"] = recall_score(grp["y_true"], grp["y_pred"], zero_division=0)
            metrics["Precision"] = precision_score(grp["y_true"], grp["y_pred"], zero_division=0)
            metrics["F1"] = f1_score(grp["y_true"], grp["y_pred"], zero_division=0)
            rows.append(metrics)

    sub_df = pd.DataFrame(rows)
    sub_df.to_csv(os.path.join(OUT_TAB, "subgroup_metrics.csv"), index=False)
    print(sub_df.to_string(index=False))

    # --- Plot: PR-AUC by subgroup ---
    overall_pr_auc = average_precision_score(y_test, proba)
    overall_recall = recall_score(y_test, y_pred)
    _plot_subgroup_metric(
        sub_df,
        "PR_AUC",
        "subgroup_pr_auc",
        "PR-AUC by Clinical Subgroup (TEST set)",
        overall_pr_auc,
    )

    # --- Plot: Recall by subgroup ---
    _plot_subgroup_metric(
        sub_df,
        "Recall",
        "subgroup_recall",
        "Recall by Clinical Subgroup (TEST set, t=0.21)",
        overall_recall,
    )

    return sub_df


def _plot_subgroup_metric(df, metric, fname, title, overall_value):
    """Grouped bar chart of a metric across stratification variables."""
    strats = df["Stratification"].unique()
    fig, axes = plt.subplots(1, len(strats), figsize=(3.5 * len(strats), 4.5),
                             sharey=True)
    if len(strats) == 1:
        axes = [axes]

    palette = [
        PALETTE["pastel_petal"],
        PALETTE["powder_blush"],
        PALETTE["glaucous"],
        PALETTE["soft_mauve"],
        PALETTE["bubblegum_pink"],
        PALETTE["deep_rose"],
        PALETTE["velvet_purple"],
    ]

    for ax, strat in zip(axes, strats):
        sub = df[df["Stratification"] == strat].copy()
        x = np.arange(len(sub))
        bars = ax.bar(x, sub[metric], color=palette[:len(sub)],
                      edgecolor=FIGURE_INK, linewidth=0.5, width=0.65)

        # Annotate values + N
        for i, (val, n) in enumerate(zip(sub[metric], sub["N"])):
            ax.text(i, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
            ax.text(i, -0.04, f"n={n}", ha="center", va="top",
                    fontsize=7, color=FIGURE_INK)

        ax.set_xticks(x)
        ax.set_xticklabels(sub["Group"], fontsize=9)
        ax.set_title(strat, fontsize=10, fontweight="bold")
        ax.set_ylim(-0.08, min(1.15, sub[metric].max() + 0.15))
        ax.axhline(
            y=overall_value,
            color=FIGURE_INK,
            linestyle="--",
            linewidth=0.8,
            alpha=0.7,
        )

    axes[0].set_ylabel(metric, fontsize=10)
    axes[-1].plot(
        [],
        [],
        color=FIGURE_INK,
        linestyle="--",
        linewidth=0.8,
        label=f"Overall TEST = {overall_value:.3f}",
    )
    axes[-1].legend(loc="upper right", fontsize=7)
    for ax in axes:
        style_axis(ax, "y")
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, fname)


# ─────────────────────────────────────────────────────────────
# 3. Decision Curve Analysis
# ─────────────────────────────────────────────────────────────
def run_decision_curve(pipe, X_test, y_test):
    """Net-benefit curves: model vs treat-all vs treat-none."""
    print("\n── Decision Curve Analysis ──")

    proba = pipe.predict_proba(X_test)[:, 1]
    n = len(y_test)
    prevalence = y_test.mean()

    thresholds = np.arange(0.01, 0.61, 0.01)
    rows = []

    for t in thresholds:
        y_pred = (proba >= t).astype(int)
        tp = int(((y_pred == 1) & (y_test == 1)).sum())
        fp = int(((y_pred == 1) & (y_test == 0)).sum())

        # Net benefit of the model
        nb_model = (tp / n) - (fp / n) * (t / (1 - t))

        # Net benefit of treat-all
        nb_all = prevalence - (1 - prevalence) * (t / (1 - t))

        rows.append({
            "threshold": t,
            "predicted_positive": int(y_pred.sum()),
            "true_positive": tp,
            "false_positive": fp,
            "nb_model": nb_model,
            "nb_treat_all": nb_all,
            "nb_treat_none": 0.0,
            "nb_advantage": nb_model - max(nb_all, 0.0),
        })

    dc_df = pd.DataFrame(rows)
    dc_df.to_csv(os.path.join(OUT_TAB, "decision_curve_data.csv"), index=False)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dc_df["threshold"], dc_df["nb_model"],
            color=FIGURE_INK, linewidth=2, label="XGBoost model")
    ax.plot(dc_df["threshold"], dc_df["nb_treat_all"],
            color=PALETTE["deep_rose"], linewidth=1.5, linestyle="--", label="Treat all")
    ax.axhline(y=0, color=PALETTE["glaucous"], linewidth=1,
               linestyle=":", label="Treat none")
    ax.axvspan(
        0.18,
        0.47,
        color=PALETTE["pastel_petal"],
        alpha=0.08,
        label="Main net-benefit interval",
    )

    # Mark the operating threshold
    t_row = dc_df.iloc[(dc_df["threshold"] - THRESHOLD).abs().argsort()[:1]]
    ax.scatter(THRESHOLD, t_row["nb_model"].values[0],
               color=FIGURE_INK, s=80, zorder=5, edgecolors=FIGURE_INK)
    ax.annotate(f"t = {THRESHOLD}",
                xy=(THRESHOLD, t_row["nb_model"].values[0]),
                xytext=(THRESHOLD + 0.06, t_row["nb_model"].values[0] + 0.03),
                fontsize=9, arrowprops=dict(arrowstyle="->", color=FIGURE_INK))

    ax.set_xlabel("Threshold Probability", fontsize=11)
    ax.set_ylabel("Net Benefit", fontsize=11)
    ax.set_title("Decision Curve Analysis — XGBoost (TEST set)", fontsize=12,
                 fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, 0.60)
    ax.set_ylim(-0.05, max(dc_df["nb_model"].max(), prevalence) + 0.05)
    style_axis(ax)
    fig.tight_layout()
    save_fig(fig, "decision_curve")

    # Print key insight
    useful_range = dc_df[dc_df["nb_advantage"] > 1e-12]
    if len(useful_range) > 0:
        useful_thresholds = useful_range["threshold"].round(2).tolist()
        intervals = []
        start = previous = useful_thresholds[0]
        for threshold in useful_thresholds[1:]:
            if round(threshold - previous, 2) > 0.01:
                intervals.append((start, previous))
                start = threshold
            previous = threshold
        intervals.append((start, previous))
        interval_text = ", ".join(
            f"{start:.2f}" if start == end else f"{start:.2f}-{end:.2f}"
            for start, end in intervals
        )
        print(
            "  Model exceeds treat-all and treat-none at thresholds: "
            f"{interval_text}"
        )
    return dc_df


# ─────────────────────────────────────────────────────────────
# 4. Learning Curves
# ─────────────────────────────────────────────────────────────
def run_learning_curves(X_dev, y_dev, groups_dev, X_test, y_test, params):
    """TEST performance as a sensitivity analysis of DEV training-set size."""
    print("\n── Learning Curves ──")

    rows = []
    repeat_rows = []

    for frac in LC_FRACTIONS:
        pr_aucs, roc_aucs, sample_sizes, prevalences = [], [], [], []

        # Repeat stratified subject-level subsampling except at full DEV size.
        n_repeats = 1 if frac == 1.0 else LC_REPEATS
        for rep in range(n_repeats):
            if frac < 1.0:
                indices = np.arange(len(X_dev))
                selected, _ = train_test_split(
                    indices,
                    train_size=frac,
                    stratify=y_dev,
                    random_state=SEED + rep,
                )
                X_sub = X_dev.iloc[selected].reset_index(drop=True)
                y_sub = y_dev.iloc[selected].reset_index(drop=True)
            else:
                X_sub = X_dev
                y_sub = y_dev

            pipe = train_xgb_pipeline(X_sub, y_sub, params)
            proba = pipe.predict_proba(X_test)[:, 1]
            pr_auc = average_precision_score(y_test, proba)
            roc_auc = roc_auc_score(y_test, proba)
            pr_aucs.append(pr_auc)
            roc_aucs.append(roc_auc)
            sample_sizes.append(len(X_sub))
            prevalences.append(float(y_sub.mean()))
            repeat_rows.append({
                "fraction": frac,
                "repeat": rep + 1,
                "n_subjects": len(X_sub),
                "rapid_prevalence": float(y_sub.mean()),
                "pr_auc": pr_auc,
                "roc_auc": roc_auc,
            })

        rows.append({
            "fraction": frac,
            "n_repeats": n_repeats,
            "n_subjects": int(round(np.mean(sample_sizes))),
            "rapid_prevalence_mean": np.mean(prevalences),
            "rapid_prevalence_std": (
                np.std(prevalences, ddof=1) if n_repeats > 1 else 0.0
            ),
            "pr_auc_mean": np.mean(pr_aucs),
            "pr_auc_std": np.std(pr_aucs, ddof=1) if n_repeats > 1 else 0.0,
            "roc_auc_mean": np.mean(roc_aucs),
            "roc_auc_std": np.std(roc_aucs, ddof=1) if n_repeats > 1 else 0.0,
        })
        print(
            f"  frac={frac:.1f}  n={round(np.mean(sample_sizes))}  "
            f"PR-AUC={np.mean(pr_aucs):.3f} ± "
            f"{(np.std(pr_aucs, ddof=1) if n_repeats > 1 else 0.0):.3f}"
        )

    lc_df = pd.DataFrame(rows)
    lc_df.to_csv(os.path.join(OUT_TAB, "learning_curve_data.csv"), index=False)
    pd.DataFrame(repeat_rows).to_csv(
        os.path.join(OUT_TAB, "learning_curve_repeats.csv"), index=False
    )

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.errorbar(lc_df["n_subjects"], lc_df["pr_auc_mean"],
                yerr=lc_df["pr_auc_std"], fmt="o-", color=FIGURE_INK,
                linewidth=2, capsize=4, markersize=6, label="PR-AUC (TEST)")
    ax.errorbar(lc_df["n_subjects"], lc_df["roc_auc_mean"],
                yerr=lc_df["roc_auc_std"], fmt="s--", color=PALETTE["soft_mauve"],
                linewidth=2, capsize=4, markersize=6, label="ROC-AUC (TEST)")

    # Annotate each point
    for _, r in lc_df.iterrows():
        ax.annotate(f"{r['pr_auc_mean']:.3f}",
                    xy=(r["n_subjects"], r["pr_auc_mean"]),
                    xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=8, color=FIGURE_INK)

    ax.set_xlabel("Training Set Size (subjects)", fontsize=11)
    ax.set_ylabel("AUC (on TEST set)", fontsize=11)
    ax.set_title("Training-Size Sensitivity — XGBoost (TEST set)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    style_axis(ax)
    ax.set_ylim(0.30, 0.80)
    fig.tight_layout()
    save_fig(fig, "learning_curve")

    return lc_df


# ─────────────────────────────────────────────────────────────
# 5. Bootstrap Confidence Intervals
# ─────────────────────────────────────────────────────────────
def run_bootstrap_ci(pipe, X_test, y_test):
    """Bootstrap 95% CIs for key TEST-set metrics."""
    print("\n── Bootstrap Confidence Intervals ──")

    proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (proba >= THRESHOLD).astype(int)
    n = len(y_test)
    rng = np.random.RandomState(SEED)

    # Point estimates
    point = {
        "PR-AUC":    average_precision_score(y_test, proba),
        "ROC-AUC":   roc_auc_score(y_test, proba),
        "Recall":    recall_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "F1":        f1_score(y_test, y_pred, zero_division=0),
        "F2":        fbeta_score(y_test, y_pred, beta=2, zero_division=0),
        "Brier":     brier_score_loss(y_test, proba),
    }

    # Bootstrap resampling
    boot = {k: [] for k in point}
    for _ in range(N_BOOTSTRAP):
        idx = rng.randint(0, n, size=n)
        yb = y_test.values[idx]
        pb = proba[idx]
        yp = y_pred[idx]

        # Skip degenerate samples (single class)
        if yb.sum() == 0 or yb.sum() == n:
            continue

        boot["PR-AUC"].append(average_precision_score(yb, pb))
        boot["ROC-AUC"].append(roc_auc_score(yb, pb))
        boot["Recall"].append(recall_score(yb, yp, zero_division=0))
        boot["Precision"].append(precision_score(yb, yp, zero_division=0))
        boot["F1"].append(f1_score(yb, yp, zero_division=0))
        boot["F2"].append(fbeta_score(yb, yp, beta=2, zero_division=0))
        boot["Brier"].append(brier_score_loss(yb, pb))

    rows = []
    for metric in point:
        vals = np.array(boot[metric])
        lo = np.percentile(vals, 100 * CI_ALPHA / 2)
        hi = np.percentile(vals, 100 * (1 - CI_ALPHA / 2))
        rows.append({
            "Metric": metric,
            "Point": point[metric],
            "CI_lower": lo,
            "CI_upper": hi,
            "CI_width": hi - lo,
        })
        print(f"  {metric:12s}  {point[metric]:.3f}  [{lo:.3f}, {hi:.3f}]")

    ci_df = pd.DataFrame(rows)
    ci_df.to_csv(os.path.join(OUT_TAB, "bootstrap_ci.csv"), index=False)
    pd.DataFrame(boot).to_csv(
        os.path.join(OUT_TAB, "bootstrap_distributions.csv"), index=False
    )

    # --- Plot: forest plot of CIs ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    y_pos = np.arange(len(ci_df))
    colors = list(PALETTE.values())

    for i, (_, r) in enumerate(ci_df.iterrows()):
        ax.errorbar(r["Point"], i,
                    xerr=[[r["Point"] - r["CI_lower"]], [r["CI_upper"] - r["Point"]]],
                    fmt="o", color=colors[i % len(colors)],
                    markersize=8, capsize=5, linewidth=2, capthick=1.5)
        ax.text(r["CI_upper"] + 0.008, i,
                f"{r['Point']:.3f} [{r['CI_lower']:.3f}, {r['CI_upper']:.3f}]",
                va="center", fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(ci_df["Metric"], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Value", fontsize=11)
    ax.set_title("Bootstrap 95% Confidence Intervals (TEST set, B=2000)",
                 fontsize=12, fontweight="bold")
    style_axis(ax, "x")
    ax.set_xlim(0, 1.05)
    fig.tight_layout()
    save_fig(fig, "bootstrap_ci")

    return ci_df


# ─────────────────────────────────────────────────────────────
# 6. Partial Dependence Plots
# ─────────────────────────────────────────────────────────────
def run_pdp(pipe, X_test):
    """Partial Dependence Plots for top-3 SHAP features."""
    print("\n── Partial Dependence Plots ──")

    # Get preprocessed feature names from the fitted pipeline
    preproc = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]

    # Transform test data
    X_transformed = preproc.transform(X_test)

    # Build feature name list from preprocessor
    num_features = preproc.transformers_[0][2]  # num column names
    cat_transformer = preproc.transformers_[1][1]
    ohe_step_name = "onehot" if "onehot" in dict(cat_transformer.steps) else "oh"
    cat_features = list(cat_transformer.named_steps[ohe_step_name].get_feature_names_out(
        preproc.transformers_[1][2]
    ))
    all_features_transformed = list(num_features) + cat_features

    # Find indices of PDP features in the transformed space
    pdp_indices = []
    pdp_labels = []
    for feat in PDP_FEATURES:
        if feat in all_features_transformed:
            pdp_indices.append(all_features_transformed.index(feat))
            pdp_labels.append(FEATURE_LABELS.get(feat, feat))
        else:
            print(f"  [WARN] Feature '{feat}' not found in transformed space, skipping.")

    if not pdp_indices:
        print("  [SKIP] No PDP features found.")
        return

    # Compute PDP manually for each feature
    fig, axes = plt.subplots(1, len(pdp_indices), figsize=(5 * len(pdp_indices), 4.5))
    if len(pdp_indices) == 1:
        axes = [axes]

    X_arr = np.array(X_transformed) if not isinstance(X_transformed, np.ndarray) else X_transformed
    pdp_rows = []

    for ax, feat, feat_idx, label in zip(
        axes, PDP_FEATURES, pdp_indices, pdp_labels
    ):
        # Use observed raw values for the grid and rug plot. Model predictions
        # are still computed in the preprocessed feature space.
        observed_vals = pd.to_numeric(X_test[feat], errors="coerce").dropna().to_numpy()
        grid = np.linspace(
            np.percentile(observed_vals, 5),
            np.percentile(observed_vals, 95),
            50,
        )

        # Compute partial dependence
        pd_values = []
        for val in grid:
            X_temp = X_arr.copy()
            X_temp[:, feat_idx] = val
            preds = clf.predict_proba(X_temp)[:, 1]
            pd_values.append(preds.mean())

        pd_values = np.array(pd_values)
        for grid_value, pd_value in zip(grid, pd_values):
            pdp_rows.append({
                "feature": feat,
                "feature_label": label,
                "grid_value": grid_value,
                "mean_predicted_probability": pd_value,
                "n_observed_test": len(observed_vals),
                "n_missing_test": int(X_test[feat].isna().sum()),
            })

        # Map grid back to original scale (undo standardisation if applicable)
        # Since XGBoost doesn't use scaling, grid is in original scale
        ax.plot(grid, pd_values, color=FIGURE_INK, linewidth=2)
        ax.fill_between(grid, pd_values, alpha=0.25, color=PALETTE["pastel_petal"])

        # Add rug plot of actual data
        ax.plot(observed_vals, np.full_like(observed_vals, pd_values.min() - 0.005),
                "|", color=PALETTE["soft_mauve"], alpha=0.45, markersize=6)

        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel("Avg. P(rapid)", fontsize=10)
        ax.set_title(label, fontsize=11, fontweight="bold")
        style_axis(ax)

        # Annotate the range
        ax.annotate(f"Δ = {pd_values.max() - pd_values.min():.3f}",
                    xy=(0.95, 0.95), xycoords="axes fraction",
                    ha="right", va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3",
                              fc=PALETTE["pastel_petal"],
                              ec=FIGURE_INK, alpha=0.8))

    fig.suptitle("Partial Dependence Plots — XGBoost (TEST set)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "pdp_top3")
    pd.DataFrame(pdp_rows).to_csv(
        os.path.join(OUT_TAB, "pdp_data.csv"), index=False
    )
    print(f"  [OK] PDP for {len(pdp_indices)} features")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    warnings.filterwarnings("ignore", category=FutureWarning)

    ensure_dirs()

    # --- Load data ---
    print("Loading data …")
    X_dev, y_dev, groups_dev, X_test, y_test, feat_cols, cutoff, raw_test = load_data()

    # --- Load XGBoost params & train on full DEV ---
    print("Training XGBoost on full DEV …")
    params = load_xgb_params()
    pipe = train_xgb_pipeline(X_dev, y_dev, params)

    # --- 1. Error Analysis ---
    error_prof = run_error_analysis(pipe, X_test, y_test, raw_test)

    # --- 2. Subgroup Analysis ---
    sub_df = run_subgroup_analysis(pipe, X_test, y_test, raw_test)

    # --- 3. Decision Curve Analysis ---
    dc_df = run_decision_curve(pipe, X_test, y_test)

    # --- 4. Learning Curves ---
    lc_df = run_learning_curves(X_dev, y_dev, groups_dev, X_test, y_test, params)

    # --- 5. Bootstrap CIs ---
    ci_df = run_bootstrap_ci(pipe, X_test, y_test)

    # --- 6. Partial Dependence Plots ---
    run_pdp(pipe, X_test)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Step 9 complete in {elapsed:.1f}s")
    print(f"  Figures → {OUT_FIG}")
    print(f"  Tables  → {OUT_TAB}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
