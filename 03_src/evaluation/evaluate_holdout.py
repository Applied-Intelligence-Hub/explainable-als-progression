"""
Step 6 — Held-Out Test Set Evaluation (6-month horizon).

Protocol
--------
1. Load DEV + TEST partitions from holdout_split_6m.csv
2. Compute binary labels using DEV 30th-percentile slope cutoff
3. For each best-config-per-family model:
   a) Train on full DEV set
   b) Predict probabilities on TEST set
   c) Compute PR-AUC, ROC-AUC, F1@0.5, F2@0.5, Brier score
4. Threshold optimisation (XGB): OOF predictions on DEV → sweep → apply to TEST
5. Generate figures & LaTeX table

Best configs (from Step 5 Optuna tuning):
  XGB  unbalanced  0.434
  RF   unbalanced  0.431
  SVM  balanced    0.426
  LGBM balanced    0.418
  LR   balanced    0.407
  DT   balanced    0.404
  KNN  unbalanced  0.387

Usage
-----
  cd g:\\TESE\\CodigoMain
  python 03_src/evaluation/evaluate_holdout.py

Outputs
-------
  04_outputs/tables/step6_test_metrics.csv
  04_outputs/figures/step6_pr_curves.pdf / .png
  04_outputs/figures/step6_roc_curves.pdf / .png
  04_outputs/figures/step6_threshold_sweep.pdf / .png
  04_outputs/figures/step6_calibration.pdf / .png
  04_outputs/figures/step6_confusion_matrix.pdf / .png
  04_outputs/tables/step6_test_metrics.tex   (LaTeX)
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis"))
from study1_style import (  # noqa: E402
    FIGURE_INK,
    PALETTE as STUDY1_PALETTE,
    SEQUENTIAL_CMAP,
    apply_study1_style,
    model_colour,
    style_axis,
)

apply_study1_style()


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
PROCESSED   = os.path.join("01_data", "processed")
OUT_TABLES  = os.path.join("04_outputs", "tables", "step6_holdout")
OUT_FIGURES = os.path.join("04_outputs", "figures", "step6_holdout")
TUNING_DIR  = os.path.join("04_outputs", "tables", "step5_tuning")
OVERLEAF_FIG = os.path.join("overleaf", "images", "figures")
OVERLEAF_TAB = os.path.join("overleaf", "images", "tables")

HORIZON    = "6m"
SEED       = 42
RAPID_FRAC = 0.30
N_SPLITS   = 5

# Columns to exclude from features
DROP_COLS = {
    "subject_id",
    "t0_delta_days",
    "vitals_delta_days",
    "fvc_delta_days",
    "slope_90d_per_30d",
    "slope_180d_per_30d",
    "ALSFRS_Responded_By",
}

# Best configuration per model family (model_key, mode)
BEST_CONFIGS = [
    ("XGB",  "unbalanced"),
    ("RF",   "unbalanced"),
    ("SVM",  "balanced"),
    ("LGBM", "balanced"),
    ("LR",   "balanced"),
    ("DT",   "balanced"),
    ("KNN",  "unbalanced"),
]

MODEL_NAMES = {
    "KNN":  "KNeighbors",
    "DT":   "DecisionTree",
    "RF":   "RandomForest",
    "SVM":  "SVM",
    "LR":   "LogisticRegression",
    "LGBM": "LightGBM",
    "XGB":  "XGBoost",
}

CLF_CLASS = {
    "KNN":  KNeighborsClassifier,
    "DT":   DecisionTreeClassifier,
    "RF":   RandomForestClassifier,
    "SVM":  SVC,
    "LR":   LogisticRegression,
    "LGBM": LGBMClassifier,
    "XGB":  XGBClassifier,
}

NEEDS_SCALING = {"KNN", "SVM", "LR"}
SUPPORTS_BALANCED = {"DT", "RF", "SVM", "LR", "LGBM", "XGB"}


# ─────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────
def load_data():
    """Load DEV + TEST partitions with features and slope."""
    ds_path = os.path.join(PROCESSED, f"dataset_{HORIZON}_v2.csv")
    hs_path = os.path.join(PROCESSED, f"holdout_split_{HORIZON}.csv")

    df = pd.read_csv(ds_path)
    hs = pd.read_csv(hs_path)

    slope_col = "slope_180d_per_30d"
    df[slope_col] = pd.to_numeric(df[slope_col], errors="coerce")
    df = df.dropna(subset=[slope_col])

    dev_ids = set(hs.loc[hs["partition"] == "dev", "subject_id"])
    test_ids = set(hs.loc[hs["partition"] == "test", "subject_id"])

    df_dev = df[df["subject_id"].isin(dev_ids)].copy()
    df_test = df[df["subject_id"].isin(test_ids)].copy()

    print(f"  DEV  n={len(df_dev)},  TEST n={len(df_test)}")

    # Label threshold from DEV only
    cutoff = float(df_dev[slope_col].quantile(RAPID_FRAC))
    print(f"  Slope cutoff (DEV 30th pctile): {cutoff:.4f}")

    feat_cols = [c for c in df.columns if c not in DROP_COLS]

    X_dev  = df_dev[feat_cols].reset_index(drop=True)
    X_test = df_test[feat_cols].reset_index(drop=True)
    y_dev  = (df_dev[slope_col] <= cutoff).astype(int).reset_index(drop=True)
    y_test = (df_test[slope_col] <= cutoff).astype(int).reset_index(drop=True)
    groups_dev = df_dev["subject_id"].reset_index(drop=True)

    prev_dev  = y_dev.mean()
    prev_test = y_test.mean()
    print(f"  Rapid prevalence — DEV: {prev_dev:.3f}  TEST: {prev_test:.3f}")

    return X_dev, y_dev, groups_dev, X_test, y_test, feat_cols, cutoff


# ─────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────
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
        ("oh",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )


# ─────────────────────────────────────────────────────────────
# Build classifier
# ─────────────────────────────────────────────────────────────
def build_clf(model_key: str, params: dict, balanced: bool,
              n_pos: int = 0, n_neg: int = 0):
    p = dict(params)
    if balanced:
        if model_key in ("DT", "RF", "SVM", "LR"):
            p["class_weight"] = "balanced"
        elif model_key == "LGBM":
            p["is_unbalance"] = True
        elif model_key == "XGB":
            if n_pos > 0:
                p["scale_pos_weight"] = n_neg / n_pos
    return CLF_CLASS[model_key](**p)


def load_best_params(model_key: str, mode: str) -> dict:
    """Load Optuna best params JSON and return full_params dict."""
    path = os.path.join(TUNING_DIR, f"{model_key}_{mode}_{HORIZON}_best.json")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["full_params"]


# ─────────────────────────────────────────────────────────────
# Metrics helper
# ─────────────────────────────────────────────────────────────
def compute_metrics(y_true, proba, threshold=0.5):
    y_pred = (proba >= threshold).astype(int)
    return {
        "pr_auc":  average_precision_score(y_true, proba),
        "roc_auc": roc_auc_score(y_true, proba),
        "f1":      f1_score(y_true, y_pred, zero_division=0),
        "f2":      fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "brier":   brier_score_loss(y_true, proba),
    }


# ─────────────────────────────────────────────────────────────
# OOF threshold sweep (on DEV)
# ─────────────────────────────────────────────────────────────
def oof_threshold_sweep(model_key, params, balanced, X_dev, y_dev, groups_dev):
    """
    Collect out-of-fold predictions on DEV via GroupKFold,
    then sweep thresholds using multiple criteria.

    Returns dict with optimal thresholds for F1, F2, and Youden's J,
    plus the full sweep arrays.
    """
    from sklearn.metrics import precision_score, recall_score

    scale = model_key in NEEDS_SCALING
    gkf = GroupKFold(n_splits=N_SPLITS)

    oof_proba = np.full(len(y_dev), np.nan)

    for tr_idx, va_idx in gkf.split(X_dev, groups=groups_dev):
        Xtr, Xva = X_dev.iloc[tr_idx], X_dev.iloc[va_idx]
        ytr = y_dev.iloc[tr_idx]

        n_pos = int(ytr.sum())
        n_neg = int(len(ytr) - n_pos)
        clf = build_clf(model_key, params, balanced, n_pos, n_neg)

        preproc = build_preprocessor(Xtr, scale=scale)
        pipe = Pipeline([("prep", preproc), ("clf", clf)])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(Xtr, ytr)

        oof_proba[va_idx] = pipe.predict_proba(Xva)[:, 1]

    # Sweep thresholds
    thresholds = np.arange(0.05, 0.96, 0.01)
    sweep = {"f1": [], "f2": [], "precision": [], "recall": [], "youden": []}

    for t in thresholds:
        y_pred = (oof_proba >= t).astype(int)
        prec = precision_score(y_dev, y_pred, zero_division=0)
        rec  = recall_score(y_dev, y_pred, zero_division=0)
        # Youden's J: TPR - FPR = recall - (FP / (FP + TN))
        fp = int(((y_pred == 1) & (y_dev == 0)).sum())
        tn = int(((y_pred == 0) & (y_dev == 0)).sum())
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        sweep["f1"].append(f1_score(y_dev, y_pred, zero_division=0))
        sweep["f2"].append(fbeta_score(y_dev, y_pred, beta=2, zero_division=0))
        sweep["precision"].append(prec)
        sweep["recall"].append(rec)
        sweep["youden"].append(rec - fpr)

    # Best thresholds per criterion
    idx_f1 = int(np.argmax(sweep["f1"]))
    idx_f2 = int(np.argmax(sweep["f2"]))
    idx_j  = int(np.argmax(sweep["youden"]))

    best = {
        "f1":     {"threshold": float(thresholds[idx_f1]),
                   "value": float(sweep["f1"][idx_f1])},
        "f2":     {"threshold": float(thresholds[idx_f2]),
                   "value": float(sweep["f2"][idx_f2])},
        "youden": {"threshold": float(thresholds[idx_j]),
                   "value": float(sweep["youden"][idx_j])},
    }

    print(f"  OOF best F1  = {best['f1']['value']:.4f} @ t = {best['f1']['threshold']:.2f}")
    print(f"  OOF best F2  = {best['f2']['value']:.4f} @ t = {best['f2']['threshold']:.2f}")
    print(f"  OOF best J   = {best['youden']['value']:.4f} @ t = {best['youden']['threshold']:.2f}")

    return best, thresholds, sweep, oof_proba


# ─────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────
MODEL_PALETTE = {key: model_colour(key) for key in MODEL_NAMES}


def plot_pr_curves(results, y_test):
    fig, ax = plt.subplots(figsize=(7, 5))
    prev = y_test.mean()
    ax.axhline(prev, ls="--", color=FIGURE_INK, lw=0.9, label=f"Baseline ({prev:.2f})")

    for key, mode, proba, _ in results:
        prec, rec, _ = precision_recall_curve(y_test, proba)
        ap = average_precision_score(y_test, proba)
        ax.plot(rec, prec, color=MODEL_PALETTE[key], lw=1.8,
                label=f"{MODEL_NAMES[key]} (AP={ap:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curves — TEST Set (6 m)")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    style_axis(ax)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_FIGURES, f"step6_pr_curves.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  → step6_pr_curves.pdf/png")


def plot_roc_curves(results, y_test):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], ls="--", color=FIGURE_INK, lw=0.9, label="Random")

    for key, mode, proba, _ in results:
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, color=MODEL_PALETTE[key], lw=1.8,
                label=f"{MODEL_NAMES[key]} (AUC={auc:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — TEST Set (6 m)")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    style_axis(ax)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_FIGURES, f"step6_roc_curves.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  → step6_roc_curves.pdf/png")


def plot_threshold_sweep(thresholds, sweep, best):
    """Multi-metric threshold sweep with F1, F2, precision, recall."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(thresholds, sweep["f1"], color=FIGURE_INK, lw=1.8, label="F₁")
    ax.plot(thresholds, sweep["f2"], color=STUDY1_PALETTE["deep_rose"], lw=1.8, label="F₂")
    ax.plot(thresholds, sweep["precision"], color=STUDY1_PALETTE["glaucous"],
            lw=1.4, ls="--", label="Precision")
    ax.plot(thresholds, sweep["recall"], color=STUDY1_PALETTE["soft_mauve"],
            lw=1.4, ls="--", label="Recall")

    # Mark best F1 threshold
    t_f1 = best["f1"]["threshold"]
    v_f1 = best["f1"]["value"]
    ax.axvline(t_f1, ls=":", color=FIGURE_INK, lw=1.0, alpha=0.8)
    ax.annotate(f"F₁ opt\nt={t_f1:.2f}\nF₁={v_f1:.3f}",
                xy=(t_f1, v_f1), fontsize=8, color=FIGURE_INK,
                xytext=(t_f1 + 0.08, v_f1 + 0.02),
                arrowprops=dict(arrowstyle="->", color=FIGURE_INK))

    # Mark best Youden threshold
    t_j = best["youden"]["threshold"]
    v_j = best["youden"]["value"]
    # Find the F1 value at the Youden threshold
    idx_j = int(np.argmin(np.abs(thresholds - t_j)))
    f1_at_j = sweep["f1"][idx_j]
    ax.axvline(t_j, ls=":", color=STUDY1_PALETTE["soft_mauve"], lw=1.0, alpha=0.8)
    ax.annotate(f"Youden J\nt={t_j:.2f}",
                xy=(t_j, f1_at_j), fontsize=8, color=STUDY1_PALETTE["soft_mauve"],
                xytext=(t_j + 0.08, f1_at_j - 0.08),
                arrowprops=dict(arrowstyle="->", color=STUDY1_PALETTE["soft_mauve"]))

    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Sweep — XGBoost on DEV (OOF)")
    ax.legend(fontsize=9, loc="center right")
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0, 1)
    style_axis(ax)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_FIGURES, f"step6_threshold_sweep.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  → step6_threshold_sweep.pdf/png")


def plot_calibration(y_test, proba, model_label):
    fig, ax = plt.subplots(figsize=(6, 5))
    prob_true, prob_pred = calibration_curve(y_test, proba, n_bins=10,
                                             strategy="uniform")
    ax.plot(prob_pred, prob_true, "o-", color=FIGURE_INK, lw=1.8,
            label=model_label)
    ax.plot([0, 1], [0, 1], "--", color=STUDY1_PALETTE["glaucous"],
            lw=1.0, label="Perfect")

    brier = brier_score_loss(y_test, proba)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(f"Calibration — {model_label} on TEST\nBrier = {brier:.4f}")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    style_axis(ax)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_FIGURES, f"step6_calibration.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  → step6_calibration.pdf/png")


def plot_confusion_matrix(y_test, proba, threshold, model_label):
    y_pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap=SEQUENTIAL_CMAP, interpolation="nearest")
    fig.colorbar(im, ax=ax, shrink=0.8)

    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=16, fontweight="bold",
                    color="white" if val > cm.max() / 2 else FIGURE_INK)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Slow (0)", "Rapid (1)"])
    ax.set_yticklabels(["Slow (0)", "Rapid (1)"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_label}\nThreshold = {threshold:.2f}")
    style_axis(ax)
    ax.grid(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_FIGURES, f"step6_confusion_matrix.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → step6_confusion_matrix.pdf/png  (TP={tp} FP={fp} FN={fn} TN={tn})")


def plot_test_bar(rows):
    """Bar chart of PR-AUC on TEST per model with numeric annotations."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = [r["model"] for r in rows]
    prauc = [r["pr_auc"] for r in rows]
    colors = [MODEL_PALETTE[r["key"]] for r in rows]

    bars = ax.barh(names, prauc, color=colors, edgecolor=FIGURE_INK, height=0.55)
    for bar, val in zip(bars, prauc):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

    prev = rows[0]["baseline_prev"]
    ax.axvline(prev, ls="--", color=FIGURE_INK, lw=0.9)
    ax.text(prev + 0.005, len(names) - 0.3, f"baseline {prev:.2f}",
            fontsize=8, color=FIGURE_INK)

    ax.set_xlabel("PR-AUC (TEST)")
    ax.set_title("Held-Out TEST Performance — PR-AUC (6 m)")
    ax.set_xlim(0, max(prauc) + 0.08)
    ax.invert_yaxis()
    style_axis(ax, "x")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_FIGURES, f"step6_test_prauc_bar.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  → step6_test_prauc_bar.pdf/png")


# ─────────────────────────────────────────────────────────────
# LaTeX table
# ─────────────────────────────────────────────────────────────
def write_latex_table(rows, filepath):
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Held-out TEST set metrics (6-month horizon, $n=279$). "
        r"Models were fitted on DEV only. PR-AUC and ROC-AUC are threshold-free; "
        r"F$_1$ and F$_2$ are evaluated at the default $t=0.50$; the Brier score "
        r"measures probabilistic error (lower is better).}",
        r"\label{tab:step6_test_metrics}",
        r"\small",
        r"\begin{tabular}{l c c c c c c}",
        r"\toprule",
        r"Model & Mode & PR-AUC & ROC-AUC & F$_1$ & F$_2$ & Brier \\",
        r"\midrule",
    ]

    best_pr = max(r["pr_auc"] for r in rows)
    for r in rows:
        pr_str = (r"\textbf{" + f'{r["pr_auc"]:.3f}' + r"}") if r["pr_auc"] == best_pr else f'{r["pr_auc"]:.3f}'
        lines.append(
            f'  {r["model"]} & {r["mode"]} & {pr_str} & '
            f'{r["roc_auc"]:.3f} & {r["f1"]:.3f} & {r["f2"]:.3f} & '
            f'{r["brier"]:.4f} \\\\'
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  → {filepath}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    os.makedirs(OUT_FIGURES, exist_ok=True)
    os.makedirs(OUT_TABLES, exist_ok=True)
    os.makedirs(OVERLEAF_FIG, exist_ok=True)
    os.makedirs(OVERLEAF_TAB, exist_ok=True)

    print("=" * 60)
    print("  Step 6 — Held-Out Test Set Evaluation (6 m)")
    print("=" * 60)

    # 1) Load data
    X_dev, y_dev, groups_dev, X_test, y_test, feat_cols, cutoff = load_data()
    baseline_prev = float(y_test.mean())

    # 2) Train + evaluate each best-per-family model
    all_rows = []       # for CSV / LaTeX
    all_results = []    # (key, mode, proba_test, metrics)

    for model_key, mode in BEST_CONFIGS:
        balanced = (mode == "balanced")
        scale = model_key in NEEDS_SCALING

        params = load_best_params(model_key, mode)
        n_pos = int(y_dev.sum())
        n_neg = int(len(y_dev) - n_pos)
        clf = build_clf(model_key, params, balanced, n_pos, n_neg)

        preproc = build_preprocessor(X_dev, scale=scale)
        pipe = Pipeline([("prep", preproc), ("clf", clf)])

        print(f"\n  Training {MODEL_NAMES[model_key]} ({mode}) …")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(X_dev, y_dev)

        proba_test = pipe.predict_proba(X_test)[:, 1]
        m = compute_metrics(y_test, proba_test, threshold=0.5)

        row = {
            "key":           model_key,
            "model":         MODEL_NAMES[model_key],
            "mode":          mode,
            "pr_auc":        m["pr_auc"],
            "roc_auc":       m["roc_auc"],
            "f1":            m["f1"],
            "f2":            m["f2"],
            "brier":         m["brier"],
            "baseline_prev": baseline_prev,
        }
        all_rows.append(row)
        all_results.append((model_key, mode, proba_test, m))

        print(f"    PR-AUC={m['pr_auc']:.4f}  ROC-AUC={m['roc_auc']:.4f}  "
              f"F1={m['f1']:.3f}  F2={m['f2']:.3f}  Brier={m['brier']:.4f}")

    # 3) Save CSV
    df_out = pd.DataFrame(all_rows)
    csv_path = os.path.join(OUT_TABLES, "step6_test_metrics.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"\n  → {csv_path}")

    # 4) Threshold optimisation for best model (XGB)
    print(f"\n{'─'*60}")
    print("  Threshold optimisation — XGBoost (DEV OOF)")
    print(f"{'─'*60}")
    xgb_params = load_best_params("XGB", "unbalanced")
    best, thresholds, sweep, oof_proba = oof_threshold_sweep(
        "XGB", xgb_params, balanced=False,
        X_dev=X_dev, y_dev=y_dev, groups_dev=groups_dev,
    )

    # Retain both development-derived operating points for comparison.
    thr_youden = best["youden"]["threshold"]
    thr_f1     = best["f1"]["threshold"]

    # Apply thresholds to TEST
    xgb_proba_test = all_results[0][2]  # XGB is first
    m_youden = compute_metrics(y_test, xgb_proba_test, threshold=thr_youden)
    m_f1opt  = compute_metrics(y_test, xgb_proba_test, threshold=thr_f1)

    print(f"  TEST @ Youden t={thr_youden:.2f}: "
          f"F1={m_youden['f1']:.3f}  F2={m_youden['f2']:.3f}")
    print(f"  TEST @ F1-opt t={thr_f1:.2f}:     "
          f"F1={m_f1opt['f1']:.3f}  F2={m_f1opt['f2']:.3f}")

    # Comprehensive threshold check at multiple points
    thr_checks = sorted(set([0.50, 0.30, thr_f1, thr_youden]))
    thr_rows = []
    for t in thr_checks:
        mt = compute_metrics(y_test, xgb_proba_test, threshold=t)
        y_pred = (xgb_proba_test >= t).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        thr_rows.append({
            "threshold": t, **mt,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
            "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
        })
    df_thr = pd.DataFrame(thr_rows)
    thr_csv = os.path.join(OUT_TABLES, "step6_xgb_threshold_test.csv")
    df_thr.to_csv(thr_csv, index=False)
    print(f"  → {thr_csv}")

    # Use F1-optimised threshold for confusion matrix (more interpretable)
    chosen_thr = thr_f1

    # 5) Figures
    print(f"\n{'─'*60}")
    print("  Generating figures")
    print(f"{'─'*60}")

    plot_pr_curves(all_results, y_test)
    plot_roc_curves(all_results, y_test)
    plot_threshold_sweep(thresholds, sweep, best)
    plot_calibration(y_test, xgb_proba_test, "XGBoost")
    plot_confusion_matrix(y_test, xgb_proba_test, chosen_thr, "XGBoost")
    plot_test_bar(all_rows)

    # 6) LaTeX table
    tex_path = os.path.join(OUT_TABLES, "step6_test_metrics.tex")
    write_latex_table(all_rows, tex_path)

    # 7) Copy to overleaf
    import shutil
    for f in [
        "step6_pr_curves.png",
        "step6_roc_curves.png",
        "step6_threshold_sweep.png",
        "step6_calibration.png",
        "step6_confusion_matrix.png",
        "step6_test_prauc_bar.png",
    ]:
        shutil.copy2(os.path.join(OUT_FIGURES, f), os.path.join(OVERLEAF_FIG, f))
    shutil.copy2(
        tex_path,
        os.path.join(OVERLEAF_TAB, "step6_test_metrics.tex"),
    )
    print("  → Copied figures/tables to overleaf/images/")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Step 6 complete  ({elapsed:.1f}s)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
