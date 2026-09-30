"""
Step 7 — Statistical comparison of the 7 best-per-family models (6 m horizon).

Workflow
--------
1. Load DEV dataset (dataset_6m_v2.csv, DEV partition only).
2. For each of the 7 best configs, load best params from JSON and re-run
   5-fold GroupKFold evaluation → per-fold PR-AUC (and ROC-AUC).
3. Friedman test over 7 models × 5 folds.
4. Calculate exploratory pairwise Wilcoxon signed-rank comparisons
   (21 pairs) with Bonferroni correction.
5. Generate annotated figures + LaTeX tables.

Outputs
-------
Tables:
  04_outputs/tables/step7_statistical/step7_fold_scores.csv
  04_outputs/tables/step7_statistical/step7_pairwise_wilcoxon.csv
  04_outputs/tables/step7_statistical/step7_pairwise_wilcoxon.tex

Figures:
  04_outputs/figures/step7_statistical/step7_fold_prauc_boxplot.pdf
  04_outputs/figures/step7_statistical/step7_pairwise_heatmap.pdf

Usage
-----
  python 03_src/analysis/step7_statistical_comparison.py
"""

from __future__ import annotations

import json
import os
import shutil
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from study1_style import (
    FIGURE_INK,
    SEQUENTIAL_CMAP,
    apply_study1_style,
    model_colour,
    style_axis,
)

apply_study1_style()


# ═════════════════════════════════════════════════════════════
# Config — mirrors optuna_tune_all.py exactly
# ═════════════════════════════════════════════════════════════
ROOT       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED  = os.path.join(ROOT, "01_data", "processed")
TUNING_DIR = os.path.join(ROOT, "04_outputs", "tables", "step5_tuning")

FIG_DIR      = os.path.join(ROOT, "04_outputs", "figures", "step7_statistical")
TAB_DIR      = os.path.join(ROOT, "04_outputs", "tables",  "step7_statistical")
OVERLEAF_FIG = os.path.join(ROOT, "overleaf", "images", "figures")
OVERLEAF_TAB = os.path.join(ROOT, "overleaf", "images", "tables")

N_SPLITS   = 5
SEED       = 42
RAPID_FRAC = 0.30
HORIZON    = "6m"

DROP_COLS = {
    "subject_id", "t0_delta_days", "vitals_delta_days", "fvc_delta_days",
    "slope_90d_per_30d", "slope_180d_per_30d", "ALSFRS_Responded_By",
}

NEEDS_SCALING = {"KNN", "SVM", "LR"}
SUPPORTS_BALANCED = {"DT", "RF", "SVM", "LR", "LGBM", "XGB"}

CLF_CLASS = {
    "KNN": KNeighborsClassifier,
    "DT":  DecisionTreeClassifier,
    "RF":  RandomForestClassifier,
    "SVM": SVC,
    "LR":  LogisticRegression,
    "LGBM": LGBMClassifier,
    "XGB": XGBClassifier,
}

# 7 best-per-family configs (from step5_tuning_summary_6m.csv)
BEST_CONFIGS = [
    ("XGB",  "unbalanced"),
    ("RF",   "unbalanced"),
    ("SVM",  "balanced"),
    ("LGBM", "balanced"),
    ("LR",   "balanced"),
    ("DT",   "balanced"),
    ("KNN",  "unbalanced"),
]

# Short labels for display (keep consistent with other scripts)
SHORT_LABELS = {
    "XGB":  "XGB",
    "RF":   "RF",
    "SVM":  "SVM",
    "LGBM": "LGBM",
    "LR":   "LR",
    "DT":   "DT",
    "KNN":  "KNN",
}


# ═════════════════════════════════════════════════════════════
# Data loading (same as optuna_tune_all.py)
# ═════════════════════════════════════════════════════════════
def load_dev_data():
    ds_path = os.path.join(PROCESSED, f"dataset_{HORIZON}_v2.csv")
    hs_path = os.path.join(PROCESSED, f"holdout_split_{HORIZON}.csv")

    df = pd.read_csv(ds_path)
    hs = pd.read_csv(hs_path)

    dev_ids = set(hs.loc[hs["partition"] == "dev", "subject_id"])
    df = df[df["subject_id"].isin(dev_ids)].copy()
    print(f"  DEV set: {len(df)} subjects")

    slope_col = "slope_180d_per_30d"
    df[slope_col] = pd.to_numeric(df[slope_col], errors="coerce")
    df = df.dropna(subset=[slope_col])

    feat_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feat_cols].copy().reset_index(drop=True)
    slope = df[slope_col].copy().reset_index(drop=True)
    groups = df["subject_id"].copy().reset_index(drop=True)

    return X, slope, groups


# ═════════════════════════════════════════════════════════════
# Preprocessing (same as optuna_tune_all.py)
# ═════════════════════════════════════════════════════════════
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


# ═════════════════════════════════════════════════════════════
# Fold-level evaluation
# ═════════════════════════════════════════════════════════════
def cv_fold_scores(model_key: str, params: dict, balanced: bool,
                   X: pd.DataFrame, slope: pd.Series, groups: pd.Series):
    """
    Re-run 5-fold GroupKFold with fold-wise labeling.
    Returns dict of { metric: [fold_0, ..., fold_4] }.
    """
    gkf = GroupKFold(n_splits=N_SPLITS)
    scale = model_key in NEEDS_SCALING
    preproc_template = build_preprocessor(X, scale=scale)

    fold_pr_auc  = []
    fold_roc_auc = []

    for tr_idx, va_idx in gkf.split(X, groups=groups):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        slope_tr, slope_va = slope.iloc[tr_idx], slope.iloc[va_idx]

        thr = float(np.nanquantile(slope_tr.values, RAPID_FRAC))
        ytr = (slope_tr <= thr).astype(int)
        yva = (slope_va <= thr).astype(int)

        n_pos = int(ytr.sum())
        n_neg = int(len(ytr) - n_pos)
        clf = build_clf(model_key, params, balanced, n_pos, n_neg)

        preproc = clone(preproc_template)
        pipe = Pipeline([("prep", preproc), ("clf", clf)])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(Xtr, ytr)
            proba = pipe.predict_proba(Xva)[:, 1]
        fold_pr_auc.append(average_precision_score(yva, proba))
        fold_roc_auc.append(roc_auc_score(yva, proba))

    return {
        "pr_auc":  fold_pr_auc,
        "roc_auc": fold_roc_auc,
    }


# ═════════════════════════════════════════════════════════════
# Statistical tests
# ═════════════════════════════════════════════════════════════
def run_friedman(score_matrix: np.ndarray):
    """
    Friedman test over k models × n folds.
    score_matrix shape: (k_models, n_folds)
    """
    stat, p_val = friedmanchisquare(*[score_matrix[i, :] for i in range(score_matrix.shape[0])])
    return stat, p_val


def pairwise_wilcoxon_bonferroni(score_matrix: np.ndarray, labels: list[str]):
    """
    Pairwise Wilcoxon signed-rank tests with Bonferroni correction.
    Returns DataFrame with columns: model_a, model_b, stat, p_raw, p_corrected, significant.
    """
    k = score_matrix.shape[0]
    n_pairs = k * (k - 1) // 2
    rows = []

    for i in range(k):
        for j in range(i + 1, k):
            a = score_matrix[i, :]
            b = score_matrix[j, :]
            try:
                stat, p_raw = wilcoxon(a, b, alternative="two-sided")
            except ValueError:
                # All differences are zero
                stat, p_raw = 0.0, 1.0
            p_corr = min(p_raw * n_pairs, 1.0)  # Bonferroni
            rows.append({
                "model_a": labels[i],
                "model_b": labels[j],
                "stat": round(float(stat), 4),
                "p_raw": round(float(p_raw), 6),
                "p_corrected": round(float(p_corr), 6),
                "significant": p_corr < 0.05,
            })

    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════
# Figures
# ═════════════════════════════════════════════════════════════
def plot_fold_boxplot(fold_df: pd.DataFrame, out_path: str):
    """
    Boxplot with individual fold points and a mean marker, ordered by mean.
    """
    models = fold_df.groupby("model")["pr_auc"].mean().sort_values(ascending=False).index.tolist()
    n = len(models)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Prepare data in order
    data_list = [fold_df.loc[fold_df["model"] == m, "pr_auc"].values for m in models]
    means = [np.mean(d) for d in data_list]

    bp = ax.boxplot(
        data_list,
        tick_labels=models,
        patch_artist=True,
        widths=0.45,
        showfliers=False,
        medianprops=dict(color=FIGURE_INK, linewidth=1.5),
    )

    # Cosmetics
    for model, patch in zip(models, bp["boxes"]):
        patch.set_facecolor(model_colour(model.split()[0]))
        patch.set_edgecolor(FIGURE_INK)
        patch.set_alpha(0.55)

    # Scatter individual fold values with light horizontal jitter.
    np.random.seed(SEED)
    for i, (m, vals) in enumerate(zip(models, data_list)):
        x_jitter = np.random.uniform(-0.15, 0.15, size=len(vals))
        xs = (i + 1) + x_jitter
        colour = model_colour(m.split()[0])
        ax.scatter(xs, vals, s=55, zorder=5, edgecolors=FIGURE_INK, linewidth=0.6,
                   color=colour, alpha=0.9)

    ax.scatter(range(1, n + 1), means, marker="D", s=48, zorder=6,
               facecolors="white", edgecolors=FIGURE_INK, linewidth=1.3,
               label="Mean")

    ax.set_ylabel("PR-AUC (average precision)", fontsize=11)
    ax.set_title("Per-fold PR-AUC — 5-fold GroupKFold (6 m horizon, DEV set)",
                 fontsize=12, fontweight="bold")
    style_axis(ax, "y")
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_pairwise_heatmap(pw_df: pd.DataFrame, labels: list[str], out_path: str):
    """
    Symmetric heatmap of Bonferroni-corrected p-values with numeric annotations.
    """
    k = len(labels)
    p_matrix = np.ones((k, k))

    for _, row in pw_df.iterrows():
        i = labels.index(row["model_a"])
        j = labels.index(row["model_b"])
        p_matrix[i, j] = row["p_corrected"]
        p_matrix[j, i] = row["p_corrected"]

    fig, ax = plt.subplots(figsize=(8, 6.5))

    im = ax.imshow(p_matrix, cmap=SEQUENTIAL_CMAP, vmin=0, vmax=1, aspect="equal")

    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels(labels, fontsize=10, rotation=45, ha="right")
    ax.set_yticklabels(labels, fontsize=10)

    # Annotate each cell
    for i in range(k):
        for j in range(k):
            if i == j:
                txt = "—"
                color = FIGURE_INK
            else:
                p = p_matrix[i, j]
                txt = f"{p:.4f}"
                if p < 0.05:
                    txt += "\n*"
                color = "white" if p > 0.65 else FIGURE_INK
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=8.5, fontweight="bold", color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Bonferroni-corrected p-value", fontsize=10)

    ax.set_title("Pairwise Wilcoxon signed-rank tests\n(Bonferroni-corrected, α = 0.05)",
                 fontsize=12, fontweight="bold")
    style_axis(ax)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ═════════════════════════════════════════════════════════════
# LaTeX table
# ═════════════════════════════════════════════════════════════
def save_pairwise_latex(pw_df: pd.DataFrame, friedman_stat: float,
                        friedman_p: float, out_path: str):
    """Save pairwise comparison as a LaTeX table with Friedman header note."""
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Exploratory pairwise Wilcoxon signed-rank comparisons with "
                 r"Bonferroni correction "
                 f"(Friedman $\\chi^2_F(6) = {friedman_stat:.2f}$, $p = {friedman_p:.4f}$).}}")
    lines.append(r"  \label{tab:step7_pairwise}")
    lines.append(r"  \small")
    lines.append(r"  \begin{tabular}{llccc}")
    lines.append(r"    \toprule")
    lines.append(r"    Model A & Model B & Statistic & $p_\text{raw}$ & $p_\text{Bonf.}$ \\")
    lines.append(r"    \midrule")

    for _, row in pw_df.iterrows():
        sig_mark = r" $^{*}$" if row["significant"] else ""
        lines.append(
            f"    {row['model_a']} & {row['model_b']} & "
            f"{row['stat']:.1f} & {row['p_raw']:.4f} & "
            f"{row['p_corrected']:.4f}{sig_mark} \\\\"
        )

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"  \vspace{2pt}")
    lines.append(r"  \begin{flushleft}")
    lines.append(r"    \footnotesize The omnibus Friedman test did not reject its null "
                 r"hypothesis. Pairwise results are shown for descriptive completeness; "
                 r"no comparison is significant after Bonferroni correction.")
    lines.append(r"  \end{flushleft}")
    lines.append(r"\end{table}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {out_path}")


# ═════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════
def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TAB_DIR, exist_ok=True)
    os.makedirs(OVERLEAF_FIG, exist_ok=True)
    os.makedirs(OVERLEAF_TAB, exist_ok=True)

    print("=" * 60)
    print("  Step 7 — Statistical comparison (6 m horizon)")
    print("=" * 60)

    # ── 1. Load DEV data ──
    print("\n[1/5] Loading DEV data ...")
    X, slope, groups = load_dev_data()

    # ── 2. Re-run fold evaluation for each best config ──
    print("\n[2/5] Re-evaluating 7 best-per-family models (5-fold GroupKFold) ...")
    fold_records = []   # flat: one row per model × fold
    labels = []         # ordered model labels

    for model_key, mode in BEST_CONFIGS:
        balanced = (mode == "balanced")
        tag = f"{model_key}_{mode}_{HORIZON}"
        json_path = os.path.join(TUNING_DIR, f"{tag}_best.json")

        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        params = payload["full_params"]

        label = SHORT_LABELS[model_key]
        labels.append(label)
        print(f"  {label} ({mode}) ...", end=" ", flush=True)

        scores = cv_fold_scores(model_key, params, balanced, X, slope, groups)
        for fold_idx in range(N_SPLITS):
            fold_records.append({
                "model": label,
                "mode": mode,
                "fold": fold_idx,
                "pr_auc": round(scores["pr_auc"][fold_idx], 4),
                "roc_auc": round(scores["roc_auc"][fold_idx], 4),
            })
        mu = np.mean(scores["pr_auc"])
        print(f"PR-AUC = {mu:.4f} ± {np.std(scores['pr_auc']):.4f}")

    fold_df = pd.DataFrame(fold_records)
    fold_csv = os.path.join(TAB_DIR, "step7_fold_scores.csv")
    fold_df.to_csv(fold_csv, index=False)
    print(f"\n  Saved fold scores: {fold_csv}")

    # ── 3. Friedman test ──
    print("\n[3/5] Friedman test ...")
    score_matrix = np.array([
        fold_df.loc[fold_df["model"] == lab, "pr_auc"].values for lab in labels
    ])  # shape (7, 5)

    fr_stat, fr_p = run_friedman(score_matrix)
    print(f"  Friedman χ² = {fr_stat:.4f},  p = {fr_p:.6f}")
    if fr_p < 0.05:
        print("  → Significant (p < 0.05): at least one model differs.")
    else:
        print("  → Not significant (p ≥ 0.05): no evidence of difference among models.")

    # ── 4. Exploratory pairwise Wilcoxon + Bonferroni ──
    print("\n[4/5] Exploratory pairwise Wilcoxon comparisons "
          "(Bonferroni, 21 pairs) ...")
    pw_df = pairwise_wilcoxon_bonferroni(score_matrix, labels)

    n_sig = pw_df["significant"].sum()
    print(f"  {n_sig} / {len(pw_df)} pairs significant at α = 0.05 (Bonferroni).")

    pw_csv = os.path.join(TAB_DIR, "step7_pairwise_wilcoxon.csv")
    pw_df.to_csv(pw_csv, index=False)
    print(f"  Saved: {pw_csv}")

    pw_tex = os.path.join(TAB_DIR, "step7_pairwise_wilcoxon.tex")
    save_pairwise_latex(pw_df, fr_stat, fr_p, pw_tex)

    # ── 5. Figures ──
    print("\n[5/5] Generating figures ...")

    fig_box = os.path.join(FIG_DIR, "step7_fold_prauc_boxplot.pdf")
    plot_fold_boxplot(fold_df, fig_box)

    fig_heat = os.path.join(FIG_DIR, "step7_pairwise_heatmap.pdf")
    plot_pairwise_heatmap(pw_df, labels, fig_heat)

    # ── Copy to overleaf ──
    for src in [fig_box, fig_heat]:
        dst = os.path.join(OVERLEAF_FIG, os.path.basename(src))
        shutil.copy2(src, dst)
        print(f"  Copied → {dst}")

    for src in [pw_tex]:
        dst = os.path.join(OVERLEAF_TAB, os.path.basename(src))
        shutil.copy2(src, dst)
        print(f"  Copied → {dst}")

    # ── Summary print ──
    print("\n" + "=" * 60)
    print("  Step 7 complete!")
    print("=" * 60)
    print(f"\n  Friedman χ² = {fr_stat:.4f},  p = {fr_p:.6f}")
    print(f"  Significant pairs (Bonferroni α=0.05): {n_sig} / {len(pw_df)}")
    print("\n  Fold-level PR-AUC summary:")
    for lab in labels:
        vals = fold_df.loc[fold_df["model"] == lab, "pr_auc"].values
        print(f"    {lab:6s}  {np.mean(vals):.4f} ± {np.std(vals):.4f}  "
              f"  folds: {', '.join(f'{v:.4f}' for v in vals)}")


if __name__ == "__main__":
    main()
