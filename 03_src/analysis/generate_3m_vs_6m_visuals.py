"""
Generate comparative 3m vs 6m visual artefacts for the Overleaf thesis.

Outputs
-------
Figures:
  04_outputs/figures/step5v2_3m_prauc_barchart.{pdf,png}
  04_outputs/figures/step5v2_3m_vs_6m_prauc.{pdf,png}
  04_outputs/figures/step5v2_3m_vs_6m_scatter.{pdf,png}

Tables (LaTeX):
  04_outputs/tables/step5v2_tuning_comparison_3m.tex
  04_outputs/tables/step5v2_3m_vs_6m_comparison.tex
"""

from __future__ import annotations
import os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from study1_style import (
    FIGURE_INK,
    apply_balanced_hatch,
    apply_study1_style,
    model_colour,
    style_axis,
)

apply_study1_style()

# ── paths ──
ROOT    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG_DIR_3M = os.path.join(ROOT, "04_outputs", "figures", "step5_3m_vs_6m")
FIG_DIR_TUNING = os.path.join(ROOT, "04_outputs", "figures", "step5_tuning")
TAB_DIR_TUNING = os.path.join(ROOT, "04_outputs", "tables", "step5_tuning")
TAB_DIR_COMP   = os.path.join(ROOT, "04_outputs", "tables", "step5_comparison")
OVERLEAF_FIG = os.path.join(ROOT, "overleaf", "images", "figures")
OVERLEAF_TAB = os.path.join(ROOT, "overleaf", "images", "tables")
for d in [FIG_DIR_3M, FIG_DIR_TUNING, TAB_DIR_TUNING, TAB_DIR_COMP, OVERLEAF_FIG, OVERLEAF_TAB]:
    os.makedirs(d, exist_ok=True)

SUMMARY_3M = os.path.join(TAB_DIR_TUNING, "step5_tuning_summary_3m.csv")
SUMMARY_6M = os.path.join(TAB_DIR_TUNING, "step5_tuning_summary_6m.csv")

t_start = time.time()

df3 = pd.read_csv(SUMMARY_3M)
df6 = pd.read_csv(SUMMARY_6M)

df3["label"] = df3["model"] + "\n" + df3["mode"]
df6["label"] = df6["model"] + "\n" + df6["mode"]

print(f"Loaded 3m: {len(df3)} rows,  6m: {len(df6)} rows")


# ════════════════════════════════════════════════════════════
# 1.  PR-AUC bar chart — 3-month horizon
# ════════════════════════════════════════════════════════════
df3s = df3.sort_values("pr_auc_mean", ascending=True).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(8, 5))
colors = [model_colour(model) for model in df3s["model"]]
bars = ax.barh(df3s["label"], df3s["pr_auc_mean"], xerr=df3s["pr_auc_std"],
               color=colors, edgecolor=FIGURE_INK, capsize=3, height=0.7)
for bar, mode, colour in zip(bars, df3s["mode"], colors):
    if mode == "balanced":
        apply_balanced_hatch(bar, colour)

# Place each value immediately after its own error bar.
for bar, val, err in zip(bars, df3s["pr_auc_mean"], df3s["pr_auc_std"]):
    ax.text(val + err + 0.004, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", ha="left", fontsize=9, fontweight="bold")

ax.set_xlabel("PR-AUC (mean ± std, 5-fold GroupKFold)", fontsize=11)
ax.set_title("Optuna-Tuned Models — 3-Month Horizon (DEV, n=1925)", fontsize=13, fontweight="bold")
best_val = df3s["pr_auc_mean"].max()
ax.axvline(x=best_val, ls="--", color=FIGURE_INK, alpha=0.7, lw=0.8)
ax.legend(handles=[Patch(facecolor="white", edgecolor=FIGURE_INK, label="Unbalanced"),
                   Patch(facecolor="white", edgecolor=FIGURE_INK, hatch="//", label="Balanced")],
          loc="lower right", fontsize=7.5, handlelength=1.6,
          handleheight=0.8, borderpad=0.35, labelspacing=0.3)
ax.set_xlim(0.30, 0.515)
style_axis(ax, "x")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR_3M, "step5v2_3m_prauc_barchart.pdf"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(FIG_DIR_3M, "step5v2_3m_prauc_barchart.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("  [OK] 3m PR-AUC bar chart saved")


# ════════════════════════════════════════════════════════════
# 2.  Side-by-side grouped bar chart: 3m vs 6m (matched models)
# ════════════════════════════════════════════════════════════
# Match models that appear in both horizons
df3_key = df3[["model", "mode", "pr_auc_mean", "pr_auc_std"]].copy()
df6_key = df6[["model", "mode", "pr_auc_mean", "pr_auc_std"]].copy()
merged = df3_key.merge(df6_key, on=["model", "mode"], suffixes=("_3m", "_6m"))
merged["label"] = merged["model"] + "\n" + merged["mode"]
model_rank = (
    merged.groupby("model")["pr_auc_mean_6m"]
    .max()
    .sort_values()
    .index
)
merged["model_rank"] = pd.Categorical(
    merged["model"], categories=model_rank, ordered=True
)
merged["mode_rank"] = merged["mode"].map({"balanced": 0, "unbalanced": 1})
merged = (
    merged.sort_values(["model_rank", "mode_rank"])
    .drop(columns=["model_rank", "mode_rank"])
    .reset_index(drop=True)
)

fig, ax = plt.subplots(figsize=(10, 6))
y = np.arange(len(merged))
bar_h = 0.35

model_colours = [model_colour(model) for model in merged["model"]]
bars_6m = ax.barh(y + bar_h / 2, merged["pr_auc_mean_6m"], bar_h,
                  xerr=merged["pr_auc_std_6m"], color=model_colours,
                  edgecolor=FIGURE_INK, capsize=2, label="6-month")
bars_3m = ax.barh(y - bar_h / 2, merged["pr_auc_mean_3m"], bar_h,
                  xerr=merged["pr_auc_std_3m"], color=model_colours,
                  edgecolor=FIGURE_INK, capsize=2, label="3-month")
for bar, colour in zip(bars_3m, model_colours):
    apply_balanced_hatch(bar, colour)

# Place each value immediately after its own error bar.
for bar, val, err in zip(
    bars_6m, merged["pr_auc_mean_6m"], merged["pr_auc_std_6m"]
):
    ax.text(val + err + 0.004, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", ha="left", fontsize=8, fontweight="bold",
            color=FIGURE_INK)
for bar, val, err in zip(
    bars_3m, merged["pr_auc_mean_3m"], merged["pr_auc_std_3m"]
):
    ax.text(val + err + 0.004, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", ha="left", fontsize=8, fontweight="bold",
            color=FIGURE_INK)

ax.set_yticks(y)
ax.set_yticklabels(merged["label"], fontsize=9)
ax.set_xlabel("PR-AUC (mean ± std)", fontsize=11)
ax.set_title("PR-AUC Comparison: 3-Month vs 6-Month Horizons (DEV)", fontsize=13, fontweight="bold")
ax.legend(loc="upper right", bbox_to_anchor=(1.0, -0.08), ncol=2, fontsize=10)
ax.set_xlim(0.30, 0.515)
style_axis(ax, "x")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR_3M, "step5v2_3m_vs_6m_prauc.pdf"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(FIG_DIR_3M, "step5v2_3m_vs_6m_prauc.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("  [OK] 3m vs 6m grouped bar chart saved")


# ════════════════════════════════════════════════════════════
# 3.  Scatter plot: 3m PR-AUC vs 6m PR-AUC per model×mode
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 7))

# Diagonal reference line
lims = [0.38, 0.45]
ax.plot(lims, lims, "--", color=FIGURE_INK, alpha=0.7, lw=1, label="y = x (parity)")

label_offsets = {
    ("LogisticRegression", "unbalanced"): (-72, -32),
    ("LogisticRegression", "balanced"): (12, -34),
    ("LightGBM", "unbalanced"): (-82, 14),
    ("LightGBM", "balanced"): (0, 34),
    ("RandomForest", "unbalanced"): (12, -36),
    ("RandomForest", "balanced"): (-82, -34),
    ("XGBoost", "unbalanced"): (12, 12),
    ("XGBoost", "balanced"): (12, 20),
}

for _, row in merged.iterrows():
    c = model_colour(row["model"])
    marker = "o" if row["mode"] == "unbalanced" else "s"
    ax.scatter(row["pr_auc_mean_6m"], row["pr_auc_mean_3m"], c=c,
               marker=marker, s=100, edgecolors=FIGURE_INK, linewidths=0.7, zorder=5)
    delta = row["pr_auc_mean_3m"] - row["pr_auc_mean_6m"]
    ax.annotate(
                f"{row['model']}\nΔ={delta:+.3f}",
                (row["pr_auc_mean_6m"], row["pr_auc_mean_3m"]),
                textcoords="offset points",
                xytext=label_offsets[(row["model"], row["mode"])],
                fontsize=7.2, color=c, fontweight="bold",
                linespacing=1.15,
                arrowprops=dict(arrowstyle="-", color=c, lw=0.6, alpha=0.8))

ax.set_xlabel("PR-AUC — 6-month (DEV)", fontsize=11)
ax.set_ylabel("PR-AUC — 3-month (DEV)", fontsize=11)
ax.set_title("3m vs 6m PR-AUC per Model (Optuna-Tuned)", fontsize=13, fontweight="bold")
ax.legend(handles=[
    plt.Line2D([0], [0], marker="o", color=FIGURE_INK, ls="", markersize=8, label="Unbalanced"),
    plt.Line2D([0], [0], marker="s", color=FIGURE_INK, ls="", markersize=8, label="Balanced"),
    plt.Line2D([0], [0], ls="--", color=FIGURE_INK, label="Parity line"),
], loc="upper left", fontsize=9)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_aspect("equal")
style_axis(ax)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR_3M, "step5v2_3m_vs_6m_scatter.pdf"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(FIG_DIR_3M, "step5v2_3m_vs_6m_scatter.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("  [OK] 3m vs 6m scatter plot saved")


# ════════════════════════════════════════════════════════════
# 4.  LaTeX table — 3m tuning comparison
# ════════════════════════════════════════════════════════════
df3_sorted = df3.sort_values("pr_auc_mean", ascending=False).reset_index(drop=True)

best_pr  = df3_sorted["pr_auc_mean"].max()
best_roc = df3_sorted["roc_auc_mean"].max()
best_f1  = df3_sorted["f1_mean"].max()
best_f2  = df3_sorted["f2_mean"].max()

def fmt_bold(val, best, std=None):
    s = f"{val:.3f}"
    if std is not None:
        s += f" $\\pm$ {std:.3f}"
    if abs(val - best) < 1e-6:
        return r"\textbf{" + s + "}"
    return s

lines = [
    r"\begin{table}[ht]",
    r"\centering",
    r"\caption[Optuna-tuned model comparison, 3-month horizon]{Optuna-tuned model comparison (3-month horizon, DEV set, GroupKFold $K\!=\!5$, 60 trials). "
    r"Models ranked by PR-AUC. Best value per metric in \textbf{bold}.}",
    r"\label{tab:step5v2_tuning_3m}",
    r"\small",
    r"\begin{tabular}{llcccc}",
    r"\toprule",
    r"Model & Mode & PR-AUC & ROC-AUC & F1\textsubscript{@0.5} & F2\textsubscript{@0.5} \\",
    r"\midrule",
]

for _, row in df3_sorted.iterrows():
    model = row["model"].replace("_", r"\_")
    mode  = row["mode"]
    pr    = fmt_bold(row["pr_auc_mean"], best_pr, row["pr_auc_std"])
    roc   = fmt_bold(row["roc_auc_mean"], best_roc)
    f1    = fmt_bold(row["f1_mean"], best_f1)
    f2    = fmt_bold(row["f2_mean"], best_f2)
    lines.append(f"{model} & {mode} & {pr} & {roc} & {f1} & {f2} \\\\")

lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

tex_path = os.path.join(TAB_DIR_TUNING, "step5v2_tuning_comparison_3m.tex")
with open(tex_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print(f"  [OK] 3m LaTeX tuning table saved → {tex_path}")


# ════════════════════════════════════════════════════════════
# 5.  LaTeX table — 3m vs 6m side-by-side for matched models
# ════════════════════════════════════════════════════════════
merged_sorted = merged.sort_values("pr_auc_mean_6m", ascending=False).reset_index(drop=True)
merged_sorted["delta"] = merged_sorted["pr_auc_mean_3m"] - merged_sorted["pr_auc_mean_6m"]

lines2 = [
    r"\begin{table}[ht]",
    r"\centering",
    r"\caption{Descriptive PR-AUC comparison across prediction horizons for matched model configurations. "
    r"Models were tuned independently on the DEV set for each horizon. "
    r"$\Delta$ = 3m $-$ 6m; positive values indicate higher performance at 3 months.}",
    r"\label{tab:3m_vs_6m}",
    r"\small",
    r"\begin{tabular}{llccc}",
    r"\toprule",
    r"Model & Mode & PR-AUC\textsubscript{6m} & PR-AUC\textsubscript{3m} & $\Delta$ \\",
    r"\midrule",
]

for _, row in merged_sorted.iterrows():
    model = row["model"].replace("_", r"\_")
    mode  = row["mode"]
    v6    = f"{row['pr_auc_mean_6m']:.3f} $\\pm$ {row['pr_auc_std_6m']:.3f}"
    v3    = f"{row['pr_auc_mean_3m']:.3f} $\\pm$ {row['pr_auc_std_3m']:.3f}"
    d     = row["delta"]
    sign  = "+" if d >= 0 else ""
    ds    = f"{sign}{d:.3f}"
    lines2.append(f"{model} & {mode} & {v6} & {v3} & {ds} \\\\")

lines2 += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

tex2_path = os.path.join(TAB_DIR_COMP, "step5v2_3m_vs_6m_comparison.tex")
with open(tex2_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines2) + "\n")
print(f"  [OK] 3m vs 6m LaTeX comparison table saved → {tex2_path}")


# ════════════════════════════════════════════════════════════
# 6.  Also regenerate 6m bar chart with numeric annotations
# ════════════════════════════════════════════════════════════
df6s = df6.sort_values("pr_auc_mean", ascending=True).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(8, 6))
colors = [model_colour(model) for model in df6s["model"]]
bars = ax.barh(df6s["label"], df6s["pr_auc_mean"], xerr=df6s["pr_auc_std"],
               color=colors, edgecolor=FIGURE_INK, capsize=3, height=0.7)
for bar, mode, colour in zip(bars, df6s["mode"], colors):
    if mode == "balanced":
        apply_balanced_hatch(bar, colour)

for bar, val, err in zip(bars, df6s["pr_auc_mean"], df6s["pr_auc_std"]):
    ax.text(val + err + 0.004, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", ha="left", fontsize=8, fontweight="bold")

ax.set_xlabel("PR-AUC (mean ± std, 5-fold GroupKFold)", fontsize=11)
ax.set_title("Optuna-Tuned Models — 6-Month Horizon (DEV, n=1113)", fontsize=13, fontweight="bold")
ax.axvline(x=df6s["pr_auc_mean"].max(), ls="--", color=FIGURE_INK, alpha=0.7, lw=0.8)
ax.legend(handles=[Patch(facecolor="white", edgecolor=FIGURE_INK, label="Unbalanced"),
                   Patch(facecolor="white", edgecolor=FIGURE_INK, hatch="//", label="Balanced")],
          loc="upper right", bbox_to_anchor=(1.0, -0.08), ncol=2, fontsize=9)
ax.set_xlim(0.30, 0.515)
style_axis(ax, "x")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR_TUNING, "step5v2_prauc_barchart.pdf"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(FIG_DIR_TUNING, "step5v2_prauc_barchart.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("  [OK] 6m PR-AUC bar chart regenerated with numeric annotations")


# ════════════════════════════════════════════════════════════
# Copy to overleaf
# ════════════════════════════════════════════════════════════
import shutil
for directory, f in [
    (FIG_DIR_3M, "step5v2_3m_prauc_barchart.png"),
    (FIG_DIR_3M, "step5v2_3m_vs_6m_prauc.png"),
    (FIG_DIR_3M, "step5v2_3m_vs_6m_scatter.png"),
    (FIG_DIR_TUNING, "step5v2_prauc_barchart.png"),
]:
    shutil.copy2(os.path.join(directory, f), os.path.join(OVERLEAF_FIG, f))
for directory, f in [
    (TAB_DIR_TUNING, "step5v2_tuning_comparison_3m.tex"),
    (TAB_DIR_COMP, "step5v2_3m_vs_6m_comparison.tex"),
]:
    shutil.copy2(os.path.join(directory, f), os.path.join(OVERLEAF_TAB, f))
print("  → Copied figures/tables to overleaf/images/")


# ════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════
elapsed = time.time() - t_start
print(f"\n{'='*60}")
print(f"  All 3m vs 6m artefacts generated in {elapsed:.1f}s")
print(f"{'='*60}")

# Quick comparison summary
print("\n── Top models per horizon ──")
print("6-month (DEV n=1113):")
top6 = df6.sort_values("pr_auc_mean", ascending=False).head(4)
for _, r in top6.iterrows():
    print(f"  {r['model']:20s} {r['mode']:12s}  PR-AUC={r['pr_auc_mean']:.3f}±{r['pr_auc_std']:.3f}  ROC-AUC={r['roc_auc_mean']:.3f}")

print("\n3-month (DEV n=1925):")
top3 = df3.sort_values("pr_auc_mean", ascending=False).head(4)
for _, r in top3.iterrows():
    print(f"  {r['model']:20s} {r['mode']:12s}  PR-AUC={r['pr_auc_mean']:.3f}±{r['pr_auc_std']:.3f}  ROC-AUC={r['roc_auc_mean']:.3f}")

print("\n── Deltas (3m - 6m) for matched configs ──")
for _, r in merged_sorted.iterrows():
    d = r["delta"]
    sign = "+" if d >= 0 else ""
    print(f"  {r['model']:20s} {r['mode']:12s}  Δ={sign}{d:.3f}")
