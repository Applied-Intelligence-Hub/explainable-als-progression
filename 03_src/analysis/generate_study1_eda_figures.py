"""Regenerate the Study I slope and treatment EDA figures."""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from study1_style import (
    ABOVE_THRESHOLD,
    BELOW_THRESHOLD,
    FIGURE_INK,
    GRID_COLOUR,
    PALETTE,
    apply_study1_style,
    save_figure,
    style_axis,
)


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(ROOT, "01_data", "processed", "dataset_6m_v2.csv")
CONMEDS_PATH = os.path.join(ROOT, "01_data", "raw", "PROACT_CONMEDS.csv")
SLOPE_OUT = os.path.join(ROOT, "04_outputs", "figures", "step3_slopes")
TREATMENT_OUT = os.path.join(ROOT, "04_outputs", "figures", "eda_treatment")
OVERLEAF_OUT = os.path.join(ROOT, "overleaf", "images", "figures")

SLOPE_COL = "slope_180d_per_30d"
RAPID_FRAC = 0.30


def load_data() -> tuple[pd.DataFrame, float]:
    df = pd.read_csv(DATA_PATH)
    df[SLOPE_COL] = pd.to_numeric(df[SLOPE_COL], errors="coerce")
    df = df.dropna(subset=[SLOPE_COL]).copy()
    threshold = float(df[SLOPE_COL].quantile(RAPID_FRAC))
    df["rapid"] = (df[SLOPE_COL] <= threshold).astype(int)
    return df, threshold


def plot_slope_distribution(df: pd.DataFrame, threshold: float) -> None:
    slow = df.loc[df["rapid"] == 0, SLOPE_COL]
    rapid = df.loc[df["rapid"] == 1, SLOPE_COL]
    bins = np.linspace(df[SLOPE_COL].min(), df[SLOPE_COL].max(), 61)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(
        rapid,
        bins=bins,
        color=ABOVE_THRESHOLD,
        edgecolor="white",
        linewidth=0.5,
        label=f"Rapid (n = {len(rapid):,})",
    )
    ax.hist(
        slow,
        bins=bins,
        color=BELOW_THRESHOLD,
        edgecolor="white",
        linewidth=0.5,
        label=f"Slow (n = {len(slow):,})",
    )
    ax.axvline(
        threshold,
        color=FIGURE_INK,
        linestyle="--",
        linewidth=1.8,
        label=f"P30 cutoff = {threshold:.2f}",
    )
    ax.annotate(
        f"P30 = {threshold:.2f} points / 30 days",
        xy=(threshold, ax.get_ylim()[1] * 0.88),
        xytext=(threshold - 2.5, ax.get_ylim()[1] * 0.94),
        arrowprops={"arrowstyle": "->", "color": FIGURE_INK},
        bbox={"boxstyle": "round,pad=0.3", "fc": "#FFF9FC", "ec": FIGURE_INK},
        color=FIGURE_INK,
        fontweight="bold",
    )
    ax.set_xlabel("ALSFRS-R slope (points / 30 days)")
    ax.set_ylabel("Number of participants")
    ax.set_title(
        f"Distribution of ALSFRS-R Slopes — 6-Month Horizon (N = {len(df):,})"
    )
    ax.legend(loc="upper left")
    style_axis(ax, "y")
    fig.tight_layout()
    save_figure(fig, SLOPE_OUT, "slope_histogram_6m", OVERLEAF_OUT)
    plt.close(fig)


def plot_arm_distribution(df: pd.DataFrame, threshold: float) -> None:
    colours = {
        "Active": PALETTE["glaucous"],
        "Placebo": PALETTE["deep_rose"],
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for arm in ("Active", "Placebo"):
        sub = df[df["study_arm"] == arm]
        axes[0].hist(
            sub[SLOPE_COL],
            bins=40,
            alpha=0.42,
            label=f"{arm} (n={len(sub)})",
            color=colours[arm],
            edgecolor="white",
            linewidth=0.5,
        )
    axes[0].axvline(
        threshold, color=FIGURE_INK, linestyle="--", linewidth=1.2,
        label=f"P30 = {threshold:.2f}",
    )
    axes[0].set_xlabel("Slope (ALSFRS-R points / 30 days)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Slope Distribution by Study Arm")
    axes[0].legend(fontsize=9)

    data_box = df[df["study_arm"].notna()].copy()
    order = ["Active", "Placebo"]
    sns.boxplot(
        data=data_box,
        x="study_arm",
        y=SLOPE_COL,
        order=order,
        palette=[colours[x] for x in order],
        hue="study_arm",
        legend=False,
        ax=axes[1],
        width=0.5,
        fliersize=3,
        boxprops={"edgecolor": FIGURE_INK},
        whiskerprops={"color": FIGURE_INK},
        capprops={"color": FIGURE_INK},
        medianprops={"color": FIGURE_INK, "linewidth": 1.5},
    )
    for i, arm in enumerate(order):
        median = data_box.loc[data_box["study_arm"] == arm, SLOPE_COL].median()
        axes[1].text(i, median + 0.05, f"med={median:.2f}", ha="center", fontweight="bold")
    axes[1].axhline(threshold, color=FIGURE_INK, linestyle="--", linewidth=1)
    axes[1].set_xlabel("Study Arm")
    axes[1].set_ylabel("Slope (points / 30 days)")
    axes[1].set_title("Slope by Study Arm")
    for ax in axes:
        style_axis(ax)
    fig.suptitle("Study Arm and ALSFRS-R Slope (6 Months)", color=FIGURE_INK)
    fig.tight_layout()
    save_figure(fig, TREATMENT_OUT, "eda_treat_arm_slope_dist", OVERLEAF_OUT)
    plt.close(fig)


def plot_arm_rate(df: pd.DataFrame) -> None:
    groups = ["Active", "Placebo", "Missing"]
    subsets = [
        df[df["study_arm"] == "Active"],
        df[df["study_arm"] == "Placebo"],
        df[df["study_arm"].isna()],
    ]
    rates = [100 * sub["rapid"].mean() for sub in subsets]
    counts = [len(sub) for sub in subsets]
    colours = [
        PALETTE["glaucous"],
        PALETTE["deep_rose"],
        PALETTE["pastel_petal"],
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        groups, rates, color=colours, edgecolor=FIGURE_INK, linewidth=0.8, width=0.55
    )
    for bar, rate, count in zip(bars, rates, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{rate:.1f}%\n(n={count})",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax.axhline(
        100 * RAPID_FRAC,
        color=FIGURE_INK,
        linestyle="--",
        linewidth=1.2,
        label="Global prevalence (30%)",
    )
    ax.set_ylabel("Rapid Progression Rate (%)")
    ax.set_title("Rapid Progression Rate by Study Arm")
    ax.set_ylim(0, max(rates) + 10)
    ax.legend(fontsize=9)
    style_axis(ax, "y")
    fig.tight_layout()
    save_figure(fig, TREATMENT_OUT, "eda_treat_arm_rapid_rate", OVERLEAF_OUT)
    plt.close(fig)


def plot_riluzole(df: pd.DataFrame, threshold: float) -> None:
    settings = [
        (0, "No riluzole", PALETTE["pastel_petal"]),
        (1, "Riluzole", PALETTE["soft_mauve"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for value, label, colour in settings:
        sub = df[df["riluzole_pre_t0"] == value]
        axes[0].hist(
            sub[SLOPE_COL],
            bins=40,
            alpha=0.65,
            label=f"{label} (n={len(sub)})",
            color=colour,
            edgecolor="white",
            linewidth=0.5,
        )
    axes[0].axvline(
        threshold, color=FIGURE_INK, linestyle="--", linewidth=1.2,
        label=f"P30 = {threshold:.2f}",
    )
    axes[0].set_xlabel("Slope (ALSFRS-R points / 30 days)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Slope Distribution by Riluzole")
    axes[0].legend(fontsize=9)

    labels, rates, counts, medians, colours = [], [], [], [], []
    for value, label, colour in settings:
        sub = df[df["riluzole_pre_t0"] == value]
        labels.append(label)
        rates.append(100 * sub["rapid"].mean())
        counts.append(len(sub))
        medians.append(sub[SLOPE_COL].median())
        colours.append(colour)
    bars = axes[1].bar(
        labels, rates, color=colours, edgecolor=FIGURE_INK, linewidth=0.8, width=0.5
    )
    for bar, rate, count, median in zip(bars, rates, counts, medians):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{rate:.1f}%\n(n={count})\nmed={median:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9,
        )
    axes[1].axhline(
        100 * RAPID_FRAC, color=FIGURE_INK, linestyle="--", linewidth=1.2,
        label="Global prevalence (30%)",
    )
    axes[1].set_ylabel("Rapid Progression Rate (%)")
    axes[1].set_title("Rapid Progression Rate by Riluzole")
    axes[1].set_ylim(0, max(rates) + 12)
    axes[1].legend(fontsize=9)
    for ax in axes:
        style_axis(ax)
    fig.suptitle("Riluzole and ALSFRS-R Slope (6 Months)", color=FIGURE_INK)
    fig.tight_layout()
    save_figure(fig, TREATMENT_OUT, "eda_treat_riluzole_slope_dist", OVERLEAF_OUT)
    plt.close(fig)


def plot_interaction(df: pd.DataFrame) -> None:
    combinations = [
        ("Active", 0, "Active\nNo", PALETTE["glaucous"]),
        ("Active", 1, "Active\nYes", PALETTE["velvet_purple"]),
        ("Placebo", 0, "Placebo\nNo", PALETTE["powder_blush"]),
        ("Placebo", 1, "Placebo\nYes", PALETTE["bubblegum_pink"]),
    ]
    rates, counts = [], []
    for arm, riluzole, _, _ in combinations:
        sub = df[(df["study_arm"] == arm) & (df["riluzole_pre_t0"] == riluzole)]
        rates.append(100 * sub["rapid"].mean())
        counts.append(len(sub))
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        range(4),
        rates,
        color=[x[3] for x in combinations],
        edgecolor=FIGURE_INK,
        linewidth=0.8,
        width=0.6,
    )
    for bar, rate, count in zip(bars, rates, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.6,
            f"{rate:.1f}%\n(n={count})",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax.set_xticks(range(4))
    ax.set_xticklabels([x[2] for x in combinations])
    ax.axhline(
        100 * RAPID_FRAC, color=FIGURE_INK, linestyle="--", linewidth=1.2,
        label="Global prevalence (30%)",
    )
    ax.set_ylabel("Rapid Progression Rate (%)")
    ax.set_title("Interaction: Study Arm × Riluzole")
    ax.set_ylim(0, max(rates) + 12)
    ax.legend(fontsize=9)
    style_axis(ax, "y")
    fig.tight_layout()
    save_figure(fig, TREATMENT_OUT, "eda_treat_interaction", OVERLEAF_OUT)
    plt.close(fig)


def plot_top_medications(df: pd.DataFrame) -> None:
    conmeds = pd.read_csv(CONMEDS_PATH)
    conmeds["Start_Delta"] = pd.to_numeric(conmeds["Start_Delta"], errors="coerce")
    conmeds = conmeds.merge(
        df[["subject_id", "t0_delta_days"]], on="subject_id", how="inner"
    )
    conmeds = conmeds[conmeds["Start_Delta"] <= conmeds["t0_delta_days"]]
    top15 = (
        conmeds.groupby("Medication_Coded")["subject_id"]
        .nunique()
        .sort_values(ascending=False)
        .head(15)
    )
    rows = []
    for medication, n_subjects in top15.items():
        ids = set(conmeds.loc[conmeds["Medication_Coded"] == medication, "subject_id"])
        sub = df[df["subject_id"].isin(ids)]
        rows.append({
            "Medication": medication,
            "N": n_subjects,
            "Rate": 100 * sub["rapid"].mean(),
        })
    plot_df = pd.DataFrame(rows).sort_values("Rate")
    colours = [
        ABOVE_THRESHOLD if rate > 100 * RAPID_FRAC else BELOW_THRESHOLD
        for rate in plot_df["Rate"]
    ]
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(
        range(len(plot_df)),
        plot_df["Rate"],
        color=colours,
        edgecolor=FIGURE_INK,
        linewidth=0.7,
        height=0.65,
    )
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(
        [f"{medication}  (n={count})" for medication, count in zip(
            plot_df["Medication"], plot_df["N"]
        )],
        fontsize=9,
    )
    for bar, rate in zip(bars, plot_df["Rate"]):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{rate:.1f}%",
            va="center",
            fontweight="bold",
        )
    ax.axvline(
        100 * RAPID_FRAC, color=FIGURE_INK, linestyle="--", linewidth=1.2,
        label="Global prevalence (30%)",
    )
    ax.set_xlabel("Rapid Progression Rate (%)")
    ax.set_title("Top 15 Pre-Baseline Medications and Rapid Progression")
    ax.set_xlim(0, plot_df["Rate"].max() + 8)
    ax.legend(fontsize=9, loc="lower right")
    style_axis(ax, "x")
    fig.tight_layout()
    save_figure(fig, TREATMENT_OUT, "eda_treat_top_meds", OVERLEAF_OUT)
    plt.close(fig)


def main() -> None:
    sns.set_theme(style="whitegrid", rc={"grid.color": GRID_COLOUR})
    apply_study1_style()
    df, threshold = load_data()
    plot_slope_distribution(df, threshold)
    plot_arm_distribution(df, threshold)
    plot_arm_rate(df)
    plot_riluzole(df, threshold)
    plot_interaction(df)
    plot_top_medications(df)
    print("Study I slope and treatment EDA figures regenerated.")


if __name__ == "__main__":
    main()
