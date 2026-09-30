"""Shared visual identity for Study I figures."""

from __future__ import annotations

import os
import shutil

import matplotlib
from matplotlib.colors import LinearSegmentedColormap, to_rgba


PALETTE = {
    "pastel_petal": "#E5C3D1",
    "powder_blush": "#F7A9A8",
    "glaucous": "#7D82B8",
    "soft_mauve": "#A66FA3",
    "bubblegum_pink": "#EF798A",
    "deep_rose": "#C95E7B",
    "velvet_purple": "#613F75",
}

MODEL_COLOURS = {
    "LogisticRegression": PALETTE["pastel_petal"],
    "Logistic Regression": PALETTE["pastel_petal"],
    "LR": PALETTE["pastel_petal"],
    "DecisionTree": PALETTE["powder_blush"],
    "Decision Tree": PALETTE["powder_blush"],
    "DT": PALETTE["powder_blush"],
    "RandomForest": PALETTE["glaucous"],
    "Random Forest": PALETTE["glaucous"],
    "RF": PALETTE["glaucous"],
    "SVM": PALETTE["soft_mauve"],
    "KNeighbors": PALETTE["bubblegum_pink"],
    "KNN": PALETTE["bubblegum_pink"],
    "LightGBM": PALETTE["deep_rose"],
    "LGBM": PALETTE["deep_rose"],
    "XGBoost": PALETTE["velvet_purple"],
    "XGB": PALETTE["velvet_purple"],
}

FIGURE_INK = PALETTE["velvet_purple"]
GRID_COLOUR = "#F6EDF2"
RISK_INCREASE = PALETTE["deep_rose"]
RISK_DECREASE = PALETTE["glaucous"]
ABOVE_THRESHOLD = PALETTE["bubblegum_pink"]
BELOW_THRESHOLD = PALETTE["glaucous"]

SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "study1_sequential",
    ["#FFFFFF", PALETTE["pastel_petal"], PALETTE["glaucous"], FIGURE_INK],
)
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "study1_diverging",
    [PALETTE["glaucous"], "#FFFFFF", PALETTE["deep_rose"]],
)
SHAP_CMAP = LinearSegmentedColormap.from_list(
    "study1_shap",
    [FIGURE_INK, PALETTE["soft_mauve"], PALETTE["powder_blush"]],
)


def model_colour(model_name: str) -> str:
    """Return the fixed colour assigned to a model family."""
    return MODEL_COLOURS.get(str(model_name), FIGURE_INK)


def apply_balanced_hatch(patch, face_colour: str, hatch: str = "///") -> None:
    """Add a balanced-mode hatch that remains visible on light and dark fills."""
    red, green, blue, _ = to_rgba(face_colour)
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    patch.set_hatch(hatch)
    patch._hatch_color = to_rgba("white" if luminance < 0.58 else FIGURE_INK)


def apply_study1_style() -> None:
    """Apply consistent publication defaults without changing plot data."""
    matplotlib.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": FIGURE_INK,
        "axes.labelcolor": FIGURE_INK,
        "axes.titlecolor": FIGURE_INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID_COLOUR,
        "grid.linewidth": 0.8,
        "grid.alpha": 1.0,
        "hatch.linewidth": 1.1,
        "xtick.color": FIGURE_INK,
        "ytick.color": FIGURE_INK,
        "text.color": FIGURE_INK,
        "legend.edgecolor": PALETTE["pastel_petal"],
        "legend.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 300,
        "font.size": 10,
    })


def style_axis(ax, grid_axis: str = "both") -> None:
    """Apply the shared axis treatment to an existing axis."""
    ax.grid(True, axis=grid_axis, color=GRID_COLOUR, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color(FIGURE_INK)
    ax.tick_params(colors=FIGURE_INK)
    ax.xaxis.label.set_color(FIGURE_INK)
    ax.yaxis.label.set_color(FIGURE_INK)
    ax.title.set_color(FIGURE_INK)


def save_figure(fig, output_dir: str, name: str, overleaf_dir: str | None = None) -> None:
    """Save PDF and PNG at publication resolution and optionally copy to Overleaf."""
    os.makedirs(output_dir, exist_ok=True)
    targets = [output_dir]
    if overleaf_dir:
        os.makedirs(overleaf_dir, exist_ok=True)
        targets.append(overleaf_dir)
    for directory in targets:
        fig.savefig(
            os.path.join(directory, f"{name}.pdf"),
            dpi=300,
            bbox_inches="tight",
        )
        fig.savefig(
            os.path.join(directory, f"{name}.png"),
            dpi=300,
            bbox_inches="tight",
        )


def copy_figure_pair(output_dir: str, overleaf_dir: str, name: str) -> None:
    """Copy an existing PDF/PNG pair into the Overleaf figure directory."""
    os.makedirs(overleaf_dir, exist_ok=True)
    for extension in ("pdf", "png"):
        source = os.path.join(output_dir, f"{name}.{extension}")
        if os.path.exists(source):
            shutil.copy2(source, os.path.join(overleaf_dir, f"{name}.{extension}"))
