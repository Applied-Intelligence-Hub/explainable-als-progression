"""
Train final Logistic Regression model (6-month horizon) for ALS progression.

This script:
  1) loads the processed 6m v2 dataset (one row per subject, 35 features)
  2) defines the binary target "rapid" as the worst 30% slopes (more negative)
  3) fits a preprocessing + Logistic Regression pipeline (Optuna best params)
  4) saves the trained model (joblib) and a small metadata JSON

Inputs:
  - 01_data/processed/dataset_6m_v2.csv
  - 04_outputs/tables/step5_tuning/LR_balanced_6m_best.json

Outputs:
  - models/final_lr_6m.joblib
  - models/final_lr_6m_metadata.json
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ----------------------------
# Configuration
# ----------------------------
DATASET_REL_PATH = os.path.join("01_data", "processed", "dataset_6m_v2.csv")
BEST_PARAMS_REL_PATH = os.path.join(
    "04_outputs", "tables", "step5_tuning", "LR_balanced_6m_best.json")

ID_COL = "subject_id"
SLOPE_COL = "slope_180d_per_30d"

# "rapid" = worst 30% slopes (more negative)
RAPID_FRAC = 0.30

# Columns to exclude from features (aligned with tune_all.py / evaluate_holdout.py)
DROP_COLS = {
    "subject_id",
    "t0_delta_days",
    "vitals_delta_days",
    "fvc_delta_days",
    "slope_90d_per_30d",
    "slope_180d_per_30d",
    "ALSFRS_Responded_By",
}

# Final decision threshold (chosen from threshold sweep in Step 6)
FINAL_THRESHOLD = 0.30

# Output artefacts
MODELS_DIR = "models"
MODEL_FILENAME = "final_lr_6m.joblib"
META_FILENAME = "final_lr_6m_metadata.json"


@dataclass
class TrainMeta:
    model_name: str
    horizon: str
    created_utc: str

    dataset_path: str
    best_params_path: str
    n_samples: int
    rapid_prevalence: float

    slope_col: str
    slope_cutoff_30pct: float

    decision_threshold: float

    feature_cols: List[str]
    num_cols: List[str]
    cat_cols: List[str]


def project_root() -> str:
    """Return the current working directory as the project root."""
    return os.getcwd()


def load_dataset(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Loaded dataset is empty.")

    for required in (ID_COL, SLOPE_COL):
        if required not in df.columns:
            raise KeyError(f"Missing required column '{required}' in dataset.")

    # Basic cleaning for slope
    df[SLOPE_COL] = pd.to_numeric(df[SLOPE_COL], errors="coerce")
    df = df.dropna(subset=[SLOPE_COL]).copy()

    if df.empty:
        raise ValueError("No rows with valid slope after cleaning.")

    return df


def build_target(df: pd.DataFrame) -> Tuple[pd.Series, float]:
    """Create y (rapid=1) and return the slope cutoff used."""
    cutoff = float(df[SLOPE_COL].quantile(RAPID_FRAC))
    y = (df[SLOPE_COL] <= cutoff).astype(int)
    return y, cutoff


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Return X and list of feature columns."""
    feat_cols = [c for c in df.columns if c not in DROP_COLS]
    if not feat_cols:
        raise ValueError("No feature columns left after dropping ID/target columns.")

    X = df[feat_cols].copy()
    return X, feat_cols


def split_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Split columns into numeric and categorical based on dtypes."""
    num_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return num_cols, cat_cols


def build_pipeline(num_cols: List[str], cat_cols: List[str],
                   best_params: dict) -> Pipeline:
    """Preprocessing + Logistic Regression with Optuna best params."""
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preproc = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )

    clf = LogisticRegression(
        C=best_params["C"],
        penalty=best_params["penalty"],
        solver=best_params.get("solver", "liblinear"),
        max_iter=best_params.get("max_iter", 3000),
        class_weight="balanced",
        random_state=best_params.get("random_state", 42),
    )

    return Pipeline(steps=[("prep", preproc), ("clf", clf)])


def save_artifacts(model: Pipeline, meta: TrainMeta) -> Tuple[str, str]:
    os.makedirs(MODELS_DIR, exist_ok=True)

    model_path = os.path.join(MODELS_DIR, MODEL_FILENAME)
    meta_path = os.path.join(MODELS_DIR, META_FILENAME)

    joblib.dump(model, model_path)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, ensure_ascii=False, indent=2)

    return model_path, meta_path


def main() -> None:
    root = project_root()
    dataset_path = os.path.join(root, DATASET_REL_PATH)
    best_path = os.path.join(root, BEST_PARAMS_REL_PATH)

    print(f"[INFO] Project root: {root}")
    print(f"[INFO] Loading dataset: {dataset_path}")
    print(f"[INFO] Loading Optuna best params: {best_path}")

    df = load_dataset(dataset_path)
    print(f"[INFO] Loaded rows: {len(df)}")

    # Load best params from Optuna JSON
    with open(best_path, "r", encoding="utf-8") as f:
        best_payload = json.load(f)
    best_params = best_payload.get("full_params", best_payload["best_params"])
    print(f"[INFO] LR best params: {best_params}")

    y, slope_cutoff = build_target(df)
    X, feat_cols = build_features(df)
    num_cols, cat_cols = split_types(X)

    print(f"[INFO] rapid prevalence: {y.mean():.3f}")
    print(f"[INFO] slope cutoff (30%): {slope_cutoff:.4f}")
    print(f"[INFO] features: {len(feat_cols)} (num={len(num_cols)}, cat={len(cat_cols)})")

    pipe = build_pipeline(num_cols, cat_cols, best_params)
    pipe.fit(X, y)

    # Sanity check
    proba = pipe.predict_proba(X)[:, 1]
    preds = (proba >= FINAL_THRESHOLD).astype(int)
    print(f"[INFO] proba: min={proba.min():.3f} mean={proba.mean():.3f} max={proba.max():.3f}")
    print(f"[INFO] positives @ threshold {FINAL_THRESHOLD:.2f}: {preds.sum()} / {len(preds)}")

    meta = TrainMeta(
        model_name="LogisticRegression(class_weight=balanced, Optuna tuned)",
        horizon="6m",
        created_utc=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        dataset_path=DATASET_REL_PATH,
        best_params_path=BEST_PARAMS_REL_PATH,
        n_samples=int(len(df)),
        rapid_prevalence=float(y.mean()),
        slope_col=SLOPE_COL,
        slope_cutoff_30pct=float(slope_cutoff),
        decision_threshold=float(FINAL_THRESHOLD),
        feature_cols=feat_cols,
        num_cols=num_cols,
        cat_cols=cat_cols,
    )

    model_path, meta_path = save_artifacts(pipe, meta)

    print(f"[OK] Saved model: {model_path}")
    print(f"[OK] Saved metadata: {meta_path}")


if __name__ == "__main__":
    main()