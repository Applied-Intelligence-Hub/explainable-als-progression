"""
Train final XGBoost model (6-month horizon) using Optuna best parameters.

This script:
  1) loads the processed 6m v2 dataset (one row per subject, 35 features)
  2) defines the binary target "rapid" as the worst 30% slopes (more negative)
  3) fits a preprocessing + tuned XGBoost pipeline
  4) saves the trained model (joblib) and a metadata JSON

Inputs:
  - 01_data/processed/dataset_6m_v2.csv
  - 04_outputs/tables/step5_tuning/XGB_unbalanced_6m_best.json

Outputs:
  - models/final_xgb_6m.joblib
  - models/final_xgb_6m_metadata.json
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier


# ----------------------------
# Config
# ----------------------------
DATASET_REL_PATH = os.path.join("01_data", "processed", "dataset_6m_v2.csv")
BEST_PARAMS_REL_PATH = os.path.join(
    "04_outputs", "tables", "step5_tuning", "XGB_unbalanced_6m_best.json")

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

ID_COL = "subject_id"
SLOPE_COL = "slope_180d_per_30d"
RAPID_FRAC = 0.30

MODELS_DIR = "models"
MODEL_FILENAME = "final_xgb_6m.joblib"
META_FILENAME = "final_xgb_6m_metadata.json"

SEED = 42
DECISION_THRESHOLD = 0.21


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

    decision_threshold: Optional[float]

    feature_cols: List[str]
    num_cols: List[str]
    cat_cols: List[str]

    # tuned parameters (as used to fit final model)
    xgb_params: dict


def load_json(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Loaded dataset is empty.")

    for required in (ID_COL, SLOPE_COL):
        if required not in df.columns:
            raise KeyError(f"Missing required column '{required}' in dataset.")

    df[SLOPE_COL] = pd.to_numeric(df[SLOPE_COL], errors="coerce")
    df = df.dropna(subset=[SLOPE_COL]).copy()
    if df.empty:
        raise ValueError("No rows with valid slope after cleaning.")

    return df


def build_target(df: pd.DataFrame) -> Tuple[pd.Series, float]:
    cutoff = float(df[SLOPE_COL].quantile(RAPID_FRAC))
    y = (df[SLOPE_COL] <= cutoff).astype(int)
    return y, cutoff


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    feat_cols = [c for c in df.columns if c not in DROP_COLS]
    if not feat_cols:
        raise ValueError(
            "No feature columns available after dropping ID/target columns.")

    X = df[feat_cols].copy()
    return X, feat_cols


def split_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    num_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return num_cols, cat_cols


def build_preproc_tree(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )


def build_xgb_from_best(best_payload: dict) -> dict:
    # payload structure: {"best_params": {...}, ...}
    params = best_payload.get("best_params", {}).copy()
    if not params:
        raise KeyError("best_params not found in Optuna payload.")

    # ensure fixed settings consistent with the Optuna script
    params.update({
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": SEED,
        "n_jobs": -1,
        "verbosity": 0,
        "tree_method": "hist",
    })
    return params


def save_artifacts(model: Pipeline, meta: TrainMeta) -> Tuple[str, str]:
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, MODEL_FILENAME)
    meta_path = os.path.join(MODELS_DIR, META_FILENAME)

    joblib.dump(model, model_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, ensure_ascii=False, indent=2)

    return model_path, meta_path


def main() -> None:
    root = os.getcwd()
    dataset_path = os.path.join(root, DATASET_REL_PATH)
    best_path = os.path.join(root, BEST_PARAMS_REL_PATH)

    print(f"[INFO] Project root: {root}")
    print(f"[INFO] Loading dataset: {dataset_path}")
    print(f"[INFO] Loading Optuna best params: {best_path}")

    df = load_dataset(dataset_path)
    best_payload = load_json(best_path)
    xgb_params = build_xgb_from_best(best_payload)

    y, slope_cutoff = build_target(df)
    X, feat_cols = build_features(df)
    num_cols, cat_cols = split_types(X)

    print(f"[INFO] Loaded rows: {len(df)}")
    print(f"[INFO] rapid prevalence: {y.mean():.3f}")
    print(f"[INFO] slope cutoff (30%): {slope_cutoff:.4f}")
    print(
        f"[INFO] features: {len(feat_cols)} (num={len(num_cols)}, cat={len(cat_cols)})")

    preproc = build_preproc_tree(num_cols, cat_cols)
    clf = XGBClassifier(**xgb_params)
    pipe = Pipeline([("prep", preproc), ("clf", clf)])

    pipe.fit(X, y)

    # quick sanity
    proba = pipe.predict_proba(X)[:, 1]
    print(
        f"[INFO] proba: min={proba.min():.3f} mean={proba.mean():.3f} max={proba.max():.3f}")

    meta = TrainMeta(
        model_name="XGBoost (Optuna tuned)",
        horizon="6m",
        created_utc=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        dataset_path=DATASET_REL_PATH,
        best_params_path=BEST_PARAMS_REL_PATH,
        n_samples=int(len(df)),
        rapid_prevalence=float(y.mean()),
        slope_col=SLOPE_COL,
        slope_cutoff_30pct=float(slope_cutoff),
        decision_threshold=DECISION_THRESHOLD,
        feature_cols=feat_cols,
        num_cols=num_cols,
        cat_cols=cat_cols,
        xgb_params=xgb_params,
    )

    model_path, meta_path = save_artifacts(pipe, meta)
    print(f"[OK] Saved model: {model_path}")
    print(f"[OK] Saved metadata: {meta_path}")


if __name__ == "__main__":
    main()
