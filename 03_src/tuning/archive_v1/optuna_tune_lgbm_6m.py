"""
Optuna tuning for LightGBM on 6-month horizon (ALS rapid progression).

Protocol:
- Dataset: 01_data/processed/dataset_6m_v1.csv
- Subject-wise CV: GroupKFold by subject_id
- Labels: rapid = worst 30% slopes (more negative) computed on TRAIN fold only
- Objective metric: PR-AUC (average precision) on validation fold

Outputs:
- 04_outputs/tables/optuna_lgbm_6m_trials.csv
- 04_outputs/tables/optuna_lgbm_6m_best_params.json
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

import optuna
from optuna.samplers import TPESampler

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from lightgbm import LGBMClassifier


# ----------------------------
# Config
# ----------------------------
DATASET_PATH = os.path.join("01_data", "processed", "dataset_6m_v1.csv")
ID_COL = "subject_id"
SLOPE_COL = "slope_180d_per_30d"
RAPID_FRAC = 0.30

N_SPLITS = 5
SEED = 42


def build_preproc_tree(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )


def make_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    if ID_COL not in df.columns or SLOPE_COL not in df.columns:
        raise KeyError(f"Dataset must contain '{ID_COL}' and '{SLOPE_COL}'.")

    df = df.copy()
    df[SLOPE_COL] = pd.to_numeric(df[SLOPE_COL], errors="coerce")
    df = df.dropna(subset=[SLOPE_COL]).copy()

    drop_cols = {ID_COL, SLOPE_COL}
    if "slope_90d_per_30d" in df.columns:
        drop_cols.add("slope_90d_per_30d")

    feat_cols = [c for c in df.columns if c not in drop_cols]
    if not feat_cols:
        raise ValueError(
            "No feature columns available after dropping ID/target columns.")

    X = df[feat_cols].copy()
    slope = df[SLOPE_COL].copy()
    groups = df[ID_COL].copy()
    return X, slope, groups


def objective_factory(X: pd.DataFrame, slope: pd.Series, groups: pd.Series, preproc: ColumnTransformer):
    gkf = GroupKFold(n_splits=N_SPLITS)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            # -1 = no limit
            "max_depth": trial.suggest_int("max_depth", -1, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 1.0, log=True),

            # fixed
            "random_state": SEED,
            "n_jobs": -1,
            "verbosity": -1,  # silencia logs
            "objective": "binary",
        }

        clf = LGBMClassifier(**params)
        pipe = Pipeline([("prep", preproc), ("clf", clf)])

        fold_scores = []

        for tr_idx, va_idx in gkf.split(X, groups=groups):
            Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
            slope_tr, slope_va = slope.iloc[tr_idx], slope.iloc[va_idx]

            # label cutoff computed ONLY on training fold
            thr_label = float(np.nanquantile(slope_tr.dropna(), RAPID_FRAC))
            ytr = (slope_tr <= thr_label).astype(int)
            yva = (slope_va <= thr_label).astype(int)

            pipe.fit(Xtr, ytr)
            proba = pipe.predict_proba(Xva)[:, 1]

            pr_auc = average_precision_score(yva, proba)
            fold_scores.append(pr_auc)

        return float(np.mean(fold_scores))

    return objective


def main():
    df = pd.read_csv(DATASET_PATH)
    X, slope, groups = make_xy(df)
    preproc = build_preproc_tree(X)

    sampler = TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    objective = objective_factory(X, slope, groups, preproc)

    n_trials = 60
    timeout_s = None

    print(
        f"[INFO] Starting Optuna for LightGBM (6m) | N={len(X)} | trials={n_trials}")
    study.optimize(objective, n_trials=n_trials,
                   timeout=timeout_s, show_progress_bar=True)

    best_params = study.best_params
    best_value = float(study.best_value)

    out_dir = os.path.join("04_outputs", "tables", "step5_tuning")
    os.makedirs(out_dir, exist_ok=True)

    trials_path = os.path.join(out_dir, "optuna_lgbm_6m_trials.csv")
    best_path = os.path.join(out_dir, "optuna_lgbm_6m_best_params.json")

    trials_df = study.trials_dataframe()
    trials_df.to_csv(trials_path, index=False)

    payload = {
        "model": "LightGBM",
        "horizon": "6m",
        "metric_optimized": "PR_AUC",
        "best_value": best_value,
        "best_params": best_params,
        "n_trials": len(study.trials),
        "n_splits": N_SPLITS,
        "rapid_frac": RAPID_FRAC,
        "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("[OK] Best PR-AUC:", best_value)
    print("[OK] Saved trials:", trials_path)
    print("[OK] Saved best params:", best_path)


if __name__ == "__main__":
    main()
