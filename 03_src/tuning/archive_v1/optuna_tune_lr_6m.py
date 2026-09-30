"""
Optuna tuning for Logistic Regression on 6-month horizon (ALS rapid progression).

Protocol:
- Dataset: 01_data/processed/dataset_6m_v1.csv
- Subject-wise CV: GroupKFold by subject_id
- Labels: rapid = worst 30% slopes (computed on TRAIN fold only)
- Objective metric: PR-AUC (average precision) on validation fold

Outputs:
- 04_outputs/tables/optuna_lr_6m_trials.csv
- 04_outputs/tables/optuna_lr_6m_best_params.json
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATASET_PATH = os.path.join("01_data", "processed", "dataset_6m_v1.csv")
ID_COL = "subject_id"
SLOPE_COL = "slope_180d_per_30d"
RAPID_FRAC = 0.30

N_SPLITS = 5
SEED = 42


def make_xy(df: pd.DataFrame):
    df = df.copy()
    df[SLOPE_COL] = pd.to_numeric(df[SLOPE_COL], errors="coerce")
    df = df.dropna(subset=[SLOPE_COL]).copy()

    drop_cols = {ID_COL, SLOPE_COL}
    if "slope_90d_per_30d" in df.columns:
        drop_cols.add("slope_90d_per_30d")

    feat_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feat_cols].copy()
    slope = df[SLOPE_COL].copy()
    groups = df[ID_COL].copy()
    return X, slope, groups


def build_preproc_scaled(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
    ])
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


def objective_factory(X, slope, groups, preproc):
    gkf = GroupKFold(n_splits=N_SPLITS)

    def objective(trial: optuna.Trial) -> float:
        # regularização
        C = trial.suggest_float("C", 1e-3, 50.0, log=True)
        penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
        class_weight = trial.suggest_categorical(
            "class_weight", [None, "balanced"])

        clf = LogisticRegression(
            max_iter=3000,
            solver="liblinear",     # suporta l1 e l2
            C=C,
            penalty=penalty,
            class_weight=class_weight,
            random_state=SEED,
        )
        pipe = Pipeline([("prep", preproc), ("clf", clf)])

        scores = []
        for tr_idx, va_idx in gkf.split(X, groups=groups):
            Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
            slope_tr, slope_va = slope.iloc[tr_idx], slope.iloc[va_idx]

            thr_label = float(np.nanquantile(slope_tr.dropna(), RAPID_FRAC))
            ytr = (slope_tr <= thr_label).astype(int)
            yva = (slope_va <= thr_label).astype(int)

            pipe.fit(Xtr, ytr)
            proba = pipe.predict_proba(Xva)[:, 1]
            scores.append(average_precision_score(yva, proba))

        return float(np.mean(scores))

    return objective


def main():
    df = pd.read_csv(DATASET_PATH)
    X, slope, groups = make_xy(df)
    preproc = build_preproc_scaled(X)

    sampler = TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    objective = objective_factory(X, slope, groups, preproc)

    n_trials = 60
    print(
        f"[INFO] Starting Optuna for LR (6m) | N={len(X)} | trials={n_trials}")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    out_dir = os.path.join("04_outputs", "tables", "step5_tuning")
    os.makedirs(out_dir, exist_ok=True)

    trials_path = os.path.join(out_dir, "optuna_lr_6m_trials.csv")
    best_path = os.path.join(out_dir, "optuna_lr_6m_best_params.json")

    study.trials_dataframe().to_csv(trials_path, index=False)

    payload = {
        "model": "LogisticRegression",
        "horizon": "6m",
        "metric_optimized": "PR_AUC",
        "best_value": float(study.best_value),
        "best_params": study.best_params,
        "n_trials": len(study.trials),
        "n_splits": N_SPLITS,
        "rapid_frac": RAPID_FRAC,
        "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("[OK] Best PR-AUC:", payload["best_value"])
    print("[OK] Saved trials:", trials_path)
    print("[OK] Saved best params:", best_path)


if __name__ == "__main__":
    main()
