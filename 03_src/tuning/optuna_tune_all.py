"""
Step 5 — Unified Optuna hyperparameter tuning: all models × balanced/unbalanced.

Protocol
--------
- Dataset: dataset_{horizon}_v2.csv, DEV partition only
- CV: GroupKFold(5), subject-wise, fold-wise label definition (30th pctile)
- Optuna objective: PR-AUC (average precision)
- After best params found: final 5-fold re-eval → PR-AUC, ROC-AUC, F1, F2

Models: KNN, DecisionTree, RandomForest, SVM, LogisticRegression, LightGBM, XGBoost
Modes : unbalanced, balanced  (KNN has no native balancing → skipped in balanced mode)

Usage
-----
  python 03_src/tuning/optuna_tune_all.py                        # all models, both modes, 6m
  python 03_src/tuning/optuna_tune_all.py --models LR XGB        # specific models only
  python 03_src/tuning/optuna_tune_all.py --unbalanced-only       # skip balanced mode
  python 03_src/tuning/optuna_tune_all.py --horizon 3m            # 3-month horizon
  python 03_src/tuning/optuna_tune_all.py --trials 30             # fewer trials (testing)

Outputs
-------
  04_outputs/tables/step5_tuning_summary_{horizon}.csv
  04_outputs/tables/step5_tuning/{model}_{mode}_{horizon}_best.json
  04_outputs/tables/step5_tuning/{model}_{mode}_{horizon}_trials.csv
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import optuna
from optuna.samplers import TPESampler

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Tree-based
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Linear / distance-based
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
PROCESSED  = os.path.join("01_data", "processed")
OUT_TABLES = os.path.join("04_outputs", "tables")
OUT_TUNING = os.path.join(OUT_TABLES, "step5_tuning")

N_SPLITS   = 5
SEED       = 42
RAPID_FRAC = 0.30

# Columns to exclude from features
DROP_COLS = {
    "subject_id",
    "t0_delta_days",
    "vitals_delta_days",
    "fvc_delta_days",
    "slope_90d_per_30d",
    "slope_180d_per_30d",
    "ALSFRS_Responded_By",   # near-constant (93 % NaN)
}

# Model short names → full names (for display)
MODEL_NAMES = {
    "KNN": "KNeighbors",
    "DT":  "DecisionTree",
    "RF":  "RandomForest",
    "SVM": "SVM",
    "LR":  "LogisticRegression",
    "LGBM": "LightGBM",
    "XGB": "XGBoost",
}

# Models that support native class balancing
SUPPORTS_BALANCED = {"DT", "RF", "SVM", "LR", "LGBM", "XGB"}

# Models that need feature scaling
NEEDS_SCALING = {"KNN", "SVM", "LR"}


# ─────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────
def load_dev_data(horizon: str):
    """Load the v2 dataset and keep DEV partition only."""
    ds_path = os.path.join(PROCESSED, f"dataset_{horizon}_v2.csv")
    hs_path = os.path.join(PROCESSED, f"holdout_split_{horizon}.csv")

    df = pd.read_csv(ds_path)
    hs = pd.read_csv(hs_path)

    dev_ids = set(hs.loc[hs["partition"] == "dev", "subject_id"])
    df = df[df["subject_id"].isin(dev_ids)].copy()
    print(f"  DEV set: {len(df)} subjects")

    slope_col = "slope_180d_per_30d" if horizon == "6m" else "slope_90d_per_30d"
    df[slope_col] = pd.to_numeric(df[slope_col], errors="coerce")
    df = df.dropna(subset=[slope_col])

    feat_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feat_cols].copy()
    slope = df[slope_col].copy().reset_index(drop=True)
    groups = df["subject_id"].copy().reset_index(drop=True)
    X = X.reset_index(drop=True)

    return X, slope, groups


# ─────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────
def build_preprocessor(X: pd.DataFrame, scale: bool) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    if scale:
        num_pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
        ])
    else:
        num_pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
        ])

    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )


# ─────────────────────────────────────────────────────────────
# Search spaces per model
# ─────────────────────────────────────────────────────────────
def suggest_knn(trial: optuna.Trial) -> dict:
    return {
        "n_neighbors": trial.suggest_int("n_neighbors", 3, 51, step=2),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "metric": trial.suggest_categorical("metric", ["euclidean", "manhattan"]),
    }


def suggest_dt(trial: optuna.Trial) -> dict:
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 30),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        "random_state": SEED,
    }


def suggest_rf(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "random_state": SEED,
        "n_jobs": -1,
    }


def suggest_svm(trial: optuna.Trial) -> dict:
    C = trial.suggest_float("C", 1e-2, 100.0, log=True)
    kernel = trial.suggest_categorical("kernel", ["rbf", "linear"])
    params = {
        "C": C,
        "kernel": kernel,
        "probability": True,
        "random_state": SEED,
    }
    if kernel == "rbf":
        params["gamma"] = trial.suggest_float("gamma", 1e-4, 1.0, log=True)
    return params


def suggest_lr(trial: optuna.Trial) -> dict:
    return {
        "C": trial.suggest_float("C", 1e-3, 50.0, log=True),
        "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
        "solver": "liblinear",
        "max_iter": 3000,
        "random_state": SEED,
    }


def suggest_lgbm(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", -1, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        # Historical runs left subsample_freq=0, so this sampled value was inactive.
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 1.0, log=True),
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": -1,
    }


def suggest_xgb(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 1.0, log=True),
        "random_state": SEED,
        "n_jobs": -1,
        "verbosity": 0,
        "tree_method": "hist",
    }


SUGGEST_FN = {
    "KNN": suggest_knn,
    "DT":  suggest_dt,
    "RF":  suggest_rf,
    "SVM": suggest_svm,
    "LR":  suggest_lr,
    "LGBM": suggest_lgbm,
    "XGB": suggest_xgb,
}

CLF_CLASS = {
    "KNN": KNeighborsClassifier,
    "DT":  DecisionTreeClassifier,
    "RF":  RandomForestClassifier,
    "SVM": SVC,
    "LR":  LogisticRegression,
    "LGBM": LGBMClassifier,
    "XGB": XGBClassifier,
}


# ─────────────────────────────────────────────────────────────
# Build classifier with optional class balancing
# ─────────────────────────────────────────────────────────────
def build_clf(model_key: str, params: dict, balanced: bool, n_pos: int = 0, n_neg: int = 0):
    """Instantiate a classifier, optionally with class balancing."""
    p = dict(params)  # shallow copy

    if balanced:
        if model_key in ("DT", "RF", "SVM", "LR"):
            p["class_weight"] = "balanced"
        elif model_key == "LGBM":
            p["is_unbalance"] = True
        elif model_key == "XGB":
            if n_pos > 0:
                p["scale_pos_weight"] = n_neg / n_pos

    return CLF_CLASS[model_key](**p)


# ─────────────────────────────────────────────────────────────
# Cross-validation evaluation (returns per-fold metrics)
# ─────────────────────────────────────────────────────────────
def cv_evaluate(
    model_key: str,
    params: dict,
    balanced: bool,
    X: pd.DataFrame,
    slope: pd.Series,
    groups: pd.Series,
    preproc_template: ColumnTransformer,
    return_all_metrics: bool = False,
):
    """
    Run 5-fold GroupKFold with fold-wise label definition.

    If return_all_metrics is False, returns mean PR-AUC only (for Optuna).
    If True, returns dict of {metric: [fold_scores]}.
    """
    gkf = GroupKFold(n_splits=N_SPLITS)
    fold_metrics = {
        "pr_auc": [],
        "roc_auc": [],
        "f1": [],
        "f2": [],
    }

    for tr_idx, va_idx in gkf.split(X, groups=groups):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        slope_tr, slope_va = slope.iloc[tr_idx], slope.iloc[va_idx]

        # Fold-wise label definition
        thr = float(np.nanquantile(slope_tr.values, RAPID_FRAC))
        ytr = (slope_tr <= thr).astype(int)
        yva = (slope_va <= thr).astype(int)

        # Build classifier (per-fold for XGB balanced to get correct scale_pos_weight)
        n_pos = int(ytr.sum())
        n_neg = int(len(ytr) - n_pos)
        clf = build_clf(model_key, params, balanced, n_pos, n_neg)

        # Build and fit pipeline
        from sklearn.base import clone
        preproc = clone(preproc_template)
        pipe = Pipeline([("prep", preproc), ("clf", clf)])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(Xtr, ytr)

        proba = pipe.predict_proba(Xva)[:, 1]
        y_pred = (proba >= 0.5).astype(int)

        fold_metrics["pr_auc"].append(average_precision_score(yva, proba))
        fold_metrics["roc_auc"].append(roc_auc_score(yva, proba))
        fold_metrics["f1"].append(f1_score(yva, y_pred, zero_division=0))
        fold_metrics["f2"].append(fbeta_score(yva, y_pred, beta=2, zero_division=0))

    if return_all_metrics:
        return fold_metrics
    else:
        return float(np.mean(fold_metrics["pr_auc"]))


# ─────────────────────────────────────────────────────────────
# Optuna objective
# ─────────────────────────────────────────────────────────────
def make_objective(model_key, balanced, X, slope, groups, preproc_template):
    suggest_fn = SUGGEST_FN[model_key]

    def objective(trial: optuna.Trial) -> float:
        params = suggest_fn(trial)
        try:
            score = cv_evaluate(
                model_key, params, balanced,
                X, slope, groups, preproc_template,
                return_all_metrics=False,
            )
        except Exception:
            return 0.0  # failed trial
        return score

    return objective


# ─────────────────────────────────────────────────────────────
# Run one model × mode combination
# ─────────────────────────────────────────────────────────────
def run_one(
    model_key: str,
    balanced: bool,
    X: pd.DataFrame,
    slope: pd.Series,
    groups: pd.Series,
    n_trials: int,
    horizon: str,
) -> dict:
    mode_label = "balanced" if balanced else "unbalanced"
    tag = f"{model_key}_{mode_label}_{horizon}"
    full_name = MODEL_NAMES[model_key]

    print(f"\n{'='*60}")
    print(f"  {full_name} | {mode_label} | {horizon} | {n_trials} trials")
    print(f"{'='*60}")

    scale = model_key in NEEDS_SCALING
    preproc = build_preprocessor(X, scale=scale)

    # --- Optuna ---
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    objective = make_objective(model_key, balanced, X, slope, groups, preproc)

    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    elapsed = time.time() - t0

    best_params = study.best_params
    best_pr_auc = float(study.best_value)

    print(f"  Best PR-AUC (Optuna): {best_pr_auc:.4f}  ({elapsed:.0f}s)")

    # --- Final evaluation with best params ---
    suggest_fn = SUGGEST_FN[model_key]

    # Reconstruct full param dict from best_params
    # (Some params like random_state are fixed and not in best_params)
    class ParamReplay:
        """Replay best_params as if Optuna suggested them."""
        def __init__(self, best):
            self._best = best
        def suggest_int(self, name, *a, **kw):
            return self._best[name]
        def suggest_float(self, name, *a, **kw):
            return self._best[name]
        def suggest_categorical(self, name, *a, **kw):
            return self._best[name]

    full_params = suggest_fn(ParamReplay(best_params))

    preproc_final = build_preprocessor(X, scale=scale)
    fold_metrics = cv_evaluate(
        model_key, full_params, balanced,
        X, slope, groups, preproc_final,
        return_all_metrics=True,
    )

    result = {
        "model": full_name,
        "model_key": model_key,
        "mode": mode_label,
        "horizon": horizon,
    }
    for metric_name, scores in fold_metrics.items():
        result[f"{metric_name}_mean"] = round(float(np.mean(scores)), 4)
        result[f"{metric_name}_std"]  = round(float(np.std(scores)), 4)
        result[f"{metric_name}_folds"] = [round(s, 4) for s in scores]

    result["optuna_best_pr_auc"] = round(best_pr_auc, 4)
    result["elapsed_s"] = round(elapsed, 1)

    # --- Save trials + best params ---
    os.makedirs(OUT_TUNING, exist_ok=True)

    trials_path = os.path.join(OUT_TUNING, f"{tag}_trials.csv")
    study.trials_dataframe().to_csv(trials_path, index=False)

    payload = {
        "model": full_name,
        "mode": mode_label,
        "horizon": horizon,
        "metric_optimized": "PR_AUC",
        "best_value": best_pr_auc,
        "best_params": best_params,
        "full_params": full_params,
        "n_trials": len(study.trials),
        "n_splits": N_SPLITS,
        "rapid_frac": RAPID_FRAC,
        "final_metrics": {
            k: {"mean": result[f"{k}_mean"], "std": result[f"{k}_std"]}
            for k in fold_metrics
        },
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    best_path = os.path.join(OUT_TUNING, f"{tag}_best.json")
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"  Final CV:  PR-AUC={result['pr_auc_mean']:.4f}±{result['pr_auc_std']:.4f}"
          f"  ROC-AUC={result['roc_auc_mean']:.4f}  F1={result['f1_mean']:.4f}"
          f"  F2={result['f2_mean']:.4f}")

    return result


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Unified Optuna tuning")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Model keys to tune (e.g. LR XGB). Default: all.")
    parser.add_argument("--horizon", default="6m", choices=["3m", "6m"])
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--unbalanced-only", action="store_true")
    parser.add_argument("--balanced-only", action="store_true")
    args = parser.parse_args()

    model_keys = args.models or list(MODEL_NAMES.keys())
    horizon = args.horizon

    # Validate model keys
    for k in model_keys:
        if k not in MODEL_NAMES:
            raise ValueError(f"Unknown model key '{k}'. Choose from: {list(MODEL_NAMES.keys())}")

    print(f"Loading DEV data for {horizon} horizon...")
    X, slope, groups = load_dev_data(horizon)
    print(f"  Features: {X.shape[1]}, Samples: {len(X)}")

    all_results = []

    for mk in model_keys:
        modes = []
        if not args.balanced_only:
            modes.append(False)
        if not args.unbalanced_only:
            if mk in SUPPORTS_BALANCED:
                modes.append(True)
            else:
                print(f"\n  [SKIP] {MODEL_NAMES[mk]} balanced — not supported natively")

        for balanced in modes:
            result = run_one(mk, balanced, X, slope, groups, args.trials, horizon)
            all_results.append(result)

    # --- Summary table ---
    if all_results:
        summary_cols = [
            "model", "mode", "horizon",
            "pr_auc_mean", "pr_auc_std",
            "roc_auc_mean", "roc_auc_std",
            "f1_mean", "f1_std",
            "f2_mean", "f2_std",
            "elapsed_s",
        ]
        summary = pd.DataFrame(all_results)[summary_cols]
        summary = summary.sort_values("pr_auc_mean", ascending=False)

        out_summary = os.path.join(OUT_TUNING, f"step5_tuning_summary_{horizon}.csv")
        summary.to_csv(out_summary, index=False)

        print(f"\n{'='*70}")
        print(f"  SUMMARY — {horizon} horizon")
        print(f"{'='*70}")
        print(summary.to_string(index=False))
        print(f"\n  Saved: {out_summary}")


if __name__ == "__main__":
    main()
