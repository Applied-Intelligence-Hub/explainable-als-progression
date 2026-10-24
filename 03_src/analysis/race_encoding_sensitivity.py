"""DEV-only Study I race-encoding sensitivity with fixed configurations.

Keep the primary pipeline unchanged. Compare numeric median imputation with
explicit recorded-zero/recorded-one/unrecorded categories and race omission.
Neither representation selection nor threshold optimisation uses TEST.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from repeated_stratified_validation import load_tuning


ROOT = Path(__file__).resolve().parents[2]
STRATEGIES = ("primary_median", "explicit_recording", "race_omitted")
CONFIGURATIONS = (("XGB", "unbalanced"), ("LR", "balanced"))
RACE_COLUMNS = (
    "Race_Caucasian", "Race_Black_African_American", "Race_Asian",
    "Race_Americ_Indian_Alaska_Native", "Race_Hawaiian_Pacific_Islander",
    "Race_Other", "Race_Unknown",
)


def represent_race(x, strategy):
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown race strategy: {strategy}")
    if not set(RACE_COLUMNS).issubset(x.columns):
        raise ValueError("The seven original race columns are required")
    observed = x[list(RACE_COLUMNS)]
    if not (observed.isna() | observed.isin([0, 1])).all().all():
        raise ValueError("Unexpected race value: expected 0, 1 or missing")
    result = x.copy()
    if strategy == "race_omitted":
        return result.drop(columns=list(RACE_COLUMNS))
    if strategy == "explicit_recording":
        for column in RACE_COLUMNS:
            result[column] = x[column].map({0: "recorded_zero", 1: "recorded_one"}).fillna(
                "unrecorded").astype(object)
    return result


def input_audit(dev, raw):
    if not raw.subject_id.is_unique:
        raise ValueError("Demographics must contain one source row per participant")
    source = raw.set_index("subject_id").loc[dev.subject_id, list(RACE_COLUMNS)].reset_index(drop=True)
    original = dev[list(RACE_COLUMNS)].reset_index(drop=True)
    if not ((source.eq(original)) | (source.isna() & original.isna())).all().all():
        raise ValueError("Race fields do not reproduce the raw demographics extract")
    affirmative = original.eq(1).sum(axis=1)
    return {
        "n_development": len(dev), "raw_fields_match": True,
        "all_seven_unrecorded": int(original.isna().all(axis=1).sum()),
        "no_affirmative_indicator": int(affirmative.eq(0).sum()),
        "multiple_affirmative_indicators": int(affirmative.gt(1).sum()),
        "explicit_unknown_indicator": int(original.Race_Unknown.eq(1).sum()),
        "fields": {column: {
            "recorded_zero": int(original[column].eq(0).sum()),
            "recorded_one": int(original[column].eq(1).sum()),
            "unrecorded": int(original[column].isna().sum()),
            "primary_dev_median": float(original[column].median()),
        } for column in RACE_COLUMNS},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--output", type=Path, default=ROOT / "04_outputs/reviewer_followup/revised_analyses")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(args.jobs))
    tuning = load_tuning()
    dataset_path = ROOT / "01_data/processed/dataset_6m_v2.csv"
    split_path = ROOT / "01_data/processed/holdout_split_6m.csv"
    raw_path = ROOT / "01_data/raw/PROACT_DEMOGRAPHICS.csv"
    data, split = pd.read_csv(dataset_path), pd.read_csv(split_path)
    dev = data.loc[data.subject_id.isin(split.loc[split.partition.eq("dev"), "subject_id"])].reset_index(drop=True)
    assert len(dev) == 1113 and dev.subject_id.is_unique
    assert set(dev.subject_id).isdisjoint(split.loc[split.partition.eq("test"), "subject_id"])
    x = dev[[column for column in dev if column not in tuning.DROP_COLS]]
    slopes = dev.slope_180d_per_30d
    provisional = slopes.le(slopes.quantile(.30)).astype(int)
    folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42).split(x, provisional))
    audit = input_audit(dev, pd.read_csv(raw_path))
    protected = [dataset_path, split_path, raw_path,
                 ROOT / "04_outputs/tables/step8_xai/test_predictions.csv",
                 ROOT / "04_outputs/tables/step8_xai/lr_coefficients.csv"]
    for model, mode in CONFIGURATIONS:
        protected.append(ROOT / f"04_outputs/tables/step5_tuning/{model}_{mode}_6m_best.json")
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    rows, coefficients, warning_rows = [], [], []
    for model, mode in CONFIGURATIONS:
        params = json.loads((ROOT / f"04_outputs/tables/step5_tuning/{model}_{mode}_6m_best.json").read_text())["full_params"]
        params = dict(params)
        if "n_jobs" in params:
            params["n_jobs"] = args.jobs
        for strategy in STRATEGIES:
            encoded = represent_race(x, strategy)
            for number, (train, valid) in enumerate(folds):
                started = time.perf_counter()
                cutoff = float(slopes.iloc[train].quantile(.30))
                y_train = slopes.iloc[train].le(cutoff).astype(int)
                y_valid = slopes.iloc[valid].le(cutoff).astype(int)
                estimator = tuning.build_clf(model, params, mode == "balanced",
                                             int(y_train.sum()), int(len(y_train) - y_train.sum()))
                pipe = Pipeline([
                    ("prep", tuning.build_preprocessor(encoded, model in tuning.NEEDS_SCALING)),
                    ("clf", estimator),
                ])
                context = {"model": model, "mode": mode, "strategy": strategy,
                           "repeat": number // 5, "fold": number % 5}
                try:
                    with warnings.catch_warnings(record=True) as caught, threadpool_limits(args.jobs):
                        warnings.simplefilter("always")
                        pipe.fit(encoded.iloc[train], y_train)
                        probability = pipe.predict_proba(encoded.iloc[valid])[:, 1]
                    if not np.isfinite(probability).all():
                        raise ValueError("Non-finite validation probabilities")
                except Exception:
                    with (args.output / "s1_race_failures.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps({**context, "traceback": traceback.format_exc()}) + "\n")
                    raise
                warning_rows.extend({**context, "category": item.category.__name__,
                                     "message": str(item.message)} for item in caught)
                names = pipe.named_steps["prep"].get_feature_names_out()
                if model == "LR":
                    coefficients.extend({**context, "feature": name, "coefficient": float(weight)}
                                        for name, weight in zip(names, estimator.coef_[0]) if "Race_" in name)
                rows.append({**context, "n_train": len(train), "n_validation": len(valid),
                             "train_positive": int(y_train.sum()), "validation_positive": int(y_valid.sum()),
                             "target_cutoff": cutoff, "n_transformed": len(names),
                             "pr_auc": average_precision_score(y_valid, probability),
                             "roc_auc": roc_auc_score(y_valid, probability),
                             "brier": brier_score_loss(y_valid, probability),
                             "seconds": time.perf_counter() - started})
            print(f"{model} {mode}: {strategy}, 15 fits completed", flush=True)
    scores = pd.DataFrame(rows)
    original = pd.read_csv(args.output / "s1_repeated_folds_6m.csv")
    for model, mode in CONFIGURATIONS:
        reference = scores.loc[scores.model.eq(model) & scores.strategy.eq("primary_median")]
        previous = original.loc[original.model.eq(model) & original["mode"].eq(mode)]
        joined = reference.merge(previous, on=["repeat", "fold"], suffixes=("_new", "_old"), validate="one_to_one")
        assert len(joined) == 15
        for metric in ["target_cutoff", "pr_auc", "roc_auc"]:
            np.testing.assert_allclose(joined[f"{metric}_new"], joined[f"{metric}_old"], rtol=0, atol=1e-12)
    summaries = []
    for (model, mode, strategy), block in scores.groupby(["model", "mode", "strategy"], sort=False):
        reference = scores.loc[scores.model.eq(model) & scores.strategy.eq("primary_median")]
        paired = block.merge(reference, on=["repeat", "fold"], suffixes=("", "_reference"), validate="one_to_one")
        differences = paired.pr_auc - paired.pr_auc_reference
        summaries.append({"model": model, "mode": mode, "strategy": strategy,
                          "pr_auc_mean": block.pr_auc.mean(), "pr_auc_population_sd": block.pr_auc.std(ddof=0),
                          "pr_auc_delta_mean": differences.mean(), "pr_auc_delta_min": differences.min(),
                          "pr_auc_delta_max": differences.max(), "roc_auc_mean": block.roc_auc.mean(),
                          "brier_mean": block.brier.mean(), "n_fits": len(block)})
    assert all(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == value for path, value in hashes.items())
    metadata = {"horizon": "6m", "n_development": len(dev), "n_splits": 5, "n_repeats": 3,
                "seed": 42, "n_fits": len(scores), "configurations": CONFIGURATIONS,
                "strategies": STRATEGIES, "input_sha256": hashes,
                "original_repeated_scores_reproduced": True, "protected_inputs_unchanged": True,
                "test_evaluated": False, "hyperparameters_retuned": False, "primary_results_replaced": False,
                "interpretation": "Post-hoc fixed-parameter sensitivity on overlapping development folds; no new selection or significance tests."}
    scores.to_csv(args.output / "s1_race_folds.csv", index=False)
    pd.DataFrame(summaries).to_csv(args.output / "s1_race_summary.csv", index=False)
    pd.DataFrame(coefficients).to_csv(args.output / "s1_race_lr_coefficients.csv", index=False)
    for name, payload in [("audit", audit), ("metadata", metadata), ("warnings", warning_rows)]:
        (args.output / f"s1_race_{name}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
