"""Fixed-hyperparameter Study I partition-stability analysis; never read TEST."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[2]


def load_tuning():
    spec = importlib.util.spec_from_file_location(
        "fixed_study1_tuning", ROOT / "03_src/tuning/optuna_tune_all.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(horizon, repeats, jobs, out):
    tuning = load_tuning()
    dataset_path = ROOT / f"01_data/processed/dataset_{horizon}_v2.csv"
    split_path = ROOT / f"01_data/processed/holdout_split_{horizon}.csv"
    dataset = pd.read_csv(dataset_path)
    split = pd.read_csv(split_path)
    dev = dataset.loc[dataset.subject_id.isin(
        split.loc[split.partition.eq("dev"), "subject_id"])].copy()
    assert dev.subject_id.is_unique
    assert set(dev.subject_id).isdisjoint(
        split.loc[split.partition.eq("test"), "subject_id"])
    slope_name = "slope_180d_per_30d" if horizon == "6m" else "slope_90d_per_30d"
    slopes = dev[slope_name].reset_index(drop=True)
    x = dev[[c for c in dev if c not in tuning.DROP_COLS]].reset_index(drop=True)
    provisional = (slopes <= slopes.quantile(.30)).astype(int)
    folds = list(RepeatedStratifiedKFold(
        n_splits=5, n_repeats=repeats, random_state=42).split(x, provisional))
    private = ROOT / "01_data/interim/reviewer_revision"
    private.mkdir(parents=True, exist_ok=True)
    membership = []
    for fold, (_, va) in enumerate(folds):
        membership.extend({"subject_id": dev.subject_id.iloc[i],
                           "repeat": fold // 5, "fold": fold % 5} for i in va)
    pd.DataFrame(membership).to_csv(private / f"s1_repeated_folds_{horizon}.csv", index=False)
    records, warning_records = [], []
    inputs = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [dataset_path, split_path]}
    configurations = sorted((ROOT / "04_outputs/tables/step5_tuning").glob(
        f"*_{horizon}_best.json"))
    assert len(configurations) == (13 if horizon == "6m" else 8)
    for path in configurations:
        inputs[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        payload = json.loads(path.read_text(encoding="utf-8"))
        key, mode = path.name.split("_")[:2]
        params = dict(payload["full_params"])
        if "n_jobs" in params:
            params["n_jobs"] = jobs
        for fold, (tr, va) in enumerate(folds):
            started = time.perf_counter()
            cutoff = float(slopes.iloc[tr].quantile(.30))
            ytr = (slopes.iloc[tr] <= cutoff).astype(int)
            yva = (slopes.iloc[va] <= cutoff).astype(int)
            model = tuning.build_clf(key, params, mode == "balanced",
                                     int(ytr.sum()), int(len(ytr) - ytr.sum()))
            pipe = Pipeline([
                ("prep", tuning.build_preprocessor(x, key in tuning.NEEDS_SCALING)),
                ("clf", model),
            ])
            context = {"horizon": horizon, "model": key, "mode": mode,
                       "repeat": fold // 5, "fold": fold % 5}
            try:
                with warnings.catch_warnings(record=True) as caught, threadpool_limits(jobs):
                    warnings.simplefilter("always")
                    pipe.fit(x.iloc[tr], ytr)
                    probability = pipe.predict_proba(x.iloc[va])[:, 1]
                if not np.isfinite(probability).all():
                    raise ValueError("Non-finite validation probabilities")
            except Exception:
                with (out / "s1_repeated_failures.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({**context, "traceback": traceback.format_exc()}) + "\n")
                raise
            warning_records.extend({**context, "category": w.category.__name__,
                                    "message": str(w.message)} for w in caught)
            records.append({**context, "n_train": len(tr), "n_validation": len(va),
                            "train_positive": int(ytr.sum()), "validation_positive": int(yva.sum()),
                            "target_cutoff": cutoff,
                            "pr_auc": average_precision_score(yva, probability),
                            "roc_auc": roc_auc_score(yva, probability),
                            "seconds": time.perf_counter() - started})
            pd.DataFrame(records).to_csv(out / f"s1_repeated_folds_{horizon}.csv", index=False)
            print(f"{horizon} {key} {mode}: {fold+1}/{len(folds)} "
                  f"AP={records[-1]['pr_auc']:.4f}", flush=True)
    df = pd.DataFrame(records)
    summaries = []
    for (model, mode), group in df.groupby(["model", "mode"], sort=False):
        by_repeat = group.groupby("repeat").pr_auc.mean()
        summaries.append({"model": model, "mode": mode, "horizon": horizon,
                          "pr_auc_mean": group.pr_auc.mean(),
                          "pr_auc_population_sd": group.pr_auc.std(ddof=0),
                          "repeat_mean_min": by_repeat.min(), "repeat_mean_max": by_repeat.max(),
                          "roc_auc_mean": group.roc_auc.mean(),
                          "original_pr_auc": json.loads((ROOT / "04_outputs/tables/step5_tuning" /
                              f"{model}_{mode}_{horizon}_best.json").read_text())["best_value"]})
    summary = pd.DataFrame(summaries).sort_values("pr_auc_mean", ascending=False)
    summary.to_csv(out / f"s1_repeated_summary_{horizon}.csv", index=False)
    (out / f"s1_repeated_warnings_{horizon}.json").write_text(
        json.dumps(warning_records, indent=2), encoding="utf-8")
    (out / f"s1_repeated_metadata_{horizon}.json").write_text(json.dumps({
        "input_sha256": inputs, "n_fits": len(records), "n_repeats": repeats,
        "n_splits": 5, "seed": 42, "jobs": jobs,
        "interpretation": "Fixed selected configurations; supplementary partition stability, not nested validation.",
        "test_accessed": False, "hyperparameters_retuned": False,
    }, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", choices=["3m", "6m", "both"], default="both")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--output", type=Path, default=ROOT / "04_outputs/reviewer_followup/revised_analyses")
    args = parser.parse_args()
    if args.repeats < 1 or args.jobs < 1:
        parser.error("--repeats and --jobs must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(args.jobs))
    for horizon in (["6m", "3m"] if args.horizon == "both" else [args.horizon]):
        run(horizon, args.repeats, args.jobs, args.output)


if __name__ == "__main__":
    main()
