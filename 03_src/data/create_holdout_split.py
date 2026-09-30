"""
Step 0 — Create a held-out test set (80/20) for both 3m and 6m horizons.

Why:
  - The final model must be evaluated on data it has NEVER seen during
    training, hyperparameter tuning, or threshold selection.
  - All CV, Optuna, and model training happen on the 80% (dev set).
  - The 20% (held-out) is touched ONCE at the very end for final metrics.

How:
  - Stratified split by binary rapid/slow label (30th-percentile cutoff)
    using sklearn's StratifiedShuffleSplit.
  - Split is done at the SUBJECT level (each subject_id appears in exactly
    one partition — no leakage).
  - The same random_state is used for reproducibility.

Outputs (in 01_data/processed/):
  - holdout_split_6m.csv   — columns: subject_id, partition (dev / test)
  - holdout_split_3m.csv   — same for 3m
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE = 0.20
RAPID_FRAC = 0.30          # 30th-percentile defines "rapid"
DATA_DIR = Path(__file__).resolve().parents[2] / "01_data" / "processed"


def create_split(dataset_path: Path, slope_col: str, horizon_label: str):
    """Create and save a held-out split for one horizon."""
    df = pd.read_csv(dataset_path)
    assert df["subject_id"].is_unique, "Duplicate subject_ids found!"

    out_path = DATA_DIR / f"holdout_split_{horizon_label}.csv"
    if out_path.exists():
        existing = pd.read_csv(out_path)
        if (not existing["subject_id"].is_unique
                or set(existing["subject_id"]) != set(df["subject_id"])
                or not existing["partition"].isin(["dev", "test"]).all()):
            raise ValueError(f"Existing split does not match the cohort: {out_path}")
        print(f"Retaining existing patient assignments: {out_path.name}")
        return existing

    # --- binary label (same logic as fold-wise, but global for stratification) ---
    cutoff = np.percentile(df[slope_col].values, RAPID_FRAC * 100)
    y_binary = (df[slope_col] <= cutoff).astype(int)  # 1 = rapid

    print(f"\n=== {horizon_label} horizon ===")
    print(f"  Dataset: {dataset_path.name}  (N={len(df)})")
    print(f"  Slope column: {slope_col}")
    print(f"  Rapid cutoff (p{int(RAPID_FRAC*100)}): {cutoff:.4f}")
    print(f"  Rapid: {y_binary.sum()}  Slow: {(1-y_binary).sum()}")

    # --- stratified split ---
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                                random_state=RANDOM_STATE)
    dev_idx, test_idx = next(sss.split(df, y_binary))

    split_df = pd.DataFrame({
        "subject_id": df["subject_id"].values,
        "partition": "dev"
    })
    split_df.loc[test_idx, "partition"] = "test"

    # --- verify proportions ---
    dev_rapid = y_binary.iloc[dev_idx].mean()
    test_rapid = y_binary.iloc[test_idx].mean()
    print(f"  Dev  set: N={len(dev_idx)}  rapid%={dev_rapid:.3f}")
    print(f"  Test set: N={len(test_idx)}  rapid%={test_rapid:.3f}")

    # --- save ---
    split_df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path.name}")

    return split_df


if __name__ == "__main__":
    # 6-month horizon
    create_split(
        dataset_path=DATA_DIR / "dataset_6m_v1.csv",
        slope_col="slope_180d_per_30d",
        horizon_label="6m"
    )

    # 3-month horizon
    create_split(
        dataset_path=DATA_DIR / "dataset_3m_v1.csv",
        slope_col="slope_90d_per_30d",
        horizon_label="3m"
    )

    print("\nDone. Held-out splits created.")
