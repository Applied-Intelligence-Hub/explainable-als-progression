"""
Fix vitals data quality issue: implausible Height/Weight/BMI values.

Problem:
  - 147 raw records have Height < 100 labelled "Centimeters" (clearly inches
    or garbage data entry).  This produces BMI > 2000.
  - Heights < 120 cm or > 220 cm are not plausible for adults.
  - Weights < 30 kg or > 200 kg are not plausible for adults.

Fix:
  1. Set Height_cm_t0 to NaN if < 120 or > 220
  2. Set Weight_kg_t0 to NaN if < 30 or > 200
  3. Recompute BMI from valid Height/Weight
  4. Overwrite features_vitals_t0.csv
  5. Regenerate v2 datasets

Run from project root:
  python 03_src/data/fix_vitals_and_rebuild.py
"""

import os
import numpy as np
import pandas as pd

PROCESSED = os.path.join("01_data", "processed")

# --- Step 1: fix vitals features ------------------------------------------
print("="*60)
print("  Fixing vitals features")
print("="*60)

vf = pd.read_csv(os.path.join(PROCESSED, "features_vitals_t0.csv"))
print(f"Vitals features loaded: {vf.shape}")

# Plausible adult bounds
BAD_HEIGHT = (vf["Height_cm_t0"] < 120) | (vf["Height_cm_t0"] > 220)
BAD_WEIGHT = (vf["Weight_kg_t0"] < 30) | (vf["Weight_kg_t0"] > 200)

n_bad_h = BAD_HEIGHT.sum()
n_bad_w = BAD_WEIGHT.sum()
print(f"  Height implausible (<120 or >220 cm): {n_bad_h}")
print(f"  Weight implausible (<30 or >200 kg):  {n_bad_w}")

vf.loc[BAD_HEIGHT, "Height_cm_t0"] = np.nan
vf.loc[BAD_WEIGHT, "Weight_kg_t0"] = np.nan

# Recompute BMI
vf["BMI_t0"] = vf["Weight_kg_t0"] / ((vf["Height_cm_t0"] / 100.0) ** 2)

print(f"  BMI after fix: {vf['BMI_t0'].describe().to_dict()}")

out_vf = os.path.join(PROCESSED, "features_vitals_t0.csv")
vf.to_csv(out_vf, index=False)
print(f"  Saved fixed: {out_vf}")


# --- Step 2: rebuild v2 datasets ------------------------------------------
print("\n" + "="*60)
print("  Rebuilding v2 datasets")
print("="*60)

INTERIM = os.path.join("01_data", "interim")

base      = pd.read_csv(os.path.join(INTERIM, "baseline_table_ALSFRS_R.csv"))
feat_vit  = pd.read_csv(os.path.join(PROCESSED, "features_vitals_t0.csv"))
feat_fvc  = pd.read_csv(os.path.join(PROCESSED, "features_fvc_t0.csv"))
feat_treat = pd.read_csv(os.path.join(PROCESSED, "features_treatment_t0.csv"))
feat_labs  = pd.read_csv(os.path.join(PROCESSED, "features_labs_t0.csv"))
targets   = pd.read_csv(os.path.join(INTERIM, "baseline_with_targets_step3_nolabelsv2.csv"))

full = (
    base
    .merge(feat_vit,   on="subject_id", how="left")
    .merge(feat_fvc,   on="subject_id", how="left")
    .merge(feat_treat, on="subject_id", how="left")
    .merge(feat_labs,  on="subject_id", how="left")
)

slope_cols = [c for c in ["subject_id", "slope_90d_per_30d", "slope_180d_per_30d"]
              if c in targets.columns]
full = full.merge(targets[slope_cols], on="subject_id", how="left")

for horizon, ds_label in [("180", "6m"), ("90", "3m")]:
    slope_col = f"slope_{horizon}d_per_30d"
    df = full.dropna(subset=[slope_col]).copy()

    out = os.path.join(PROCESSED, f"dataset_{ds_label}_v2.csv")
    df.to_csv(out, index=False)
    print(f"  {ds_label}: {df.shape} -> {out}")
    bmi_valid = df["BMI_t0"].dropna()
    print(f"    BMI range: [{bmi_valid.min():.1f}, {bmi_valid.max():.1f}]")
    print(f"    BMI NaN: {df['BMI_t0'].isna().sum()}")

print("\nDone!")
