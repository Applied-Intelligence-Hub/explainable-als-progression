"""
Step 4b — Build new feature blocks (Treatment + Labs) and re-generate datasets v2.
Run from project root: python 03_src/data/build_features_v2.py
"""
import os
import numpy as np
import pandas as pd

RAW        = os.path.join("01_data", "raw")
INTERIM    = os.path.join("01_data", "interim")
PROCESSED  = os.path.join("01_data", "processed")
OUT_TABLES = os.path.join("04_outputs", "tables", "step4")

os.makedirs(OUT_TABLES, exist_ok=True)

# ---------- Load sources ----------
base = pd.read_csv(os.path.join(INTERIM, "baseline_table_ALSFRS_R.csv"))
print("Baseline subjects:", base["subject_id"].nunique())

vitals_feat = pd.read_csv(os.path.join(PROCESSED, "features_vitals_t0.csv"))
fvc_feat    = pd.read_csv(os.path.join(PROCESSED, "features_fvc_t0.csv"))
targets     = pd.read_csv(os.path.join(INTERIM, "baseline_with_targets_step3_nolabelsv2.csv"))

riluzole  = pd.read_csv(os.path.join(RAW, "PROACT_RILUZOLE.csv"))
treatment = pd.read_csv(os.path.join(RAW, "PROACT_TREATMENT.csv"))
conmeds   = pd.read_csv(os.path.join(RAW, "PROACT_CONMEDS.csv"))
labs      = pd.read_csv(os.path.join(RAW, "PROACT_LABS.csv"))

t0_map = dict(zip(base["subject_id"], base["t0_delta_days"]))

# ==================== TREATMENT BLOCK ====================
print("\n--- Treatment Block ---")

# 1a) Riluzole (binary: used before or at t0)
ril = riluzole.copy()
ril["t0"] = ril["subject_id"].map(t0_map)
ril_pre = ril[
    (ril["Subject_used_Riluzole"] == "Yes") &
    (ril["Riluzole_use_Delta"] <= ril["t0"])
]
ril_subjects = set(ril_pre["subject_id"].unique())

# 1b) Study Arm
treat = treatment.drop_duplicates(subset="subject_id", keep="first")
treat_feat = treat[["subject_id", "Study_Arm"]].rename(
    columns={"Study_Arm": "study_arm"}
)

# 1c) Number of ConMeds pre-baseline
cm = conmeds.copy()
cm["t0"] = cm["subject_id"].map(t0_map)
cm["Start_Delta"] = pd.to_numeric(cm["Start_Delta"], errors="coerce")
cm_pre = cm.dropna(subset=["Start_Delta", "t0"])
cm_pre = cm_pre[cm_pre["Start_Delta"] <= cm_pre["t0"]]
n_conmeds = (
    cm_pre.groupby("subject_id")["Medication_Coded"]
    .nunique()
    .reset_index()
)
n_conmeds.columns = ["subject_id", "n_conmeds_pre_t0"]

# Assemble treatment feature table
treatment_feat = base[["subject_id"]].copy()
treatment_feat["riluzole_pre_t0"] = (
    treatment_feat["subject_id"].isin(ril_subjects).astype(int)
)
treatment_feat = treatment_feat.merge(treat_feat, on="subject_id", how="left")
treatment_feat = treatment_feat.merge(n_conmeds, on="subject_id", how="left")
treatment_feat["n_conmeds_pre_t0"] = (
    treatment_feat["n_conmeds_pre_t0"].fillna(0).astype(int)
)

out_treat = os.path.join(PROCESSED, "features_treatment_t0.csv")
treatment_feat.to_csv(out_treat, index=False)

ril_pct = treatment_feat["riluzole_pre_t0"].mean() * 100
cm_mean = treatment_feat["n_conmeds_pre_t0"].mean()
print(f"  Saved: {out_treat} | shape: {treatment_feat.shape}")
print(f"  riluzole=Yes: {ril_pct:.1f}%")
print(f"  study_arm:\n{treatment_feat['study_arm'].value_counts(dropna=False).to_string()}")
print(f"  n_conmeds_pre_t0: mean={cm_mean:.1f}")

# ==================== LABS BLOCK ====================
print("\n--- Labs Block ---")

LAB_TESTS = {
    "Creatinine": "creatinine_t0",
    "ALT(SGPT)":  "alt_t0",
}

labs_sub = labs[labs["Test_Name"].isin(LAB_TESTS.keys())].copy()
labs_sub["Laboratory_Delta"] = pd.to_numeric(labs_sub["Laboratory_Delta"], errors="coerce")
labs_sub["Test_Result"]      = pd.to_numeric(labs_sub["Test_Result"], errors="coerce")
labs_sub["t0"] = labs_sub["subject_id"].map(t0_map)

# Pre-baseline only
labs_sub = labs_sub.dropna(subset=["Laboratory_Delta", "t0", "Test_Result"])
labs_sub = labs_sub[labs_sub["Laboratory_Delta"] <= labs_sub["t0"]]

# Last pre-baseline per subject × test
labs_sub = labs_sub.sort_values(["subject_id", "Test_Name", "Laboratory_Delta"])
labs_last = labs_sub.groupby(["subject_id", "Test_Name"], as_index=False).tail(1)

# Pivot to wide
labs_wide = labs_last.pivot(
    index="subject_id", columns="Test_Name", values="Test_Result"
)
labs_wide = labs_wide.rename(columns=LAB_TESTS).reset_index()

# Merge onto baseline
labs_feat = base[["subject_id"]].merge(labs_wide, on="subject_id", how="left")

out_labs = os.path.join(PROCESSED, "features_labs_t0.csv")
labs_feat.to_csv(out_labs, index=False)

print(f"  Saved: {out_labs} | shape: {labs_feat.shape}")
for col, name in LAB_TESTS.items():
    n = labs_feat[name].notna().sum()
    total = len(labs_feat)
    mean_val = labs_feat[name].mean()
    med_val  = labs_feat[name].median()
    print(f"  {name}: {n}/{total} ({100*n/total:.1f}%) | mean={mean_val:.2f}, median={med_val:.2f}")

# ==================== FULL MERGE ====================
print("\n--- Full Merge ---")

full = (
    base
    .merge(vitals_feat,    on="subject_id", how="left")
    .merge(fvc_feat,       on="subject_id", how="left")
    .merge(treatment_feat, on="subject_id", how="left")
    .merge(labs_feat,      on="subject_id", how="left")
)

slope_cols = [c for c in ["subject_id", "slope_90d_per_30d", "slope_180d_per_30d"]
              if c in targets.columns]
full = full.merge(targets[slope_cols], on="subject_id", how="left")

print(f"  Full table: {full.shape}")
print(f"  Columns ({full.shape[1]}): {list(full.columns)}")

# Filter to horizon-specific datasets
df_6m = full.dropna(subset=["slope_180d_per_30d"]).copy()
df_3m = full.dropna(subset=["slope_90d_per_30d"]).copy()

out_6m = os.path.join(PROCESSED, "dataset_6m_v2.csv")
out_3m = os.path.join(PROCESSED, "dataset_3m_v2.csv")
df_6m.to_csv(out_6m, index=False)
df_3m.to_csv(out_3m, index=False)

print(f"  dataset_6m_v2: {df_6m.shape}")
print(f"  dataset_3m_v2: {df_3m.shape}")

v1_cols = set(pd.read_csv(os.path.join(PROCESSED, "dataset_6m_v1.csv"), nrows=0).columns)
new_cols = [c for c in df_6m.columns if c not in v1_cols]
print(f"  New columns in v2: {new_cols}")

# ==================== COVERAGE TABLE ====================
print("\n--- Coverage ---")

def coverage_report(df, label):
    N = len(df)
    rows = []
    n_vit = df[["Weight_kg_t0", "Pulse"]].notna().any(axis=1).sum()
    rows.append(("Vitals", n_vit, round(n_vit / N * 100, 1)))
    n_fvc = df["FVC_Liters_best_t0"].notna().sum()
    rows.append(("FVC", n_fvc, round(n_fvc / N * 100, 1)))
    n_arm = df["study_arm"].notna().sum()
    rows.append(("Treatment (study_arm)", n_arm, round(n_arm / N * 100, 1)))
    n_ril = int(df["riluzole_pre_t0"].sum())
    rows.append(("Treatment (riluzole=Yes)", n_ril, round(n_ril / N * 100, 1)))
    for col in ["creatinine_t0", "alt_t0"]:
        n = df[col].notna().sum()
        rows.append((f"Labs ({col})", n, round(n / N * 100, 1)))
    return pd.DataFrame(rows, columns=["block", f"n_{label}", f"pct_{label}"])

cov_6m = coverage_report(df_6m, "6m")
cov_3m = coverage_report(df_3m, "3m")
cov_all = cov_6m.merge(cov_3m, on="block", how="outer")
print(cov_all.to_string(index=False))

cov_path = os.path.join(OUT_TABLES, "step4_featureblock_coverage_v2.csv")
cov_all.to_csv(cov_path, index=False)
print(f"\n  Saved: {cov_path}")

# ==================== VERIFY HELD-OUT SPLITS ====================
print("\n--- Verify Splits ---")

split_6m = pd.read_csv(os.path.join(PROCESSED, "holdout_split_6m.csv"))
split_3m = pd.read_csv(os.path.join(PROCESSED, "holdout_split_3m.csv"))

assert set(df_6m["subject_id"]) == set(split_6m["subject_id"]), "6m mismatch!"
assert set(df_3m["subject_id"]) == set(split_3m["subject_id"]), "3m mismatch!"
print("  Held-out splits match v2 datasets perfectly.")

dev_6m = split_6m[split_6m["partition"] == "dev"]
test_6m = split_6m[split_6m["partition"] == "test"]
print(f"  6m: {len(dev_6m)} dev + {len(test_6m)} test = {len(split_6m)}")

dev_3m = split_3m[split_3m["partition"] == "dev"]
test_3m = split_3m[split_3m["partition"] == "test"]
print(f"  3m: {len(dev_3m)} dev + {len(test_3m)} test = {len(split_3m)}")

print("\nDone.")
