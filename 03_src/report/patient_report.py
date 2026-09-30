"""
Patient Risk Report Generator — ALS Rapid-Progression Screening.

Loads the trained XGBoost pipeline and generates a textual report for a
single patient, including:
  1. Probability of rapid progression (6-month horizon)
  2. Binary classification at the selected DEV threshold
  3. Top-5 contributing features (SHAP local explanation)
  4. Contextual review flags based on DEV percentile comparisons

Usage
-----
  # From workspace root:
  python 03_src/report/patient_report.py                       # demo patient
  python 03_src/report/patient_report.py --patient-id 12345    # from dataset
  python 03_src/report/patient_report.py --json patient.json   # from JSON file
  python 03_src/report/patient_report.py --save                # also save to 04_outputs/reports/

Outputs
-------
  Printed textual report to stdout.
  Optionally saved to 04_outputs/reports/<patient_id>.txt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import shap

warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────────────────────
# Paths (relative to workspace root)
# ─────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join("models", "final_xgb_6m.joblib")
META_PATH  = os.path.join("models", "final_xgb_6m_metadata.json")
DATA_PATH  = os.path.join("01_data", "processed", "dataset_6m_v2.csv")
OUT_DIR    = os.path.join("04_outputs", "reports")

# Selected operating threshold from the DEV OOF analysis.
DECISION_THRESHOLD = 0.21

N_TOP = 5       # number of SHAP features to show

# ─────────────────────────────────────────────────────────────
# Human-readable feature labels
# ─────────────────────────────────────────────────────────────
FEATURE_LABELS = {
    "ALSFRS_R_t0":                     "ALSFRS-R total (baseline)",
    "Age":                             "Age (years)",
    "Sex":                             "Sex",
    "Ethnicity":                       "Ethnicity",
    "Race_Caucasian":                  "Race: Caucasian",
    "Race_Black_African_American":     "Race: Black/African-American",
    "Race_Asian":                      "Race: Asian",
    "Race_Americ_Indian_Alaska_Native":"Race: Am. Indian/Alaska Native",
    "Race_Hawaiian_Pacific_Islander":  "Race: Hawaiian/Pacific Isl.",
    "Race_Other":                      "Race: Other",
    "Race_Unknown":                    "Race: Unknown",
    "R_1_Dyspnea":                     "ALSFRS-R item: Dyspnea",
    "R_2_Orthopnea":                   "ALSFRS-R item: Orthopnea",
    "R_3_Respiratory_Insufficiency":   "ALSFRS-R item: Resp. Insufficiency",
    "Mode_of_Administration":          "Mode of Administration",
    "Weight_kg_t0":                    "Weight (kg)",
    "Height_cm_t0":                    "Height (cm)",
    "BMI_t0":                          "BMI",
    "Pulse":                           "Pulse (bpm)",
    "Respiratory_Rate":                "Respiratory Rate",
    "Temperature":                     "Temperature (°C)",
    "Blood_Pressure_Systolic":         "BP Systolic (mmHg)",
    "Blood_Pressure_Diastolic":        "BP Diastolic (mmHg)",
    "Baseline_Standing_BP_Systolic":   "Standing BP Systolic",
    "Baseline_Standing_BP_Diastolic":  "Standing BP Diastolic",
    "Baseline_Supine_BP_Systolic":     "Supine BP Systolic",
    "Baseline_Supine_BP_Diastolic":    "Supine BP Diastolic",
    "FVC_Liters_best_t0":             "FVC (litres)",
    "FVC_pctNormal_best_t0":          "FVC (% predicted normal)",
    "subject_normal":                  "FVC subject normal flag",
    "riluzole_pre_t0":                 "Riluzole (pre-baseline)",
    "study_arm":                       "Study arm",
    "n_conmeds_pre_t0":                "Concomitant medications (n)",
    "alt_t0":                          "ALT (U/L, baseline)",
    "creatinine_t0":                   "Creatinine (µmol/L, baseline)",
}

# Descriptive review rules keyed by feature name. These cutoffs summarise
# the DEV distribution; they are not clinical reference ranges.
REVIEW_RULES = {
    "ALSFRS_R_t0": {
        "low_p25": 35.0,
        "message": "Value is at or below the DEV 25th percentile.",
    },
    "FVC_Liters_best_t0": {
        "low_p25": 2.78,
        "message": "Value is at or below the DEV 25th percentile (2.78 L).",
    },
    "FVC_pctNormal_best_t0": {
        "low_p25": 77.0,
        "message": "Value is at or below the DEV 25th percentile (77%).",
    },
    "Respiratory_Rate": {
        "high_p75": 20.0,
        "message": "Value is at or above the DEV 75th percentile (20).",
    },
    "BMI_t0": {
        "low_p25": 23.3,
        "message": "Value is at or below the DEV 25th percentile (23.3).",
    },
    "Age": {
        "high_p75": 65.0,
        "message": "Value is at or above the DEV 75th percentile (65 years).",
    },
    "R_1_Dyspnea": {
        "low_p25": 3.0,
        "message": "Value is at or below the DEV 25th percentile.",
    },
    "R_2_Orthopnea": {
        "low_p25": 4.0,
        "message": "Value is at or below the DEV 25th percentile.",
    },
    "R_3_Respiratory_Insufficiency": {
        "low_p25": 4.0,
        "message": "Value is at or below the DEV 25th percentile.",
    },
    "Weight_kg_t0": {
        "low_p25": 68.0,
        "message": "Value is at or below the DEV 25th percentile (68 kg).",
    },
    "Pulse": {
        "high_p75": 83.0,
        "message": "Value is at or above the DEV 75th percentile.",
    },
    "creatinine_t0": {
        "low_p25": 53.0,
        "message": "Value is at or below the DEV 25th percentile.",
    },
    "alt_t0": {
        "high_p75": 45.0,
        "message": "Value is at or above the DEV 75th percentile (45 U/L).",
    },
}


def readable(name: str) -> str:
    return FEATURE_LABELS.get(name, name)


# ─────────────────────────────────────────────────────────────
# Load model & metadata
# ─────────────────────────────────────────────────────────────
def load_model():
    pipe = joblib.load(MODEL_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    feat_cols = meta["feature_cols"]
    slope_cutoff = meta["slope_cutoff_30pct"]
    return pipe, meta, feat_cols, slope_cutoff


# ─────────────────────────────────────────────────────────────
# Build patient vector
# ─────────────────────────────────────────────────────────────
def patient_from_dataset(patient_id: int, feat_cols: list[str]):
    """Load a patient from the processed dataset by subject_id."""
    df = pd.read_csv(DATA_PATH)
    row = df[df["subject_id"] == patient_id]
    if row.empty:
        sys.exit(f"ERROR: patient {patient_id} not found in dataset.")
    row = row.iloc[0]
    X = row[feat_cols].to_frame().T.reset_index(drop=True)
    slope_val = row.get("slope_180d_per_30d", None)
    return X, slope_val


def patient_from_json(json_path: str, feat_cols: list[str]):
    """Load a patient from a JSON file with feature key–value pairs."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    missing = [c for c in feat_cols if c not in data]
    if missing:
        print(f"WARNING: {len(missing)} features missing from JSON — "
              f"will be NaN (imputed by pipeline): {missing[:5]}...")
    X = pd.DataFrame([{c: data.get(c, np.nan) for c in feat_cols}])
    return X, None


# ─────────────────────────────────────────────────────────────
# SHAP local explanation
# ─────────────────────────────────────────────────────────────
def compute_shap(pipe, X, feat_cols):
    """Compute SHAP values for a single patient."""
    prep = pipe.named_steps["prep"]
    clf  = pipe.named_steps["clf"]
    X_trans = prep.transform(X)

    explainer = shap.TreeExplainer(clf)
    sv = explainer(X_trans)

    # Map transformed feature names back to original
    if hasattr(prep, "get_feature_names_out"):
        trans_names = list(prep.get_feature_names_out())
    else:
        trans_names = feat_cols

    vals  = sv.values[0]   # shape: (n_transformed_features,)
    base  = sv.base_values[0]

    # Aggregate one-hot encoded features back to original
    agg = {}
    for i, tn in enumerate(trans_names):
        orig = _map_to_original(tn, feat_cols)
        agg[orig] = agg.get(orig, 0.0) + vals[i]

    shap_series = pd.Series(agg)
    return shap_series, base


def _map_to_original(trans_name: str, feat_cols: list[str]) -> str:
    """Map a transformed feature name back to the original feature name."""
    # Direct match
    if trans_name in feat_cols:
        return trans_name
    # One-hot: "cat__Sex_Male" → try "Sex"
    for fc in feat_cols:
        if trans_name.startswith(f"cat__{fc}") or trans_name.startswith(f"num__{fc}"):
            return fc
        if fc in trans_name:
            return fc
    return trans_name


# ─────────────────────────────────────────────────────────────
# Contextual review flags
# ─────────────────────────────────────────────────────────────
def generate_review_flags(top_features: list[tuple], X: pd.DataFrame):
    """Flag top contributors outside descriptive DEV percentile cutoffs."""
    flags = []
    for feat, shap_val in top_features:
        rule = REVIEW_RULES.get(feat)
        if rule is None:
            continue
        val = X[feat].iloc[0] if feat in X.columns else None
        if val is None or pd.isna(val):
            continue

        triggered = False
        if "low_p25" in rule and val <= rule["low_p25"]:
            triggered = True
        if "high_p75" in rule and val >= rule["high_p75"]:
            triggered = True

        if triggered:
            flags.append({
                "feature": feat,
                "readable": readable(feat),
                "value": val,
                "shap": shap_val,
                "message": rule["message"],
            })
    return flags


# ─────────────────────────────────────────────────────────────
# Report formatting
# ─────────────────────────────────────────────────────────────
def format_report(patient_id, proba, shap_series, base_value, X,
                  slope_val, slope_cutoff):
    """Build the full textual report."""
    W = 62  # column width
    lines = []

    # ── Header ──
    lines.append("=" * W)
    lines.append("  ALS RAPID-PROGRESSION RISK REPORT")
    lines.append("  Model: XGBoost (Optuna-tuned) — 6-month horizon")
    lines.append("=" * W)

    # ── 1. Risk assessment ──
    lines.append("")
    lines.append("─── RISK ASSESSMENT " + "─" * (W - 20))
    lines.append(f"  Patient ID:         {patient_id}")
    lines.append(f"  P(rapid, 6 months): {proba:.4f}  ({proba*100:.1f}%)")

    classification = "RAPID" if proba >= DECISION_THRESHOLD else "SLOW"
    lines.append(f"  Classification (selected threshold, t={DECISION_THRESHOLD}):")
    lines.append(f"    {classification} PROGRESSOR")

    if slope_val is not None:
        true_label = "RAPID" if slope_val <= slope_cutoff else "SLOW"
        lines.append(f"  Retrospective outcome: {true_label}  "
                     f"(slope={slope_val:.4f}, cutoff={slope_cutoff:.4f})")
        lines.append("  Note: this record was included in the final full-cohort refit;")
        lines.append("  the comparison above is descriptive, not out-of-sample evaluation.")

    lines.append("─" * W)

    # ── 2. Top contributing features ──
    lines.append("")
    lines.append("─── TOP-5 CONTRIBUTING FEATURES (SHAP) " + "─" * (W - 39))
    lines.append(f"  Base value (population average log-odds): {base_value:.4f}")
    lines.append("")

    top_abs = shap_series.abs().sort_values(ascending=False).head(N_TOP)
    top_feats = [(feat, shap_series[feat]) for feat in top_abs.index]

    lines.append(f"  {'#':<3} {'Feature':<32} {'Value':>8}  {'SHAP':>7}  Direction")
    lines.append(f"  {'─'*3} {'─'*32} {'─'*8}  {'─'*7}  {'─'*15}")

    for i, (feat, sv) in enumerate(top_feats, 1):
        val = X[feat].iloc[0] if feat in X.columns else "N/A"
        val_str = f"{val:.1f}" if isinstance(val, (int, float, np.floating)) and not pd.isna(val) else str(val)
        direction = "↑ increases risk" if sv > 0 else "↓ decreases risk"
        lines.append(f"  {i:<3} {readable(feat):<32} {val_str:>8}  {sv:>+7.4f}  {direction}")

    lines.append("─" * W)

    # ── 3. Contextual review flags ──
    lines.append("")
    lines.append("─── CONTEXTUAL REVIEW FLAGS " + "─" * (W - 28))

    flags = generate_review_flags(top_feats, X)
    if flags:
        lines.append("  Descriptive checks among the top model contributors:")
        lines.append("")
        for s in flags:
            val_str = f"{s['value']:.1f}" if isinstance(s['value'], (int, float, np.floating)) else str(s['value'])
            lines.append(f"  • {s['readable']} = {val_str}")
            lines.append(f"    → {s['message']}")
            lines.append("")
    else:
        lines.append("  No predefined descriptive cutoff was triggered among")
        lines.append("  the five largest SHAP contributors.")
        lines.append("")

    lines.append("─" * W)

    # ── Disclaimer ──
    lines.append("")
    lines.append("DISCLAIMER: This report is generated by a machine-learning")
    lines.append("model trained on PRO-ACT data for research purposes only.")
    lines.append("The probability, classification, SHAP values, and percentile")
    lines.append("flags have not been prospectively or clinically validated.")
    lines.append("They do NOT constitute medical advice or a care recommendation.")
    lines.append("=" * W)

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Demo patient (default)
# ─────────────────────────────────────────────────────────────
def get_demo_patient_id():
    """Pick a random dataset record for demonstration."""
    rng = np.random.default_rng()
    patient_ids = pd.read_csv(DATA_PATH, usecols=["subject_id"])["subject_id"]
    return int(rng.choice(patient_ids.to_numpy()))


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Generate ALS rapid-progression risk report for a patient."
    )
    parser.add_argument("--patient-id", type=int, default=None,
                        help="subject_id from the dataset")
    parser.add_argument("--json", type=str, default=None,
                        help="Path to JSON file with patient features")
    parser.add_argument("--save", action="store_true",
                        help="Save report to 04_outputs/reports/")
    args = parser.parse_args()

    print("Loading model...")
    pipe, meta, feat_cols, slope_cutoff = load_model()

    # Build patient vector
    if args.json:
        X, slope_val = patient_from_json(args.json, feat_cols)
        # Use JSON filename (without extension) as patient ID
        json_name = os.path.splitext(os.path.basename(args.json))[0]
        patient_id = f"JSON_{json_name}"
    elif args.patient_id:
        X, slope_val = patient_from_dataset(args.patient_id, feat_cols)
        patient_id = args.patient_id
    else:
        patient_id = get_demo_patient_id()
        print(f"No patient specified — using dataset record {patient_id} for demonstration.")
        X, slope_val = patient_from_dataset(patient_id, feat_cols)

    # Predict
    proba = float(pipe.predict_proba(X)[:, 1][0])

    # SHAP explanation
    print("Computing SHAP explanation...")
    shap_series, base_value = compute_shap(pipe, X, feat_cols)

    # Generate report
    report = format_report(patient_id, proba, shap_series, base_value,
                           X, slope_val, slope_cutoff)
    print()
    print(report)

    # Optionally save
    if args.save:
        os.makedirs(OUT_DIR, exist_ok=True)
        out_path = os.path.join(OUT_DIR, f"report_{patient_id}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\nReport saved to: {out_path}")


if __name__ == "__main__":
    main()
