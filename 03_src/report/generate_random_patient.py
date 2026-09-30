"""
Generate a random fictitious patient JSON for demo/testing purposes.

Each feature is sampled from approximate distributions calibrated on the
full PRO-ACT dataset (N=1,392). The generator reproduces selected marginal
summaries and a small number of dependencies; it is intended only for
software demonstration, not for clinical simulation or privacy protection.

Usage
-----
  python 03_src/report/generate_random_patient.py                  # prints JSON
  python 03_src/report/generate_random_patient.py --out patient.json  # saves to file
"""

import argparse
import json
import os
import sys

import numpy as np

DEFAULT_OUT_DIR = os.path.join("04_outputs", "random_patients")


def _trunc_normal(rng, mean, std, lo, hi):
    """Sample from a truncated normal distribution."""
    while True:
        x = rng.normal(mean, std)
        if lo <= x <= hi:
            return x


def generate_random_patient(rng: np.random.Generator | None = None) -> dict:
    """Return a dict with all 35 features sampled from approximate distributions.

    Distributions are calibrated from the PRO-ACT dataset (N=1,392):
    - Continuous variables: truncated normal matching observed mean/std/range
    - Categorical variables: probabilities matching observed frequencies
    - Selected dependencies: BMI from weight/height and FVC conditioned on sex

    The full multivariable dependence structure is not reproduced.
    """
    if rng is None:
        rng = np.random.default_rng()

    # ── Demographics ────────────────────────────────────────
    # ALSFRS-R: mean=38.2, std=5.5, range [17, 48]
    alsfrs = int(round(_trunc_normal(rng, 38.2, 5.5, 17, 48)))

    # Age: mean=56.5, std=11.4, range [19, 82]
    age = int(round(_trunc_normal(rng, 56.5, 11.4, 19, 82)))

    # Sex: 64% Male (1), 36% Female (0)
    sex = int(rng.choice([0, 1], p=[0.36, 0.64]))

    # Ethnicity: 93% Non-Hispanic (0), 5% Hispanic (1), 2% Unknown
    ethnicity = int(rng.choice([0, 1], p=[0.95, 0.05]))

    # Race (one-hot, mutually exclusive — real frequencies)
    race_probs = {
        "Race_Caucasian": 0.85,
        "Race_Black_African_American": 0.04,
        "Race_Asian": 0.03,
        "Race_Americ_Indian_Alaska_Native": 0.01,
        "Race_Hawaiian_Pacific_Islander": 0.005,
        "Race_Other": 0.04,
        "Race_Unknown": 0.025,
    }
    race_keys = list(race_probs.keys())
    chosen = rng.choice(race_keys, p=list(race_probs.values()))
    race = {k: (1 if k == chosen else 0) for k in race_keys}

    # Respiratory sub-scores (real frequencies from dataset)
    r1_dyspnea = int(rng.choice(
        [0, 1, 2, 3, 4],
        p=[0.003, 0.018, 0.094, 0.171, 0.714]))
    r2_orthopnea = int(rng.choice(
        [0, 1, 2, 3, 4],
        p=[0.005, 0.003, 0.034, 0.080, 0.878]))
    r3_resp_insuf = int(rng.choice(
        [0, 1, 2, 3, 4],
        p=[0.001, 0.001, 0.034, 0.018, 0.946]))

    # Mode of administration: 0 or 1 (82% in-person, 18% electronic/paper)
    mode_admin = int(rng.choice([0, 1], p=[0.82, 0.18]))

    # ── Anthropometrics & Vitals ────────────────────────────
    # Weight: mean=78.7, std=16.3, range [40, 150]
    weight = round(_trunc_normal(rng, 78.7, 16.3, 40, 150), 1)

    # Height: mean=170.5, std=10.6, range [145, 196]
    # Adjust slightly by sex: males ~175, females ~163
    if sex == 1:  # Male
        height = round(_trunc_normal(rng, 175.0, 8.0, 150, 196), 1)
    else:         # Female
        height = round(_trunc_normal(rng, 163.0, 7.5, 145, 185), 1)

    # BMI: derived from weight/height (realistic correlation)
    bmi = round(weight / (height / 100) ** 2, 1)

    # Pulse: mean=75.7, std=12.3, range [45, 132]
    pulse = int(round(_trunc_normal(rng, 75.7, 12.3, 45, 132)))

    # Respiratory Rate: mean=17.4, std=3.1, range [10, 36]
    # Distribution is left-modal (median=16, P75=20)
    resp_rate = int(round(_trunc_normal(rng, 17.4, 3.1, 10, 36)))

    # Temperature: mean=36.5, observed IQR 36.3–36.8 (tight)
    temp = round(_trunc_normal(rng, 36.5, 0.3, 35.5, 38.0), 1)

    # ── Blood Pressure ──────────────────────────────────────
    # Systolic: mean=130.4, std=16.0
    bp_sys = int(round(_trunc_normal(rng, 130.4, 16.0, 89, 200)))
    # Diastolic: mean=80.5, std=10.1
    bp_dia = int(round(_trunc_normal(rng, 80.5, 10.1, 49, 115)))
    # Standing/Supine: small offsets from seated
    stand_sys = bp_sys + int(rng.integers(-5, 6))
    stand_dia = bp_dia + int(rng.integers(-5, 6))
    sup_sys = bp_sys + int(rng.integers(-8, 3))
    sup_dia = bp_dia + int(rng.integers(-5, 3))

    # ── Pulmonary Function ──────────────────────────────────
    # FVC litres: mean=3.5, std=1.0, range [1.4, 6.9]
    # Correlates with sex: males ~3.9L, females ~2.8L
    if sex == 1:  # Male
        fvc_liters = round(_trunc_normal(rng, 3.9, 0.9, 1.4, 6.9), 1)
    else:
        fvc_liters = round(_trunc_normal(rng, 2.8, 0.7, 1.4, 5.5), 1)

    # FVC % predicted: mean=88.5, std=15.5, range [47, 146]
    fvc_pct = int(round(_trunc_normal(rng, 88.5, 15.5, 47, 146)))

    # Subject normal FVC: mean=4.0, std=0.9, range [2.1, 6.5]
    subj_normal = round(_trunc_normal(rng, 4.0, 0.9, 2.1, 6.5), 1)

    # ── Treatment ───────────────────────────────────────────
    # Riluzole: 65% yes (1), 35% no (0)
    riluzole = int(rng.choice([0, 1], p=[0.35, 0.65]))
    # Study arm: 60% Placebo (0), 40% Active (1)
    study_arm = int(rng.choice([0, 1], p=[0.60, 0.40]))
    # Concomitant meds: median=4, many 0s, right-skewed
    n_conmeds = int(round(max(0, _trunc_normal(rng, 5.3, 7.2, 0, 30))))

    # ── Labs ────────────────────────────────────────────────
    # ALT: mean=36.4, std=20.1 — right-skewed, range [8, 182]
    alt = int(round(max(8, _trunc_normal(rng, 36.4, 20.1, 8, 120))))
    # Creatinine: mean=69.8, std=18.7, range [18, 159]
    creatinine = round(_trunc_normal(rng, 69.8, 18.7, 18, 159), 1)

    patient = {
        "ALSFRS_R_t0": alsfrs,
        "Age": age,
        "Sex": sex,
        "Ethnicity": ethnicity,
        **race,
        "R_1_Dyspnea": r1_dyspnea,
        "R_2_Orthopnea": r2_orthopnea,
        "R_3_Respiratory_Insufficiency": r3_resp_insuf,
        "Mode_of_Administration": mode_admin,
        "Weight_kg_t0": weight,
        "Height_cm_t0": height,
        "BMI_t0": bmi,
        "Pulse": pulse,
        "Respiratory_Rate": resp_rate,
        "Temperature": temp,
        "Blood_Pressure_Systolic": bp_sys,
        "Blood_Pressure_Diastolic": bp_dia,
        "Baseline_Standing_BP_Systolic": stand_sys,
        "Baseline_Standing_BP_Diastolic": stand_dia,
        "Baseline_Supine_BP_Systolic": sup_sys,
        "Baseline_Supine_BP_Diastolic": sup_dia,
        "FVC_Liters_best_t0": fvc_liters,
        "FVC_pctNormal_best_t0": fvc_pct,
        "subject_normal": subj_normal,
        "riluzole_pre_t0": riluzole,
        "study_arm": study_arm,
        "n_conmeds_pre_t0": n_conmeds,
        "alt_t0": alt,
        "creatinine_t0": creatinine,
    }
    return patient


def main():
    parser = argparse.ArgumentParser(
        description="Generate a random fictitious ALS patient JSON."
    )
    parser.add_argument("--out", type=str, default=None,
                        help="Output JSON file path (default: auto-save to 04_outputs/random_patients/)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    patient = generate_random_patient(rng)

    json_str = json.dumps(patient, indent=2, ensure_ascii=False)

    if args.out:
        out_path = args.out
    else:
        os.makedirs(DEFAULT_OUT_DIR, exist_ok=True)
        existing = [f for f in os.listdir(DEFAULT_OUT_DIR) if f.startswith("random_patient_") and f.endswith(".json")]
        idx = len(existing) + 1
        out_path = os.path.join(DEFAULT_OUT_DIR, f"random_patient_{idx:03d}.json")

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json_str + "\n")
    print(f"Random patient saved to: {out_path}")


if __name__ == "__main__":
    main()
