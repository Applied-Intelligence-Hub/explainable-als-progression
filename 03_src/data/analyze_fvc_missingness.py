"""
Step 4c — FVC Missingness Analysis

Purpose:
  Check whether subjects WITH vs WITHOUT FVC data differ systematically on
  key clinical variables. If the missingness is informative (MNAR), it means
  FVC-absent subjects may be a biased subgroup, and median imputation could
  introduce bias.

Tests:
  - Continuous variables: Mann-Whitney U test (non-parametric, robust)
  - Categorical variables: Chi-squared test
  - Effect sizes: Cohen's d (continuous), Cramér's V (categorical)

Output:
  04_outputs/tables/step4_fvc_missingness_analysis.csv

Run from project root:
  python 03_src/data/analyze_fvc_missingness.py
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

PROCESSED  = os.path.join("01_data", "processed")
OUT_TABLES = os.path.join("04_outputs", "tables", "step4")


def cohens_d(group1, group2):
    """Cohen's d for two independent groups."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (group1.mean() - group2.mean()) / pooled_std


def cramers_v(contingency_table):
    """Cramér's V from a contingency table."""
    chi2 = stats.chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1
    if min_dim == 0 or n == 0:
        return 0.0
    return np.sqrt(chi2 / (n * min_dim))


def analyze_horizon(dataset_path, horizon_label, slope_col):
    """Run missingness analysis for one horizon."""
    df = pd.read_csv(dataset_path)
    df["has_fvc"] = df["FVC_Liters_best_t0"].notna().astype(int)

    n_total = len(df)
    n_with = df["has_fvc"].sum()
    n_without = n_total - n_with

    print(f"\n{'='*60}")
    print(f"  {horizon_label} horizon: N={n_total}")
    print(f"  With FVC: {n_with} ({100*n_with/n_total:.1f}%)")
    print(f"  Without FVC: {n_without} ({100*n_without/n_total:.1f}%)")
    print(f"{'='*60}")

    results = []

    # --- Continuous variables ---
    cont_vars = [
        ("ALSFRS_R_t0", "ALSFRS-R at baseline"),
        ("Age", "Age"),
        (slope_col, "Slope (target)"),
        ("creatinine_t0", "Creatinine"),
        ("alt_t0", "ALT"),
        ("BMI_t0", "BMI"),
        ("Weight_kg_t0", "Weight"),
        ("n_conmeds_pre_t0", "N conmeds"),
    ]

    for col, label in cont_vars:
        if col not in df.columns:
            continue
        g_with = df.loc[df["has_fvc"] == 1, col].dropna()
        g_without = df.loc[df["has_fvc"] == 0, col].dropna()

        if len(g_with) < 5 or len(g_without) < 5:
            continue

        stat_u, p_val = stats.mannwhitneyu(g_with, g_without, alternative="two-sided")
        d = cohens_d(g_with, g_without)

        results.append({
            "variable": label,
            "type": "continuous",
            "n_with_fvc": len(g_with),
            "n_without_fvc": len(g_without),
            "mean_with": round(g_with.mean(), 3),
            "mean_without": round(g_without.mean(), 3),
            "test": "Mann-Whitney U",
            "statistic": round(stat_u, 1),
            "p_value": round(p_val, 6),
            "effect_size": round(abs(d), 3),
            "effect_type": "Cohen d",
            "significant_005": p_val < 0.05,
        })

        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"  {label:20s}: with={g_with.mean():.2f}  without={g_without.mean():.2f}  "
              f"|d|={abs(d):.3f}  p={p_val:.4f} {sig}")

    # --- Categorical variables ---
    cat_vars = [
        ("Sex", "Sex"),
        ("study_arm", "Study Arm"),
        ("riluzole_pre_t0", "Riluzole"),
    ]

    for col, label in cat_vars:
        if col not in df.columns:
            continue
        sub = df[[col, "has_fvc"]].dropna()
        if sub[col].nunique() < 2:
            continue

        ct = pd.crosstab(sub[col], sub["has_fvc"])
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            continue

        chi2, p_val, dof, _ = stats.chi2_contingency(ct)
        v = cramers_v(ct)

        results.append({
            "variable": label,
            "type": "categorical",
            "n_with_fvc": int(sub.loc[sub["has_fvc"] == 1].shape[0]),
            "n_without_fvc": int(sub.loc[sub["has_fvc"] == 0].shape[0]),
            "mean_with": "",
            "mean_without": "",
            "test": "Chi-squared",
            "statistic": round(chi2, 1),
            "p_value": round(p_val, 6),
            "effect_size": round(v, 3),
            "effect_type": "Cramer V",
            "significant_005": p_val < 0.05,
        })

        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"  {label:20s}: chi2={chi2:.1f}  V={v:.3f}  p={p_val:.4f} {sig}")

    return pd.DataFrame(results)


if __name__ == "__main__":
    results_all = []

    for ds, label, slope_col in [
        (os.path.join(PROCESSED, "dataset_6m_v2.csv"), "6m", "slope_180d_per_30d"),
        (os.path.join(PROCESSED, "dataset_3m_v2.csv"), "3m", "slope_90d_per_30d"),
    ]:
        res = analyze_horizon(ds, label, slope_col)
        res.insert(0, "horizon", label)
        results_all.append(res)

    final = pd.concat(results_all, ignore_index=True)
    out = os.path.join(OUT_TABLES, "step4_fvc_missingness_analysis.csv")
    final.to_csv(out, index=False)
    print(f"\nSaved: {out}")

    # Summary
    n_sig = final["significant_005"].sum()
    n_total = len(final)
    print(f"\nSignificant differences (p<0.05): {n_sig}/{n_total}")
    if n_sig > 0:
        print("Variables with significant differences:")
        sig_vars = final[final["significant_005"]]
        for _, row in sig_vars.iterrows():
            print(f"  [{row['horizon']}] {row['variable']}: p={row['p_value']:.4f}, "
                  f"effect={row['effect_size']} ({row['effect_type']})")
