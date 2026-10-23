"""Study I aggregate reviewer checks; no model refits or patient-level exports.

Analysis functions are exported from the dissertation reviewer_followup.py.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import binomtest, chi2_contingency, fisher_exact, mannwhitneyu
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
INPUTS: dict[str, str] = {}


def fingerprint(path: Path) -> None:
    INPUTS[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    fingerprint(path)
    return pd.read_csv(path)


def write_table(out: Path, name: str, caption: str, label: str,
                headers: list[str], rows: list[list[str]], note: str = "") -> None:
    body = [r"\begin{table}[ht!]", r"\centering", r"\small",
            rf"\caption[{caption}]{{{caption}}}", rf"\label{{{label}}}",
            r"\setlength{\tabcolsep}{4pt}",
            r"\begin{tabular}{l" + "r" * (len(headers) - 1) + "}", r"\toprule",
            " & ".join(headers) + r" \\", r"\midrule"]
    body.extend(" & ".join(row) + r" \\" for row in rows)
    body.extend([r"\bottomrule", r"\end{tabular}"])
    if note:
        body.extend([r"\par\smallskip", r"\begin{minipage}{0.98\linewidth}\footnotesize",
                     note, r"\end{minipage}"])
    body.append(r"\end{table}")
    (out / f"{name}.tex").write_text("\n".join(body) + "\n", encoding="utf-8")


def summary(series: pd.Series) -> str:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return "--"
    q = s.quantile([0.25, 0.5, 0.75])
    return (r"\shortstack{" + f"{q.loc[0.5]:.1f} [{q.loc[0.25]:.1f}, {q.loc[0.75]:.1f}]"
            + r" \\ $n=" + str(len(s)) + "$}")


def calibration_diagnostics(y: np.ndarray, p: np.ndarray) -> dict:
    logit = np.log(np.clip(p, 1e-8, 1 - 1e-8) / np.clip(1 - p, 1e-8, 1))
    def loss(beta):
        eta = beta[0] + beta[1] * logit
        return np.sum(np.logaddexp(0, eta) - y * eta)
    fit = minimize(loss, [0.0, 1.0], method="BFGS", tol=1e-7)
    # A small numerical gradient is acceptable when BFGS reports precision loss.
    eta = fit.x[0] + fit.x[1] * logit
    grad = np.array([np.sum(expit(eta) - y), np.sum((expit(eta) - y) * logit)])
    if not np.isfinite(fit.fun) or np.linalg.norm(grad) > 1e-3:
        raise RuntimeError(f"Calibration diagnostic did not converge: {fit.message}")
    return {"intercept": float(fit.x[0]), "slope": float(fit.x[1])}


def study1(out: Path, boots: int) -> dict:
    base = read_csv(ROOT / "01_data/processed/features_baseline_v1.csv")
    six = read_csv(ROOT / "01_data/processed/dataset_6m_v2.csv")
    three = read_csv(ROOT / "01_data/processed/dataset_3m_v2.csv")
    pred = read_csv(ROOT / "04_outputs/tables/step8_xai/test_predictions.csv")
    split = read_csv(ROOT / "01_data/processed/holdout_split_6m.csv")
    assert len(base) == 5436 and len(six) == 1392 and len(three) == 2407
    assert len(set(three.subject_id) & set(six.subject_id)) == 720
    columns = ["Age", "Sex", "ALSFRS_R_t0", "FVC_Liters_best_t0", "FVC_pctNormal_best_t0"]
    for cohort in [three, six]:
        assert cohort.subject_id.is_unique
        current = cohort.set_index("subject_id")[columns]
        pd.testing.assert_frame_equal(base.set_index("subject_id").loc[current.index, columns],
                                      current, check_dtype=False)
    assert not pred.subject_id.duplicated().any()
    assert set(pred.subject_id) == set(split.loc[split.partition.eq("test"), "subject_id"])
    test = pred.merge(six, on="subject_id", validate="one_to_one")
    y, p = test.y_true.to_numpy(), test.proba_rapid.to_numpy()
    assert confusion_matrix(y, p >= 0.21).ravel().tolist() == [62, 133, 12, 72]
    dev = six.loc[six.subject_id.isin(split.loc[split.partition.eq("dev"), "subject_id"])]
    cutoff = dev.slope_180d_per_30d.quantile(0.30)
    assert np.array_equal(y, (test.slope_180d_per_30d <= cutoff).astype(int))
    cohorts = [base.loc[base.subject_id.isin(three.subject_id)],
               base.loc[base.subject_id.isin(six.subject_id)],
               base.loc[~base.subject_id.isin(six.subject_id)]]
    rows = [[r"Participants", *[f"{len(c):,}" for c in cohorts]]]
    for col, label in [("Age", "Age (years)"), ("ALSFRS_R_t0", "Baseline ALSFRS-R"),
                       ("FVC_Liters_best_t0", "FVC (litres)"),
                       ("FVC_pctNormal_best_t0", r"FVC (\% normal)")]:
        rows.append([label, *[summary(c[col]) for c in cohorts]])
    rows.append(["Male / recorded sex", *[f"{c.Sex.eq('Male').sum()} / {c.Sex.notna().sum()}" for c in cohorts]])
    write_table(out, "review_s1_baseline", "Study I baseline characteristics and eligibility",
                "tab:review_s1_baseline", ["Characteristic", r"\shortstack{3-month\\eligible}", r"\shortstack{6-month\\eligible}", r"\shortstack{6-month\\excluded}"], rows,
                "Continuous values are median [interquartile range]; number observed. Exclusions are relative to the 5,436 participants with a valid ALSFRS-R assessment. The two eligible cohorts overlap (720 participants).")
    baseline_tests = []
    for col in ["Age", "ALSFRS_R_t0", "FVC_Liters_best_t0", "FVC_pctNormal_best_t0"]:
        a, b = cohorts[1][col].dropna(), cohorts[2][col].dropna()
        u, pv = mannwhitneyu(a, b, alternative="two-sided")
        baseline_tests.append({"feature": col, "n_eligible": len(a), "n_excluded": len(b),
                               "p": float(pv), "rank_biserial": float(2*u/(len(a)*len(b))-1)})
    pd.DataFrame(baseline_tests).to_csv(out / "review_s1_selection_tests.csv", index=False)

    # Match the fixed cut-points used by the original supplementary script.
    groups = {
        "Age": pd.cut(test.Age, [0, 49, 57, 65, np.inf],
                      labels=["<=49", "50--57", "58--65", ">65"]).astype(object).fillna("Missing"),
        "Sex": test.Sex.fillna("Missing"),
        "ALSFRS-R": pd.cut(test.ALSFRS_R_t0, [-np.inf, 35, 39, 42, np.inf],
                           labels=["<=35", "36--39", "40--42", ">42"]),
        "FVC": pd.cut(test.FVC_pctNormal_best_t0, [-np.inf, 77, 87, 98, np.inf],
                      labels=["<=77", "78--87", "88--98", ">98"]).astype(object).fillna("Missing"),
    }
    rng = np.random.default_rng(42)
    subgroup_rows, records = [], []
    for family, strata in groups.items():
        for group in pd.unique(strata):
            mask = strata.eq(group).to_numpy()
            ys, ps = y[mask], p[mask]
            n, positives = len(ys), int(ys.sum())
            if n < 10 or positives < 3:
                continue
            indices = rng.integers(0, n, size=(boots, n))
            by, bp = ys[indices], ps[indices] >= .21
            denominators = by.sum(axis=1)
            recall = (by*bp).sum(axis=1)[denominators > 0] / denominators[denominators > 0]
            lo, hi = np.quantile(recall, [.025, .975])
            est = float(np.sum(ys*(ps >= .21))/positives)
            records.append(dict(stratification=family, group=group, n=n, positives=positives,
                                recall=est, lower=lo, upper=hi, valid_bootstraps=len(recall)))
            display = str(group).replace("<=", r"$\leq$").replace(">", r"$>$")
            subgroup_rows.append([f"{family}: {display}", str(n), str(positives),
                                  f"{est:.3f} [{lo:.3f}, {hi:.3f}]"])
    order = {
        "Age": ["<=49", "50--57", "58--65", ">65", "Missing"],
        "Sex": ["Female", "Male", "Missing"],
        "ALSFRS-R": ["<=35", "36--39", "40--42", ">42"],
        "FVC": ["<=77", "78--87", "88--98", ">98", "Missing"],
    }
    indices = sorted(range(len(records)), key=lambda i: (
        list(order).index(records[i]["stratification"]),
        order[records[i]["stratification"]].index(records[i]["group"])))
    records = [records[i] for i in indices]
    subgroup_rows = [subgroup_rows[i] for i in indices]
    pd.DataFrame(records).to_csv(out / "review_s1_subgroup_ci.csv", index=False)
    write_table(out, "review_s1_subgroup_ci", "Study I subgroup recall with bootstrap intervals",
                "tab:review_s1_subgroup_ci", ["Stratum", "$N$", "Rapid", "Recall [95\\% CI]"], subgroup_rows,
                f"Pointwise percentile intervals from {boots:,} within-stratum patient resamples at the fixed threshold 0.21. Single-class resamples without rapid progressors are omitted. These are not simultaneous intervals or tests of subgroup differences.")

    thresholds = np.round(np.arange(.01, .61, .01), 2)
    n = len(y)
    idx = rng.integers(0, n, size=(boots, n))
    by, bp = y[idx], p[idx]
    dca = []
    for t in thresholds:
        decisions = bp >= t
        nb = ((by*decisions).sum(axis=1) - ((1-by)*decisions).sum(axis=1)*t/(1-t))/n
        treat_all = by.mean(axis=1) - (1-by.mean(axis=1))*t/(1-t)
        delta = nb - treat_all
        pred_t = p >= t
        point = (np.sum(y*pred_t)-np.sum((1-y)*pred_t)*t/(1-t))/n
        lo, hi = np.quantile(nb, [.025, .975])
        dl, dh = np.quantile(delta, [.025, .975])
        dca.append(dict(threshold=t, net_benefit=point, lower=lo, upper=hi,
                        difference_vs_all=point-(y.mean()-(1-y.mean())*t/(1-t)),
                        difference_lower=dl, difference_upper=dh))
    dc = pd.DataFrame(dca)
    dc.to_csv(out / "review_s1_dca_ci.csv", index=False)
    from study1_style import apply_study1_style, PALETTE
    apply_study1_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.fill_between(dc.threshold, dc.lower, dc.upper, color=PALETTE["soft_mauve"], alpha=.2, label="Pointwise 95% interval")
    ax.plot(dc.threshold, dc.net_benefit, color=PALETTE["velvet_purple"], label="XGBoost (fixed fit)")
    ax.plot(thresholds, y.mean()-(1-y.mean())*thresholds/(1-thresholds), color=PALETTE["deep_rose"], ls="--", label="Treat all")
    ax.axhline(0, color="0.4", ls=":", label="Treat none")
    ax.axvline(.21, color="0.4", ls=":", lw=.8)
    ax.set(xlabel="Threshold probability", ylabel="Net benefit", ylim=(-.1, .36), xlim=(.01, .6))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "review_s1_dca_ci.pdf", bbox_inches="tight")
    fig.savefig(out / "review_s1_dca_ci.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    continuous = ["ALSFRS_R_t0", "Age", "slope_180d_per_30d", "creatinine_t0",
                  "alt_t0", "BMI_t0", "Weight_kg_t0", "n_conmeds_pre_t0"]
    observed = six.FVC_Liters_best_t0.notna()
    effects = []
    for col in continuous:
        a, b = six.loc[observed, col].dropna(), six.loc[~observed, col].dropna()
        u, pv = mannwhitneyu(a, b, alternative="two-sided")
        effects.append([col.replace("_", r"\_"), str(len(a)), str(len(b)),
                        f"{2*u/(len(a)*len(b))-1:+.3f}"])
    write_table(out, "review_s1_rank_effects", "Rank-based effects in the FVC missingness comparison",
                "tab:review_s1_rank_effects", ["Variable", "FVC observed", "FVC absent", "Rank-biserial $r$"], effects,
                "$r=2U/(n_1n_2)-1$, with the observed-FVC group first. Positive values indicate higher ranks in that group. Missing comparison values are omitted separately for each variable.")
    categorical_rows = []
    for col, label in [("Sex", "Sex"), ("study_arm", "Study arm"), ("riluzole_pre_t0", "Riluzole")]:
        table = pd.crosstab(six[col], observed)
        chi_uncorrected, _, _, _ = chi2_contingency(table, correction=False)
        v = np.sqrt(chi_uncorrected / (table.to_numpy().sum()*min(np.array(table.shape)-1)))
        categorical_rows.append([label, str(int(table.to_numpy().sum())), f"{v:.3f}"])
    write_table(out, "review_s1_cramers_v", "Uncorrected Cramer's V for FVC availability",
                "tab:review_s1_cramers_v", ["Variable", "$N$", r"Cram\'er's $V$"], categorical_rows,
                "Calculated from the uncorrected Pearson statistic. Original reported test p-values retain the Yates correction for two-by-two tables.")

    arm = pd.crosstab(six.study_arm, six.slope_180d_per_30d <= six.slope_180d_per_30d.quantile(.3))
    assert sorted(arm.sum(axis=1).tolist()) == [506, 762]
    cy, py, _, _ = chi2_contingency(arm, correction=True)
    cu, pu, _, _ = chi2_contingency(arm, correction=False)
    odds, pf = fisher_exact(arm)
    race = []
    for col in [c for c in dev if c.startswith("Race_")]:
        race.append({"feature": col, "missing_dev": int(dev[col].isna().sum()),
                     "median_dev": float(dev[col].median()), "recorded_dev": int(dev[col].notna().sum())})
    pd.DataFrame(race).to_csv(out / "review_s1_race_encoding.csv", index=False)
    null = y.mean()*(1-y.mean())
    return dict(n_test=len(y), average_precision=average_precision_score(y,p),
                brier=brier_score_loss(y,p), null_brier=null,
                scaled_brier=1-brier_score_loss(y,p)/null,
                probability_range=[float(p.min()), float(p.max())],
                calibration=calibration_diagnostics(y,p),
                treatment_tests=dict(yates_chi2=cy, yates_p=py, uncorrected_chi2=cu,
                                     uncorrected_p=pu, fisher_p=pf),
                dca_at_021=dc.loc[dc.threshold.eq(.21)].iloc[0].to_dict())


def trials(out: Path) -> dict:
    records = []
    for path in sorted((ROOT / "04_outputs/tables/step5_tuning").glob("*_trials.csv")):
        d = read_csv(path)
        records.append(dict(search=path.stem.replace("_trials",""),n=len(d),
                            zero_scores=int(d.value.eq(0).sum()), missing_scores=int(d.value.isna().sum()),
                            states=d.state.value_counts().to_dict(),best_trial_zero_based=int(d.loc[d.value.idxmax(),"number"])))
    pd.DataFrame(records).to_csv(out / "review_s1_trial_audit.csv", index=False)
    return dict(study1=records)



def main() -> None:
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    ROOT = args.project_root.resolve()
    out = args.output or ROOT / "04_outputs/reviewer_followup"
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "scope": "Study I fixed-cohort, saved-prediction checks; no model refits.",
        "bootstrap_resamples": args.bootstrap,
        "study1": study1(out, args.bootstrap),
        "trials": trials(out),
        "packages": {p: importlib.metadata.version(p) for p in
                     ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib"]},
    }
    result["input_sha256"] = INPUTS
    (out / "review_followup_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["study1"], indent=2))


if __name__ == "__main__":
    main()
