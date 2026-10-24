"""Recompute only categorical-aware LIME and comparable grouped SHAP rankings."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

import step8_explainability as xai
from lime_categorical import CategoricalView, build_explainer


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "04_outputs/reviewer_followup/revised_analyses")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    xdev, ydev, xtest, ytest, ids, _, _ = xai.load_data()
    parameters, _ = xai.load_params("XGB", "unbalanced")
    pipe = Pipeline([("prep", xai.build_preprocessor(xdev)),
                     ("clf", XGBClassifier(**parameters))])
    pipe.fit(xdev, ydev)
    probability = pipe.predict_proba(xtest)[:, 1]
    saved = pd.read_csv(ROOT / "04_outputs/tables/step8_xai/test_predictions.csv").set_index("subject_id")
    assert np.array_equal(saved.loc[ids, "y_true"].to_numpy(), ytest)
    np.testing.assert_allclose(probability, saved.loc[ids, "proba_rapid"], rtol=0, atol=1e-7)
    prediction = (probability >= .21).astype(int)
    assert confusion_matrix(ytest, prediction).ravel().tolist() == [62, 133, 12, 72]
    cases = xai.pick_case_studies(ytest, prediction, probability)
    profiles = pd.read_csv(ROOT / "04_outputs/tables/step8_xai/case_study_profiles.csv").set_index("case")
    for case, index in cases.items():
        assert int(ids.iloc[index]) == int(profiles.loc[case, "subject_id"])
    development = pipe.named_steps["prep"].transform(xdev)
    test = pipe.named_steps["prep"].transform(xtest)
    view = CategoricalView(pipe.named_steps["prep"], development)
    for matrix in [development, test]:
        np.testing.assert_array_equal(view.decode(view.encode(matrix)), matrix)
    transformed_shap = shap.TreeExplainer(pipe.named_steps["clf"])(test).values
    grouped_shap = view.aggregate_shap(transformed_shap)
    np.testing.assert_allclose(grouped_shap.sum(axis=1),
                               transformed_shap.astype(np.float64).sum(axis=1), atol=1e-10)
    explainer = build_explainer(view, development, xai.readable)
    encoded_test = view.encode(test)
    details, concordance, fidelity = [], [], []

    def predict(samples):
        return pipe.named_steps["clf"].predict_proba(view.decode(samples))

    for case, index in cases.items():
        explanation = explainer.explain_instance(
            encoded_test[index], predict, num_features=15, num_samples=2000, labels=(1,))
        weights = explanation.as_map()[1]
        lime_top = [view.names[i] for i, _ in weights[:10]]
        shap_top = [view.names[i] for i in np.argsort(np.abs(grouped_shap[index]))[::-1][:10]]
        overlap = set(lime_top) & set(shap_top)
        concordance.append({"case": case, "overlap_count": len(overlap),
                            "shap_top_k": ", ".join(xai.readable(f) for f in shap_top),
                            "lime_top_k": ", ".join(xai.readable(f) for f in lime_top),
                            "overlap_features": ", ".join(sorted(xai.readable(f) for f in overlap))})
        fidelity.append({"case": case, "local_weighted_r2": explanation.score,
                         "model_probability": float(probability[index]),
                         "surrogate_probability": float(explanation.local_pred[0]),
                         "absolute_probability_error": float(abs(explanation.local_pred[0] - probability[index]))})
        for position, (feature, weight) in enumerate(weights):
            details.append({"case": case, "rank": position + 1,
                            "feature": view.names[feature], "weight": weight})
        figure = explanation.as_pyplot_figure(label=1)
        xai.recolour_directional_bars(figure)
        figure.set_size_inches(9, 6)
        figure.suptitle(f"Categorical-aware LIME: {case} case", fontsize=11, fontweight="bold")
        figure.tight_layout()
        for extension in ["pdf", "png"]:
            figure.savefig(args.output / f"lime_local_{case}.{extension}", dpi=300, bbox_inches="tight")
        plt.close(figure)
        print(case, fidelity[-1], "overlap", len(overlap), flush=True)
    pd.DataFrame(concordance).to_csv(args.output / "s1_lime_concordance.csv", index=False)
    pd.DataFrame(fidelity).to_csv(args.output / "s1_lime_fidelity.csv", index=False)
    private = ROOT / "01_data/interim/reviewer_revision"
    private.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(details).to_csv(private / "s1_lime_weights.csv", index=False)
    metadata = {"seed":42, "neighbours_per_case":2000, "n_original_variables":len(view.names),
                "n_transformed_columns":view.width, "categorical_variables":[view.names[i] for i in view.categorical_indices],
                "exact_encode_decode":True, "original_test_predictions_reproduced":True,
                "test_average_precision":average_precision_score(ytest,probability),
                "concordance_unit":"Original variables; one-hot SHAP contributions summed before absolute ranking.",
                "source_prediction_sha256":hashlib.sha256((ROOT / "04_outputs/tables/step8_xai/test_predictions.csv").read_bytes()).hexdigest(),
                "limitation":"Valid categorical groups do not ensure all perturbed clinical combinations are plausible."}
    (args.output / "s1_lime_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
