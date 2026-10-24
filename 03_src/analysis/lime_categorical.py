"""LIME input view with valid categorical groups and an exact inverse mapping."""

from dataclasses import dataclass

import numpy as np
from lime.lime_tabular import LimeTabularExplainer


ORDINAL_FIELDS = {"R_1_Dyspnea", "R_2_Orthopnea", "R_3_Respiratory_Insufficiency"}


@dataclass
class FeatureGroup:
    name: str
    indices: list[int]
    kind: str
    domain: np.ndarray | None = None
    labels: list[str] | None = None


class CategoricalView:
    """Collapse one-hot groups for sampling, then reconstruct classifier inputs."""

    def __init__(self, preprocessor, processed_development):
        self.width = processed_development.shape[1]
        self.groups = []
        for name, transformer, columns in preprocessor.transformers_:
            if name == "remainder":
                continue
            position = preprocessor.output_indices_[name].start
            if name == "num":
                assert preprocessor.output_indices_[name].stop - position == len(columns)
                for offset, column in enumerate(columns):
                    index = position + offset
                    domain = np.unique(processed_development[:, index])
                    discrete = len(domain) <= 2 or column in ORDINAL_FIELDS
                    self.groups.append(FeatureGroup(
                        column, [index], "numeric_category" if discrete else "continuous",
                        domain if discrete else None,
                        [f"{value:g}" for value in domain] if discrete else None))
            elif name == "cat":
                encoder = transformer.named_steps.get("oh")
                if encoder is None:
                    encoder = transformer.named_steps["onehot"]
                for column, categories in zip(columns, encoder.categories_):
                    count = len(categories)
                    self.groups.append(FeatureGroup(
                        column, list(range(position, position + count)), "onehot",
                        np.arange(count + 1), [str(value) for value in categories] + ["unseen category"]))
                    position += count
            else:
                raise ValueError(f"Unsupported preprocessing block: {name}")
        assert sorted(i for group in self.groups for i in group.indices) == list(range(self.width))

    @property
    def names(self):
        return [group.name for group in self.groups]

    @property
    def categorical_indices(self):
        return [i for i, group in enumerate(self.groups) if group.kind != "continuous"]

    @property
    def categorical_names(self):
        return {i: group.labels for i, group in enumerate(self.groups) if group.labels is not None}

    def encode(self, processed):
        encoded = np.empty((len(processed), len(self.groups)))
        for column, group in enumerate(self.groups):
            values = processed[:, group.indices]
            if group.kind == "continuous":
                encoded[:, column] = values[:, 0]
            elif group.kind == "numeric_category":
                matches = values[:, 0, None] == group.domain[None, :]
                if not matches.any(axis=1).all():
                    raise ValueError(f"Unseen numeric category in {group.name}")
                encoded[:, column] = matches.argmax(axis=1)
            else:
                if not np.isin(values, [0, 1]).all() or not np.isin(values.sum(axis=1), [0, 1]).all():
                    raise ValueError(f"Invalid source one-hot group: {group.name}")
                encoded[:, column] = np.where(values.sum(axis=1) == 0,
                                              len(group.indices), values.argmax(axis=1))
        return encoded

    def decode(self, encoded):
        processed = np.zeros((len(encoded), self.width))
        for column, group in enumerate(self.groups):
            values = encoded[:, column]
            if group.kind == "continuous":
                processed[:, group.indices[0]] = values
                continue
            codes = np.rint(values).astype(int)
            if not np.array_equal(codes, values) or (codes < 0).any() or (codes >= len(group.domain)).any():
                raise ValueError(f"Invalid categorical perturbation: {group.name}")
            if group.kind == "numeric_category":
                processed[:, group.indices[0]] = group.domain[codes]
            else:
                known = codes < len(group.indices)
                processed[np.flatnonzero(known), np.array(group.indices)[codes[known]]] = 1
        return processed

    def aggregate_shap(self, transformed_shap):
        transformed_shap = np.asarray(transformed_shap, dtype=np.float64)
        return np.column_stack([transformed_shap[:, group.indices].sum(axis=1)
                                for group in self.groups])


def build_explainer(view, processed_development, readable, seed=42):
    return LimeTabularExplainer(
        training_data=view.encode(processed_development),
        feature_names=[readable(name) for name in view.names],
        categorical_features=view.categorical_indices,
        categorical_names=view.categorical_names,
        class_names=["Slow", "Rapid"], mode="classification",
        random_state=seed, discretize_continuous=True,
    )
