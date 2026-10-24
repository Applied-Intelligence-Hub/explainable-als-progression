"""Synthetic reviewer-regression fixtures; no study models are retrained.

Exported from the dissertation's test_reviewer_revisions.py.
"""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[2]
from race_encoding_sensitivity import RACE_COLUMNS, represent_race


class RaceRepresentationTests(unittest.TestCase):
    def fixture(self):
        frame = pd.DataFrame(np.nan, index=range(3), columns=RACE_COLUMNS)
        frame.loc[0, "Race_Caucasian"] = 1
        frame.loc[0, "Race_Other"] = 1
        frame.loc[1, "Race_Caucasian"] = 0
        frame.loc[1, "Race_Unknown"] = 1
        frame["Age"] = [40., 50., 60.]
        return frame

    def test_missing_zero_one_and_multiple_affirmatives_remain_distinct(self):
        source = self.fixture()
        result = represent_race(source, "explicit_recording")
        self.assertEqual(result.Race_Caucasian.tolist(),
                         ["recorded_one", "recorded_zero", "unrecorded"])
        self.assertEqual(result.loc[0, "Race_Other"], "recorded_one")
        self.assertEqual(result.loc[1, "Race_Unknown"], "recorded_one")
        self.assertEqual(result.loc[2, "Race_Unknown"], "unrecorded")
        self.assertFalse(result.isna().any().any())
        pd.testing.assert_frame_equal(source, self.fixture())

    def test_omission_removes_only_race(self):
        source = self.fixture()
        result = represent_race(source, "race_omitted")
        pd.testing.assert_frame_equal(result, source[["Age"]])
        pd.testing.assert_frame_equal(represent_race(source, "primary_median"), source)

    def test_invalid_values_are_not_silently_recoded(self):
        source = self.fixture()
        source.loc[0, "Race_Asian"] = 2
        with self.assertRaisesRegex(ValueError, "Unexpected race value"):
            represent_race(source, "explicit_recording")

class LimeRepresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from lime_categorical import CategoricalView
            cls.view_class = CategoricalView
        except ModuleNotFoundError as error:
            raise unittest.SkipTest(str(error))

    def test_exact_roundtrip_and_mutually_exclusive_categories(self):
        development = pd.DataFrame({"Age": [40., 50., 60.], "flag": [0., 1., 0.],
                                     "Sex": ["Female", "Male", "Female"]})
        prep = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), ["Age", "flag"]),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), ["Sex"]),
        ])
        processed = prep.fit_transform(development)
        view = self.view_class(prep, processed)
        np.testing.assert_array_equal(view.decode(view.encode(processed)), processed)
        other = development.iloc[[0]].assign(Sex="unseen")
        unknown = prep.transform(other)
        np.testing.assert_array_equal(view.decode(view.encode(unknown)), unknown)
        samples = np.tile(view.encode(processed)[0], (30, 1))
        rng = np.random.default_rng(42)
        for column in view.categorical_indices:
            samples[:, column] = rng.integers(0, len(view.groups[column].domain), len(samples))
        restored = view.decode(samples)
        self.assertTrue(np.isin(restored[:, 1], [0, 1]).all())
        self.assertTrue(np.isin(restored[:, 2:].sum(axis=1), [0, 1]).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
