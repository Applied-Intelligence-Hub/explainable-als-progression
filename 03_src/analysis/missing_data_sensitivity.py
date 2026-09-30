"""Focused missing-data sensitivity analysis for the selected Study I model.

The tuned unbalanced XGBoost configuration is kept fixed while the numeric
missing-data treatment changes. The analysis compares the primary fold-fitted
median imputation, median imputation with missingness indicators, and five
stochastic iterative imputations whose model probabilities are averaged.
Categorical predictors retain fold-fitted most-frequent imputation and one-hot
encoding. This is a post-hoc sensitivity analysis, not a new model search.
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.compose import ColumnTransformer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(ROOT, '01_data', 'processed')
TUNING_DIR = os.path.join(ROOT, '04_outputs', 'tables', 'step5_tuning')
OUT_DIR = os.path.join(ROOT, '04_outputs', 'tables', 'missing_data_sensitivity')
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
N_SPLITS = 5
N_IMPUTATIONS = 5
RAPID_FRAC = 0.30
SLOPE_COL = 'slope_180d_per_30d'

DROP_COLS = {
    'subject_id',
    't0_delta_days',
    'vitals_delta_days',
    'fvc_delta_days',
    'slope_90d_per_30d',
    'slope_180d_per_30d',
    'ALSFRS_Responded_By',
}


def load_data():
    data = pd.read_csv(os.path.join(DATA_DIR, 'dataset_6m_v2.csv'))
    split = pd.read_csv(os.path.join(DATA_DIR, 'holdout_split_6m.csv'))
    data[SLOPE_COL] = pd.to_numeric(data[SLOPE_COL], errors='coerce')
    data = data.dropna(subset=[SLOPE_COL]).copy()

    dev_ids = set(split.loc[split['partition'] == 'dev', 'subject_id'])
    test_ids = set(split.loc[split['partition'] == 'test', 'subject_id'])
    dev = data[data['subject_id'].isin(dev_ids)].reset_index(drop=True)
    test = data[data['subject_id'].isin(test_ids)].reset_index(drop=True)
    features = [column for column in data.columns if column not in DROP_COLS]
    return dev, test, features


def load_params():
    path = os.path.join(TUNING_DIR, 'XGB_unbalanced_6m_best.json')
    with open(path, encoding='utf-8') as handle:
        params = json.load(handle)['full_params']
    return params


def make_preprocessor(X_train, strategy, seed):
    numeric = X_train.select_dtypes(include=[np.number, 'bool']).columns.tolist()
    categorical = [column for column in X_train.columns if column not in numeric]

    if strategy == 'median':
        numeric_imputer = SimpleImputer(strategy='median')
    elif strategy == 'median_indicator':
        numeric_imputer = SimpleImputer(
            strategy='median', add_indicator=True)
    elif strategy == 'iterative':
        lower = X_train[numeric].min(skipna=True).fillna(0).to_numpy()
        upper = X_train[numeric].max(skipna=True).fillna(0).to_numpy()
        upper = np.where(upper <= lower, lower + 1e-12, upper)
        numeric_imputer = IterativeImputer(
            max_iter=10,
            sample_posterior=True,
            random_state=seed,
            skip_complete=True,
            keep_empty_features=True,
            min_value=lower,
            max_value=upper,
        )
    else:
        raise ValueError(f'Unknown strategy: {strategy}')

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(
            strategy='most_frequent')),
        ('encoder', OneHotEncoder(
            handle_unknown='ignore', sparse_output=False)),
    ])
    return ColumnTransformer([
        ('numeric', numeric_imputer, numeric),
        ('categorical', categorical_pipeline, categorical),
    ])


def fit_predict(X_train, y_train, X_valid, strategy, params):
    seeds = range(SEED, SEED + N_IMPUTATIONS) if strategy == 'iterative' else [SEED]
    probabilities = []
    for seed in seeds:
        preprocessor = make_preprocessor(X_train, strategy, seed)
        model = XGBClassifier(**params)
        pipeline = Pipeline([
            ('preprocessing', preprocessor),
            ('classifier', model),
        ])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pipeline.fit(X_train, y_train)
        probabilities.append(pipeline.predict_proba(X_valid)[:, 1])
    return np.mean(probabilities, axis=0)


def evaluate_cv(dev, features, strategy, params):
    splitter = GroupKFold(n_splits=N_SPLITS)
    X = dev[features]
    slopes = dev[SLOPE_COL]
    groups = dev['subject_id']
    pr_auc, roc_auc = [], []

    for fold, (train_idx, valid_idx) in enumerate(
            splitter.split(X, groups=groups), start=1):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        train_slopes = slopes.iloc[train_idx]
        valid_slopes = slopes.iloc[valid_idx]
        cutoff = float(train_slopes.quantile(RAPID_FRAC))
        y_train = (train_slopes <= cutoff).astype(int)
        y_valid = (valid_slopes <= cutoff).astype(int)
        probability = fit_predict(
            X_train, y_train, X_valid, strategy, params)
        pr_auc.append(average_precision_score(y_valid, probability))
        roc_auc.append(roc_auc_score(y_valid, probability))
        print(f'{strategy}: fold {fold}/{N_SPLITS} complete')

    return pr_auc, roc_auc


def evaluate_test(dev, test, features, strategy, params):
    cutoff = float(dev[SLOPE_COL].quantile(RAPID_FRAC))
    y_dev = (dev[SLOPE_COL] <= cutoff).astype(int)
    y_test = (test[SLOPE_COL] <= cutoff).astype(int)
    probability = fit_predict(
        dev[features], y_dev, test[features], strategy, params)
    return (
        average_precision_score(y_test, probability),
        roc_auc_score(y_test, probability),
    )


def main():
    dev, test, features = load_data()
    params = load_params()
    labels = {
        'median': 'Median imputation (primary)',
        'median_indicator': 'Median imputation + missingness indicators',
        'iterative': f'Iterative multiple imputation (m={N_IMPUTATIONS})',
    }

    rows = []
    for strategy, label in labels.items():
        cv_pr, cv_roc = evaluate_cv(dev, features, strategy, params)
        test_pr, test_roc = evaluate_test(
            dev, test, features, strategy, params)
        rows.append({
            'strategy': label,
            'dev_cv_pr_auc_mean': np.mean(cv_pr),
            'dev_cv_pr_auc_std': np.std(cv_pr),
            'dev_cv_roc_auc_mean': np.mean(cv_roc),
            'dev_cv_roc_auc_std': np.std(cv_roc),
            'test_pr_auc': test_pr,
            'test_roc_auc': test_roc,
            'n_imputations': N_IMPUTATIONS if strategy == 'iterative' else 1,
        })

    results = pd.DataFrame(rows)
    path = os.path.join(OUT_DIR, 'study1_missing_data_sensitivity.csv')
    results.to_csv(path, index=False)
    print()
    print(results.to_string(index=False))
    print(f'\nSaved: {path}')


if __name__ == '__main__':
    main()
