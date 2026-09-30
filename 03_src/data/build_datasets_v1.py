"""Build the intermediate v1 datasets needed to reproduce the fixed splits.

This extracts the dataset assembly from the development notebook. The v1
feature-table ordering must be retained because the seeded split uses row
positions. Primary model development subsequently uses the v2 predictors.
"""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    processed = ROOT / '01_data' / 'processed'
    features = pd.read_csv(processed / 'features_baseline_v1.csv')
    interim = ROOT / '01_data' / 'interim'
    original_targets = interim / 'baseline_with_targets_step3_nolabels.csv'
    target_path = original_targets if original_targets.exists() else (
        interim / 'baseline_with_targets_step3_nolabelsv2.csv'
    )
    targets = pd.read_csv(target_path)
    print(f'Split-preparation targets: {target_path.name}')
    slope_columns = ['subject_id', 'slope_90d_per_30d', 'slope_180d_per_30d']
    combined = features.merge(
        targets[slope_columns], on='subject_id', how='inner',
        sort=False, validate='one_to_one',
    )
    for horizon, days in [('3m', 90), ('6m', 180)]:
        dataset = combined.dropna(subset=[f'slope_{days}d_per_30d'])
        destination = processed / f'dataset_{horizon}_v1.csv'
        dataset.to_csv(destination, index=False)
        print(f'Saved {len(dataset):,} {horizon} records to {destination}')


if __name__ == '__main__':
    main()
