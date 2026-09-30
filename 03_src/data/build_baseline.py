"""Build the Study I baseline table from the authorised PRO-ACT extract.

Baseline is the earliest assessment with a valid ALSFRS-R total and date.
The first demographic record supplies the recorded demographic attributes.
Stable sorting preserves the source ordering when assessment dates tie.
"""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CLINICAL_COLUMNS = [
    'subject_id', 'ALSFRS_Delta', 'ALSFRS_R_Total',
    'R_1_Dyspnea', 'R_2_Orthopnea', 'R_3_Respiratory_Insufficiency',
    'Mode_of_Administration', 'ALSFRS_Responded_By',
]
DEMOGRAPHIC_COLUMNS = [
    'subject_id', 'Age', 'Sex', 'Ethnicity', 'Race_Caucasian',
    'Race_Black_African_American', 'Race_Asian',
    'Race_Americ_Indian_Alaska_Native', 'Race_Hawaiian_Pacific_Islander',
    'Race_Other', 'Race_Unknown',
]
OUTPUT_COLUMNS = [
    'subject_id', 't0_delta_days', 'ALSFRS_R_t0',
    *DEMOGRAPHIC_COLUMNS[1:], *CLINICAL_COLUMNS[3:],
]


def build_baseline(raw_directory: Path) -> pd.DataFrame:
    als = pd.read_csv(raw_directory / 'PROACT_ALSFRS.csv')
    demographics = pd.read_csv(raw_directory / 'PROACT_DEMOGRAPHICS.csv')
    valid = als.dropna(subset=['ALSFRS_R_Total', 'ALSFRS_Delta'])
    first = valid.sort_values(
        ['subject_id', 'ALSFRS_Delta'], kind='stable',
    ).drop_duplicates('subject_id', keep='first')
    baseline = first[CLINICAL_COLUMNS].rename(columns={
        'ALSFRS_Delta': 't0_delta_days', 'ALSFRS_R_Total': 'ALSFRS_R_t0',
    })
    baseline = baseline.merge(
        demographics[DEMOGRAPHIC_COLUMNS].drop_duplicates('subject_id'),
        on='subject_id', how='left', sort=False, validate='one_to_one',
    )
    return baseline[OUTPUT_COLUMNS]


def main() -> None:
    baseline = build_baseline(ROOT / '01_data' / 'raw')
    destination = ROOT / '01_data' / 'interim' / 'baseline_table_ALSFRS_R.csv'
    destination.parent.mkdir(parents=True, exist_ok=True)
    baseline.to_csv(destination, index=False)
    print(f'Saved {len(baseline):,} baseline records to {destination}')


if __name__ == '__main__':
    main()
