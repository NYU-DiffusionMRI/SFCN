from pathlib import Path
from typing import Optional

import pandas as pd

# final dataframe always includes the following columns:
# - subject_ID
# - MR_ID
# - age
# - T1w_path (relative to the raw data directory)


def map_scan_path(mr_id: str, raw_dir: Path) -> Optional[Path]:
    """Map MR ID to the corresponding T1w scan path."""
    # OASIS-3 structure: ${raw_dir}/OAS3xxxx_MR_dxxxx/anat${d}/sub-OAS3xxxx_ses-dxxxx_run-xx_T1w.nii.gz
    # Note: there can be multiple anat directories (anat, anat2, anat3, etc.)
    # Always choose the last (highest numbered) anat directory
    subject_id = mr_id.split('_')[0]  # e.g., OAS30001
    session_dir = raw_dir / mr_id

    if not session_dir.exists():
        print(f"Session directory {session_dir} does not exist. Setting to None.")
        return None

    # Find all anat directories and choose the highest numbered one
    anat_dirs = [d for d in session_dir.iterdir() if d.is_dir() and d.name.startswith('anat')]
    if not anat_dirs:
        print(f"No anat directories found for {mr_id}. Setting to None.")
        return None

    anat_dir = max(anat_dirs, key=lambda d: int(d.name[4:]))

    # Find T1w files in this anat directory
    t1w_files = list(anat_dir.glob('*_T1w.nii.gz'))
    assert len(t1w_files) == 1, f"Found {len(t1w_files)} T1w files in {anat_dir} for MR ID {mr_id}"

    scan_path = t1w_files[0]
    return scan_path.relative_to(raw_dir)


def main():
    scan_csv = Path('data/OASIS_3/OASIS3_MR_scans.csv')
    clinical_csv = Path('data/OASIS_3/OASIS3_UDSd1_diagnoses.csv')
    out_file = Path('data/OASIS_3/OASIS3_cleaned.csv')
    assert not out_file.exists(), f"File {out_file} already exists"

    scan_df = pd.read_csv(scan_csv)
    clinical_df = pd.read_csv(clinical_csv)

    # 1. Extract base clinical visit data (days_to_visit == 0) for age calculation
    base_clinical_df = clinical_df[clinical_df['days_to_visit'] == 0]
    clinical_df_to_merge = base_clinical_df.loc[:, ['OASISID', 'age at visit']].copy()
    clinical_df_to_merge.rename(columns={'age at visit': 'base_age'}, inplace=True)

    # 2. Filter only T1-weighted scans
    t1_scan_df = scan_df.loc[scan_df['Scans'].notna() & scan_df['Scans'].str.contains('T1w')].copy()

    # 3. Extract days_to_visit from MR ID and calculate precise age
    parse_days_to_visit = lambda row: int(row['MR ID'].split('_')[-1][1:])
    t1_scan_df['days_to_visit'] = t1_scan_df.apply(parse_days_to_visit, axis=1)

    # Merge with base clinical data to get base age
    t1_scan_df = t1_scan_df.merge(clinical_df_to_merge, left_on='Subject', right_on='OASISID', how='inner')
    t1_scan_df = t1_scan_df.drop(columns=['OASISID'])

    # Calculate precise age from base age and days since baseline
    t1_scan_df['Age'] = (t1_scan_df['base_age'] + t1_scan_df['days_to_visit'] / 365).round(2)

    # Keep only relevant columns for merging
    t1_scan_df = t1_scan_df[['MR ID', 'Subject', 'Age', 'Scanner', 'Scans']]

    # 4. Map scans to nearest clinical visit (within 1 year tolerance) for cognitive status
    mr_sorted = t1_scan_df.sort_values(by=['Age', 'Subject']).reset_index(drop=True)
    udsd_sorted = clinical_df.sort_values(by=['age at visit', 'OASISID']).reset_index(drop=True)
    udsd_sorted = udsd_sorted[['OASISID', 'age at visit', 'NORMCOG']]

    matched = pd.merge_asof(
        mr_sorted,
        udsd_sorted,
        left_by="Subject",
        right_by="OASISID",
        left_on="Age",
        right_on="age at visit",
        direction="nearest",
        tolerance=1.0   # years
    )

    # 5. Keep only cognitively normal scans (NORMCOG == 1.0)
    matched = matched[matched['NORMCOG'] == 1.0]
    matched.drop(columns=['OASISID', 'age at visit', 'NORMCOG'], inplace=True)
    matched.sort_values(by=['Subject', 'Age'], inplace=True)

    # 6. Rename columns to standard format
    matched.rename(columns={
        'Subject': 'subject_ID',
        'MR ID': 'MR_ID',
        'Age': 'age'
    }, inplace=True)

    # 7. Add T1w_path column (relative to the raw data directory)
    raw_dir = Path('data/OASIS_3/raw')
    matched['T1w_path'] = matched['MR_ID'].apply(lambda id: map_scan_path(id, raw_dir))

    matched.to_csv(out_file, index=False)
    print(f"Saved {len(matched)} rows to {out_file}")


if __name__ == "__main__":
    main()