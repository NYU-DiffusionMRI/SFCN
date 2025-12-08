from pathlib import Path
from typing import Optional

import pandas as pd

# final dataframe always includes the following columns:
# - subject_ID
# - MR_ID
# - age
# - T1w_path (relative to the raw data directory)



def map_scan_path(subject_ID: str, raw_dir: Path) -> Optional[Path]:
    pattern = f'{subject_ID}*.nii.gz'
    matching_files = list(raw_dir.glob(pattern))

    if len(matching_files) != 1:
        print(f"Found {len(matching_files)} files for subject {subject_ID}. Setting to None.")
        return None
    else:
        scan_path = matching_files[0]
        return scan_path.relative_to(raw_dir)


def main():
    demo_csv = Path('data/IXI/IXI_Demographic.csv')
    out_file = Path('data/IXI/IXI_cleaned.csv')
    assert not out_file.exists(), f"File {out_file} already exists"

    demo = pd.read_csv(demo_csv)

    demo.rename(columns={'IXI_ID': 'subject_ID', 'AGE': 'age'}, inplace=True)

    # 1. drop age with null values
    demo_clean = demo.dropna(subset=['age'])

    # 2. drop duplicate subject_ID based on age consistency:
    # - If duplicate IXI_IDs have the same age: keep first occurrence
    # - If duplicate IXI_IDs have different ages: remove all
    # - Non-duplicate IXI_IDs are kept
    grp = demo_clean.groupby('subject_ID')
    cnt = grp['age'].transform('size')
    same_age = grp['age'].transform('nunique') == 1
    # ensured in the analysis notebook that the first row is the valid one for duplicate subject_ID
    first_of_id = ~demo_clean.duplicated('subject_ID', keep='first')
    keep_mask = (cnt == 1) | (same_age & first_of_id)
    demo_clean = demo_clean.loc[keep_mask].copy()

    # 3. update subject_ID prepending 'IXI' + 3 digits string
    demo_clean['subject_ID'] = 'IXI' + demo_clean['subject_ID'].astype(str).str.zfill(3)

    # 4. add MR_ID column (same as subject_ID since one scan for eacch subject for IXI)
    demo_clean['MR_ID'] = demo_clean['subject_ID']

    # 5. add T1w_path column (relative to the raw data directory)
    raw_dir = Path('data/IXI/raw')
    demo_clean['T1w_path'] = demo_clean['subject_ID'].apply(lambda id: map_scan_path(id, raw_dir))
    demo_clean = demo_clean[demo_clean['T1w_path'].notna()]

    demo_clean.to_csv(out_file, index=False)
    print(f"Saved {len(demo_clean)} rows to {out_file}")


if __name__ == "__main__":
    main()
