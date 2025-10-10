from pathlib import Path
from typing import List, Optional

import pandas as pd


def sample(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=['AGE'])
    df = df[(44 <= df['AGE']) & (df['AGE'] <= 80)]

    return df

def add_image_paths(df: pd.DataFrame, image_dir: Path) -> Optional[Path]:
    def get_image_path(IXI_ID: int, image_dir: Path):
        pattern = f'IXI{IXI_ID:03d}*.nii.gz'
        matching_files = list(image_dir.glob(pattern))

        if len(matching_files) != 1:
            print(f"Found {len(matching_files)} files for IXI_ID {IXI_ID}. Setting to None.")
            return None
        else:
            return matching_files[0]

    df["IMAGE_PATH"] = df["IXI_ID"].apply(lambda id: get_image_path(id, image_dir))
    df = df[df["IMAGE_PATH"].notna()]
    print(f"Found {len(df)} rows with image paths")
    return df


def drop_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate IXI_ID rows based on AGE consistency:
    - If duplicate IXI_IDs have the same AGE: keep first occurrence
    - If duplicate IXI_IDs have different AGEs: remove all
    - Non-duplicate IXI_IDs are kept
    Notes:
    - AGE NaNs are ignored in the uniqueness check (same as pandas' default nunique(dropna=True)).
    """
    grp = df.groupby('IXI_ID')

    # How many rows per IXI_ID
    cnt = grp['AGE'].transform('size')

    # True if all non-null AGE values within IXI_ID are the same
    same_age = grp['AGE'].transform('nunique') == 1

    # Marks the first row for each IXI_ID
    first_of_id = ~df.duplicated('IXI_ID', keep='first')

    # Keep:
    # - all unique IXI_IDs (cnt == 1)
    # - for duplicate IXI_IDs where ages agree, keep only the first row
    keep_mask = (cnt == 1) | (same_age & first_of_id)

    return df.loc[keep_mask].copy()


def create_sampled_symlinks(files: List[Path], target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for file in files:
        symlink_path = target_dir / file.name
        if symlink_path.exists():
            print(f"ℹ️ Skipping {symlink_path} — already exists")
            continue

        try:
            symlink_path.symlink_to(file.resolve())
            print(f"✅ Created symlink: {symlink_path} → {file}")
        except OSError as e:
            print(f"❌ Failed to create symlink: {symlink_path} → {file}: {e}")


def main():
    demographic_csv = Path('data/IXI/IXI_Demographic.csv')
    df = pd.read_csv(demographic_csv)
    df = sample(df)

    image_dir = Path('data/IXI/T1/raw')
    df = add_image_paths(df, image_dir)

    df = df[['IXI_ID', 'AGE', 'IMAGE_PATH']]    # keep only the columns we need

    df = drop_duplicate_rows(df)

    out_file = Path('data/IXI/sampled_age_44_80_no_duplicates.csv')
    df.to_csv(out_file, index=False)
    print(f"Saved {len(df)} rows to {out_file}")

    sampled_dir = Path('data/IXI/T1/sampled_age_44_80_no_duplicates')
    create_sampled_symlinks(list(df["IMAGE_PATH"]), sampled_dir)
    print(f"Created {len(df)} symlinks in {sampled_dir}")

if __name__ == "__main__":
    main()