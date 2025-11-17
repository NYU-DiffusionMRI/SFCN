from pathlib import Path
from typing import List, Optional

import pandas as pd


def sample(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=['Age'])
    df = df[(44 <= df['Age']) & (df['Age'] <= 80)]

    return df

def add_image_paths(df: pd.DataFrame, image_dir: Path) -> Optional[Path]:
    def get_image_path(subject_id: str, image_dir: Path):
        file = image_dir / f'{subject_id}/anat/NIfTI/mprage.nii.gz'
        return file

    df["Image Path"] = df["Subject"].apply(lambda id: get_image_path(id, image_dir))
    df = df[df["Image Path"].notna()]
    print(f"Found {len(df)} rows with image paths")
    return df


def create_sampled_symlinks(subjects: List[str], image_paths: List[Path], target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for subject_id, file in zip(subjects, image_paths):
        symlink_path = target_dir / f'{subject_id}_mprage.nii.gz'
        if symlink_path.exists():
            print(f"ℹ️ Skipping {symlink_path} — already exists")
            continue

        try:
            symlink_path.symlink_to(file.resolve())
            print(f"✅ Created symlink: {symlink_path} → {file}")
        except OSError as e:
            print(f"❌ Failed to create symlink: {symlink_path} → {file}: {e}")


def main():
    demographic_csv = Path('data/ABIDE_1/merged_metadata.csv')
    df = pd.read_csv(demographic_csv)
    df = sample(df)

    image_dir = Path('data/ABIDE_1/raw')
    df = add_image_paths(df, image_dir)

    df = df[['Subject', 'Age', 'Image Path']]    # keep only the columns we need

    out_file = Path('data/ABIDE_1/sampled_age_44_80.csv')
    df.to_csv(out_file, index=False)
    print(f"Saved {len(df)} rows to {out_file}")

    sampled_dir = Path('data/ABIDE_1/sampled_age_44_80')
    create_sampled_symlinks(list(df['Subject']), list(df['Image Path']), sampled_dir)
    print(f"Created {len(df)} symlinks in {sampled_dir}")

if __name__ == "__main__":
    main()
