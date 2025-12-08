from pathlib import Path
from argparse import ArgumentParser

import pandas as pd
from sklearn.model_selection import train_test_split


UNIFIED_DATA_CSV = Path('data/unified_data.csv')

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42


def split_data(csv_file: Path, dataset_name: str) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    df = df[['MR_ID', 'subject_ID', 'age', 'T1w_path']]

    # drop duplicates of the subject_ID
    unique_subjects = df['subject_ID'].drop_duplicates()

    train_subj, temp_subj = train_test_split(
        unique_subjects, test_size=(VAL_RATIO + TEST_RATIO),
        random_state=RANDOM_SEED
    )
    val_subj, test_subj = train_test_split(
        temp_subj, test_size=(TEST_RATIO / (VAL_RATIO + TEST_RATIO)),
        random_state=RANDOM_SEED
    )
    train_subj = set(train_subj)
    val_subj = set(val_subj)
    test_subj = set(test_subj)


    def assign_split(row):
        sid = row["subject_ID"]
        if sid in train_subj: return "train"
        if sid in val_subj:   return "val"
        if sid in test_subj:  return "test"
        return "unused"


    df["split"] = df.apply(assign_split, axis=1)
    df["dataset"] = dataset_name

    return df


def main():
    parser = ArgumentParser(description='Split dataset into train, validation and test sets')
    parser.add_argument('--input-csv', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--dataset-name', type=str, required=True, help='Name of the dataset')
    args = parser.parse_args()

    in_csv = Path(args.input_csv)
    df = split_data(in_csv, args.dataset_name)

    # append the split to the unified data and save
    if UNIFIED_DATA_CSV.exists():
        unified_data = pd.read_csv(UNIFIED_DATA_CSV)
        unified_data = pd.concat([unified_data, df], ignore_index=True)
    else:
        unified_data = df

    unified_data.to_csv(UNIFIED_DATA_CSV, index=False)


if __name__ == "__main__":
    main()
