from . import dp_utils
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import ants

# ----- config -----
BIN_RANGE = [42, 82]
BIN_STEP = 1
# get bin centers exactly like training (40 bins: 42.5, ..., 81.5)
_, BIN_CENTERS = dp_utils.num2vect(0.0, BIN_RANGE, BIN_STEP, sigma=0)  # ignore the index; we want centers
BIN_CENTERS = np.asarray(BIN_CENTERS, dtype=np.float32)  # (40,)
CROP_SHAPE = (160, 192, 160)


class T1wDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        root_dirs: dict[str, str],
        crop_shape: tuple[int, int, int] = (160, 192, 160)
    ):
        """
        Dataset for T1-weighted MRI brain images from multiple datasets.

        Args:
            df: DataFrame with columns:
                - MR_ID: unique identifier for the scan
                - age: age of the subject
                - T1w_path: path relative to the dataset root directory
                - dataset: dataset name (e.g., 'IXI', 'OASIS_3')
            root_dirs: Dictionary mapping dataset name to its root directory path (as string).
                      Example: {'IXI': '/data/IXI', 'OASIS_3': '/data/OASIS_3'}
            crop_shape: Crop shape for preprocessing. Default (160, 192, 160)
                       matches the original SFCN training.
        """
        self.df = df
        # Convert string paths to Path objects
        self.root_dirs = {k: Path(v) for k, v in root_dirs.items()}
        self.crop_shape = crop_shape

        # Validate that all datasets in df have corresponding root_dirs
        datasets_in_df = set(df['dataset'].unique())
        missing = datasets_in_df - set(root_dirs.keys())
        if missing:
            raise ValueError(f"Missing root directories for datasets: {missing}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, str]:
        row = self.df.iloc[idx]

        # Get the appropriate root directory based on dataset
        root_dir = self.root_dirs[row['dataset']]
        t1w_path = root_dir / row["T1w_path"]

        vol = ants.image_read(str(t1w_path)).numpy().astype(np.float32)
        vol = vol / (vol.mean() + 1e-8)
        # vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
        vol = dp_utils.crop_center(vol, self.crop_shape)
        vol = vol[None, ...]  # (1, D, H, W)
        age = np.float32(row["age"])

        return torch.from_numpy(vol), torch.tensor(age, dtype=torch.float32), row["MR_ID"]
