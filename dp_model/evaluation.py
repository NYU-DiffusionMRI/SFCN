from . import dp_utils

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import ants

# ----- config -----
BIN_RANGE = [42, 82]
BIN_STEP = 1
# get bin centers exactly like training (40 bins: 42.5, ..., 81.5)
_, BIN_CENTERS = dp_utils.num2vect(0.0, BIN_RANGE, BIN_STEP, sigma=0)  # ignore the index; we want centers
BIN_CENTERS = np.asarray(BIN_CENTERS, dtype=np.float32)  # (40,)
CROP_SHAPE = (160, 192, 160)


class BrainAgeDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, str]:
        row = self.df.iloc[idx]
        vol = ants.image_read(str(row["preproc_image"])).numpy().astype(np.float32)
        vol = vol / (vol.mean() + 1e-8)
        # vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
        vol = dp_utils.crop_center(vol, CROP_SHAPE)
        vol = vol[None, ...]  # (1, D, H, W)
        age = np.float32(row["age"])

        return torch.from_numpy(vol), torch.tensor(age, dtype=torch.float32), row["scan_id"]


@torch.no_grad()
def predict_and_eval(model, df: pd.DataFrame, device="cuda", batch_size=4, num_workers=2):
    model = model.to(device).eval()

    dataset = BrainAgeDataset(df)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    bc_t = torch.tensor(BIN_CENTERS, dtype=torch.float32, device=device)  # (40,)
    preds, trues, ids = [], [], []

    for x, y, subj_ids in dataloader:
        x = x.to(device, non_blocking=True) # (B,1,160,192,160)
        out = model(x)[0]   # (B, 40, 1, 1, 1)
        out = out.flatten(1) # (B, 40)
        assert out.size(1) == bc_t.numel()


        # out is logp-probs (log softmax)
        probs = out.exp()                                    # (B, 40)
        ages = (probs * bc_t).sum(dim=1)                     # (B,)

        preds.extend(ages.detach().cpu().numpy().tolist())
        trues.extend(y.detach().cpu().numpy().tolist())
        ids.extend(list(subj_ids))

    preds = np.asarray(preds, dtype=np.float32).reshape(-1) # (N,)
    trues = np.asarray(trues, dtype=np.float32).reshape(-1) # (N,)
    mae = np.mean(np.abs(preds - trues))

    return pd.DataFrame({"IXI_ID": ids, "true_age": trues, "pred_age": preds}), mae
