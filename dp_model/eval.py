from .datasets import T1wDataset, BIN_CENTERS

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader


@torch.no_grad()
def predict_and_eval(
    model,
    df: pd.DataFrame,
    root_dirs: dict[str, str],
    device="cuda",
    batch_size=4,
    num_workers=2
):
    """
    Predict brain age and evaluate model performance.

    Args:
        model: Trained SFCN model
        df: DataFrame with scan information (must include 'dataset' column)
        root_dirs: Dictionary mapping dataset name to root directory path (as string)
        device: Device to run inference on
        batch_size: Batch size for inference
        num_workers: Number of worker processes for data loading

    Returns:
        Tuple of (predictions_df, mae)
    """
    model = model.to(device).eval()

    dataset = T1wDataset(df, root_dirs=root_dirs)
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

    return pd.DataFrame({"MR_ID": ids, "true_age": trues, "pred_age": preds}), mae
