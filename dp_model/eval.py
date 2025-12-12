from .datasets import T1wDataset, BIN_CENTERS

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr


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


def bias_correct(cal_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, float, float, tuple[float, float]]:
    """
    Apply bias correction using calibration set.

    Args:
        cal_df: Calibration DataFrame with 'true_age' and 'pred_age' columns
        test_df: Test DataFrame with 'true_age' and 'pred_age' columns

    Returns:
        Tuple of (corrected_test_df, mae_before, mae_after, (a, b))
        where corrected_test_df has 'pred_age_corrected' column added
    """
    # Fit linear model on calibration: pred = a * true + b
    a, b = np.polyfit(cal_df['true_age'], cal_df['pred_age'], 1)

    # Apply correction on test: corrected = (pred - b) / a
    test_df = test_df.copy()
    test_df['pred_age_corrected'] = (test_df['pred_age'] - b) / a

    # Compute MAEs
    mae_before = np.abs(test_df['pred_age'] - test_df['true_age']).mean()
    mae_after = np.abs(test_df['pred_age_corrected'] - test_df['true_age']).mean()

    return test_df, mae_before, mae_after, (a, b)


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean absolute error."""
    return np.mean(np.abs(y_true - y_pred))


def compute_pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Pearson correlation coefficient."""
    return pearsonr(y_true, y_pred)[0]


def compute_spearman_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Spearman correlation coefficient."""
    return spearmanr(y_true, y_pred)[0]


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute coefficient of determination (R^2)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)
