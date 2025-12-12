#!/usr/bin/env python3
"""
Evaluation script for SFCN brain age prediction model.

This script evaluates the model on test splits from multiple datasets (IXI, OASIS_3, or both).
It loads a unified CSV file with dataset and split columns, filters by the specified datasets
and split, and computes evaluation metrics.

Usage:
    PYTHONPATH=. python scripts/evaluate.py --config configs/default.yaml --datasets IXI OASIS_3 --split test
    PYTHONPATH=. python scripts/evaluate.py --datasets IXI  # Evaluate on IXI test split only
    PYTHONPATH=. python scripts/evaluate.py --datasets OASIS_3  # Evaluate on OASIS_3 test split only
    PYTHONPATH=. python scripts/evaluate.py  # Evaluate on all datasets (IXI and OASIS_3) test split
"""

import argparse
import yaml
import pandas as pd
import torch
from pathlib import Path

from dp_model.model_files.sfcn import SFCN
from dp_model.eval import predict_and_eval, bias_correct, compute_mae, compute_pearson_corr, compute_spearman_corr, compute_r2


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_and_filter_data(
    csv_path: str,
    datasets: list[str] = None,
    split: str = "test"
) -> pd.DataFrame:
    """
    Load unified CSV and filter by dataset and split.

    Args:
        csv_path: Path to unified CSV file
        datasets: List of dataset names to include (e.g., ['IXI', 'OASIS_3'])
                 If None, includes all datasets
        split: Split to evaluate on (default: 'test')

    Returns:
        Filtered DataFrame
    """
    df = pd.read_csv(csv_path)

    # Filter by split
    df_filtered = df[df['split'] == split].copy()

    # Filter by datasets if specified
    if datasets is not None:
        df_filtered = df_filtered[df_filtered['dataset'].isin(datasets)]

    # SFCN age range
    sfcn_df = df_filtered[(df_filtered['age'] >= 44) & (df_filtered['age'] <= 80)]

    if len(sfcn_df) == 0:
        raise ValueError(f"No data found for split='{split}' and datasets={datasets}")

    return sfcn_df


def load_model(checkpoint_path: str, device: str = "cuda") -> torch.nn.Module:
    """
    Load SFCN model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint (.p or .pth file)
        device: Device to load model on

    Returns:
        Loaded model
    """
    model = SFCN()

    # Check if model was saved with DataParallel
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Handle DataParallel state dict
    if any(k.startswith('module.') for k in checkpoint.keys()):
        # Checkpoint was saved with DataParallel
        model = torch.nn.DataParallel(model)
        model.load_state_dict(checkpoint)
    else:
        # Checkpoint was saved without DataParallel
        model.load_state_dict(checkpoint)

    return model


def save_results(
    results_df: pd.DataFrame,
    mae: float,
    output_dir: str,
    datasets: list[str],
    split: str,
    config_info: dict = None,
    bias_correction: dict = None
):
    """
    Save evaluation results to files.

    Args:
        results_df: DataFrame with predictions
        mae: Mean absolute error
        output_dir: Directory to save results
        datasets: List of evaluated datasets
        split: Split that was evaluated
        config_info: Dictionary with configuration information to include in summary
        bias_correction: Dictionary with bias correction results (mae_before, mae_after, a, b, n_cal)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create filename based on datasets
    if datasets is None or len(datasets) == 0:
        dataset_str = "all"
    else:
        dataset_str = "_".join(sorted(datasets))

    # Save predictions CSV
    pred_file = output_path / f"predictions_{dataset_str}_{split}.csv"
    results_df.to_csv(pred_file, index=False)
    print(f"Saved predictions to: {pred_file}")

    # Save summary statistics
    summary_file = output_path / f"summary_{dataset_str}_{split}.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Evaluation Results\n")
        f.write(f"==================\n\n")

        # Write configuration information
        if config_info:
            f.write(f"Configuration:\n")
            f.write(f"--------------\n")
            for key, value in config_info.items():
                f.write(f"{key}: {value}\n")
            f.write(f"\n")

        f.write(f"Data:\n")
        f.write(f"-----\n")
        f.write(f"Datasets: {', '.join(datasets) if datasets else 'all'}\n")
        f.write(f"Split: {split}\n")
        f.write(f"Number of MR scans: {len(results_df)}\n\n")

        f.write(f"Results:\n")
        f.write(f"--------\n")

        # Bias correction results if available
        if bias_correction:
            f.write(f"MAE (before correction):                 {bias_correction['mae_before']:.4f} years\n")
            f.write(f"MAE (after correction):                  {bias_correction['mae_after']:.4f} years\n")
            f.write(f"R² (before correction):                  {bias_correction['r2_before']:.4f}\n")
            f.write(f"R² (after correction):                   {bias_correction['r2_after']:.4f}\n")
            f.write(f"Pearson r (before correction):           {bias_correction['pearson_before']:.4f}\n")
            f.write(f"Pearson r (after correction):            {bias_correction['pearson_after']:.4f}\n")
            f.write(f"Spearman ρ(Δ,age) (before):              {bias_correction['spearman_delta_before']:.4f}\n")
            f.write(f"Spearman ρ(Δ,age) (after):               {bias_correction['spearman_delta_after']:.4f}\n")
            f.write(f"Pearson r(Δ,age) (before):               {bias_correction['pearson_delta_before']:.4f}\n")
            f.write(f"Pearson r(Δ,age) (after):                {bias_correction['pearson_delta_after']:.4f}\n")
            f.write(f"  [Δ = predicted age - true age; measures age-related bias]\n")
            f.write(f"Bias fit: pred ≈ {bias_correction['a']:.3f} * true + {bias_correction['b']:.3f}\n")
            f.write(f"Calibration samples (MR scans): {bias_correction['n_cal']}\n\n")
        else:
            r2_overall = compute_r2(results_df['true_age'].values, results_df['pred_age'].values)
            f.write(f"Mean Absolute Error (MAE): {mae:.4f} years\n")
            f.write(f"Coefficient of Determination (R²): {r2_overall:.4f}\n\n")

        # Per-dataset statistics if multiple datasets
        if 'dataset' in results_df.columns:
            f.write(f"Per-Dataset Statistics:\n")
            f.write(f"-----------------------\n")
            for dataset in results_df['dataset'].unique():
                df_ds = results_df[results_df['dataset'] == dataset]
                if 'pred_age_corrected' in df_ds.columns:
                    mae_before = compute_mae(df_ds['true_age'].values, df_ds['pred_age'].values)
                    mae_after = compute_mae(df_ds['true_age'].values, df_ds['pred_age_corrected'].values)
                    r2_before = compute_r2(df_ds['true_age'].values, df_ds['pred_age'].values)
                    r2_after = compute_r2(df_ds['true_age'].values, df_ds['pred_age_corrected'].values)

                    delta_before = df_ds['pred_age'].values - df_ds['true_age'].values
                    spearman_delta_before = compute_spearman_corr(df_ds['true_age'].values, delta_before)
                    pearson_delta_before = compute_pearson_corr(df_ds['true_age'].values, delta_before)
                    pearson_before = compute_pearson_corr(df_ds['true_age'].values, df_ds['pred_age'].values)

                    delta_after = df_ds['pred_age_corrected'].values - df_ds['true_age'].values
                    spearman_delta_after = compute_spearman_corr(df_ds['true_age'].values, delta_after)
                    pearson_delta_after = compute_pearson_corr(df_ds['true_age'].values, delta_after)
                    pearson_after = compute_pearson_corr(df_ds['true_age'].values, df_ds['pred_age_corrected'].values)

                    f.write(f"{dataset}:\n")
                    f.write(f"  MAE: {mae_before:.4f} → {mae_after:.4f} years\n")
                    f.write(f"  R²: {r2_before:.4f} → {r2_after:.4f}\n")
                    f.write(f"  Pearson r: {pearson_before:.4f} → {pearson_after:.4f}\n")
                    f.write(f"  Spearman ρ(Δ,age): {spearman_delta_before:.4f} → {spearman_delta_after:.4f}\n")
                    f.write(f"  Pearson r(Δ,age): {pearson_delta_before:.4f} → {pearson_delta_after:.4f}\n")
                    f.write(f"  (n={len(df_ds)})\n\n")
                else:
                    mae_ds = compute_mae(df_ds['true_age'].values, df_ds['pred_age'].values)
                    r2_ds = compute_r2(df_ds['true_age'].values, df_ds['pred_age'].values)
                    delta_ds = df_ds['pred_age'].values - df_ds['true_age'].values
                    spearman_ds = compute_spearman_corr(df_ds['true_age'].values, delta_ds)
                    pearson_delta_ds = compute_pearson_corr(df_ds['true_age'].values, delta_ds)
                    pearson_ds = compute_pearson_corr(df_ds['true_age'].values, df_ds['pred_age'].values)
                    f.write(f"{dataset}:\n")
                    f.write(f"  MAE: {mae_ds:.4f} years\n")
                    f.write(f"  R²: {r2_ds:.4f}\n")
                    f.write(f"  Pearson r: {pearson_ds:.4f}\n")
                    f.write(f"  Spearman ρ(Δ,age): {spearman_ds:.4f}\n")
                    f.write(f"  Pearson r(Δ,age): {pearson_delta_ds:.4f}\n")
                    f.write(f"  (n={len(df_ds)})\n\n")

    print(f"Saved summary to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SFCN model on test splits from multiple datasets"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config YAML file (default: configs/default.yaml)"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets to evaluate on (e.g., IXI OASIS_3). If not specified, evaluates on all datasets."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split to evaluate on (default: test)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (overrides config)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for evaluation (overrides config)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (overrides config)"
    )

    args = parser.parse_args()

    # Load configuration
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)

    # Extract settings from config (allow command-line overrides)
    eval_config = config.get('evaluation', {})
    batch_size = args.batch_size if args.batch_size else eval_config.get('batch_size', 8)
    num_workers = eval_config.get('num_workers', 4)
    output_dir = args.output_dir if args.output_dir else eval_config.get('output_dir', 'results')

    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path = args.checkpoint if args.checkpoint else config['evaluation']['ckpt_path']
    unified_csv = config['data']['unified_csv']

    # Get root directories for datasets
    root_dirs = {
        dataset_name: info['root']
        for dataset_name, info in config['datasets'].items()
    }

    # Print evaluation settings
    print("\n" + "="*60)
    print("EVALUATION SETTINGS")
    print("="*60)
    print(f"Datasets: {', '.join(args.datasets) if args.datasets else 'all'}")
    print(f"Split: {args.split}")
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers}")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output directory: {output_dir}")
    print("="*60 + "\n")

    # Load and filter data
    print(f"Loading data from: {unified_csv}")
    df = load_and_filter_data(
        csv_path=unified_csv,
        datasets=args.datasets,
        split=args.split
    )
    print(f"Found {len(df)} MR scans for evaluation")

    # Print dataset breakdown
    if 'dataset' in df.columns:
        print("\nDataset breakdown:")
        for dataset, count in df['dataset'].value_counts().items():
            print(f"  {dataset}: {count} MR scans")
    print()

    # Validate that all datasets in df have root directories
    datasets_in_df = df['dataset'].unique()
    missing_roots = set(datasets_in_df) - set(root_dirs.keys())
    if missing_roots:
        raise ValueError(f"Missing root directories in config for datasets: {missing_roots}")

    # Filter root_dirs to only include datasets we're evaluating
    root_dirs_filtered = {k: v for k, v in root_dirs.items() if k in datasets_in_df}

    # Load model
    print(f"Loading model from: {checkpoint_path}")
    model = load_model(checkpoint_path, device=device)
    print("Model loaded successfully\n")

    # Run evaluation on test set
    print("Running evaluation on test set...")
    results_df, mae = predict_and_eval(
        model=model,
        df=df,
        root_dirs=root_dirs_filtered,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers
    )

    # Add dataset column to results if needed
    if 'dataset' in df.columns:
        results_df = results_df.merge(
            df[['MR_ID', 'dataset']],
            on='MR_ID',
            how='left'
        )

    # Bias correction using validation set as calibration
    bias_correction_results = None
    print("\nApplying bias correction using validation set...")
    try:
        df_val = load_and_filter_data(csv_path=unified_csv, datasets=args.datasets, split='val')
        print(f"Loaded {len(df_val)} validation MR scans for calibration")

        cal_df, _ = predict_and_eval(
            model=model,
            df=df_val,
            root_dirs=root_dirs_filtered,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers
        )

        results_df, mae_before, mae_after, (a, b) = bias_correct(cal_df, results_df)

        # Compute correlations before and after bias correction
        pearson_r = compute_pearson_corr(results_df['true_age'].values, results_df['pred_age'].values)
        pearson_r_after = compute_pearson_corr(results_df['true_age'].values, results_df['pred_age_corrected'].values)
        delta_before = results_df['pred_age'].values - results_df['true_age'].values
        spearman_delta_before = compute_spearman_corr(results_df['true_age'].values, delta_before)
        pearson_delta_before = compute_pearson_corr(results_df['true_age'].values, delta_before)

        delta_after = results_df['pred_age_corrected'].values - results_df['true_age'].values
        spearman_delta_after = compute_spearman_corr(results_df['true_age'].values, delta_after)
        pearson_delta_after = compute_pearson_corr(results_df['true_age'].values, delta_after)

        r2_before = compute_r2(results_df['true_age'].values, results_df['pred_age'].values)
        r2_after = compute_r2(results_df['true_age'].values, results_df['pred_age_corrected'].values)

        bias_correction_results = {
            'mae_before': mae_before,
            'mae_after': mae_after,
            'r2_before': r2_before,
            'r2_after': r2_after,
            'pearson_before': pearson_r,
            'pearson_after': pearson_r_after,
            'spearman_delta_before': spearman_delta_before,
            'spearman_delta_after': spearman_delta_after,
            'pearson_delta_before': pearson_delta_before,
            'pearson_delta_after': pearson_delta_after,
            'a': a,
            'b': b,
            'n_cal': len(cal_df)
        }
        print(f"Bias fit: pred ≈ {a:.3f} * true + {b:.3f}")

    except Exception as e:
        print(f"Warning: Could not apply bias correction: {e}")
        print("Proceeding without bias correction...")

    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    if bias_correction_results:
        print(f"MAE (before correction):                 {bias_correction_results['mae_before']:.4f} years")
        print(f"MAE (after correction):                  {bias_correction_results['mae_after']:.4f} years")
        print(f"R² (before correction):                  {bias_correction_results['r2_before']:.4f}")
        print(f"R² (after correction):                   {bias_correction_results['r2_after']:.4f}")
        print(f"Pearson r (before correction):           {bias_correction_results['pearson_before']:.4f}")
        print(f"Pearson r (after correction):            {bias_correction_results['pearson_after']:.4f}")
        print(f"Spearman ρ(Δ,true age) (before):         {bias_correction_results['spearman_delta_before']:.4f}")
        print(f"Spearman ρ(Δ,true age) (after):          {bias_correction_results['spearman_delta_after']:.4f}")
        print(f"Pearson r(Δ,true age) (before):          {bias_correction_results['pearson_delta_before']:.4f}")
        print(f"Pearson r(Δ,true age) (after):           {bias_correction_results['pearson_delta_after']:.4f}")
        print(f"MAE improvement:                         {bias_correction_results['mae_before'] - bias_correction_results['mae_after']:.4f} years")
    else:
        r2_overall = compute_r2(results_df['true_age'].values, results_df['pred_age'].values)
        print(f"Overall MAE: {mae:.4f} years")
        print(f"Overall R²: {r2_overall:.4f}")

    # Per-dataset results
    if 'dataset' in results_df.columns:
        print("\nPer-Dataset Results:")
        for dataset in sorted(results_df['dataset'].unique()):
            df_ds = results_df[results_df['dataset'] == dataset]
            if 'pred_age_corrected' in df_ds.columns:
                mae_before = compute_mae(df_ds['true_age'].values, df_ds['pred_age'].values)
                mae_after = compute_mae(df_ds['true_age'].values, df_ds['pred_age_corrected'].values)
                r2_before = compute_r2(df_ds['true_age'].values, df_ds['pred_age'].values)
                r2_after = compute_r2(df_ds['true_age'].values, df_ds['pred_age_corrected'].values)

                delta_before = df_ds['pred_age'].values - df_ds['true_age'].values
                spearman_delta_before = compute_spearman_corr(df_ds['true_age'].values, delta_before)
                pearson_delta_before = compute_pearson_corr(df_ds['true_age'].values, delta_before)
                pearson_before = compute_pearson_corr(df_ds['true_age'].values, df_ds['pred_age'].values)

                delta_after = df_ds['pred_age_corrected'].values - df_ds['true_age'].values
                spearman_delta_after = compute_spearman_corr(df_ds['true_age'].values, delta_after)
                pearson_delta_after = compute_pearson_corr(df_ds['true_age'].values, delta_after)
                pearson_after = compute_pearson_corr(df_ds['true_age'].values, df_ds['pred_age_corrected'].values)

                print(f"  {dataset}:")
                print(f"    MAE: {mae_before:.4f} → {mae_after:.4f} years")
                print(f"    R²: {r2_before:.4f} → {r2_after:.4f}")
                print(f"    Pearson r: {pearson_before:.4f} → {pearson_after:.4f}")
                print(f"    Spearman ρ(Δ,age): {spearman_delta_before:.4f} → {spearman_delta_after:.4f}")
                print(f"    Pearson r(Δ,age): {pearson_delta_before:.4f} → {pearson_delta_after:.4f}")
                print(f"    (n={len(df_ds)})")
            else:
                mae_ds = compute_mae(df_ds['true_age'].values, df_ds['pred_age'].values)
                r2_ds = compute_r2(df_ds['true_age'].values, df_ds['pred_age'].values)
                delta_ds = df_ds['pred_age'].values - df_ds['true_age'].values
                spearman_ds = compute_spearman_corr(df_ds['true_age'].values, delta_ds)
                pearson_delta_ds = compute_pearson_corr(df_ds['true_age'].values, delta_ds)
                pearson_ds = compute_pearson_corr(df_ds['true_age'].values, df_ds['pred_age'].values)

                print(f"  {dataset}:")
                print(f"    MAE: {mae_ds:.4f} years")
                print(f"    R²: {r2_ds:.4f}")
                print(f"    Pearson r: {pearson_ds:.4f}")
                print(f"    Spearman ρ(Δ,age): {spearman_ds:.4f}")
                print(f"    Pearson r(Δ,age): {pearson_delta_ds:.4f}")
                print(f"    (n={len(df_ds)})")

    print("="*60 + "\n")

    # Prepare configuration info for summary
    config_info = {
        "Config file": args.config,
        "Checkpoint": checkpoint_path,
        "Device": device,
        "Batch size": batch_size,
        "Num workers": num_workers,
        "Output directory": output_dir
    }

    # Save results
    save_results(
        results_df=results_df,
        mae=mae,
        output_dir=output_dir,
        datasets=args.datasets,
        split=args.split,
        config_info=config_info,
        bias_correction=bias_correction_results
    )

    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()

