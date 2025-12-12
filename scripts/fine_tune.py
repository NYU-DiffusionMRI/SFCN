import yaml
from pathlib import Path
from argparse import ArgumentParser

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dp_model.model_files.sfcn import SFCN
from dp_model.datasets import T1wDataset, BIN_CENTERS
from dp_model import dp_loss
from dp_model import dp_utils as dpu


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def evaluate(model, loader, device):
    """Evaluate model and return loss and MAE."""
    model.eval()
    bc_t = torch.tensor(BIN_CENTERS, dtype=torch.float32, device=device)
    all_preds, all_trues, total_loss = [], [], 0.0

    with torch.no_grad():
        for inputs, ages, _ in tqdm(loader, desc='Validating'):
            inputs = inputs.to(device)  # (B, 1, 160, 192, 160)

            # Convert ages to soft labels for loss computation
            targets = []
            for age in ages:
                y, _ = dpu.num2vect(age.item(), [42, 82], 1, sigma=1)
                targets.append(y)
            targets = torch.tensor(np.array(targets), dtype=torch.float32).to(device)

            out = model(inputs)[0].flatten(1)  # (B, 40) - log probabilities
            total_loss += dp_loss.my_KLDivLoss(out, targets).item() * len(inputs)

            # Convert to predicted ages for MAE
            probs = out.exp()  # (B, 40)
            pred_ages = (probs * bc_t).sum(dim=1)  # (B,)

            all_preds.extend(pred_ages.cpu().numpy())
            all_trues.extend(ages.numpy())

    avg_loss = total_loss / len(all_preds)
    mae = np.mean(np.abs(np.array(all_preds) - np.array(all_trues)))
    return avg_loss, mae

def train(config, experiment_name, device):
    # Load and filter data (ages 44-80, IXI + OASIS_3 only)
    df = pd.read_csv(config['data']['unified_csv'])
    df = df[(df['age'] >= 44) & (df['age'] <= 80)]
    df = df[df['dataset'].isin(['IXI', 'OASIS_3'])]

    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']

    # Create datasets
    root_dirs = {name: info['root'] for name, info in config['datasets'].items()}
    train_ds = T1wDataset(train_df, root_dirs)
    val_ds = T1wDataset(val_df, root_dirs)

    train_loader = DataLoader(train_ds, batch_size=config['fine_tune']['batch_size'],
                              shuffle=True, num_workers=config['fine_tune']['num_workers'])
    val_loader = DataLoader(val_ds, batch_size=config['fine_tune']['batch_size'],
                           shuffle=False, num_workers=config['fine_tune']['num_workers'])

    # Load pretrained model
    model = SFCN()
    model = nn.DataParallel(model)
    checkpoint = torch.load(config['model']['checkpoint_path'], map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model = model.to(device)

    # Freeze all feature_extractor except last block (conv_5), unfreeze classifier
    for param in model.module.feature_extractor.parameters():
        param.requires_grad = False
    for param in model.module.feature_extractor.conv_5.parameters():
        param.requires_grad = True
    for param in model.module.classifier.parameters():
        param.requires_grad = True

    # Training setup
    lr_max = config['fine_tune']['lr_max']
    optimizer = optim.AdamW(
        [
            {'params': model.module.feature_extractor.conv_5.parameters(), 'lr': lr_max * 0.1},
            {'params': model.module.classifier.parameters(), 'lr': lr_max},
        ],
        weight_decay=config['fine_tune']['weight_decay']
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['fine_tune']['max_epochs'],
        eta_min=config['fine_tune']['lr_min']
    )

    # Training loop with early stopping
    best_mae = float('inf')
    best_epoch = 0
    patience_counter = 0

    num_epochs = config['fine_tune']['max_epochs']
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for inputs, ages, _ in pbar:
            inputs = inputs.to(device)  # (B, 1, 160, 192, 160)

            # Convert ages to soft labels
            targets = []
            for age in ages:
                y, _ = dpu.num2vect(age.item(), [42, 82], 1, sigma=1)
                targets.append(y)
            targets = torch.tensor(np.array(targets), dtype=torch.float32).to(device)

            optimizer.zero_grad()
            out = model(inputs)[0].flatten(1)  # (B, 40)
            loss = dp_loss.my_KLDivLoss(out, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(inputs)

        train_loss /= len(train_loader.dataset)

        # Validate
        val_loss, val_mae = evaluate(model, val_loader, device)
        print(f'Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val MAE = {val_mae:.2f}')


        # Early stopping
        if val_mae < best_mae:
            best_mae = val_mae
            best_epoch = epoch
            patience_counter = 0

            ckpt_dir = Path(config['fine_tune']['ckpt_dir']) / experiment_name
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"ckpt_ep{epoch+1}.pth"
            torch.save(model.state_dict(), ckpt_path)

            print(f' New best model saved to {ckpt_path}')

        else:
            patience_counter += 1
            print(f'  No improvement for {patience_counter} consecutive epoch(s)')
            if patience_counter >= config['fine_tune']['patience']:
                print(f'Early stopping at epoch {epoch+1}')
                break

        scheduler.step()

    print(f'Best Val MAE: {best_mae:.2f} at epoch {best_epoch+1}')
    return best_mae


def main():
    parser = ArgumentParser()
    parser.add_argument('--experiment-name', type=str, required=True)
    args = parser.parse_args()

    config = load_config('configs/default.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train(config, args.experiment_name, device)


if __name__ == '__main__':
    main()

