import torch
import optuna
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from your_encoder_module import suggest_encoder_config, build_encoder_dict
from your_dataset import YourDataset
from model import ETRIHumanUnderstandModel  # Wraps encoders + heads
from your_trainer import train_one_epoch, evaluate  # Define these separately

modal_list = [
    'mGps', 'mAmbience', 'mLight', 'mScreenStatus',
    'mUsageStats', 'mWifi', 'wHr', 'wLight', 'wPedo', 'mActivity', 'mACStatus', 'mAppUsage'
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def objective(trial):
    # === Step 1: Get encoder_dict from Optuna trial === #
    encoder_config = suggest_encoder_config(trial, modal_list)
    encoder_dict = build_encoder_dict(encoder_config)

    # === Step 2: Build full model (MoE-style or unified) === #
    model = ETRIHumanUnderstandModel(encoder_dict=encoder_dict)
    model = model.to(device)

    # === Step 3: Dataloader === #
    train_loader = DataLoader(YourDataset(split="train"), batch_size=8, shuffle=True)
    val_loader   = DataLoader(YourDataset(split="val"), batch_size=8)

    # === Step 4: Optimizer === #
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler()

    # === Step 5: Training === #
    best_val = -float("inf")
    for epoch in range(5):
        train_one_epoch(model, train_loader, optimizer, scaler)
        val_metric = evaluate(model, val_loader)  # e.g. accuracy or R^2

        if val_metric > best_val:
            best_val = val_metric
            torch.save(model.state_dict(), f"./checkpoints/best_model_trial_{trial.number}.pt")

        # Optuna pruning
        trial.report(val_metric, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")  # or "minimize"
    study.optimize(objective, n_trials=30)

    print("Best trial:")
    print(study.best_trial)