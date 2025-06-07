import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset

from EarlyStopping import EarlyStopping
from model import ETRIHumanUnderstandModel
from LifelogDataset import H5LifelogDataset
from train import train, evaluate


def run_kfold_cross_validation(
    h5_save_dir: str = "Img_Data",
    batch_size: int = 8,
    num_workers: int = 4,
    epochs: int = 50,
    lr: float = 1e-3,
    n_splits: int = 5,
    patience: int = 5,
    min_delta: float = 1e-4,
    device: str = 'cuda',
    save_dir: str = "./checkpoints"
):
    # 전체 train 데이터셋
    full_dataset = H5LifelogDataset(os.path.join(h5_save_dir, "train"))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
        print(f"\n🔁 Fold {fold+1}/{n_splits}")

        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True)

        model = ETRIHumanUnderstandModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        criterion_list = [nn.CrossEntropyLoss() for _ in range(6)]

        # 학습 수행
        train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=epochs,
            optimizer=optimizer,
            criterion_list=criterion_list,
            device=device,
            log_wandb=False,
            patience=patience,
            min_delta=min_delta,
            early_stopping=EarlyStopping(patience=5, min_delta=1e-4)
        )

        # 모델 저장
        os.makedirs(f"{save_dir}/fold{fold+1}", exist_ok=True)
        torch.save(model.state_dict(), f"{save_dir}/fold{fold+1}/model_last.pt")

        # 성능 평가
        val_metrics = evaluate(model, val_loader, criterion_list, device)
        f1_macro_avg = np.mean([m['f1_macro'] for m in val_metrics])
        print(f"✅ Fold {fold+1} F1-macro: {f1_macro_avg:.4f}")

        fold_results.append({
            "fold": fold + 1,
            "f1_macro": f1_macro_avg,
            "metrics": val_metrics
        })

    return fold_results


if __name__ == "__main__":
    results = run_kfold_cross_validation()
    mean_f1 = np.mean([r['f1_macro'] for r in results])
    print(f"\n🎯 Average F1-macro over {len(results)} folds: {mean_f1:.4f}")
