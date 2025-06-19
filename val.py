import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset

from util.EarlyStopping import EarlyStopping
from model import ETRIHumanUnderstandModel
from util.LifelogDataset import H5LifelogDataset
from train import get_param_groups, train, evaluate, get_all_labels, get_class_weights
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

def run_kfold_cross_validation(
    early_stopping=None,
    model=None,
    dropout_ratio=0.5,
    base_backbone="mobilenetv3_small_050",
    base_block=None,
    multimodal_feature_dim=128,
    fusion='attention',
    freeze=20,
    use_moe=True,
    proj_dim=128,
    MHT_heads_hidden_layer_list=[128//2, 128//2, 128//2, 128//4],
    MHT_back_bone_hidden_layer_list=[128//2, 128, 128, 128, 128//2, 128],
    MHT_input_header_hidden_layer_list=[128*2 , 128, 128, 128//2],
    h5_save_dir: str = "Img_Data",
    batch_size: int = 8,
    num_workers: int = 8,
    epochs: int = 20,
    lr: float = 1e-3,
    n_splits: int = 5,
    patience: int = 5,
    min_delta: float = 1e-4,
    log = False,
    scheduler = None,
    device: str = 'cuda',
    use_amp: bool = True,
    label_smoothing: float = 0.01,
    weight_decy=1e-4,
    multi_dim: int = 128,
    save_dir: str = "./checkpoints",
    gamma=2.0,
    heads= None,
    gn = None,
    experts = None,
    num_experts = 6,
    moe_gating_type = 'soft',
    moe_hidden_dim = 256,
    moe_k = 3,
    moe_noise_std = 0,
    moe_lambda_bal = 0,
    seed = 42
):
    
    assert base_block is not None, "Base block must be specified."

    # 전체 train 데이터셋
    full_dataset = H5LifelogDataset(os.path.join(h5_save_dir, "train"), seed=seed)
    labels = get_all_labels(full_dataset)
    mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(mskf.split(X=np.zeros(len(labels)), y=labels)):
        print(f"\n🔁 Fold {fold+1}/{n_splits}")

        model = ETRIHumanUnderstandModel(
            dropout_ratio=dropout_ratio,
            base_backbone=base_backbone,
            base_block=base_block,
            multimodal_feature_dim=multi_dim,
            fusion=fusion,
            use_moe=use_moe,
            proj_dim=proj_dim,
            MHT_heads_hidden_layer_list=MHT_heads_hidden_layer_list,
            MHT_back_bone_hidden_layer_list=MHT_back_bone_hidden_layer_list,
            MHT_input_header_hidden_layer_list=MHT_input_header_hidden_layer_list,
            num_experts = num_experts,
            heads= heads,
            gn = gn,
            experts = experts,
            moe_hidden_dim=moe_hidden_dim,
            moe_gating_type = moe_gating_type,
            moe_k = moe_k,
            moe_noise_std = moe_noise_std,
            moe_lambda_bal = moe_lambda_bal,
            seed = seed
        )
        model.to(device=torch.device(device))

        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True, drop_last=True, 
                                  persistent_workers=True, prefetch_factor=2)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True, drop_last=True,
                                persistent_workers=True,prefetch_factor=2)

        labels_train = get_all_labels(train_subset)
        dtype = torch.float16 if use_amp else torch.float32
        class_weight_list = get_class_weights(labels=labels_train, ref_dtype=dtype, ref_device=device)

        optimizer = torch.optim.AdamW(get_param_groups(model, base_lr=1e-4, decay_rate=0.3), lr=lr, weight_decay=weight_decy)
        criterion_list = [FocalLoss(label_smoothing=label_smoothing, gamma=gamma, alpha=class_weight_list[i]) for i in range(6)]
    
        if scheduler is not None and scheduler == WarmupCosineScheduler:
            scheduler_ins=scheduler(optimizer, warmup_epochs = 10, total_epochs=epochs, min_lr=5e-6)
        if scheduler is not None and scheduler == ReduceLROnPlateau:
            scheduler_ins=scheduler(optimizer, mode='max', factor=0.5, patience=7)

        # 학습 수행
        f1_score_log = train(
            early_stopping=early_stopping,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=epochs,
            scheduler=scheduler_ins,
            freeze_epoch=freeze,
            optimizer=optimizer,
            criterion_list=criterion_list,
            device=device,
            log_wandb=False,
            log=log,
            after_freeze_lr=1e-5
        )

        # 모델 저장
        os.makedirs(f"{save_dir}/fold{fold+1}", exist_ok=True)
        torch.save(model.state_dict(), f"{save_dir}/fold{fold+1}/model_last.pt")

        # 성능 평가
        val_metrics = evaluate(model, val_loader, criterion_list, device)
        f1_macro_avg = np.mean([m['f1_macro'] for m in val_metrics])

        print(f"Fold {fold+1} - F1 Macro: {f1_macro_avg:.4f} Best F1 Macro: {f1_score_log['best_1']:.4f} Best-5 F1 Macro: {np.mean(f1_score_log['best_5']):.4f}")

        fold_results.append({
            "fold": fold + 1,
            "f1_macro": f1_macro_avg,
            "metrics": val_metrics
        })

    return fold_results


if __name__ == "__main__":
    from Model.Encoder import EffNetTransformerEncoder, EffNetSimpleEncoder, EffNetPerformerEncoder
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from util.WarmupCosineScheduler import WarmupCosineScheduler
    from util.loss import FocalLoss

    multi_dim = 128
    scheduler = ReduceLROnPlateau
    results = run_kfold_cross_validation(
            batch_size=10,
            dropout_ratio=0.2,
            label_smoothing=0.01,
            weight_decy=1e-5,
            base_backbone="mobilenetv2_100",
            base_block=EffNetPerformerEncoder,
            multimodal_feature_dim=multi_dim,
            freeze=25,
            fusion='attention',
            use_moe=True,
            proj_dim=multi_dim,
            MHT_heads_hidden_layer_list=[multi_dim//2, multi_dim//2, multi_dim//2, multi_dim//4],
            MHT_back_bone_hidden_layer_list=[multi_dim//2, multi_dim, multi_dim, multi_dim, multi_dim//2, multi_dim],
            MHT_input_header_hidden_layer_list=[multi_dim*2 , multi_dim, multi_dim, multi_dim//2],
            scheduler=ReduceLROnPlateau,
            experts = None, 
            num_experts = 5,
            moe_gating_type = 'noisy_topk',  # 'soft', 'topk', 'noisy_topk'
            moe_hidden_dim = 256,
            moe_k = 2,
            moe_noise_std = 0.1,
            moe_lambda_bal = 0.005,
            seed = 42,
            epochs=60,
            early_stopping=EarlyStopping(patience=7, min_delta=0.001), 
            log=True)
    
    mean_f1 = np.mean([r['f1_macro'] for r in results])
    print(f"\n Average F1-macro over {len(results)} folds: {mean_f1:.4f}")
