import os
import queue
import numpy as np
import packaging
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset

from util.EarlyStopping import EarlyStopping
from model import ETRIHumanUnderstandModel
from util.LifelogDataset import H5LifelogDataset
from train import get_param_groups, train, evaluate, get_all_labels, get_class_weights, maybe_compile_model
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import multiprocessing
from torch.nn.functional import softmax

def worker(fold, gpu_id, train_idx, val_idx, common_kwargs, queue:queue.Queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    result, output = run_single_fold(
        fold=fold,
        gpu_device=str(gpu_id),
        train_idx=train_idx,
        val_idx=val_idx,
        **common_kwargs
    )
    print(f"Fold {fold} finished on GPU {gpu_id}: {result}")
    queue.put((fold, val_idx, output))

def run_single_fold(
    fold: int,
    gpu_device: str,
    train_idx,
    val_idx,
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
    # 기타 필요한 인자들은 필요시 추가
):
    assert base_block is not None, "Base block must be specified."

    # GPU 설정: 환경변수로 이미 지정되었을 것. device_str은 "cuda"로 두면 CUDA_VISIBLE_DEVICES 덕분에 올바른 GPU 사용.
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"==> Fold {fold}: using device {device} (CUDA_VISIBLE_DEVICES={gpu_device})")

    # 전체 dataset 및 label
    full_dataset = H5LifelogDataset(os.path.join(h5_save_dir, "train"), seed=seed)
    
    print(f"\n🔁 Running fold {fold+1}/{n_splits}")

    # 모델 생성 및 device 이동
    model = ETRIHumanUnderstandModel(
            dropout_ratio=dropout_ratio,
            base_backbone=base_backbone,
            base_block=base_block,
            multimodal_feature_dim=multimodal_feature_dim,
            fusion=fusion,
            use_moe=use_moe,
            proj_dim=proj_dim,
            MHT_heads_hidden_layer_list=MHT_heads_hidden_layer_list,
            MHT_back_bone_hidden_layer_list=MHT_back_bone_hidden_layer_list,
            MHT_input_header_hidden_layer_list=MHT_input_header_hidden_layer_list,
            num_experts=num_experts,
            moe_hidden_dim=moe_hidden_dim,
            moe_gating_type=moe_gating_type,
            moe_k=moe_k,
            moe_noise_std=moe_noise_std,
            moe_lambda_bal=moe_lambda_bal,
            seed=seed
        )
    model.to(device)

    # DataLoader 준비
    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)
    train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=False,
            persistent_workers=True, prefetch_factor=2
    )
    val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True, drop_last=False,
            persistent_workers=True, prefetch_factor=2
    )

        # 클래스 가중치
    labels_train = get_all_labels(train_subset)
    dtype = torch.float16 if use_amp else torch.float32
    class_weight_list = get_class_weights(labels=labels_train, ref_dtype=dtype, ref_device=device)

    # Optimizer 및 criterion
    optimizer = torch.optim.AdamW(get_param_groups(model, base_lr=1e-4, decay_rate=0.3),
                                      lr=lr, weight_decay=weight_decy)
    criterion_list = [
        FocalLoss(label_smoothing=label_smoothing, gamma=gamma, alpha=class_weight_list[i])
        for i in range(len(class_weight_list))
    ]
    # TF32 사용 권장
    torch.set_float32_matmul_precision('high')

        # Scheduler 인스턴스
    scheduler_ins = None
    if scheduler is not None:
        if scheduler == WarmupCosineScheduler:
                scheduler_ins = WarmupCosineScheduler(optimizer, warmup_epochs=10, total_epochs=epochs, min_lr=5e-6)
        elif scheduler== ReduceLROnPlateau:
                scheduler_ins = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=7)

    # 학습 수행: train 함수 내부에서 maybe_compile_model 호출
    f1_score_log = train(
            early_stopping=early_stopping,
            fold=fold+1,
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
            log=False,
            after_freeze_lr=1e-5
    )

    # 모델 저장
    fold_save_dir = os.path.join(save_dir, f"fold{fold+1}")
    os.makedirs(fold_save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(fold_save_dir, "model_last.pt"))

    # 평가
    val_metrics, output = evaluate(model, val_loader, criterion_list, device)
    f1_macro_avg = np.mean([m['f1_macro'] for m in val_metrics])
    print(f"Fold {fold+1} - Last F1 Macro: {f1_macro_avg:.4f} Best F1 Macro: {f1_score_log['best_1']:.4f} Best-5 F1 Macro: {np.mean(f1_score_log['best_5']):.4f}")

    checkpoint = torch.load(os.path.join(fold_save_dir, "model_best_1.pt"), map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device).eval()
    
    _, output_best = evaluate(model, val_loader, criterion_list, device)
    return {
            "fold": fold + 1,
            "Best F1 Macro": f1_score_log['best_1'],
            "Best-5 F1 Macro": np.mean(f1_score_log['best_5']),
            "metrics": val_metrics
    }, output_best


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
    
    config = {
        'early_stopping': early_stopping,
        'dropout_ratio': dropout_ratio,
        'base_backbone': base_backbone,
        'base_block': base_block,
        'multimodal_feature_dim': multimodal_feature_dim,
        'fusion': fusion,
        'freeze': freeze,
        'use_moe': use_moe,
        'proj_dim': proj_dim,
        'MHT_heads_hidden_layer_list': MHT_heads_hidden_layer_list,
        'MHT_back_bone_hidden_layer_list': MHT_back_bone_hidden_layer_list,
        'MHT_input_header_hidden_layer_list': MHT_input_header_hidden_layer_list,
        'h5_save_dir': h5_save_dir,
        'batch_size': batch_size,
        'num_workers': num_workers,
        'epochs': epochs,
        'lr': lr,
        'n_splits': n_splits,
        'patience': patience,
        'min_delta': min_delta,
        'log': log,
        'scheduler': scheduler,
        'device': device,
        'use_amp': use_amp,
        'label_smoothing': label_smoothing,
        'weight_decy': weight_decy,
        'multi_dim': multi_dim,
        'save_dir': save_dir,
        'gamma': gamma,
        'heads': heads,
        'gn': gn,
        'experts': experts,
        'num_experts': num_experts,
        'moe_gating_type': moe_gating_type,
        'moe_hidden_dim': moe_hidden_dim,
        'moe_k': moe_k,
        'moe_noise_std': moe_noise_std,
        'moe_lambda_bal': moe_lambda_bal,
        'seed': seed,
    }   
    
    assert base_block is not None, "Base block must be specified."

    # 전체 train 데이터셋
    full_dataset = H5LifelogDataset(os.path.join(h5_save_dir, "train"), seed=seed)
    labels = get_all_labels(full_dataset)
    mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_results = []
    class_counts = [2, 2, 2, 3, 2, 2]
    n_tasks = len(class_counts)
    n_samples = len(labels)

    oof_list = [
        np.zeros((n_samples, n_cls)) 
        for n_cls in class_counts
    ]

    processes = []

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        result_queue = queue.Queue()
        for fold, (train_idx, val_idx) in enumerate(mskf.split(X=np.zeros(len(labels)), y=labels)):
            gpu_id = fold % torch.cuda.device_count()  # GPU ID for this fold
            p = multiprocessing.Process(target=worker, args=(fold, gpu_id, train_idx, val_idx, config, result_queue))
            p.start()
            processes.append(p)


        for p in processes:
            p.join()


            while not result_queue.empty():
                fold, val_idx, preds = result_queue.get()
                for task_i, preds_val in enumerate(preds):
                    # preds_val.shape == (len(va_idx), class_counts[task_i])
                    oof_list[task_i][val_idx] = preds_val

    else:
        for fold, (train_idx, val_idx) in enumerate(mskf.split(X=np.zeros(len(labels)), y=labels)):
            (result, output) = run_single_fold(
                fold=fold,
                gpu_device='0',  # 단일 GPU 사용 시
                train_idx=train_idx,
                val_idx=val_idx,
                **config
            )
            for task_i, preds_val in enumerate(output):
                # preds_val.shape == (len(va_idx), class_counts[task_i])
                oof_list[task_i][val_idx] = preds_val.cpu().numpy()

            fold_results.append(result)
            print(f"Fold {fold+1} finished.")
        
    return fold_results


if __name__ == "__main__":
    from Model.Encoder import EffNetTransformerEncoder, EffNetSimpleEncoder, EffNetPerformerEncoder
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from util.WarmupCosineScheduler import WarmupCosineScheduler
    from util.loss import FocalLoss
    from util.StackingEnsemble import StackingEnsemble

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
            epochs=0,
            early_stopping=EarlyStopping(patience=7, min_delta=0.001), 
            log=True)

    ensemble_model = StackingEnsemble()

