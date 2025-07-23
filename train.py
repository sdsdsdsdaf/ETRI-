import io
import os
import PIL
import seaborn as sns
from matplotlib import pyplot as plt
import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import wandb
from typing import Tuple
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from util.LifelogDataset import H5LifelogDataset
from torch.utils.data import DataLoader
from util.SAM import convert_bn_to_fp32
from model import ETRIHumanUnderstandModel
from util.EarlyStopping import EarlyStopping
from Model.Encoder import EffNetPerformerEncoder, EffNetSimpleEncoder, EffNetTransformerEncoder
from util.WarmupCosineScheduler import WarmupCosineScheduler
import packaging.version
from pytorch_optimizer import SAM
from util.loss import FocalLoss
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

def check_grad_nan_inf(model):
    has_issue = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"⚠️ NaN detected in grad of parameter: {name} | shape: {param.grad.shape}")
                has_issue = True
            if torch.isinf(param.grad).any():
                print(f"⚠️ Inf detected in grad of parameter: {name} | shape: {param.grad.shape}")
                has_issue = True


def maybe_compile_model(model):
    try:
        # PyTorch 버전 확인
        version = packaging.version.parse(torch.__version__.split("+")[0])
        if version >= packaging.version.parse("2.0.0") and hasattr(torch, "compile"):
            # torch.compile이 사용 가능한 경우
            model = torch.compile(model)
            print(f"Applied torch.compile on PyTorch {torch.__version__}")
        else:
            print(f"Skipping torch.compile: PyTorch {torch.__version__} does not support it")
    except Exception as e:
        print(f"Could not apply torch.compile: {e}")
    return model

def get_all_labels(dataset) -> np.ndarray:
    """전체 label (multi-task)을 numpy array로 반환"""
    all_labels = []

    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    for _, label in tqdm(loader, desc="Collecting labels"):
        all_labels.append(label)

    return torch.cat(all_labels, dim=0).cpu().numpy()

def get_multitask_stratified_kfold_splits(
    labels: np.ndarray,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42
):
    


    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    X_dummy = np.zeros((labels.shape[0], 1))  # X는 필요 없으니 dummy로
    return mskf.split(X_dummy, labels)


def get_param_groups(model, base_lr=5e-4, decay_rate=0.2):
    param_groups = []

    # ① Encoders (EfficientNet, MobileNet 등)
    for name, encoder in model.multimodal_model.encoders.items():
        param_groups.append({
            'params': encoder.parameters(),
            'lr': base_lr * decay_rate
        })

    # ② Projection
    if hasattr(model.multimodal_model, "projections"):
        param_groups.append({
            'params': model.multimodal_model.projections.parameters(),
            'lr': base_lr
        })

    # ③ MoE
    if model.use_moe and model.moe is not None:
        # MoE의 gating network와 experts는 다른 lr로 설정
        param_groups.append({
            'params': model.moe.gating_network.parameters(),
            'lr': base_lr * 1.5
        })

    # ④ MultiHeadTask
    param_groups.append({
        'params': model.MHT.parameters(),
        'lr': base_lr * 1.2
    })

    return param_groups

def freeze_backbone(model):
    for name, module in model.named_modules():
        if 'backbone' in name:
            for param in module.parameters():
                param.requires_grad = False
    print("✅ Backbone frozen.")

def reset_optimizer(model, base_lr=1e-5, weight_decy=5e-6):

    #TODO SAM Optimizer로 변경
    new_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = SAM(
        #get_param_groups(model, base_lr=lr, decay_rate=0.3),
        new_params,
        torch.optim.AdamW,
        lr=base_lr,
        weight_decay=weight_decy, 
        adaptive=False,
        use_gc=True,
        rho=0.001,)
    return optimizer
   


def train_one_epoch(
        model,
        epoch,
        num_epochs, 
        train_loader, 
        val_loader=None, 
        optimizer=None,
        criterion_list=None,
        scaler:GradScaler=None,
        num_tasks=6,
        device='cuda',) -> Tuple[float, float]:
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train])")
    model.train()
    num_tasks = len(criterion_list)
    total_losses = [0.0 for _ in range(num_tasks)]
    corrects = [0 for _ in range(num_tasks)]
    totals = [0 for _ in range(num_tasks)]
    preds_all = [[] for _ in range(num_tasks)]
    labels_all = [[] for _ in range(num_tasks)]

    for data, labels in pbar:
        inputs = {k: v.to(device=torch.device(device), non_blocking=True) for k, v in data.items() if isinstance(v, torch.Tensor)}
        inputs['modality_names'] = data['modality_names']
        labels = labels.to(device).long()  # (B, 6)

        # print(inputs['modality_names'].min(), inputs['modality_names'].max())
        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device, enabled=True):
            outputs, bal_loss = model(inputs)  # ➜ List of outputs: [out0, out1, ..., out5]
            losses = [
                criterion_list[i](outputs[i], labels[:, i])
                for i in range(num_tasks)
            ]
            total_loss = sum(losses) + bal_loss
        
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        # check_grad_nan_inf(model)
        optimizer.first_step(zero_grad=True)

        """
        total_loss.backward()    
        # check_grad_nan_inf(model)
        optimizer.first_step(zero_grad=True)
        """

        with torch.amp.autocast(device_type=device, enabled=True):
            outputs, bal_loss = model(inputs)  # ➜ List of outputs: [out0, out1, ..., out5]
            losses = [
                criterion_list[i](outputs[i], labels[:, i])
                for i in range(num_tasks)
            ]
            total_loss = sum(losses) + bal_loss

        scaler.scale(total_loss).backward()
        # check_grad_nan_inf(model)
        optimizer.second_step(zero_grad=True)
        scaler.update()

        for i in range(num_tasks):
            total_losses[i] += losses[i].item()

            preds = outputs[i].argmax(dim=1)
            corrects[i] += (preds == labels[:, i]).sum().item()
            totals[i] += labels.size(0)

            preds_all[i].extend(preds.detach().cpu().tolist())
            labels_all[i].extend(labels[:, i].detach().cpu().tolist())

        pbar.set_postfix(total_loss=total_loss.item() / num_tasks)

    avg_losses = [l / len(train_loader) for l in total_losses]
    task_accs = [c / t if t > 0 else 0.0 for c, t in zip(corrects, totals)]
    mean_acc = sum(task_accs) / num_tasks
    return avg_losses, mean_acc, preds_all, labels_all
    

def get_class_weights(labels: np.ndarray, ref_dtype=torch.float32, ref_device="cuda"):
    """
    labels: np.ndarray, shape (N, num_tasks)
    ref_dtype: AMP나 autocast 여부에 따라 float16 or float32 등 지정
    """
    num_tasks = labels.shape[1]
    weights = []
    for i in range(num_tasks):
        task_labels = labels[:, i]
        class_counts = np.bincount(task_labels)
        total = class_counts.sum()
        class_weight = total / (len(class_counts) * class_counts)
        weight_tensor = torch.tensor(class_weight, dtype=ref_dtype, device=ref_device)
        weights.append(weight_tensor)
    return weights

def train(
        model,
        fold, 
        train_loader, 
        val_loader=None,
        batch_size=32, 
        optimizer=None,
        weight_decy=5e-6, 
        criterion_list=None, 
        scheduler=None,
        num_epochs=10,
        freeze_epoch=10,
        device='cuda',
        log_wandb=False,
        wandb_project_name = "etri-human-understanding",
        early_stopping: EarlyStopping = None,
        log=False,
        after_freeze_lr = 1e-4,
        save_dir="./checkpoints"):
    

    if isinstance(device, torch.device):
        device = device.type

    model.to(device)
    scaler = torch.amp.GradScaler(device=device, enabled=True, init_scale=2.0, growth_interval=2000)
    f1_score_log = {'best_1': float('-inf'), 'best_5': [float('-inf')] * 5}
    f1_score_log['best_5'] = sorted(f1_score_log['best_5'], reverse=False)
    
    for epoch in range(num_epochs):


        if epoch == freeze_epoch:
            freeze_backbone(model)
            optimizer = reset_optimizer(model, base_lr=after_freeze_lr, weight_decy=weight_decy)
            if scheduler is not None and isinstance(scheduler, WarmupCosineScheduler):
                scheduler = scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=0, total_epochs=num_epochs, min_lr=5e-5)
            if scheduler is not None and isinstance(scheduler, ReduceLROnPlateau):
                scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.75, patience=5)

        train_loss, train_acc, preds_all, labels_all = train_one_epoch(
            model,
            epoch,
            num_epochs, 
            train_loader, 
            val_loader, 
            optimizer, 
            criterion_list,
            scaler,
            device,
        )

        if val_loader is not None:
            val_metrics, output = evaluate(model, val_loader, criterion_list, device,log=log)
            val_f1_macro = sum(m['f1_macro'] for m in val_metrics) / len(val_metrics) if val_metrics else 0
            val_acc = sum(m['acc_macro'] for m in val_metrics) / len(val_metrics) if val_metrics else 0
        else:
            continue

        if f1_score_log['best_1'] < val_f1_macro:
            os.makedirs(save_dir, exist_ok=True)
            if not isinstance(fold, str):
                os.makedirs(os.path.join(save_dir, "fold"+str(fold)), exist_ok=True)
            else:
                os.makedirs(os.path.join(save_dir, fold), exist_ok=True)

            print(f"🏆 New best F1-macro: {val_f1_macro:.4f} (previous: {f1_score_log['best_1']:.4f})")
 
            if isinstance(fold, str):
                print(os.path.join(save_dir, fold, "model_best_1.pt"))
                torch.save(model.state_dict(), os.path.join(save_dir, fold, "model_best_1.pt"))
            else:
                torch.save(model.state_dict(), os.path.join(save_dir, "fold"+str(fold), f"model_best_1.pt"))

        f1_score_log['best_1'] = max(f1_score_log['best_1'], val_f1_macro)
        f1_score_log['best_5'].append(val_f1_macro)
        f1_score_log['best_5'] = sorted(f1_score_log['best_5'], reverse=True)[:5]

        log_save_dir = "report_logs"
        os.makedirs(log_save_dir, exist_ok=True)
        if log:

            # 🔹 classification_report 저장
            for m in val_metrics:
                with open(os.path.join(log_save_dir, f"classification_report_task{m['task']}.txt"), "w", encoding="utf-8") as f:
                    f.write(f"[Task {m['task']}] Classification Report\n\n")
                    f.write(m['classification_report'])
                    f.write("\n\n[Confusion Matrix]\n")
                    f.write(np.array2string(m['confusion_matrix'], separator=', '))
                    f.write("\n\n[Predictions]\n")
                    f.write("y_true: " + ', '.join(map(str, labels_all[m['task']])) + "\n")
                    f.write("y_pred: " + ', '.join(map(str, preds_all[m['task']])) + "\n")

        

        if log_wandb and epoch == 0:
            wandb.init(proj=wandb_project_name, config={"epochs": num_epochs, "batch_size": batch_size})
            log_data = {
                "epoch": epoch,
                "train_loss_avg": sum(train_loss) / len(train_loss),
                "train_acc": train_acc,
                "val_f1_macro": val_f1_macro,
            }
            for m in val_metrics:
                log_data.update({
                    f'train_loss_task{m["task"]}': train_loss[m['task']],
                    f'train_acc_task{m["task"]}' : m['acc_macro'],
                    f"val_loss_task{m['task']}": m['loss'],
                    f"f1_macro_task{m['task']}": m['f1_macro'],
                    f"f1_micro_task{m['task']}": m['f1_micro'],
                    f"classification_report_task{m['task']}": wandb.Html(f"<pre>{m['classification_report']}</pre>")
                })

                # confusion matrix heatmap 이미지로 wandb.Image 저장
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(m['confusion_matrix'], annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
                ax.set_title(f"Confusion Matrix - Task {m['task']}")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")

                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                img = PIL.Image.open(buf)
                log_data.update({
                    f"confusion_matrix_task{m['task']}_image": wandb.Image(img)
                })
                plt.close(fig)

            wandb.log(log_data)
        
        if scheduler is not None and isinstance(scheduler, ReduceLROnPlateau) and epoch > 1:
            scheduler.step(val_f1_macro)
            print(f"[Epoch {epoch}] Scheduler triggered Current LR: {scheduler.get_last_lr()}")

        elif scheduler is not None and epoch > 1:
            scheduler.step()
            print(f"[Epoch {epoch}] Scheduler triggered Current LR: {scheduler.get_last_lr()}")

        train_f1_macro_list = [
            f1_score(labels_all[i], preds_all[i], average='macro')
            for i in range(len(labels_all))
        ]
        train_f1_macro = sum(train_f1_macro_list) / len(train_f1_macro_list)
        total_val_loss = sum(m['loss'] for m in val_metrics) if val_metrics else 0
        total_val_loss /= len(val_metrics)
        print(f"Epoch {epoch+1}: Train Loss {np.mean(train_loss):.4f},  Val Loss {total_val_loss:.4f}, Train Acc {train_acc:.4f}, Val Acc {val_acc:.4f},Train_f1_macro {train_f1_macro:.4f}, Val_f1_macro {val_f1_macro:.4f}")

        val_loss = total_val_loss / len(val_metrics)

        if epoch > 25:
            if early_stopping is not None:
                early_stopping(val_loss, val_f1_macro, model)
            if early_stopping is not None and early_stopping.early_stop:
                print("🛑 Early stopping triggered")
                break

    return f1_score_log
        

def evaluate(model, val_loader, criterion_list, device='cuda', log = False):
    model.eval()
    num_tasks = len(criterion_list)

    all_preds = [[] for _ in range(num_tasks)]
    all_labels = [[] for _ in range(num_tasks)]
    total_losses = [0.0 for _ in range(num_tasks)]

    with torch.no_grad(), torch.amp.autocast(device_type=device if isinstance(device, str) else device.type, enabled=False):
        for batch in tqdm(val_loader, desc="Evaluating"):
            data, labels = batch  # labels: Tensor (B, 6)
            inputs = {k: v.to(device, non_blocking=True) for k, v in data.items()if isinstance(v, torch.Tensor)}
            inputs['modality_names'] = data['modality_names']
            labels = labels.to(device)  # (B, 6)

            outputs, bal_loss = model(inputs)  # List of 6 outputs, each (B, num_classes_i)

            for i in range(num_tasks):
                label_i = labels[:, i]             # (B,)
                output_i = outputs[i]              # (B, num_classes_i)
                loss_i = criterion_list[i](output_i, label_i)
                total_losses[i] += loss_i.item() 

                pred_i = output_i.argmax(dim=1)    # (B,)
                all_preds[i].extend(pred_i.cpu().numpy())
                all_labels[i].extend(label_i.cpu().numpy())
                outputs[i] = torch.softmax(output_i, dim=1)  # (B, num_classes_i)

    # 평가 지표 계산
    metrics = []
    for i in range(num_tasks):
        f1_macro = f1_score(all_labels[i], all_preds[i], average='macro')
        f1_micro = f1_score(all_labels[i], all_preds[i], average='micro')
        cm = confusion_matrix(all_labels[i], all_preds[i])
        report = classification_report(all_labels[i], all_preds[i], zero_division=0)

        acc_macro = np.mean([
            (np.array(all_preds[i]) == np.array(all_labels[i])).mean()
            for i in range(num_tasks)
        ])

        metrics.append({
            'task': i,
            'loss': total_losses[i] / len(val_loader),
            'acc_macro': acc_macro,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'confusion_matrix': cm,
            'classification_report': report
        })

    if log:
        save_dir = "report_logs"
        os.makedirs(save_dir, exist_ok=True)

        # 🔹 classification_report 저장
        with open(os.path.join(save_dir, f"classification_report_task{i}.txt"), "w", encoding="utf-8") as f:
            f.write(f"[Task {i}] Classification Report\n\n")
            f.write(report)
            f.write("\n\n[Confusion Matrix]\n")
            f.write(np.array2string(cm, separator=', '))
            f.write("\n\n[Predictions]\n")
            f.write("y_true: " + ', '.join(map(str, all_labels[i])) + "\n")
            f.write("y_pred: " + ', '.join(map(str, all_preds[i])) + "\n")

    return metrics, outputs

def train_optuna_setting(trials, epochs ,batch_size=32, num_workers=4, seed=42, is_train=None):
    BASE_BLOCK_CLASSES = {
        "EffNetPerformerEncoder": EffNetPerformerEncoder,
        "EffNetSimpleEncoder": EffNetSimpleEncoder,
        "EffNetTransformerEncoder": EffNetTransformerEncoder,
    }
    sss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)

    if os.path.exists("Weights"):
        print("Weights directory already exists. Skipping creation.")
        return sss

    for i ,trial in enumerate(trials):
        params = trial.params
        trial_number = trial.number

        print(f"\n [{i+1}/{len(trials)}] ==== Single training with Trial {trial_number} params ====\n")
        print(params)

        # param extraction + type cast
        lr = float(params.get("lr", 1e-4))
        dropout_ratio = float(params.get("dropout_ratio", 0.5))
        label_smoothing = float(params.get("label_smoothing", 0.0))
        weight_decay = float(params.get("weight_decay", 1e-4))
        base_block = params.get("base_block", "EffNetPerformerEncoder")
        multimodal_feature_dim = int(params.get("multimodal_feature_dim", 128))
        fusion = params.get("fusion", "concat")
        num_experts = int(params.get("num_experts", 4))
        moe_gating_type = params.get("moe_gating_type", "soft")
        moe_hidden_dim = int(params.get("moe_hidden_dim", 128))
        moe_k = int(params.get("moe_k", 4))
        moe_noise_std = float(params.get("moe_noise_std", 0.0))
        moe_lambda_bal = float(params.get("moe_lambda_bal", 0.0))
        base_backbone = params.get("base_backbone", "efficientformerv2_s0")
        use_moe = True
        device = 'cuda'
        use_amp = True

      
        if moe_k > num_experts:
            moe_k = num_experts

        # 여기서 cross-validation 아닌 단일 run

        model = ETRIHumanUnderstandModel(
            dropout_ratio=dropout_ratio,
            base_backbone=base_backbone,
            base_block=BASE_BLOCK_CLASSES[base_block],
            multimodal_feature_dim=multimodal_feature_dim,
            fusion=fusion,
            use_moe=use_moe,
            proj_dim=multimodal_feature_dim,
            MHT_heads_hidden_layer_list=[multimodal_feature_dim//2, multimodal_feature_dim//2, multimodal_feature_dim//2, multimodal_feature_dim//4],
            MHT_back_bone_hidden_layer_list=[multimodal_feature_dim//2, multimodal_feature_dim, multimodal_feature_dim, multimodal_feature_dim, multimodal_feature_dim//2, multimodal_feature_dim],
            MHT_input_header_hidden_layer_list=[multimodal_feature_dim*2, multimodal_feature_dim, multimodal_feature_dim, multimodal_feature_dim//2],
            num_experts=num_experts,
            moe_gating_type=moe_gating_type,
            moe_hidden_dim=moe_hidden_dim,
            moe_k=moe_k,
            moe_noise_std=moe_noise_std,
            moe_lambda_bal=moe_lambda_bal,
            seed=seed
        )
        model.to(device)
        model.apply(convert_bn_to_fp32)

        dataset = H5LifelogDataset(os.path.join("Img_Data", "train"))
        labels = get_all_labels(dataset)

        if is_train:
            # 80/20 stratified split
            train_idx, val_idx = next(sss.split(X=np.zeros(len(labels)), y=labels))

            train_subset = torch.utils.data.Subset(dataset, train_idx)
            val_subset = torch.utils.data.Subset(dataset, val_idx)

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
        else:
            # 전체 dataset으로 train만
            train_loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=True, drop_last=False,
                persistent_workers=True, prefetch_factor=2
            )
            val_loader = None

        if is_train:
            labels_train = get_all_labels(train_subset)
        else:
            labels_train = get_all_labels(dataset)

        dtype = torch.float16 if use_amp else torch.float32
        class_weight_list = get_class_weights(labels=labels_train, ref_dtype=dtype, ref_device=device)

        param_groups = get_param_groups(model, base_lr=lr, decay_rate=0.3)

        for group in param_groups:
            group['params'] = [p for p in group['params'] if p.requires_grad]

        optimizer = SAM(
        #get_param_groups(model, base_lr=lr, decay_rate=0.3),
            param_groups,
            torch.optim.AdamW,
            lr=lr,
            weight_decay=weight_decay, 
            adaptive=False,
            use_gc=True,
            rho=0.001,)

        criterion_list = [
            FocalLoss(label_smoothing=label_smoothing, gamma=2.0, alpha=class_weight_list[i])
            for i in range(len(class_weight_list))
        ]
        early_stopping = EarlyStopping(patience=7, min_delta=0.001)
        torch.set_float32_matmul_precision('high')

        train(
            num_epochs=epochs,
            early_stopping=early_stopping if is_train else None,
            fold=f'trial_{trial_number}',
            model=model,
            train_loader=train_loader,
            weight_decy=weight_decay,
            val_loader=val_loader, 
            freeze_epoch=15,
            optimizer=optimizer,
            criterion_list=criterion_list,
            device=device,
            log_wandb=False,
            log=False,
            after_freeze_lr=1e-5,
            save_dir = "Weights"
        )

    return sss

def build_meta_features(top_trials, loader, device):
    meta_feat = []
    labels_list = None

    for t_idx, trial in enumerate(top_trials):
        params = trial.params
        trial_number = trial.number
        print(f"\n[{t_idx + 1}/{len(top_trials)}]==== Building meta features for Trial {trial_number} ====")

        # Model 복원 (위 코드와 동일)
        base_block = params.get("base_block", "EffNetPerformerEncoder")
        multimodal_feature_dim = int(params.get("multimodal_feature_dim", 128))
        fusion = params.get("fusion", "concat")
        num_experts = int(params.get("num_experts", 4))
        moe_gating_type = params.get("moe_gating_type", "soft")
        moe_hidden_dim = int(params.get("moe_hidden_dim", 128))
        moe_k = int(params.get("moe_k", 4))
        moe_noise_std = float(params.get("moe_noise_std", 0.0))
        moe_lambda_bal = float(params.get("moe_lambda_bal", 0.0))
        base_backbone = params.get("base_backbone", "efficientformerv2_s0")
        use_moe = True

        if moe_k > num_experts:
            moe_k = num_experts

        BASE_BLOCK_CLASSES = {
            "EffNetPerformerEncoder": EffNetPerformerEncoder,
            "EffNetSimpleEncoder": EffNetSimpleEncoder,
            "EffNetTransformerEncoder": EffNetTransformerEncoder,
        }

        model = ETRIHumanUnderstandModel(
            dropout_ratio=float(params.get("dropout_ratio", 0.5)),
            base_backbone=base_backbone,
            base_block=BASE_BLOCK_CLASSES[base_block],
            multimodal_feature_dim=multimodal_feature_dim,
            fusion=fusion,
            use_moe=use_moe,
            proj_dim=multimodal_feature_dim,
            MHT_heads_hidden_layer_list=[multimodal_feature_dim//2]*3 + [multimodal_feature_dim//4],
            MHT_back_bone_hidden_layer_list=[multimodal_feature_dim//2, multimodal_feature_dim, multimodal_feature_dim, multimodal_feature_dim, multimodal_feature_dim//2, multimodal_feature_dim],
            MHT_input_header_hidden_layer_list=[multimodal_feature_dim*2, multimodal_feature_dim, multimodal_feature_dim, multimodal_feature_dim//2],
            num_experts=num_experts,
            moe_gating_type=moe_gating_type,
            moe_hidden_dim=moe_hidden_dim,
            moe_k=moe_k,
            moe_noise_std=moe_noise_std,
            moe_lambda_bal=moe_lambda_bal,
            seed=42
        )
        model.to(device)

        weight_path = os.path.join("Weights", f"trial_{trial_number}","model_best_1.pt")
        state_dict = torch.load(weight_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        preds = []
        with torch.no_grad():
            for batch in tqdm(loader, leave=False):
                data, labels = batch
                inputs = {k: v.to(device, non_blocking=True) for k, v in data.items() if isinstance(v, torch.Tensor)}
                inputs['modality_names'] = data['modality_names']
                outputs, _ = model(inputs)  # outputs: list of Tensors per task

                # → 하나의 batch를 (B, n_tasks, n_classes_per_task)로 구성
                batch_size = next(iter(outputs)).shape[0]
                batch_pred = np.zeros((batch_size, 6, 3), dtype=np.float32)

                for i, o in enumerate(outputs):
                    probs = o.softmax(-1).cpu().numpy()  # (B, cls_count[i])
                    batch_pred[:, i, :probs.shape[1]] = probs  # (B, 1, cls_count[i])

                preds.append(batch_pred)
                if t_idx == 0:
                    if labels_list is None:
                        labels_list = []
                    labels_list.append(labels.cpu().numpy())

        # 전체 concat
        preds_all = np.concatenate(preds, axis=0)  # (N, n_tasks, cls_count[i])
        meta_feat.append(preds_all)
        print(f"✅ Meta features for Trial {trial_number} built with shape: {preds_all.shape}")

    # print("Labels List: ",labels_list)
    X_meta = np.concatenate(meta_feat, axis=2)  # (N, n_tasks, cls_count[i])
    y_meta = np.concatenate(labels_list, axis=0)  # (N, n_tasks)

    print(f"✅ Meta features built complete with shape: {X_meta.shape} label shape{y_meta.shape}")
    return X_meta, y_meta

def ensemble_training(top_trials, num_workers = 4, batch_size = 8, epochs=2, is_train=True):
    h5_save_dir = "Img_Data"
    train_or_test = {True: "train", False: "test"}
    lr = 1e-3
    use_amp = True
    class_counts = [2, 2, 2, 3, 2, 2]
    device = 'cuda'
    seed=42

    if os.path.exists("Weights"):
        print("Weights directory already exists. Skipping creation.")
        return None
    
    sss = train_optuna_setting(top_trials, epochs=epochs, num_workers = 4, batch_size = 32, is_train=is_train)
    dataset = H5LifelogDataset(os.path.join(h5_save_dir, 'train'))
    labels_all = get_all_labels(dataset)

    if is_train:
        train_idx, val_idx = next(sss.split(X=np.zeros(len(labels_all)), y=labels_all))
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        X_meta_train, y_meta_train = build_meta_features(top_trials, train_loader, device)
        X_meta_val, y_meta_val = build_meta_features(top_trials, val_loader, device)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        X_meta_train, y_meta_train = build_meta_features(top_trials, loader, device)
        X_meta_val, y_meta_val = None, None

    ensemble_model = StackingEnsemble(
        base_models=None,
        meta_model=LGBMClassifier(
            n_estimators=20, learning_rate=0.1, num_leaves=3, max_depth=2,
            random_state=42, verbosity=-1, class_weight='balanced'
        ),
        use_proba=True,
        cls_count=class_counts
    )
    ensemble_model.fit_meta(X_meta_train, y_meta_train)

    if is_train:
        # 4️⃣ predict & f1-score 계산
        y_pred = ensemble_model.meta_model.predict(X_meta_val)

        f1_scores = []
        for i in range(len(class_counts)):
            pred_i = y_pred[:, i]
            true_i = y_meta_val[:, i]
            score = f1_score(true_i, pred_i, average='macro')
            f1_scores.append(score)

        avg_f1 = np.mean(f1_scores)
        print(f"\n✅ F1 scores per task: {f1_scores}")
        print(f"✅ Mean F1 score: {avg_f1:.4f}")

    print("\n✅ Ensemble training complete")

    return ensemble_model


def ensemble_test(top_trials, ensemble_model, batch_size=32, num_workers=4):
    device = 'cuda'
    test_dataset = H5LifelogDataset(os.path.join("Img_Data", "test"))
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
        persistent_workers=True, prefetch_factor=2
    )

    # 1️⃣ meta-feature 생성
    X_meta_test, _ = build_meta_features(top_trials, test_loader, device)

    # 2️⃣ prediction
    y_pred = ensemble_model.meta_model.predict(X_meta_test)  # shape: (n_samples, n_tasks)

    print(f"\n✅ Test prediction shape: {y_pred.shape}")

    # 3️⃣ 반환 또는 파일 저장
    return y_pred


if __name__ == "__main__":

    #TODO 후에 이 train으로 데이터 전처리부터 끝까지 할 수 있게 수정정

    from model import ETRIHumanUnderstandModel
    from util.LifelogDataset import H5LifelogDataset
    from torch.utils.data import DataLoader
    import torch.nn as nn
    import optuna
    from util.StackingEnsemble import StackingEnsemble
    from lightgbm import LGBMClassifier

    N = 10  # 상위 N개의 trial을 선택
    epochs = 50
    batch_size = 32
    trials = optuna.load_study(
        study_name='moe_model_tuning',
        storage='sqlite:////home/ubuntu/ETRI-/moe_model_tuning.db'
    )

    completed_trials = [t for t in trials.trials if t.state.name == "COMPLETE"]

    # 2️⃣ value 기준으로 정렬 (내림차순: best가 먼저)
    top_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)[:N]
    print(f"\nTop {N} trials based on value: {[t.number for t in top_trials]}\n")

    ensemble_model = ensemble_training(top_trials, batch_size=batch_size, epochs=epochs, num_workers=4)
    # final_result = ensemble_test(top_trials, ensemble_model, batch_size=batch_size, num_workers=4)


    



            


     