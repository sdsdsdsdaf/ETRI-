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

from EarlyStopping import EarlyStopping


def get_param_groups(model, base_lr=1e-3, decay_rate=0.7):
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
    param_groups.append({
        'params': model.moe.parameters(),
        'lr': base_lr * 1.5
    })

    # ④ MultiHeadTask
    param_groups.append({
        'params': model.MHT.parameters(),
        'lr': base_lr * 1.2
    })

    return param_groups


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
    total_samples = 0

    for data, labels in pbar:
        inputs = {k: v.to(device, non_blocking=True) for k, v in data.items() if isinstance(v, torch.Tensor)}
        inputs['modality_names'] = data['modality_names']
        labels = labels.to(device).long()  # (B, 6)

        optimizer.zero_grad()

        with autocast():
            outputs = model(inputs)  # ➜ List of outputs: [out0, out1, ..., out5]
            losses = [
                criterion_list[i](outputs[i], labels[:, i])
                for i in range(num_tasks)
            ]
            total_loss = sum(losses)

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        for i in range(num_tasks):
            total_losses[i] += losses[i].item()
        total_samples += labels.size(0)

        pbar.set_postfix(total_loss=total_loss.item() / num_tasks)

    avg_losses = [l / len(train_loader) for l in total_losses]
    return avg_losses, total_samples
    



def train(
        model, 
        train_loader, 
        val_loader=None, 
        optimizer=None, 
        criterion_list=None, 
        scheduler=None,
        num_epochs=10,
        device='cuda',
        log_wandb=False,
        wandb_project_name = "etri-human-understanding",
        early_stopping = None,):
    

    device = torch.device(device)
    model.to(device)
    scaler = GradScaler()\
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
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
            val_metrics = evaluate(model, val_loader, criterion_list, device)
            val_f1_macro = sum(m['f1_macro'] for m in val_metrics) / len(val_metrics) if val_metrics else 0
        else:
            continue
        total_val_loss = -1.

        if log_wandb:
            wandb.init(proj=wandb_project_name, config={"epochs": num_epochs, "batch_size": batch_size})
            log_data = {
                "epoch": epoch,
                "train_loss_avg": sum(train_loss) / len(train_loss),
                "train_acc": train_acc,
                "val_f1_macro": val_f1_macro,
            }
            for m in val_metrics:
                log_data.update({
                    f"val_loss_task{m['task']}": m['loss'],
                    f"f1_macro_task{m['task']}": m['f1_macro'],
                    f"f1_micro_task{m['task']}": m['f1_micro'],
                    f"classification_report_task{m['task']}": wandb.Html(f"<pre>{m['classification_report']}</pre>")
                })

                total_val_loss += m['loss']

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
        
        if scheduler:
            scheduler.step()

        print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.4f}, Val Loss {val_metrics['loss']:.4f}, Val_f1_macro {val_f1_macro:.4f}")

        val_loss = total_val_loss / len(val_metrics)

        if early_stopping is not None:
            early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("🛑 Early stopping triggered")
            break

        

def evaluate(model, val_loader, criterion_list, device='cuda'):
    model.eval()
    num_tasks = len(criterion_list)

    all_preds = [[] for _ in range(num_tasks)]
    all_labels = [[] for _ in range(num_tasks)]
    total_losses = [0.0 for _ in range(num_tasks)]

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            inputs, labels = batch  # labels: Tensor (B, 6)
            inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
            labels = labels.to(device)  # (B, 6)

            with torch.cuda.amp.autocast():
                outputs = model(inputs)  # List of 6 outputs, each (B, num_classes_i)

            for i in range(num_tasks):
                label_i = labels[:, i]             # (B,)
                output_i = outputs[i]              # (B, num_classes_i)
                loss_i = criterion_list[i](output_i, label_i)
                total_losses[i] += loss_i.item()

                pred_i = output_i.argmax(dim=1)    # (B,)
                all_preds[i].extend(pred_i.cpu().numpy())
                all_labels[i].extend(label_i.cpu().numpy())

    # 평가 지표 계산
    metrics = []
    for i in range(num_tasks):
        f1_macro = f1_score(all_labels[i], all_preds[i], average='macro')
        f1_micro = f1_score(all_labels[i], all_preds[i], average='micro')
        cm = confusion_matrix(all_labels[i], all_preds[i])
        report = classification_report(all_labels[i], all_preds[i], zero_division=0)

        metrics.append({
            'task': i,
            'loss': total_losses[i] / len(val_loader),
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'confusion_matrix': cm,
            'classification_report': report
        })

    return metrics


if __name__ == "__main__":

    #TODO 후에 이 train으로 데이터 전처리부터 끝까지 할 수 있게 수정정

    from model import ETRIHumanUnderstandModel
    from LifelogDataset import H5LifelogDataset
    from torch.utils.data import DataLoader
    import torch.nn as nn


    h5_save_dir = "Img_Data"
    train_or_test = {True: "train", False: "test"}
    batch_size = 8
    num_workers = 4
    epochs = 50
    lr = 1e-3

    train_dataset = H5LifelogDataset(os.path.join(h5_save_dir, train_or_test[True]))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    model = ETRIHumanUnderstandModel()
    optimizer = torch.optim.AdamW(get_param_groups(model, base_lr=5e-4, decay_rate=0.7))
    criterion_list = [nn.CrossEntropyLoss() for _ in range(6)]

    train(model=model,
          train_loader=train_loader,
          num_epochs=epochs,
          optimizer=optimizer,
          criterion_list=criterion_list,
          device='cuda',
          log_wandb=False
          )

            


    