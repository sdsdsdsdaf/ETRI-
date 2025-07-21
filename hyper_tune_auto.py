import os
from lightgbm import LGBMClassifier
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from sklearn.metrics import f1_score
import torch
from Model.Encoder import EffNetPerformerEncoder, EffNetSimpleEncoder, EffNetTransformerEncoder
from model import ETRIHumanUnderstandModel
from train import get_all_labels
from util.EarlyStopping import EarlyStopping
from util.LifelogDataset import H5LifelogDataset
from util.StackingEnsemble import StackingEnsemble
from val import run_kfold_cross_validation
from torch.optim.lr_scheduler import ReduceLROnPlateau
from util.loss import FocalLoss

BASE_BLOCK_CLASSES = {
    "EffNetPerformerEncoder": EffNetPerformerEncoder,
    "EffNetSimpleEncoder": EffNetSimpleEncoder,
    "EffNetTransformerEncoder": EffNetTransformerEncoder,
}

def objective(trial: optuna.Trial):

    lr = trial.suggest_float("lr", 1e-6, 5e-4, log=True)
    dropout_ratio = trial.suggest_float("dropout_ratio", 0.1, 0.7)
    label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.1)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    base_block = trial.suggest_categorical("base_block", ["EffNetPerformerEncoder", "EffNetSimpleEncoder", "EffNetTransformerEncoder"])
    multimodal_feature_dim = trial.suggest_categorical("multimodal_feature_dim", [128, 256, 512])
    fusion = trial.suggest_categorical("fusion", ['attention', 'concat', 'projection'])
    num_experts = trial.suggest_int("num_experts", 2, 16)
    moe_gating_type = trial.suggest_categorical("moe_gating_type", ['soft', 'topk', 'noisy_topk'])
    moe_hidden_dim = trial.suggest_categorical("moe_hidden_dim", [128, 256, 512])
    moe_k = trial.suggest_int("moe_k", 2, 16)
    moe_noise_std = trial.suggest_float("moe_noise_std", 0.0, 0.5)
    use_moe = trial.suggest_categorical("use_moe", [True, False])
    moe_lambda_bal = trial.suggest_float("moe_lambda_bal", 0.0, 1.0)
    seed = 42
    base_backbone = trial.suggest_categorical("base_backbone", ["efficientformerv2_s0", "regnety_004"])
    
    if moe_k > num_experts:
        moe_k = num_experts

    class_counts = [2, 2, 2, 3, 2, 2]
    oof_results, mskf = run_kfold_cross_validation(
            lr=lr,
            batch_size=4,
            dropout_ratio=dropout_ratio,
            label_smoothing=label_smoothing,
            weight_decy=weight_decay,
            base_backbone=base_backbone,
            base_block= BASE_BLOCK_CLASSES[base_block],
            multimodal_feature_dim=multimodal_feature_dim,
            freeze=5,
            fusion=fusion,
            use_moe=True,  
            proj_dim=multimodal_feature_dim,
            MHT_heads_hidden_layer_list=[multimodal_feature_dim//2, multimodal_feature_dim//2, multimodal_feature_dim//2, multimodal_feature_dim//4],
            MHT_back_bone_hidden_layer_list=[multimodal_feature_dim//2, multimodal_feature_dim, multimodal_feature_dim, multimodal_feature_dim, multimodal_feature_dim//2, multimodal_feature_dim],
            MHT_input_header_hidden_layer_list=[multimodal_feature_dim*2, multimodal_feature_dim, multimodal_feature_dim, multimodal_feature_dim//2],
            scheduler=ReduceLROnPlateau,
            experts = None, 
            num_experts = num_experts,
            moe_gating_type = 'noisy_topk',  # 'soft', 'topk', 'noisy_topk'
            moe_hidden_dim = moe_hidden_dim,
            moe_k = moe_k,
            moe_noise_std = moe_noise_std,
            moe_lambda_bal = moe_lambda_bal,
            seed = seed,
            epochs=5,
            early_stopping=EarlyStopping(patience=7, min_delta=0.001), 
            log=False)
    
    labels = get_all_labels(H5LifelogDataset(os.path.join("Img_Data", "train")))
    f1_result = 0
    for i in range(len(class_counts)):
        f1_result += f1_score(y_true=labels[:, i], y_pred=np.argmax(oof_results[:, i, :], axis=-1), average='macro')
    f1_result /= len(class_counts)
    print(f"Before Ensemble OOF F1 Macro Score: {f1_result:.4f}")
    # ensemble_model = StackingEnsemble(meta_model=LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=31, max_depth=-1, random_state=seed, verbosity=-1))
    ensemble_model = StackingEnsemble(
        base_models=None,
        meta_model=LGBMClassifier(
            n_estimators=20,
            learning_rate=0.1,
            num_leaves=3,
            max_depth=2,
            random_state=seed,
            verbosity=-1,
            class_weight='balanced',    
        ),
        use_proba=True,
        cls_count=class_counts
    )

    f1_macro_score = ensemble_model.score(X=oof_results, y=labels, cv=mskf, mode='f1_macro')
    accuracy = ensemble_model.score(X=oof_results, y=labels, cv=mskf, mode='accuracy')
    print(f"Final OOF results f1_macro: {f1_macro_score:.4f} accuracy: {accuracy:.4f}")

    return f1_macro_score


if __name__ == "__main__":
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler

    # (Optional) seed 고정 → reproducibility
    optuna.logging.set_verbosity(optuna.logging.INFO)
    sampler = TPESampler(seed=42)

    study = optuna.create_study(
        direction='maximize',  # f1_macro_score를 maximize 하니까
        sampler=sampler,
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        study_name='moe_model_tuning'
    )

    study.optimize(objective, n_trials=50, timeout=None)

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value (f1_macro): {trial.value:.4f}")
    print(f"  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")