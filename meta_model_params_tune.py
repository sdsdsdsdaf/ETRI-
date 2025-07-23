import os
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import torch
from xgboost import XGBClassifier
from train import build_meta_features, get_all_labels
from util.VotingEnsemble import VotingEnsemble
from util.LifelogDataset import H5LifelogDataset
from util.StackingEnsemble import StackingEnsemble
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from tqdm.auto import tqdm

train_loader = torch.utils.data.DataLoader(
    H5LifelogDataset(os.path.join("Img_Data", "train")),
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

def objective(trial: optuna.Trial):

    device = 'cuda'
    moe_trials = optuna.load_study(
        study_name='moe_model_tuning',
        storage='sqlite:////home/ubuntu/ETRI-/moe_model_tuning.db'
    )
    N = trial.suggest_int('N', 3, 10)  # 상위 N개의 trial을 선택
    completed_trials = [t for t in moe_trials.trials if t.state.name == "COMPLETE"]
    top_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)[:N]

    X_meta_train, y_meta_train = build_meta_features(top_trials, train_loader, device)

    ensemble_mode = trial.suggest_categorical('ensemble_mode', ['stacking', 'voting'])
    meta_model_type = trial.suggest_categorical('meta_model', ['LGBM', 'RandomForest', 'XGBoost', 'CatBoost'])
    
    if ensemble_mode == "voting":
        voting = trial.suggest_categorical('voting', ['soft', 'hard'])
    use_weights = trial.suggest_categorical('use_weights', [True, False])
    if use_weights:
        weights = [trial.suggest_float(f"weight_{i}", 0.1, 2.0) for i in range(N)]
    else:
        weights = None



    cv = MultilabelStratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )
    
    meta_model = None
    if meta_model_type == 'LGBM':
        meta_model = LGBMClassifier(
            n_estimators=trial.suggest_int('n_estimators', 20, 100),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.2),
            num_leaves=trial.suggest_int('num_leaves', 3, 31),
            max_depth=trial.suggest_int('max_depth', 2, 6),
            random_state=42,
            verbosity=-1,
            class_weight='balanced',
        )
    elif meta_model_type == 'LogisticRegression':
        meta_model = LogisticRegression(
            C=trial.suggest_float('lr_C', 0.01, 10),
            max_iter=1000,
            class_weight='balanced',
        )
    elif meta_model_type == 'RandomForest':
        meta_model = RandomForestClassifier(
            n_estimators=trial.suggest_int('rf_estimators', 20, 100),
            max_depth=trial.suggest_int('rf_max_depth', 2, 10),
            class_weight='balanced',
            random_state=42,
        )
    elif meta_model_type == 'XGBoost':
        meta_model = XGBClassifier(
            n_estimators=trial.suggest_int('xgb_n_estimators', 20, 200),
            learning_rate=trial.suggest_float('xgb_learning_rate', 0.01, 0.3),
            max_depth=trial.suggest_int('xgb_max_depth', 2, 10),
            subsample=trial.suggest_float('xgb_subsample', 0.6, 1.0),
            colsample_bytree=trial.suggest_float('xgb_colsample_bytree', 0.6, 1.0),
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=42,
        )

    elif meta_model_type == 'CatBoost':
        meta_model = CatBoostClassifier(
            iterations=trial.suggest_int('catboost_iterations', 20, 200),
            learning_rate=trial.suggest_float('catboost_learning_rate', 0.01, 0.3),
            depth=trial.suggest_int('catboost_depth', 2, 10),
            l2_leaf_reg=trial.suggest_float('catboost_l2_leaf_reg', 1.0, 10.0),
            random_seed=42,
            verbose=0,
        )


    if ensemble_mode == 'stacking':
        # stacking용 ensemble
        ensemble = StackingEnsemble(
            base_models=None,
            meta_model=meta_model,
            use_proba=True,
            cls_count=[2, 2, 2, 3, 2, 2]
        )
        scores = ensemble.score(cv=cv, X=X_meta_train, y=y_meta_train, mode='f1_macro')
    else:
        # voting ensemble 직접 처리
        cls_count = [2, 2, 2, 3, 2, 2]

        voting_ensemble = VotingEnsemble(
            voting=voting,
            cls_count=cls_count,
            num_weak_learners = N,
            weights = weights
        )
        scores = voting_ensemble.score(
            X=X_meta_train,
            y=y_meta_train,
            cv=cv,
            mode='f1_macro'
        )

    return scores

if __name__ == "__main__":
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler

    # (Optional) seed 고정 → reproducibility
    optuna.logging.set_verbosity(optuna.logging.INFO)
    sampler = TPESampler(seed=42)

    study = optuna.create_study(
        direction='maximize',  # f1_macro_score를 maximize 하니까
        sampler=sampler,
        storage="sqlite:///meta_model_tuning.db",
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=4),
        study_name='meta_model_tuning',
        load_if_exists=True
    )

    study.optimize(objective, n_trials=100, timeout=None)
    print("Best trial:")
    trial = study.best_trial

    print(f"  Value (f1_macro): {trial.value:.4f}")
    print(f"  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

