#TODO: data폴더에서부터 받게 수정

import os
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import numpy as np
import optuna
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import torch
from xgboost import XGBClassifier
from train import build_meta_features, ensemble_training, train_optuna_setting
from util.LifelogDataset import H5LifelogDataset
from util.TimeSeriesDataPreProcess import data_load_and_split_test_and_train, filter_data_and_labels_by_modalities
from util.tensor_to_img import process_all_samples
import pickle as pkl
from util.StackingEnsemble import StackingEnsemble
from util.VotingEnsemble import VotingEnsemble
import pandas as pd

def reconstruct_meta_model(best_trial_params):
    model_type = best_trial_params['meta_model']
    N = best_trial_params['N']
    ensemble_mode = best_trial_params['ensemble_mode']
    meta_model_type = best_trial_params['meta_model']

    if ensemble_mode == "voting":
        voting = best_trial_params['voting']
        use_weights = best_trial_params['use_weights']
        if use_weights:
            weights = [best_trial_params[f"weight_{i}"] for i in range(N)]
    else:
        weights = None

    meta_model = None
    if model_type == 'LGBM':
        model = LGBMClassifier(
            n_estimators=best_trial_params['n_estimators'],
            learning_rate=best_trial_params['learning_rate'],
            num_leaves=best_trial_params['num_leaves'],
            max_depth=best_trial_params['max_depth'],
            random_state=42,
            verbosity=-1,
            class_weight='balanced'
        )

    elif model_type == 'RandomForest':
        model = RandomForestClassifier(
            n_estimators=best_trial_params['rf_estimators'],
            max_depth=best_trial_params['rf_max_depth'],
            class_weight='balanced',
            random_state=42
        )

    elif model_type == 'XGBoost':
        model = XGBClassifier(
            n_estimators=best_trial_params['xgb_n_estimators'],
            learning_rate=best_trial_params['xgb_learning_rate'],
            max_depth=best_trial_params['xgb_max_depth'],
            subsample=best_trial_params['xgb_subsample'],
            colsample_bytree=best_trial_params['xgb_colsample_bytree'],
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=42
        )

    elif model_type == 'CatBoost':
        model = CatBoostClassifier(
            iterations=best_trial_params['catboost_iterations'],
            learning_rate=best_trial_params['catboost_learning_rate'],
            depth=best_trial_params['catboost_depth'],
            l2_leaf_reg=best_trial_params['catboost_l2_leaf_reg'],
            random_seed=42,
            verbose=0
        )

    else:
        raise ValueError(f"Unknown meta_model type: {model_type}")

    return model



if __name__ == "__main__":
    os.system("pip install -r requirements.txt")


    """
    dir = 'Data_Dict'
    method = 'linear'
    is_daliy = True
    is_continuous = True
    mask_type = 'ratio_mask' if is_continuous else 'bit_mask'
    daliy_or_all_day = "daily" if is_daliy else "all_day"
    save_csv = True

    train_x_file_name =f"train_data_{daliy_or_all_day}_{method}_{mask_type}.pkl"
    train_y_file_name = f"train_label_{daliy_or_all_day}_{method}_{mask_type}.pkl"
    test_x_file_name = f"test_data_{daliy_or_all_day}_{method}_{mask_type}.pkl"
    test_y_file_name = f"test_label_{daliy_or_all_day}_{method}_{mask_type}.pkl"

    os.makedirs(dir, exist_ok=True)

    train_x_file_path = os.path.join(dir, train_x_file_name)
    train_y_file_path = os.path.join(dir, train_y_file_name)
    test_x_file_path = os.path.join(dir, test_x_file_name)
    test_y_file_path = os.path.join(dir, test_y_file_name)

    print(f'Find {daliy_or_all_day}_{method}_{mask_type} Data')
    if (not os.path.exists(train_x_file_path) or 
        not os.path.exists(train_y_file_path) or
        not os.path.exists(test_x_file_path) or
        not os.path.exists(test_y_file_path)
        ):
        print(f'{daliy_or_all_day}_{method}_{mask_type} Data Not Found')
        print(f'Create {daliy_or_all_day}_{method}_{mask_type} Data')

        train_data, train_label, test_data, test_label = data_load_and_split_test_and_train(save_csv=save_csv, is_continuous=is_continuous)
        with open(train_x_file_path, 'wb') as f:
            pkl.dump(train_data, f)
        with open(train_y_file_path, 'wb') as f:
            pkl.dump(train_label, f)
        with open(test_x_file_path, 'wb') as f:
            pkl.dump(test_data, f)
        with open(test_y_file_path, 'wb') as f:
            pkl.dump(test_label, f)
    else:
        print(f'{daliy_or_all_day}_{method}_{mask_type} Data Found')
        print(f'Load {daliy_or_all_day}_{method}_{mask_type} Data')
        with open(train_x_file_path, 'rb') as f:
            train_data = pkl.load(f)
        with open(train_y_file_path, 'rb') as f:
            train_label = pkl.load(f)
        with open(test_x_file_path, 'rb') as f:
            test_data = pkl.load(f)
        with open(test_y_file_path, 'rb') as f:
            test_label = pkl.load(f)


    print(f"[train_dict] size: {len(train_data)}")
    print(f"[train_label] size: {len(train_label)}")
    print(f"[test_dict] size: {len(test_data)}")
    print(f"[test_label] size: {len(test_label)}")

    train_data, train_label = filter_data_and_labels_by_modalities(train_data, train_label)

    train_x_file_name = f"train_data_filtered_{daliy_or_all_day}_{method}_{mask_type}.pkl"
    train_x_file_path = os.path.join(dir, train_x_file_name)
    with open(train_x_file_path, 'wb') as f:
            pkl.dump(train_data, f)

    process_all_samples(
        pkl_path="Data_Dict/train_data_filtered_daily_linear_ratio_mask.pkl",
        label_path = "Data_Dict/train_label_daily_linear_ratio_mask.pkl",
        save_dir="Img_Data/train",
        resize=(224, 224),
        img_save=False,
        debug = False
    )

    print("✅ Data preprocessing and image generation completed successfully.")

    """

    meta_model_trials = optuna.load_study(
        study_name='meta_model_tuning',
        storage='sqlite:////home/ubuntu/ETRI-/meta_model_tuning.db'
    )
    best_meta_setting = meta_model_trials.best_trial

    base_learner_trials = optuna.load_study(
        study_name='moe_model_tuning',
        storage='sqlite:////home/ubuntu/ETRI-/moe_model_tuning.db'
    )
    N = best_meta_setting.params['N']
    batch_size = 32

    top_trials = sorted(base_learner_trials.trials, key=lambda t: t.value, reverse=True)[:N]
    train_dataset = H5LifelogDataset(os.path.join("Img_Data", "train"))
    test_dataset = H5LifelogDataset(os.path.join("Img_Data", "test"))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    meta_model = reconstruct_meta_model(best_meta_setting.params)
    # Base Learner Learning
    print("\n"+"="*15+"Start Base Model Training"+"="*15)
    ensemble_training(top_trials, num_workers=4, batch_size=batch_size, epochs=50, is_train=True)


    # Meta Model Learning
    print("\n"+"="*15+"Start Meta Model Training"+"="*15)
    base_trials = sorted(base_learner_trials.trials, key=lambda t: t.value, reverse=True)[:N]
    X_meta_train, y_meta_train = build_meta_features(base_trials, loader=train_loader, device='cuda')
    ensemble_model = StackingEnsemble(
        base_models=None,
        meta_model=meta_model,
        use_proba=True,
        cls_count=[2, 2, 2, 3, 2, 2]
    )
    ensemble_model.fit_meta(X_meta_train, y_meta_train)

    # Meta Model Evaluation
    print("\n"+"="*15+"Start Meta Model Test"+"="*15)
    X_meta_test, _ = build_meta_features(base_trials, loader=test_loader, device='cuda')
    prediction = ensemble_model.predict(X_meta_test)
    print(f"Meta Model Predictions Shape: {prediction.shape}")

    print(prediction[:10])

    sample_df = pd.read_csv(os.path.join("data","ch2025_submission_sample.csv"))
    columns = sample_df.columns.tolist()
    print("Columns in sample:", sample_df.columns.tolist())

    # 2️⃣ dummy prediction 생성 (예: 0으로 채우기)
    #    여기서 너가 실제 예측 결과 리스트/배열로 넣으면 됨
    num_rows = sample_df.shape[0]
    id_cols = ['subject_id', 'sleep_date', 'lifelog_date']
    pred_cols = [col for col in sample_df.columns if col not in id_cols]
    
    submission = pd.DataFrame()
    for col in id_cols:
        submission[col] = sample_df[col]

    for i, col in enumerate(pred_cols):
        submission[col] = prediction[:, i]

    # 4️⃣ CSV로 저장
    submission.to_csv("submission.csv", index=False)
    print(submission.head(10))

    


