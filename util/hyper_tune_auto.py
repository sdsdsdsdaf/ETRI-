from Model.Encoder import EffNetPerformerEncoder, EffNetSimpleEncoder, EffNetTransformerEncoder
import torch.nn as nn
from optuna import Trial


encoder_class_dict = {
    "simple": (
        EffNetSimpleEncoder,
        {"model_name", "out_dim", "dropout_ratio", "act"}
    ),
    "transformer": (
        EffNetTransformerEncoder,
        {"model_name", "out_dim", "seq_len", "nhead", "num_layers", "act", "use_learnable_pe"}
    ),
    "performer": (
        EffNetPerformerEncoder,
        {"model_name", "out_dim", "seq_len", "nhead", "num_layers", "act", "dropout_ratio", "use_learnable_pe"}
    ),
}

def suggest_encoder_config(trial:Trial, modal_list):
    encoder_config = {}

    for modal in modal_list:
        encoder_type = trial.suggest_categorical(f"{modal}_encoder_type", list(encoder_class_dict.keys()))
        encoder_class, required_keys = encoder_class_dict[encoder_type]

        # 직접 trial에서 뽑아야 Optuna가 기록함
        model_name = trial.suggest_categorical(f"{modal}_model_name", ["mobilenetv3_small_050", "efficientnet_b0"])
        out_dim = trial.suggest_categorical(f"{modal}_out_dim", [128, 256])
        dropout = trial.suggest_float(f"{modal}_dropout", 0.1, 0.5)

        kwargs = {
            "model_name": model_name,
            "out_dim": out_dim,
            "act": nn.GELU,
            "dropout_ratio": dropout,
            "seq_len": 49,
            "nhead": trial.suggest_int(f"{modal}_nhead", 4, 8),
            "num_layers": trial.suggest_int(f"{modal}_num_layers", 2, 4),
            "use_learnable_pe": trial.suggest_categorical(f"{modal}_use_pe", [True, False]),
        }

        # 필요한 키만 필터링
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in required_keys}

        encoder_config[modal] = (encoder_class, filtered_kwargs)

    return encoder_config


def build_encoder_dict(encoder_config):
    return {
        modal: encoder_class(**kwargs)
        for modal, (encoder_class, kwargs) in encoder_config.items()
    }