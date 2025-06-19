from typing import Optional,Union
from Model.AutoEncoder import Conv1DAutoencoder, FCAutoencoder
from Model.Encoder import EffNetTransformerEncoder
from Model.MoE import MoE
from Model.MultimodalModel import MultimodalModel
from Model.MultiHeadTask import MultiHeadTask
from Model.Encoder import EffNetPerformerEncoder, EffNetSimpleEncoder, EffNetTransformerEncoder

import torch
import torch.nn as nn


def init_except_autoencoder(m, act=nn.GELU, exclude_backbone_ids=None):
    if exclude_backbone_ids and id(m) in exclude_backbone_ids:
        return

    if isinstance(m, (FCAutoencoder, Conv1DAutoencoder)):
        return

    if isinstance(m, nn.Linear):
        if act == nn.GELU:
            nn.init.xavier_uniform_(m.weight)
        elif act == nn.ReLU:
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    # LayerNorm 초기화
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class ETRIHumanUnderstandModel(nn.Module):
    def __init__(self,
                 mBle_in = 10,
                 img_size = 224,
                 base_backbone = None,
                 encoder_dict: dict = None, 
                 encoder_config: dict = None, 
                 multimodal_feature_dim: int = 256,
                 ae_dict: nn.ModuleDict = None,
                 base_block: Optional[Union[EffNetTransformerEncoder, EffNetSimpleEncoder, EffNetPerformerEncoder]]=None, 
                 fusion: str = 'projection', 
                 proj_dim: int = 256,
                 fc_expand_feature: int = 128,
                 dropout_ratio: float = 0.3,
                 act=nn.GELU,
                 hidden_layer_list:Optional[list[int]] = [128],
                 MHT_input_header_hidden_layer_list:Optional[list[int]] = [128],
                 MHT_heads_hidden_layer_list:Optional[list[int]] = [128],
                 MHT_back_bone_hidden_layer_list:Optional[list[int]] = [128],
                 num_experts: int = 6,
                 use_moe: bool = True,
                 moe_hidden_dim: int = 256,
                 heads = None,
                 gn = None,
                 experts=None,
                 moe_gating_type: str = 'soft',   # 'soft', 'topk', 'noisy_topk'
                 moe_k: int = 3,
                 moe_noise_std: float = 0.0,
                 moe_lambda_bal: float = 0.0,
                 seed: int = 42):
        super().__init__()

        if seed is not None:
            import random, numpy as np
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        

        self.multimodal_model = MultimodalModel(
           mBle_in=mBle_in,
           img_size=img_size,
           encoder_dict=encoder_dict,
           base_block = base_block,
           base_model=base_backbone,
           encoder_config=encoder_config,
           out_feature=multimodal_feature_dim,
           ae_dict=ae_dict,
           fusion=fusion,
           proj_dim=proj_dim,
           fc_expand_feature=fc_expand_feature,
           dropout_ratio=dropout_ratio,
           act=act,
           hidden_layer_list=hidden_layer_list
        )

        self.use_moe = use_moe
        if self.use_moe:

            self.norm_before_moe = nn.Sequential(
                nn.LayerNorm(multimodal_feature_dim),
                nn.Dropout(dropout_ratio)
            )

            self.moe = MoE(
                input_dim=self.multimodal_model.shared_dim,
                num_experts=num_experts,
                gating_network=gn,
                experts=experts,
                act=act,
                gating_type=moe_gating_type,   # 'soft', 'topk', 'noisy_topk'
                k=moe_k,
                hidden_dim=moe_hidden_dim,
                noise_std=moe_noise_std,
                lambda_bal=moe_lambda_bal
            )
        else:
            self.moe = None

        self.MHT = MultiHeadTask(
            in_feature=self.multimodal_model.shared_dim,
            heads_hidden_layer_list=MHT_heads_hidden_layer_list,
            input_header_hidden_layer_list=MHT_input_header_hidden_layer_list,
            back_bone_hidden_layer_list=MHT_back_bone_hidden_layer_list,
            act=act,
            dropout_ratio=dropout_ratio,
            heads=heads,  # Define heads as needed
            ae_dict=ae_dict
        )
        for module in self.modules():
            init_except_autoencoder(module, exclude_backbone_ids=None, act=act)

    def forward(self, inputs) -> torch.Tensor:

        sleep_date_inputs = inputs['tensor_sleep']
        lifelog_date_inputs = inputs['tensor_lifelog']
        mBle_sleep_inputs = inputs['mble_data_sleep']
        mBle_lifelog_inputs = inputs['mble_data_lifelog']
        modal_list = inputs['modality_names']


        multimodal_features_sleep_date = self.multimodal_model(sleep_date_inputs, mBle_sleep_inputs, modal_list)
        multimodal_features_life_date = self.multimodal_model(lifelog_date_inputs, mBle_lifelog_inputs, modal_list)

        if self.use_moe:

            multimodal_features_sleep_date = self.norm_before_moe(multimodal_features_sleep_date)
            multimodal_features_life_date = self.norm_before_moe(multimodal_features_life_date)
            
            # Apply MoE to both sleep and life date features
            multimodal_features_sleep_date, sleep_date_bal_loss = self.moe(multimodal_features_sleep_date, training=self.training)
            multimodal_features_life_date, life_date_bal_loss = self.moe(multimodal_features_life_date, training=self.training)
            
            sleep_date_bal_loss = sleep_date_bal_loss if sleep_date_bal_loss is not None else torch.tensor(0.0, device=multimodal_features_sleep_date.device)
            life_date_bal_loss = life_date_bal_loss  if life_date_bal_loss  is not None else torch.tensor(0.0, device=multimodal_features_life_date.device)
            bal_loss = (sleep_date_bal_loss + life_date_bal_loss) / 2

        output = self.MHT([multimodal_features_sleep_date, multimodal_features_life_date])

        return output, bal_loss

if __name__ == "__main__":
    model = ETRIHumanUnderstandModel(
        proj_dim=128,
        use_moe=True,
        num_experts=3
    )
    
    print(model)