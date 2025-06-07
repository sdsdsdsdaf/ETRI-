from typing import Optional
from Model.AutoEncoder import Conv1DAutoencoder, FCAutoencoder
from Model.Encoder import EffNetTransformerEncoder
from Model.MoE import MoE
from Model.MultimodalModel import MultimodalModel
from Model.MultiHeadTask import MultiHeadTask

import torch
import torch.nn as nn


def init_except_autoencoder(m, act=nn.GELU, exclude_backbone_ids=None):
    if exclude_backbone_ids and id(m) in exclude_backbone_ids:
        return

    if isinstance(m, (FCAutoencoder, Conv1DAutoencoder)):
        return

    if isinstance(m, nn.Linear):
        if isinstance(act, nn.GELU):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(act, nn.ReLU):
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
                 encoder_dict: dict = None,  # ✅ 실제 nn.ModuleDict
                 encoder_config: dict = None, 
                 multimodal_feature_dim: int = 256,
                 ae_dict: nn.ModuleDict = None,
                 fusion: str = 'projection', 
                 proj_dim: int = 256,
                 fc_expand_feature: int = 128,
                 dropout_ratio: float = 0.3,
                 act=nn.GELU,
                 hidden_layer_list:Optional[list[int]] = [128],
                 MHT_hidden_layer_list:Optional[list[int]] = [128],
                 num_experts: int = 6,
                 use_moe: bool = True,
                 heads = None,
                 gn = None,
                 experts=None):
        super().__init__()
        

        self.multimodal_model = MultimodalModel(
           mBle_in=mBle_in,
           img_size=img_size,
           encoder_dict=encoder_dict,
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
                act=act
            )
        else:
            self.moe = None

        self.MHT = MultiHeadTask(
            in_feature=self.multimodal_model.shared_dim,
            hidden_layer_list=MHT_hidden_layer_list,
            act=act,
            dropout_ratio=dropout_ratio,
            heads=heads,  # Define heads as needed
            ae_dict=ae_dict
        )

        init_except_autoencoder(self, act=act)

    def forward(self, inputs) -> torch.Tensor:
        # TODO: 날짜 별로 데이터 나누어  multimodal_features_sleep_date ,multimodal_features_life_date 다른 피처 생성

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
            multimodal_features_sleep_date = self.moe(multimodal_features_sleep_date)
            multimodal_features_life_date = self.moe(multimodal_features_life_date)

        output = self.MHT([multimodal_features_sleep_date, multimodal_features_life_date])

        return output

if __name__ == "__main__":
    model = ETRIHumanUnderstandModel(
        proj_dim=128,
        use_moe=True,
        num_experts=3
    )
    
    print(model)