from Model.MoE import MoE
from Model.MultimodalModel import MultimodalModel
from Model.Block import MultiHeadTask

import torch
import torch.nn as nn

class ETRIHumanUnderstandModel(nn.Module):
    def __init__(self, 
                 in_feature_dict:dict,
                 multimodal_feature_dim:int = 128, 
                 encoder_dict: dict = None, 
                 ae_dict: nn.ModuleDict = None,
                 fusion: str = 'projection', 
                 proj_dim: int = 128,
                 num_experts: int = 3,
                 fc_expand_feature: int = 128,
                 dropout_ratio: float = 0.3,
                 act= nn.ReLU,
                 use_moe: bool = True,
                 gn = None,
                 experts=None):
        super().__init__()
        
        self.multimodal_model = MultimodalModel(
            in_feature_dict=in_feature_dict,
            out_feature=multimodal_feature_dim,
            encoder_dict=encoder_dict, 
            ae_dict=ae_dict, 
            fusion=fusion, 
            proj_dim=proj_dim,
            fc_expand_feature=fc_expand_feature,
            dropout_ratio=dropout_ratio,
            act=act,
        )

        self.use_moe = use_moe
        if self.use_moe:

            self.norm_before_moe = nn.Sequential(
                nn.LayerNorm(multimodal_feature_dim),
                nn.Dropout(0.1)
            )

            self.moe = MoE(
                input_dim=multimodal_feature_dim,
                num_experts=num_experts,
                gating_network=gn,
                experts=experts,
                act=act
            )
        else:
            self.moe = None

        self.MHT = MultiHeadTask(
            in_feature=multimodal_feature_dim,
            expand_feature=fc_expand_feature,
            act=act,
            dropout_ratio=dropout_ratio,
            heads=None,  # Define heads as needed
            ae_dict=ae_dict
        )

    def forward(self, sleep_date_inputs: dict, lifelog_date_inputs: dict) -> torch.Tensor:
        # TODO: 날짜 별로 데이터 나누어  multimodal_features_sleep_date ,multimodal_features_life_date 다른 피처 생성


        multimodal_features_sleep_date = self.multimodal_model(sleep_date_inputs)
        multimodal_features_life_date = self.multimodal_model(lifelog_date_inputs)

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