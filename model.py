from Model.MoE import MoE
from Model.MutilmodalModel import MultimodalModel

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, 
                 encoder_dict: dict = None, 
                 ae_dict: nn.ModuleDict = None,
                 fusion: str = 'projection', 
                 proj_dim: int = 128,
                 use_moe: bool = True,
                 num_experts: int = 3):
        super().__init__()
        
        self.multimodal_model = MultimodalModel(
            encoder_dict=encoder_dict, 
            ae_dict=ae_dict, 
            fusion=fusion, 
            proj_dim=proj_dim
        )
        
        if use_moe:
            self.moe = MoE(
                input_dim=self.multimodal_model.shared_dim,
                num_experts=num_experts
            )
        else:
            self.moe = None

    def forward(self, inputs: dict):

        # TODO: 날짜 별로 데이터 나누어 moe1, moe2로 결과 생성
        features = self.multimodal_model(inputs)
        
        if self.moe is not None:
            features = self.moe(features)
        
        return features
    

if __name__ == "__main__":
    model = Model(
        proj_dim=128,
        use_moe=True,
        num_experts=3
    )
    
    print(model)