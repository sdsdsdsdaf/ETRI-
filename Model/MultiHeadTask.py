import torch.nn as nn
from .Block import ResidualFCBlock
from .Encoder import ResidualFCEncoder
import torch


class MultiHeadTask(nn.Module):
    def __init__(self, 
                 in_feature: int, 
                 expand_feature: int = 128, 
                 act=nn.ReLU, 
                 dropout_ratio: float = 0.3,
                 heads: nn.ModuleDict = None,
                 ae_dict: nn.ModuleDict = None,
                 hidden_layer_list=None):
        super().__init__()

        self.q_need__data_list = [1, 0, 0, 1, 1, 1] #1 둘다 필요 0 sleep만
        
        self.input_header = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(in_feature * (1 + q)),
                ResidualFCBlock(in_feature * (1 + q), in_feature, act=act, dropout_ratio=dropout_ratio)
            )
            for q in self.q_need__data_list
        ])

        self.backbone = nn.Sequential(
            nn.LayerNorm(in_feature),
            ResidualFCEncoder(
                in_feature=in_feature, 
                out_feature=in_feature, 
                act=act, 
                dropout_ratio=dropout_ratio,
                hidden_layer_list=hidden_layer_list,
                ae=ae_dict
            )
        )

        q_list = [2, 2, 2, 3, 2, 2]
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(in_feature),
                ResidualFCBlock(in_feature, expand_feature, act=act, dropout_ratio=dropout_ratio),
                nn.Dropout(dropout_ratio),
                nn.Linear(expand_feature, out_dim)  # ✅ out_dim = 2 or 3
            )   
            for out_dim in q_list   
        ])

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        x_concat = torch.cat(x, dim=-1)  # (B, 2*D)
        outputs = []

        for i, need_concat in enumerate(self.q_need__data_list):
            x_q = x_concat if need_concat else x[0]
            x_q = self.input_header[i](x_q)
            x_q = self.backbone(x_q)
            x_q = self.heads[i](x_q)  # (B, output_dim_i)
            outputs.append(x_q)

        return outputs