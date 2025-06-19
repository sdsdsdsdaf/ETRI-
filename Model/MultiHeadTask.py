from typing import Optional
import torch.nn as nn
from .Block import ResidualFCBlock
from .Encoder import ResidualFCEncoder
import torch


class MultiHeadTask(nn.Module):
    def __init__(self, 
                 in_feature: int, 
                 act=nn.ReLU, 
                 dropout_ratio: float = 0.3,
                 heads: nn.ModuleDict = None,
                 ae_dict: nn.ModuleDict = None,
                 input_header_hidden_layer_list:Optional[list[int]] = [128],
                 heads_hidden_layer_list:Optional[list[int]] = [128],
                 back_bone_hidden_layer_list:Optional[list[int]] = [128]):
        super().__init__()

        self.q_need__data_list = [1, 0, 0, 1, 1, 1] #1 둘다 필요 0 sleep만
        
        self.input_header = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(in_feature * (1 + q)),
                ResidualFCEncoder(
                    in_feature=in_feature * (1 + q),
                    out_feature=back_bone_hidden_layer_list[0],
                    act=act,
                    dropout_ratio=dropout_ratio,
                    hidden_layer_list=input_header_hidden_layer_list,
                    ae=ae_dict
                )
            )
            for q in self.q_need__data_list
        ])

        self.backbone = nn.Sequential(
            nn.LayerNorm(back_bone_hidden_layer_list[0]),
            ResidualFCEncoder(
                in_feature=back_bone_hidden_layer_list[0], 
                out_feature=back_bone_hidden_layer_list[-1], 
                act=act, 
                dropout_ratio=dropout_ratio,
                hidden_layer_list=back_bone_hidden_layer_list[1:-1],
                ae=ae_dict
            )
        )

        q_list = [2, 2, 2, 3, 2, 2]

        self.heads = heads
        if heads is None:
            self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(in_feature),
                    ResidualFCEncoder(
                        in_feature=back_bone_hidden_layer_list[-1], 
                        out_feature=out_dim, 
                        act=act, 
                        dropout_ratio=dropout_ratio,
                        hidden_layer_list=heads_hidden_layer_list,
                        ae=ae_dict
                    )
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