from typing import Optional, Union
import torch
import torch.nn as nn

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .AutoEncoder import FCAutoencoder, Conv1DAutoencoder
from .Block import ResidualFCBlock, Conv1dBlock

class ResidualFCEncoder(nn.Module):
    def __init__(
        self, 
        in_feature:int, 
        out_feature:int,
        expand_feature:int = 128, 
        act=nn.ReLU, 
        dropout_ratio=0.3,
        ae:Optional[Union[FCAutoencoder, Conv1DAutoencoder]]=None, # Autoencoder
    ):
        
        super().__init__()
        self.ae = ae
        if ae is not None:
            in_feature = ae.encoder[-1].out_features
        
        self.out_features = out_feature

        self.fcBlock1 = ResidualFCBlock(
            in_feature, 
            out_feature, 
            expand_feature=expand_feature, 
            act=act, 
            dropout_ratio=dropout_ratio,
        )

        if dropout_ratio != 0:
            self.drop = nn.Dropout(dropout_ratio)

    def forward(self, x:torch.Tensor):
        if self.ae is not None:
            x, _ = self.ae(x)
        out = self.fcBlock1(x)
        return out



class Conv1dEncoder(nn.Module): #Out Channel 64~128
    def __init__(
            self, 
            in_ch:int,
            out_ch=128,
            kernel_size = 3,
            padding = 1,
            act=nn.ReLU,
            ae:Optional[Union[FCAutoencoder, Conv1DAutoencoder]]=None
            ):
        #TODO: 필요하다면 stride, dilation, groups 추가
        
        super().__init__()
        self.ae = ae  # Autoencoder for feature embedding
        self.out_features = out_ch

        if ae is not None:
            in_ch = ae.encoder[-1].out_features

        self.net = nn.Sequential(
            Conv1dBlock(in_ch, out_ch//4, kernel_size=kernel_size, padding=padding, act=act),  # t=24
            Conv1dBlock(out_ch//4, out_ch//4, kernel_size=3, padding=1, act=act),     # t=24
            nn.AvgPool1d(kernel_size=2),                       # → t=12

            Conv1dBlock(out_ch//4, out_ch//2, kernel_size=kernel_size, padding=padding, act=act),    # t=12
            Conv1dBlock(out_ch//2, out_ch//2, kernel_size=kernel_size, padding=padding, act=act),   # t=12
            nn.AvgPool1d(kernel_size=2),                       # → t=6

            Conv1dBlock(out_ch//2, out_ch, kernel_size=kernel_size, padding=padding, act=act),   # t=6
            Conv1dBlock(out_ch, out_ch, kernel_size=3, padding=1, act=act),  # t=6
            nn.AdaptiveAvgPool1d(1),   # t=6
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity=act.__name__.lower())
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)  # (B, C, 1) → (B, C)
    


class TabTransformerEncoder(nn.Module):
    def __init__(self,
                  num_features:int, 
                  dim=64, 
                  depth=2, 
                  heads=4,
                  dropout=0.1, 
                  use_cls=False, 
                  act=nn.ReLU,
                  ae:Optional[Union[FCAutoencoder, Conv1DAutoencoder]]=None):
        super().__init__()
        self.num_features = num_features
        self.use_cls = use_cls

        self.ae = ae  # Autoencoder for feature embedding
        self.out_features = dim
        
        # 각 feature를 개별적으로 임베딩 (1D → dim)
        if ae is not None:
            self.feature_embeddings = nn.ModuleList([
                nn.Linear(ae.encoder[-1].out_features, dim) for _ in range(num_features)
            ])
        else:
            self.feature_embeddings = nn.ModuleList([
                nn.Linear(1, dim) for _ in range(num_features)
            ])
        
        if use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.norm = nn.LayerNorm(dim)
        self.output_proj = nn.Linear(dim, dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity=act.__name__.lower())
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):  # x: (B, F)
        B, F = x.shape
        assert F == self.num_features, f"Expected {self.num_features} features, got {F}"

        # 각 feature별 임베딩 (독립적으로 Linear)
        token_list = [self.feature_embeddings[i](x[:, i:i+1]) for i in range(F)]
        x = torch.stack(token_list, dim=1)  # (B, F, D)

        if self.use_cls:
            cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
            x = torch.cat([cls_token, x], dim=1)          # (B, F+1, D)

        x = self.transformer(x)                           # (B, F(+1), D)

        if self.use_cls:
            out = x[:, 0]  # (B, D)
        else:
            out = x.mean(dim=1)  # (B, D) — mean pooling

        out = self.norm(out)
        out = self.output_proj(out)
        return out