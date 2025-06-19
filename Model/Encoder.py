from typing import Optional, Union
import torch
import torch.nn as nn
import timm
import torch.nn.functional as F

from .AutoEncoder import FCAutoencoder, Conv1DAutoencoder
from .Block import ResidualFCBlock, LearnablePositionalEncoding, SinusoidalPositionalEncoding, PerformerWithFFNBlock


def get_activation_function(act_module):
    if act_module == nn.ReLU:
        return F.relu
    elif act_module == nn.GELU:
        return F.gelu
    elif act_module == nn.LeakyReLU:
        return F.leaky_relu
    elif act_module  == nn.Sigmoid:
        return F.sigmoid
    elif act_module == nn.Tanh:
        return F.tanh
    else:
        return F.relu

class EffNetTransformerEncoder(nn.Module):
    def __init__(
            self, 
            model_name="efficientnet_b1",
            seq_len=49, 
            out_dim=256, 
            act=nn.GELU, 
            nhead=8, 
            num_layers=3, 
            use_learnable_pe=True):
        
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True)
        self.backbone.reset_classifier(0)

        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            feat = self.backbone.forward_features(dummy_input)
        feature_dim = feat.shape[1]

        self.proj = nn.Linear(feature_dim, out_dim)
        self.pe = LearnablePositionalEncoding(seq_len=seq_len, dim=out_dim) if use_learnable_pe else SinusoidalPositionalEncoding(seq_len=seq_len, dim=out_dim)
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=out_dim, nhead=nhead, activation=get_activation_function(act),batch_first=True)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(out_dim))

    def forward(self, x:torch.Tensor):
        x = self.backbone.forward_features(x)   #(B, 1280, 7, 7)
        x = x.flatten(2).transpose(1, 2)        #(B, 49, 1280)
        x = self.proj(x)                        #(B, 49, out_dim)
        x = self.pe(x)                          #(B, 49, out_dim)
        out = self.transformer(x).mean(dim=1)   #(B, 49, out_dim)
        return out
    
class EffNetSimpleEncoder(nn.Module):
    def __init__(self, 
                 model_name="efficientnet_b1", 
                 act=nn.GELU, 
                 out_dim=128,
                 dropout_ratio=0.1):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True)
        self.backbone.reset_classifier(0)

        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            feat = self.backbone.forward_features(dummy_input)
        feature_dim = feat.shape[1]

        self.proj = nn.Linear(feature_dim, out_dim)
        self.act = act()
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(p=dropout_ratio)

    def forward(self, x:torch.Tensor):
        x = self.backbone.forward_features(x)  # (B, C, H, W) or (B, C, 1, 1)
        x = x.mean(dim=(2, 3)) if x.ndim == 4 else x  # GAP
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        out = self.dropout(x)
        return out
    

try:
    from performer_pytorch import SelfAttention
except ImportError:
    raise ImportError("Please install performer-pytorch: pip install performer-pytorch")

class EffNetPerformerEncoder(nn.Module):
    def __init__( 
            self, 
            model_name="efficientnet_b1",
            seq_len=49, 
            out_dim=256, 
            nhead=8, 
            act=nn.GELU, 
            num_layers=3, 
            dropout_ratio=0.1,
            use_learnable_pe=True):
        
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True)
        self.backbone.reset_classifier(0)
        latent_dim = 4*out_dim  

        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            feat = self.backbone.forward_features(dummy_input)
        feature_dim = feat.shape[1]

        self.proj = nn.Linear(feature_dim, out_dim)
        self.pe = LearnablePositionalEncoding(seq_len=seq_len, dim=out_dim) if use_learnable_pe else SinusoidalPositionalEncoding(seq_len=seq_len, dim=out_dim)
        
        self.attn_blocks = nn.ModuleList([
            PerformerWithFFNBlock(d_model=out_dim, nhead=nhead, dim_feedforward=latent_dim, dropout=dropout_ratio, act=act)
            for _ in range(num_layers)
        ])

    def forward(self, x:torch.Tensor):
        x = self.backbone.forward_features(x)   #(B, 1280, 7, 7)
        x = x.flatten(2).transpose(1, 2)        #(B, 49, 1280)
        x = self.proj(x)                        #(B, 49, out_dim)
        x = self.pe(x)                          #(B, 49, out_dim)
        for attn_block in self.attn_blocks:     #(B, 49, out_dim)
            x = attn_block(x)
        
        out = x.mean(dim=1)                     #(B, out_dim)
        return out



class ResidualFCEncoder(nn.Module):
    def __init__(
        self, 
        in_feature:int, 
        out_feature:int,
        act=nn.ReLU, 
        dropout_ratio=0.3,
        ae:Optional[Union[FCAutoencoder]]=None,
        use_bn = False,
        hidden_layer_list: Optional[list[int]] = None, # Autoencoder
    ):
        
        assert len(hidden_layer_list) % 3 == 1, "hidden_layer_list must be 3n + 1."
        self.hidden_layer_list = hidden_layer_list.copy()

        super().__init__()
        self.ae = ae
        if ae is not None:
            in_feature = ae.out_features

        if hidden_layer_list is None:
            self.hidden_layer_list = [128]

        self.hidden_layer_list.append(out_feature)  # 마지막 레이어는 out_feature로 설정
        self.hidden_layer_list.insert(0, in_feature)  # 첫번째 레이어는 in_feature로 설정
        
        self.out_features = out_feature
        layers = []
        for i in range(len(self.hidden_layer_list) // 3):
            layers.append(
                ResidualFCBlock(
                    in_feature=self.hidden_layer_list[3*i], 
                    out_feature=self.hidden_layer_list[3*i + 2], 
                    expand_feature=self.hidden_layer_list[3*i + 1], 
                    act=act,
                    use_bn=use_bn,
                    dropout_ratio=dropout_ratio
                )
            )


        self.net = nn.Sequential(*layers) 

    def forward(self, x:torch.Tensor):
        if self.ae is not None:
           _, x, _ = self.ae(x)
        out = self.net(x)
        return out


