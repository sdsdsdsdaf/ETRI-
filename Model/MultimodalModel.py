import torch
import torch.nn as nn
from typing import Optional, Union

from .Encoder import ResidualFCEncoder, Conv1dEncoder, TabTransformerEncoder

def get_ae(ae_dict, key):
    return ae_dict[key] if ae_dict and key in ae_dict else None

class MultimodalModel(nn.Module):
    def __init__(self,
                 in_feature_dict: dict = None,
                 out_feature: int = 128,
                 encoder_dict: dict = None, 
                 ae_dict: nn.ModuleDict = None,
                 fusion: str = 'projection', 
                 proj_dim: int = 128,
                 fc_expand_feature: int = 128,
                 dropout_ratio: float = 0.3,
                 act=nn.ReLU,
                 hidden_layer_list:Optional[list[int]] = [128]):
        """
        encoder_dict: Dictionary mapping modality name to its encoder (nn.Module). 
                      If None, a default encoder_dict will be constructed.
        fusion: One of ['concat', 'attention', 'projection']
        proj_dim: If fusion is 'projection', output dim of each modality projection
        """
        super().__init__()

        assert fusion in ['concat', 'attention', 'projection'], f"Unsupported fusion type: {fusion}"
        assert not len(in_feature_dict) == 0, "in_feature_dict must be provided"

        if encoder_dict is None:
            encoder_dict = {
                'mAmbience': ResidualFCEncoder(in_feature=in_feature_dict['mAmbience'], hidden_layer_list=hidden_layer_list,act=act, dropout_ratio=dropout_ratio, out_feature=out_feature, ae=get_ae(ae_dict, 'mAmbience')),
                'mBle': ResidualFCEncoder(in_feature=in_feature_dict['mBle'], hidden_layer_list=hidden_layer_list, expand_feature=fc_expand_feature, act=act, dropout_ratio=dropout_ratio, out_feature=out_feature, ae=get_ae(ae_dict, 'mBle')),
                'mGps': Conv1dEncoder(in_ch=in_feature_dict['mGps'], out_ch=out_feature, ae=get_ae(ae_dict, 'mGps'), act=act),
                'mLight': ResidualFCEncoder(in_feature=in_feature_dict['mLight'], hidden_layer_list=hidden_layer_list, out_feature=out_feature, expand_feature=fc_expand_feature, act=act, dropout_ratio=dropout_ratio, ae=get_ae(ae_dict, 'mLight')),
                'mScreenStatus': ResidualFCEncoder(in_feature=in_feature_dict['mScreenStatus'], hidden_layer_list=hidden_layer_list, out_feature=out_feature, expand_feature=fc_expand_feature, act=act, dropout_ratio=dropout_ratio, ae=get_ae(ae_dict, 'mScreenStatus')),
                'mUsageStats': TabTransformerEncoder(num_features=in_feature_dict['mUsageStats'], dim=out_feature, ae=get_ae(ae_dict, 'mUsageStats')),
                'mWifi': ResidualFCEncoder(in_feature=in_feature_dict['mWifi'], out_feature=out_feature, hidden_layer_list=hidden_layer_list, act=act, dropout_ratio=dropout_ratio, ae=get_ae(ae_dict, 'mWifi')),
                'wHr': Conv1dEncoder(in_ch=in_feature_dict['wHr'], out_ch=out_feature, act=act, ae=get_ae(ae_dict, 'wHr')),
                'wLight': Conv1dEncoder(in_ch=in_feature_dict['wLight'], out_ch=out_feature, act=act,ae=get_ae(ae_dict, 'wLight')),
                'wPedo': Conv1dEncoder(in_ch=in_feature_dict['wPedo'], out_ch=out_feature, act=act, ae=get_ae(ae_dict, 'wPedo')),
            }
            
        self.encoders = nn.ModuleDict(encoder_dict)
        self.fusion_type = fusion

        if fusion == 'projection':
            self.projections = nn.ModuleDict({
                name: nn.Linear(encoder.out_features, proj_dim)
                for name, encoder in self.encoders.items()
            })
            self.shared_dim = proj_dim

        elif fusion == 'concat':
            self.shared_dim = sum(encoder.out_features for encoder in self.encoders.values())

        elif fusion == 'attention':
            self.shared_dim = list(self.encoders.values())[0].out_features
            self.attention = nn.MultiheadAttention(embed_dim=self.shared_dim, num_heads=4, batch_first=True)

        else:
            raise ValueError(f"Unsupported fusion type: {fusion}")

        self.shared_norm = nn.LayerNorm(self.shared_dim)

    def forward(self, inputs: dict):
        """
        inputs: dict of {modality_name: tensor of shape (B, ...)}
        """
        encoded = {}
        for name, x in inputs.items():
            encoded[name] = self.encoders[name](x)

        if self.fusion_type == 'concat':
            fused = torch.cat([encoded[name] for name in self.encoders], dim=-1)

        elif self.fusion_type == 'projection':
            fused = torch.stack([
                self.projections[name](encoded[name]) for name in self.encoders
            ], dim=1).mean(dim=1)  # Mean pooling

        elif self.fusion_type == 'attention':
            tokens = torch.stack([encoded[name] for name in self.encoders], dim=1)
            attn_out, _ = self.attention(tokens, tokens, tokens)
            fused = attn_out.mean(dim=1)

        return self.shared_norm(fused)

