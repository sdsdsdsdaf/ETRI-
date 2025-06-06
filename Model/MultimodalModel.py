import torch
import torch.nn as nn
from typing import Optional, Union

from .Encoder import ResidualFCEncoder, EffNetTransformerEncoder, EffNetSimpleEncoder, EffNetPerformerEncoder



def get_ae(ae_dict, key):
    return ae_dict[key] if ae_dict and key in ae_dict else None

class MultimodalModel(nn.Module):
    def __init__(self,
                 mBle_in = 5,
                 img_size = 224,
                 CNNEncoder = EffNetTransformerEncoder,
                 CNN_hyper_dict = None,
                 out_feature: int = 128,
                 encoder_dict: dict = None, 
                 ae_dict: nn.ModuleDict = None,
                 fusion: str = 'projection', 
                 proj_dim: int = 128,
                 fc_expand_feature: int = 128,
                 dropout_ratio: float = 0.3,
                 act=nn.GELU,
                 hidden_layer_list:Optional[list[int]] = [128]):
        """
        encoder_dict: Dictionary mapping modality name to its encoder (nn.Module). 
                      If None, a default encoder_dict will be constructed.
        fusion: One of ['concat', 'attention', 'projection']
        proj_dim: If fusion is 'projection', output dim of each modality projection
        """
        super().__init__()

        assert fusion in ['concat', 'attention', 'projection'], f"Unsupported fusion type: {fusion}"
        if CNN_hyper_dict == None:
            CNN_hyper_dict = {}

        CNN_hyper_dict['out_dim'] = out_feature
        CNN_hyper_dict['model_name'] = 'mo'

        if encoder_dict is None:
            encoder_config = {
            'mAmbience': ("mobilenetv3_small_050", 128),
            'mGps':      ("mobilenetv3_small_050", 128),
            'mLight':    ("mobilenetv3_small_050", 128),
            'mScreenStatus': ("mobilenetv3_small_050", 128),
            'mUsageStats':   ("mobilenetv3_small_050", 128),
            'mWifi':     ("mobilenetv3_small_050", 128),
            'wHr':       ("mobilenetv3_small_050", 128),
            'wLight':    ("mobilenetv3_small_050", 128),
            'wPedo':     ("mobilenetv3_small_050", 128),
            'mActivity': ("mobilenetv3_small_050", 128),
            'mACStatus': ("mobilenetv3_small_050", 128),
            'mAppUsage': ("mobilenetv3_small_050", 128),
        }

            # 2. encoder_dict 생성 (공통 파라미터는 고정)
            encoder_dict = {
                modal: CNNEncoder(
                    model_name=model_name,
                    out_dim=out_dim,
                    act=ae_dict,
                )
                for modal, (model_name, out_dim) in encoder_config.items()
            }

            encoder_dict['mBle'] = ResidualFCEncoder(
                in_feature=mBle_in,
                hidden_layer_list=hidden_layer_list,
                expand_feature=fc_expand_feature,
                out_feature=out_feature,
                act=act,
                dropout_ratio=dropout_ratio,
                ae=None
            )
            
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

