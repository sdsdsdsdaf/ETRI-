import torch
import torch.nn as nn
from typing import Optional, Union

from .Encoder import ResidualFCEncoder, EffNetTransformerEncoder, EffNetSimpleEncoder, EffNetPerformerEncoder
from .Block import PerformerWithFFNBlock as Perfo




def get_ae(ae_dict, key):
    return ae_dict[key] if ae_dict and key in ae_dict else None

class MultimodalModel(nn.Module):
    def __init__(self,
                 mBle_in=10,
                 img_size=224,
                 base_model = None,
                 base_block = EffNetSimpleEncoder,
                 encoder_dict: dict = None,  
                 encoder_config: dict = None,  
                 out_feature: int = 256,
                 ae_dict: nn.ModuleDict = None,
                 fusion: str = 'projection',
                 proj_dim: int = 256,
                 fc_expand_feature: int = 128,
                 dropout_ratio: float = 0.3,
                 act=nn.GELU,
                 hidden_layer_list: Optional[list[int]] = [128]):
        super().__init__()
        assert fusion in ['concat', 'attention', 'projection'], f"Unsupported fusion type: {fusion}"
        self.batchNorm = nn.BatchNorm1d(mBle_in)
        if base_model is None:
            base_model = "mobilenetv3_small_100"


        # ✅ encoder_dict가 없으면 encoder_config를 우선 사용
        if encoder_dict is None:
            encoder_dict = {}

            if encoder_config is not None:
                # 🔹 Optuna 설정 기반: (encoder_class, kwargs)
                for modal, (encoder_class, kwargs) in encoder_config.items():
                    encoder_dict[modal] = encoder_class(**kwargs)
            else:
                # 🔹 Baseline 설정: 모두 동일한 CNN encoder 사용
                default_modalities = [
                    'mAmbience', 'mGps', 'mLight', 'mScreenStatus', 'mUsageStats', 'mWifi',
                    'wHr', 'wLight', 'wPedo', 'mActivity', 'mACStatus'
                ]
                for modal in default_modalities:
                    encoder_dict[modal] = base_block(
                        model_name=base_model,
                        out_dim=out_feature,
                        act=act,
                    )

            # 🔹 mBle는 항상 ResidualFCEncoder 사용
            encoder_dict["mBle"] = ResidualFCEncoder(
                in_feature=mBle_in,
                hidden_layer_list=hidden_layer_list,
                out_feature=out_feature,
                act=act,
                dropout_ratio=dropout_ratio,
                ae=ae_dict
            )

        self.encoders = nn.ModuleDict(encoder_dict)
        self.fusion_type = fusion

        if fusion == 'projection':
            self.projections = nn.ModuleDict({
                name: nn.Linear(out_feature, proj_dim)
                for name, encoder in self.encoders.items()
            })
            self.shared_dim = proj_dim

        elif fusion == 'concat':
            self.shared_dim = sum(out_feature for encoder in self.encoders.values())

        elif fusion == 'attention':
            self.shared_dim = out_feature
            self.attention = Perfo(d_model=self.shared_dim, nhead=4)

        else:
            raise ValueError(f"Unsupported fusion type: {fusion}")

        self.shared_norm = nn.LayerNorm(self.shared_dim)

    def forward(self, inputs, mBle_inputs=None, modal_listname=None):
        """
        
        """
        modal_listname_list = [modal_listitem[0] for modal_listitem in modal_listname]
        modal_listname_list.remove('mBle')
        encoded = {}
        for idx, name in enumerate(modal_listname_list):
            x = inputs[:,idx]
            encoded[name] = self.encoders[name](x)
        
        mBle_inputs = mBle_inputs.squeeze(1)
        encoded['mBle'] = self.encoders['mBle'](self.batchNorm(mBle_inputs))

        if self.fusion_type == 'concat':
            fused = torch.cat([encoded[name] for name in self.encoders], dim=-1)

        elif self.fusion_type == 'projection':
            fused = torch.stack([
                self.projections[name](encoded[name]) for name in self.encoders
            ], dim=1).mean(dim=1)  # Mean pooling

        elif self.fusion_type == 'attention':
            tokens = torch.stack([encoded[name] for name in self.encoders], dim=1)
            attn_out = self.attention(tokens)
            fused = attn_out.mean(dim=1)

        return self.shared_norm(fused)

