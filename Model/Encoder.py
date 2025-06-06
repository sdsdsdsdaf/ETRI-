from typing import Optional, Union
import torch
import torch.nn as nn
import timm

from .AutoEncoder import FCAutoencoder, Conv1DAutoencoder
from .Block import ResidualFCBlock, Conv1dBlock, LearnablePositionalEncoding, SinusoidalPositionalEncoding, PerformerWithFFNBlock


def init_except_autoencoder(m, act="GELU"):
    # Autoencoder 모듈 제외
    if isinstance(m, (FCAutoencoder, Conv1DAutoencoder)):
        return
    # Linear 초기화
    if isinstance(m, nn.Linear) and act == "GELU":
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear) and act == "ReLU":
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    # LayerNorm 초기화
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

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

        self.proj = nn.Linear(self.backbone.num_features, out_dim)
        self.pe = LearnablePositionalEncoding(seq_len=seq_len, dim=out_dim) if use_learnable_pe else SinusoidalPositionalEncoding(seq_len=seq_len, dim=out_dim)
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=out_dim, nhead=nhead, activation=act,batch_first=True)
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
        self.proj = nn.Linear(self.backbone.num_features, out_dim)
        self.act = act()
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(p=dropout_ratio)

    def forward(self, x:torch.Tensor):
        x = self.backbone(x)
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

        self.proj = nn.Linear(self.backbone.num_features, out_dim)
        self.pe = LearnablePositionalEncoding(seq_len=seq_len, dim=out_dim) if use_learnable_pe else SinusoidalPositionalEncoding(seq_len=seq_len, dim=out_dim)
        
        self.attn_blocks = nn.ModuleList([
            PerformerWithFFNBlock(d_model=out_dim, heads=nhead, dim_feedforward=latent_dim, dropout=dropout_ratio, act=act)
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
        hidden_layer_list: Optional[list[int]] = None, # Autoencoder
    ):
        
        assert len(hidden_layer_list) % 3 == 1, "hidden_layer_list must be 3n + 1."


        super().__init__()
        self.ae = ae
        if ae is not None:
            in_feature = ae.out_features

        if hidden_layer_list is None:
            hidden_layer_list = [128]

        hidden_layer_list.append(out_feature)  # 마지막 레이어는 out_feature로 설정
        hidden_layer_list.insert(0, in_feature)  # 첫번째 레이어는 in_feature로 설정
        
        self.out_features = out_feature
        layers = []
        for i in range(len(hidden_layer_list) // 3):
            layers.append(
                ResidualFCBlock(
                    in_feature=hidden_layer_list[3*i], 
                    out_feature=hidden_layer_list[3*i + 2], 
                    expand_feature=hidden_layer_list[3*i + 1], 
                    act=act, 
                    dropout_ratio=dropout_ratio
                )
            )


        self.net = nn.Sequential(*layers) 

    def forward(self, x:torch.Tensor):
        if self.ae is not None:
           _, x, _ = self.ae(x)
        out = self.net(x)
        return out


def _make_stage(in_ch:int, 
                out_ch:int, 
                kernel_size:int = 3, 
                padding:int = 1, 
                act=nn.ReLU, 
                num_blocks:int = 2):
    """
    Helper function to create a stage of Conv1dBlocks.
    """
    layers = []
    for _ in range(num_blocks):
        layers.append(Conv1dBlock(in_ch, out_ch, kernel_size=kernel_size, padding=padding, act=act))
        in_ch = out_ch  # Downsample after each stage
    return nn.Sequential(*layers)



class Conv1dEncoder(nn.Module): #Out Channel 64~128
    def __init__(
            self, 
            in_ch:int,
            out_ch=128,
            kernel_size = 3,
            padding = 1,
            act=nn.ReLU,
            ae:Optional[Union[Conv1DAutoencoder]]=None,
            cnn_list:list = [2, 2, 2]  # 각 단계의 CNN 레이어 수
            ):
        #TODO: 필요하다면 stride, dilation, groups 추가
        
        super().__init__()
        self.ae = ae  # Autoencoder for feature embedding
        self.out_features = out_ch

        if ae is not None:
            in_ch = ae.out_features

        if len(cnn_list) != 3:
            raise ValueError("cnn_list must contain exactly 3 integers, e.g., [2, 2, 2]")

        stage1 = _make_stage(in_ch, out_ch//4, kernel_size=kernel_size, padding=padding, act=act, num_blocks=cnn_list[0])  # t=24
        stage2 = _make_stage(out_ch//4, out_ch//2, kernel_size=kernel_size, padding=padding, act=act, num_blocks=cnn_list[1])
        stage3 = _make_stage(out_ch//2, out_ch, kernel_size=kernel_size, padding=padding, act=act, num_blocks=cnn_list[2])

        self.net = nn.Sequential(
            stage1,   # t=24
            nn.AvgPool1d(kernel_size=2),                       # → t=12

            stage2,  # t=12
            nn.AvgPool1d(kernel_size=2),                       # → t=6

            stage3, # t=6
            nn.AdaptiveAvgPool1d(1),   # t=6
        )

    def forward(self, x):
        if self.ae is not None:
            x, _, _ = self.ae(x)
        return self.net(x).squeeze(-1)  # (B, C, 1) → (B, C)
    


class TabTransformerEncoder(nn.Module): #TODO: AE의 구조를 조금 더 유연하게 변경할 수 있도록 개선
    def __init__(self,
                  num_features:int, 
                  dim=64, 
                  depth=2, 
                  heads=4,
                  dropout=0.1, 
                  use_cls=False, 
                  act=nn.ReLU,
                  ae_list:Optional[list[Union[FCAutoencoder]]]=None):
        super().__init__()
        self.num_features = num_features
        self.use_cls = use_cls

        self.ae = ae_list  # Autoencoder for feature embedding
        self.out_features = dim
        
        if self.ae is not None: # self.ae는 ae_list를 가리킴
            assert len(self.ae) == num_features, \
        f"Length of ae_list ({len(self.ae)}) must match num_features ({num_features})."
                
            for idx, single_ae in enumerate(self.ae):
                in_feature_of_ae = None

                assert isinstance(single_ae, FCAutoencoder), \
                f"Element {idx} in ae_list is not an FCAutoencoder instance."
            # FCAutoencoder의 encoder는 nn.Sequential, 첫번째는 Linear
                if hasattr(single_ae, 'encoder') and \
                isinstance(single_ae.encoder, nn.Sequential) and \
                len(single_ae.encoder) > 0 and \
                isinstance(single_ae.encoder[0], nn.Linear):
                    in_feature_of_ae = single_ae.encoder[0].in_features
                    assert in_feature_of_ae == 1, \
                        f"Input feature dimension for FCAutoencoder at index {idx} must be 1, got {in_feature_of_ae}."
                else: # FCAutoencoder 구조가 예상과 다른 경우
                    raise ValueError(f"Unexpected structure for FCAutoencoder at index {idx}.")

        # 각 feature를 개별적으로 임베딩 (1D → dim) 그레서 AE를 사용하는 경우, AE의 출력 차원에 맞춰 Linear 레이어를 정의
        if self.ae is not None:
            self.feature_embeddings = nn.ModuleList([
                nn.Linear(self.ae[i].out_features, dim) for i in range(num_features) 
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
        
    def forward(self, x_input: torch.Tensor) -> torch.Tensor:
        B, F_orig = x_input.shape
        assert F_orig == self.num_features, \
            f"Expected {self.num_features} features, got {F_orig}"

        processed_tokens = []
        if self.ae is not None:
            # AE 사용 시: 각 feature에 AE 적용 후, 해당 feature_embedding 적용
            # 가정: self.ae는 FCAutoencoder(in_dim=1, latent_dim=AE_LATENT_DIM)
            #       self.feature_embeddings[i]는 Linear(AE_LATENT_DIM, dim)
            for i in range(F_orig):
                current_feature_slice = x_input[:, i:i+1] # (B, 1)
                latent_representation, _, _ = self.ae[i](current_feature_slice) # (B, 1) -> (B, AE_LATENT_DIM)
                token = self.feature_embeddings[i](latent_representation) # (B, dim)
                processed_tokens.append(token)
        else:
            # AE 미사용 시: 각 feature에 feature_embedding 직접 적용
            # 가정: self.feature_embeddings[i]는 Linear(1, dim)
            for i in range(F_orig):
                current_feature_slice = x_input[:, i:i+1] # (B, 1)
                token = self.feature_embeddings[i](current_feature_slice) # (B, dim)
                processed_tokens.append(token)

        x = torch.stack(processed_tokens, dim=1) # (B, F_orig, dim)

        if self.use_cls:
            cls_token = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_token, x], dim=1)      # (B, F_orig + 1, dim)

        x = self.transformer(x)                       # (B, F_orig + 1, dim)

        if self.use_cls:
            out = x[:, 0]                             # (B, dim)
        else:
            out = x.mean(dim=1)                       # (B, dim)

        out = self.norm(out)
        # output_proj 레이어가 __init__에 정의되어 있다면 사용
        if hasattr(self, 'output_proj'):
            out = self.output_proj(out)
            
        return out