import torch
import torch.nn as nn
import math

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, seq_len=49, dim=256):
        super().__init__()
        pe = torch.zeros(seq_len, dim)
        position = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = nn.Parameter(pe.unsqueeze(0), requires_grad=True)


    def forward(self, x):
        return  x + self.pe[:, :x.size(1)]
    
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, seq_len=49, dim=256):
        super().__init__()
        pe = torch.zeros(seq_len, dim)
        position = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, seq_len, dim)

    def forward(self, x):  # x: (B, seq_len, dim)
        return x + self.pe[:, :x.size(1)]
    
from performer_pytorch import SelfAttention

class PerformerWithFFNBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, act=nn.GELU):
        super().__init__()
        self.self_attn = SelfAttention(dim=d_model, heads=nhead)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # Position-wise Feed Forward Network (FFN)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = act()
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, src):                       # src (B, 49, d_model)
        # Self Attention + Residual + Norm
        src2 = self.self_attn(src)                # (B, 49, d_model)
        src = src + self.dropout1(src2)           # (B, 49, d_model)
        src = self.norm1(src)                     # (B, 49, d_model)
        
        # Feed Forward Network + Residual + Norm
        src2 = self.linear2(
            self.dropout2(
                self.activation(
                    self.linear1(src))))          # (B, 49, d_model)
        
        
        src = src + self.dropout3(src2)           # (B, 49, d_model)
        src = self.norm2(src)                     # (B, 49, d_model)
        
        return src



#TODO 후에 합칠 때  -> feature 개수 생각
class ResidualFCBlock(nn.Module):
    def __init__(
            self, 
            in_feature:int, 
            out_feature:int,
            expand_feature:int=128, 
            act=nn.ReLU, 
            dropout_ratio=0.3,
            use_bn=False, # Autoencoder
        ):
        
        super().__init__()

        self.fc1 = nn.Linear(in_feature, expand_feature)
        self.act = act()
        self.fc2 = nn.Linear(expand_feature, out_feature)

        if dropout_ratio is not None and dropout_ratio > 0:
            self.drop = nn.Dropout(dropout_ratio)
        if use_bn:
            self.norm1 = nn.BatchNorm1d(expand_feature)
            self.norm2 = nn.BatchNorm1d(out_feature)
            
        self.proj = nn.Linear(in_feature, out_feature) if in_feature != out_feature else nn.Identity()

    def forward(self, x:torch.Tensor):
        residual = self.proj(x)
        out = self.fc1(x)
        if hasattr(self, 'norm1'):
            out = self.norm1(out)
        out = self.act(out)
        out = self.fc2(out)
        if hasattr(self, 'norm2'):
            out = self.norm2(out)
        out = self.act(out)

        if self.drop is not None:
            out = self.drop(out)

        out += residual
        return out
    
class Conv1dBlock(nn.Module): #Residual Block로 변경 가능
    def __init__( self, 
            in_ch:int,
            out_ch:int,
            kernel_size:int = 3,
            stride:int = 1,
            padding:int = 1,
            act=nn.ReLU,):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, bias=False)
        self.norm = nn.BatchNorm1d(out_ch)
        self.act = act()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.act(out)
        return out
    
