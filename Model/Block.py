import torch
import torch.nn as nn

#TODO 후에 합칠 때  -> feature 개수 생각
class ResidualFCBlock(nn.Module):
    def __init__(
            self, 
            in_feature:int, 
            out_feature:int,
            expand_feature:int = 128, 
            act=nn.ReLU, 
            dropout_ratio=0.3,
        ):
        
        super().__init__()

        self.fc1 = nn.Linear(in_feature, expand_feature)
        self.act = act()
        self.fc2 = nn.Linear(expand_feature, out_feature)

        if dropout_ratio != 0:
            self.drop = nn.Dropout(dropout_ratio)

        self.proj = nn.Linear(in_feature, out_feature) if in_feature != out_feature else nn.Identity()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity=act.__name__.lower())
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x:torch.Tensor):
        residual = self.proj(x)
        out = self.fc1(x)
        out = self.act(out)
        out = self.fc2(out)

        if not self.drop is None:
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
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size//2, bias=False)
        self.norm = nn.BatchNorm1d(out_ch)
        self.act = act()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.act(out)
        return out