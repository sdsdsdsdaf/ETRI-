import torch.nn as nn
import torch

class GatingNetwork(nn.Module):
    def __init__(self, input_dim: int, num_experts: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_experts)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        return self.softmax(logits)

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, act = nn.GELU):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):  # x: (B, D)
        return self.layer(x)

class MoE(nn.Module): #Input: (B, D), Output: (B, D)
    def __init__(self, 
                 input_dim: int,
                 num_experts: int = 3,
                 gating_network: nn.Module = None,
                 experts: nn.ModuleList = None,
                 act=nn.GELU):
        super().__init__()

        if gating_network is None:
            self.gating_network = GatingNetwork(input_dim, num_experts)
        else:
            self.gating_network = gating_network

        if experts is None:
            self.experts = nn.ModuleList([Expert(input_dim, act=act) for _ in range(num_experts)])
        else:
            self.experts = experts
        self.num_experts = num_experts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_scores = self.gating_network(x) # (B, E)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # (B, E, D)
        output = (expert_outputs * gate_scores.unsqueeze(-1)).sum(dim=1)             # (B, D)
        return output

    

