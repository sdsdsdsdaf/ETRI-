import torch.nn as nn
import torch
import torch.nn.functional as F

def soft_gating(logits: torch.Tensor):
    """
    Args:
        logits: [B, E]
    Returns:
        gating_weights: [B, E] (softmax 확률)
        probs: same as gating_weights (load balancing 등에 사용)
    """
    probs = F.softmax(logits, dim=-1)
    return probs, probs

def top_k_gating(logits: torch.Tensor, k: int, eps: float = 1e-9):
    """
    Args:
        logits: [B, E]
        k: select top-k
    Returns:
        gating_weights: [B, E], top-k만 남고 정규화된 확률
        probs: [B, E], softmax(logits) 확률 (load balancing용)
    """
    probs = F.softmax(logits, dim=-1)  # [B, E]
    topk_vals, topk_idx = torch.topk(probs, k, dim=-1)  # [B, k]
    mask = torch.zeros_like(probs)
    mask.scatter_(1, topk_idx, 1.0)
    masked = probs * mask
    denom = masked.sum(dim=-1, keepdim=True).clamp(min=eps)
    gating_weights = masked / denom
    return gating_weights, probs

def noisy_top_k_gating(logits: torch.Tensor, k: int, noise_std: float, training: bool = True, eps: float = 1e-9):
    """
    Args:
        logits: [B, E]
        k: select top-k
        noise_std: float
        training: bool
    Returns:
        gating_weights: [B, E], noisy top-k 결과
        probs: [B, E], softmax(logits) 확률 (noise 전, load balancing용)
    """
    probs = F.softmax(logits, dim=-1)
    if training and noise_std > 0:
        noise = torch.randn_like(logits) * noise_std
        logits_noisy = logits + noise
    else:
        logits_noisy = logits
    probs_noisy = F.softmax(logits_noisy, dim=-1)
    topk_vals, topk_idx = torch.topk(probs_noisy, k, dim=-1)
    mask = torch.zeros_like(probs_noisy)
    mask.scatter_(1, topk_idx, 1.0)
    masked = probs_noisy * mask
    denom = masked.sum(dim=-1, keepdim=True).clamp(min=eps)
    gating_weights = masked / denom
    return gating_weights, probs

def load_balancing_loss(probs: torch.Tensor):
    """
    Args:
        probs: [B, E], softmax(logits) 확률
    Returns:
        scalar tensor
    """
    B, E = probs.shape
    importance = probs.mean(dim=0)  # [E]
    loss = E * torch.sum(importance * importance)
    return loss

class GatingNetwork(nn.Module):
    def __init__(self, input_dim: int, num_experts: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_experts)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        return logits

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
                 hidden_dim = 64,
                 num_experts: int = 3,
                 gating_network: nn.Module = None,
                 experts: nn.ModuleList = None,
                 act=nn.GELU,
                 gating_type: str = 'soft',   # 'soft', 'topk', 'noisy_topk'
                 k: int = 1,
                 noise_std: float = 0.0,
                 lambda_bal: float = 0.0,):
        super().__init__()

        self.gating_type = gating_type.lower()

        assert self.gating_type in ['soft', 'topk', 'noisy_topk'], \
            f"Unsupported gating_type: {self.gating_type}. Choose from 'soft', 'topk', 'noisy_topk'."
        
        self.k = k
        self.noise_std = noise_std
        self.lambda_bal = lambda_bal

        if gating_network is None:
            self.gating_network = GatingNetwork(input_dim, num_experts)
        else:
            self.gating_network = gating_network

        if experts is None:
            self.experts = nn.ModuleList([Expert(input_dim, hidden_dim=hidden_dim,act=act) for _ in range(num_experts)])
        else:
            self.experts = experts
        self.num_experts = num_experts

    def forward(self, x: torch.Tensor, training: bool = True):
        """
        Args:
            x: [B, D]
            training: bool (noisy 적용 여부)
        Returns:
            output: [B, D]
            bal_loss: scalar tensor or None
        """
        logits = self.gating_network(x)  # [B, E]

        # gating 방식 분기
        if self.gating_type == 'soft':
            gating_weights, probs = soft_gating(logits)
        elif self.gating_type == 'topk':
            gating_weights, probs = top_k_gating(logits, self.k)
        elif self.gating_type == 'noisy_topk':
            gating_weights, probs = noisy_top_k_gating(logits, self.k, self.noise_std, training)
        else:
            raise ValueError(f"Unsupported gating_type: {self.gating_type}")

        # Experts 결과
        # expert(x) 반복 호출. 메모리 비용 고려해 필요시 최적화
        expert_outs = torch.stack([expert(x) for expert in self.experts], dim=1)  # [B, E, D]
        output = (expert_outs * gating_weights.unsqueeze(-1)).sum(dim=1)         # [B, D]

        # Load balancing loss: topk/noisy_topk 시에만 의미, soft에도 더할 순 있지만 일반적으론 topk 계열에서 사용
        bal_loss = None
        if self.lambda_bal and self.lambda_bal > 0:
            # soft일 때도 계산 가능하나, soft gating 시 균등 강제 의도가 다를 수 있음.
            # 필요하면 soft에도 사용.
            bal_loss = load_balancing_loss(probs) * self.lambda_bal

        return output, bal_loss

    

