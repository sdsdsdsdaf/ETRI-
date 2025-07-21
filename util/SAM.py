import torch
import torch.nn as nn

def convert_bn_to_fp32(model):
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            m.float()
    return model

class SAM(torch.optim.Optimizer):
    def __init__(self, base_optimizer, model, rho=0.05, adaptive=False, eps=1e-12):
        self.base_optimizer = base_optimizer
        self.model = model
        self.rho = rho
        self.adaptive = adaptive
        self.eps = eps
        self.state = {}

    @torch.no_grad()
    def first_step(self, scaler, zero_grad=True):
        grad_norm = self._grad_norm()
        scale = self.rho / (grad_norm + self.eps)

        for p in self.model.parameters():
            if p.grad is None:
                continue
            e_w = (torch.pow(p, 2) if self.adaptive else 1.0) * p.grad * scale
            p.add_(e_w)  # perturb weight
            self.state[p] = e_w  # save perturbation

        if zero_grad:
            self.model.zero_grad()

    @torch.no_grad()
    def second_step(self, scaler, zero_grad=True):
        for p in self.model.parameters():
            if p in self.state:
                p.sub_(self.state[p])  # restore original weights

        scaler.step(self.base_optimizer)
        scaler.update()

        if zero_grad:
            self.model.zero_grad()

    def _grad_norm(self):
        shared_device = self.model.parameters().__next__().device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if self.adaptive else 1.0) * p.grad).norm(p=2).to(shared_device)
                for p in self.model.parameters() if p.grad is not None
            ]),
            p=2
        )
        return norm
