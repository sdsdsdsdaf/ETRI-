import torch
import torch.nn as nn

class FCAutoencoder(nn.Module):
    def __init__(self, in_dim, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, in_dim)
        )
        

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat

class Conv1DAutoencoder(nn.Module):
    def __init__(self, in_ch=1, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=3, padding=1),  # (B, 32, T)
            nn.ReLU(),
            nn.AvgPool1d(2),                                 # (B, 32, T//2)
            nn.Conv1d(32, 64, kernel_size=3, padding=1),     # (B, 64, T//2)
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),                         # (B, 64, 1)
        )
        self.proj = nn.Linear(64, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, in_ch, kernel_size=3, padding=1)
        )

    def forward(self, x):  # x: (B, C, T)
        z_feat = self.encoder(x).squeeze(-1)  # (B, 64)
        z = self.proj(z_feat)                 # (B, latent_dim)
        x_hat = self.decoder(z.unsqueeze(-1)) # (B, C, T)
        return z, x_hat