import torch
import torch.nn as nn

class FCAutoencoder(nn.Module):
    def __init__(self, in_dim, latent_dim=64):
        super().__init__()

        self.out_features = latent_dim

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
        return z, z, x_hat

class Conv1DAutoencoder(nn.Module):
    def __init__(self, in_ch=1, latent_dim=64):
        super().__init__()

        self.out_features = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=3, padding=1),  # (B, 32, T)
            nn.ReLU(),
            nn.AvgPool1d(2),                                 # (B, 32, T//2)
            nn.Conv1d(32, latent_dim, kernel_size=3, padding=1),     # (B, output_c, T//2)
        )
        self.proj = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # (B, outout_c, 1)
            nn.Flatten(),          # (B, output_c)
            nn.Linear(latent_dim, latent_dim)) # (B, latent_dim)

        self.decode_projection = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, in_ch, kernel_size=3, padding=1)
        )

    def forward(self, x):  # x: (B, C, T)
        z_feat = self.encoder(x)  # (B, latent_dim, T//2)
        z_proj = self.proj(z_feat)   # (B, latent_dim)
        z_hat = self.decode_projection(z_proj).unsqueeze(-1)  # (B, latent_dim, 1)
        x_hat = self.decoder(z_hat) # (B, C, T)
        return z_feat, z_proj, x_hat