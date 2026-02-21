import torch
import torch.nn as nn

class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=16):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        # Add noise during training for "Denoising" effect
        if self.training:
            noise = torch.randn_like(x) * 0.1
            x_noisy = x + noise
        else:
            x_noisy = x
            
        latent = self.encoder(x_noisy)
        reconstructed = self.decoder(latent)
        return latent, reconstructed
