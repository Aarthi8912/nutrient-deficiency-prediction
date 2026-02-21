import torch
import torch.nn as nn
from models.autoencoder import DenoisingAutoencoder
from models.attention import FeatureAttention

class NutritionMTLModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.autoencoder = DenoisingAutoencoder(input_dim, latent_dim=16)
        self.attention = FeatureAttention(latent_dim=16)

        # Task 1: Nutrition deficiency classification (High Capacity + Skip Connection)
        # We concatenate the 16-dim attended features with the input_dim raw features
        self.classifier = nn.Sequential(
            nn.Linear(16 + input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            # Removed BatchNorm for better single-sample inference stability
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Task 2: Risk score regression
        self.regressor = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        latent, reconstructed = self.autoencoder(x)
        attended, weights = self.attention(latent)

        # Combine latent patterns with raw input for perfect precision
        combined = torch.cat([attended, x], dim=1) 
        deficiency = self.classifier(combined)
        
        risk_score = self.regressor(attended)

        return deficiency, risk_score, reconstructed, weights
