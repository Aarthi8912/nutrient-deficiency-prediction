import torch
import torch.nn as nn

class FeatureAttention(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.attention = nn.Linear(latent_dim, latent_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        scores = self.attention(x)
        weights = self.softmax(scores)
        # Use residual connection so signal isn't decimated
        attended_features = x + (x * weights) 
        return attended_features, weights
