"""Value network for diffusion models"""

import torch
import torch.nn as nn

class DiffusionValueNetwork(nn.Module):
    """
    Value Function for Diffusion Models:
        - Estimates how good a prompt/context is for generating diverse images
        - Takes text embeddings as input (like state in vanilla PPO)
    Input:
        - Text embedding or features
    Output:
        - Scalar value V(prompt): expected diversity reward for this prompt
    """
    def __init__(self, feature_dim: int = 512):
        super(DiffusionValueNetwork, self).__init__()
        
        # Simplified network to prevent overfitting
        self.network = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Estimate value from prompt features"""
        return self.network(features)