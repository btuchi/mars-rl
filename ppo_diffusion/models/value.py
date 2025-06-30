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
        
        # Initialize the value network
        self.network = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.LayerNorm(1024),
            # LeakyReLu or GeLU?
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(256, 128),
            nn.GELU(),
            
            nn.Linear(128, 1)
        )
        
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Estimate value from prompt features"""
        return self.network(features)