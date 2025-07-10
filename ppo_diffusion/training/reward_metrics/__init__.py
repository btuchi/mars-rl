"""
Reward metrics package for PPO-diffusion training
"""

import numpy as np
from typing import Optional

from .mi import calculate_individual_mi_rewards
from .mmd import calculate_individual_mmd_rewards

class RewardMetric:
    """Base class for reward metrics"""
    
    def __init__(self, name: str):
        self.name = name
    
    def calculate_rewards(self, generated_features: np.ndarray, 
                         reference_features: np.ndarray, 
                         **kwargs) -> np.ndarray:
        """Calculate individual rewards for each generated feature"""
        raise NotImplementedError

class MMDRewardMetric(RewardMetric):
    """Maximum Mean Discrepancy reward metric"""
    
    def __init__(self):
        super().__init__("MMD")
    
    def calculate_rewards(self, generated_features: np.ndarray, 
                         reference_features: np.ndarray, 
                         gamma: Optional[float] = None) -> np.ndarray:
        """Calculate MMD-based individual diversity rewards"""
        return calculate_individual_mmd_rewards(
            generated_features, reference_features, gamma=gamma
        )

class MIRewardMetric(RewardMetric):
    """Mutual Information reward metric"""
    
    def __init__(self):
        super().__init__("MI")
    
    def calculate_rewards(self, generated_features: np.ndarray, 
                         reference_features: np.ndarray, 
                         gamma: Optional[float] = None) -> np.ndarray:
        """Calculate MI-based individual diversity rewards"""
        return calculate_individual_mi_rewards(
            generated_features, reference_features, gamma
        )

class FIDRewardMetric(RewardMetric):
    """Fréchet Inception Distance reward metric (placeholder for future)"""
    
    def __init__(self):
        super().__init__("FID")
    
    def calculate_rewards(self, generated_features: np.ndarray, 
                         reference_features: np.ndarray, 
                         **kwargs) -> np.ndarray:
        """Calculate FID-based rewards (TODO: implement)"""
        print("FID reward metric not implemented yet")
        return np.zeros(len(generated_features))

class LPIPSRewardMetric(RewardMetric):
    """Learned Perceptual Image Patch Similarity reward metric (placeholder)"""
    
    def __init__(self):
        super().__init__("LPIPS")
    
    def calculate_rewards(self, generated_features: np.ndarray, 
                         reference_features: np.ndarray, 
                         **kwargs) -> np.ndarray:
        """Calculate LPIPS-based rewards (TODO: implement)"""
        print("LPIPS reward metric not implemented yet")
        return np.zeros(len(generated_features))

# Registry of available reward metrics
REWARD_METRICS = {
    "MMD": MMDRewardMetric,
    "MI": MIRewardMetric,
    "FID": FIDRewardMetric,
    "LPIPS": LPIPSRewardMetric
}

def get_reward_metric(metric_name: str) -> RewardMetric:
    """Factory function to get reward metric by name"""
    if metric_name not in REWARD_METRICS:
        raise ValueError(f"Unknown reward metric: {metric_name}. Available: {list(REWARD_METRICS.keys())}")
    
    return REWARD_METRICS[metric_name]()

def list_available_metrics() -> list:
    """List all available reward metrics"""
    return list(REWARD_METRICS.keys())

__all__ = [
    'RewardMetric',
    'MMDRewardMetric', 
    'MIRewardMetric',
    'FIDRewardMetric',
    'LPIPSRewardMetric',
    'get_reward_metric',
    'list_available_metrics',
    'calculate_individual_mi_rewards', 
    'calculate_individual_mmd_rewards'
]