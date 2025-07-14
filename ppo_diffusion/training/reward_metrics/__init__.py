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

class MMD_MIRewardMetric(RewardMetric):
    """Combined MMD and MI reward metric with configurable weights"""
    
    def __init__(self, mmd_weight: float = 0.7, mi_weight: float = 0.3):
        super().__init__("MMD_MI")
        self.mmd_weight = mmd_weight
        self.mi_weight = mi_weight
        self.mmd_metric = MMDRewardMetric()
        self.mi_metric = MIRewardMetric()
        
        print(f"🎯 Combined MMD_MI metric: {mmd_weight:.1f} MMD + {mi_weight:.1f} MI")
    
    def calculate_rewards(self, generated_features: np.ndarray, 
                         reference_features: np.ndarray, 
                         gamma: Optional[float] = None,
                         images: np.ndarray = None,
                         ref_images: np.ndarray = None) -> np.ndarray:
        """Calculate weighted combination of MMD and MI rewards"""
        # Calculate MMD rewards using features
        mmd_rewards = self.mmd_metric.calculate_rewards(
            generated_features, reference_features, gamma=gamma
        )
        
        # Calculate MI rewards using images
        if images is not None and ref_images is not None:
            mi_rewards = self.mi_metric.calculate_rewards(
                images, ref_images, gamma=gamma
            )
        else:
            print("⚠️ No images provided for MI calculation, using zero MI rewards")
            mi_rewards = np.zeros(len(generated_features))
        
        # Weighted combination
        combined_rewards = (
            self.mmd_weight * mmd_rewards + 
            self.mi_weight * mi_rewards
        )
        
        print(f"🎯 MMD: [{mmd_rewards.min():.4f}, {mmd_rewards.max():.4f}]")
        print(f"🎯 MI:  [{mi_rewards.min():.4f}, {mi_rewards.max():.4f}]")
        print(f"🎯 Combined: [{combined_rewards.min():.4f}, {combined_rewards.max():.4f}]")
        
        return combined_rewards

# Registry of available reward metrics
REWARD_METRICS = {
    "MMD": MMDRewardMetric,
    "MI": MIRewardMetric,
    "FID": FIDRewardMetric,
    "LPIPS": LPIPSRewardMetric,
    "MMD_MI": MMD_MIRewardMetric
}

def get_reward_metric(metric_name: str, **kwargs) -> RewardMetric:
    """Factory function to get reward metric by name"""
    if metric_name not in REWARD_METRICS:
        raise ValueError(f"Unknown reward metric: {metric_name}. Available: {list(REWARD_METRICS.keys())}")
    
    # Pass kwargs for metrics that need configuration (like MMD_MI weights)
    return REWARD_METRICS[metric_name](**kwargs)

def list_available_metrics() -> list:
    """List all available reward metrics"""
    return list(REWARD_METRICS.keys())

__all__ = [
    'RewardMetric',
    'MMDRewardMetric', 
    'MIRewardMetric',
    'FIDRewardMetric',
    'LPIPSRewardMetric',
    'MMD_MIRewardMetric',
    'get_reward_metric',
    'list_available_metrics',
    'calculate_individual_mi_rewards', 
    'calculate_individual_mmd_rewards'
]