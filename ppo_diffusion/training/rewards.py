"""Reward functions for diffusion training"""

import numpy as np
from typing import List
from ..core.trajectory import DiffusionTrajectory
from ..core.features import FeatureExtractor
from .reward_metrics import get_reward_metric, list_available_metrics

class DiffusionRewardFunction:
    """
    Calculate diversity reward for a batch of trajectories using ResNet-18 visual features
    Args:
        feature_extractor: ResNet-18 model to extract visual features
        ref_features: Reference features for diversity calculation
        buffer_size: Size of the reward history buffer for normalization
        
    Returns:
        Individual diversity rewards for each trajectory
    """
    def __init__(self, ref_features: np.ndarray, feature_extractor: FeatureExtractor, buffer_size: int = 50, reward_metric: str = "MMD"):
        self.ref_features = ref_features
        self.buffer_size = buffer_size
        self.reward_history = []
        self.feature_extractor = feature_extractor
        
        # Configurable reward metric system
        self.reward_metric_name = reward_metric
        self.reward_metric = get_reward_metric(reward_metric)
        
        print(f"🎯 Using reward metric: {reward_metric}")
        print(f"🎯 Available metrics: {list_available_metrics()}")
    
    def calculate_batch_rewards(self, trajectories: List[DiffusionTrajectory], prompt: str) -> np.ndarray:
        """
        Calculate individual diversity rewards for a batch of trajectories
        Uses pure visual diversity (ResNet-18 features) without text comparison
        Args:
            trajectories: List of diffusion trajectories
            prompt: Text prompt (unused in pure visual diversity approach)
        Returns:
            individual_rewards: Array of rewards for each trajectory
        """
        # Extract visual features from all trajectories
        batch_features = []

        for trajectory in trajectories:
            # Extract ResNet-18 features from final image
            features = self.feature_extractor.extract_trajectory_features(trajectory)
            batch_features.append(features.reshape(1, -1))

        # batch_features = [traj_feat1, traj_feat2, traj_feat3, traj_feat4, ... traj_featM]
        # traj_feat_i = [f1, f2, f3, .., fn] -> one final image
        # Stack into single np array (batch_size x feature_dim)
        batch_features_array = np.vstack(batch_features)

        # PHASE 4: Calculate rewards using selected metric
        print(f"🎯 [PHASE 4] Computing {self.reward_metric_name} rewards...")
        individual_rewards = self.reward_metric.calculate_rewards(
            batch_features_array,
            self.ref_features,
            gamma=None  # Auto-set gamma
        )

        # Use diversity rewards from selected metric
        diversity_rewards = individual_rewards
        print(f"🎯 [PHASE 4] {self.reward_metric_name} rewards: [{diversity_rewards.min():.6f}, {diversity_rewards.max():.6f}]")
        
        # Apply z-score normalization with extended warm-up
        normalized_rewards = self.normalize_batch_rewards(diversity_rewards)
        
        return normalized_rewards
    
    def normalize_reward(self, reward: float) -> float:
        """Normalize reward based on running statistics"""
        self.reward_history.append(reward)
        if len(self.reward_history) > self.buffer_size:
            self.reward_history.pop(0)
        
        if len(self.reward_history) < 10:
            return reward * 0.1
        
        mean_reward = np.mean(self.reward_history)
        std_reward = np.std(self.reward_history)
        return (reward - mean_reward) / (std_reward + 1e-8)

    def normalize_batch_rewards(self, rewards: np.ndarray) -> np.ndarray:
        """
        Gentle reward normalization to improve critic learning without causing negative reward shock
        """
        # Add all rewards to history for running statistics
        for reward in rewards:
            self.reward_history.append(reward)
            if len(self.reward_history) > self.buffer_size:
                self.reward_history.pop(0)
        
        # Extended warm-up period - no normalization for first 100 episodes
        if len(self.reward_history) < 100:
            # Just return raw rewards during warm-up
            print(f"🔍 WARM-UP ({len(self.reward_history)}/100): Raw rewards: [{rewards.min():.6f}, {rewards.max():.6f}]")
            return rewards
        
        # Use z-score normalization after warm-up
        mean_reward = np.mean(self.reward_history)
        std_reward = np.std(self.reward_history)
        
        # Avoid division by zero
        if std_reward < 1e-8:
            print(f"🔍 WARNING: Very low std ({std_reward:.8f}), returning raw rewards")
            return rewards
        
        # Z-score normalization
        normalized_rewards = (rewards - mean_reward) / std_reward
        
        # Clip extreme values to prevent instability
        normalized_rewards = np.clip(normalized_rewards, -3.0, 3.0)
        
        # DEBUG: Print diagnostics
        print(f"🔍 RAW rewards: [{rewards.min():.6f}, {rewards.max():.6f}], mean={rewards.mean():.6f}")
        print(f"🔍 Z-SCORE normalized: [{normalized_rewards.min():.6f}, {normalized_rewards.max():.6f}], mean={normalized_rewards.mean():.6f}")
        print(f"🔍 Running stats: mean={mean_reward:.6f}, std={std_reward:.6f}")
        
        return normalized_rewards