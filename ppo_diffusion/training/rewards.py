"""Reward functions for diffusion training"""

import numpy as np
import torch
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
    def __init__(self, ref_features: np.ndarray, feature_extractor: FeatureExtractor, buffer_size: int = 50, reward_metric: str = "MMD", ref_images: np.ndarray = None):
        self.ref_features = ref_features
        self.ref_images = ref_images  # Reference images for MI calculation
        self.buffer_size = buffer_size
        self.reward_history = []
        self.feature_extractor = feature_extractor
        
        # Check if we have reference images when needed
        if reward_metric in ["MI", "MMD_MI"] and ref_images is None:
            print("⚠️ Warning: MI-based metrics need reference images, but none provided!")
            print("⚠️ You may need to modify agent initialization to pass reference images.")
        
        # Configurable reward metric system
        self.reward_metric_name = reward_metric
        
        # Pass weights for MMD_MI metric and FID scale
        if reward_metric == "MMD_MI":
            from ..utils.constants import MMD_WEIGHT, MI_WEIGHT
            self.reward_metric = get_reward_metric(reward_metric, mmd_weight=MMD_WEIGHT, mi_weight=MI_WEIGHT)
        elif reward_metric == "FID":
            from ..utils.constants import FID_REWARD_SCALE
            self.reward_metric = get_reward_metric(reward_metric, reward_scale=FID_REWARD_SCALE)
        else:
            self.reward_metric = get_reward_metric(reward_metric)
        
        print(f"🎯 Using reward metric: {reward_metric}")
        print(f"🎯 Available metrics: {list_available_metrics()}")
    
    def calculate_batch_rewards(self, trajectories: List[DiffusionTrajectory], prompt: str) -> np.ndarray:
        """
        Calculate individual diversity rewards for a batch of trajectories
        Args:
            trajectories: List of diffusion trajectories
            prompt: Text prompt (unused in pure visual diversity approach)
        Returns:
            individual_rewards: Array of rewards for each trajectory
        """
        # Check if we need images (for MI, FID) or features (for MMD)
        needs_images = self.reward_metric_name in ["MI", "MMD_MI", "FID"]
        
        if needs_images:
            # Extract raw images for MI calculation
            batch_images = []
            batch_features = []
            
            for trajectory in trajectories:
                # Get raw image tensor (3, 512, 512)
                # CRITICAL FIX: Ensure image is detached from computation graph
                image_tensor = trajectory.final_image.clone().detach().squeeze(0)  # Remove batch dim
                # Convert to numpy (3, 512, 512) -> (512, 512, 3) for MI
                image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
                batch_images.append(image_np)
                
                # Also extract features for MMD component (if MMD_MI)
                # CRITICAL FIX: Use detached features to prevent computation graph contamination
                if self.reward_metric_name == "MMD_MI":
                    with torch.no_grad():
                        detached_image = trajectory.final_image.clone().detach()
                        from ..core.trajectory import DiffusionTrajectory
                        detached_trajectory = DiffusionTrajectory(
                            steps=[],
                            final_image=detached_image,
                            condition=trajectory.condition.clone().detach() if trajectory.condition is not None else None
                        )
                        features = self.feature_extractor.extract_trajectory_features(detached_trajectory)
                        batch_features.append(features.reshape(1, -1))
            
            batch_images_array = np.stack(batch_images)  # (batch_size, 512, 512, 3)
            
            if self.reward_metric_name == "MMD_MI":
                # For combined metric, pass both images and features
                batch_features_array = np.vstack(batch_features)
                # The MMD_MI metric will handle calling the right sub-metrics
                individual_rewards = self.reward_metric.calculate_rewards(
                    batch_features_array,  # For MMD component
                    self.ref_features,     # Reference features for MMD
                    gamma=None,
                    images=batch_images_array,  # For MI component
                    ref_images=self.ref_images  # Reference images for MI
                )
            elif self.reward_metric_name == "FID":
                # FID needs images and reference images
                individual_rewards = self.reward_metric.calculate_rewards(
                    batch_images_array,
                    self.ref_images,  # Need reference images for FID
                    device=self.feature_extractor.device if hasattr(self.feature_extractor, 'device') else 'cuda'
                )
            else:
                # Pure MI - pass images
                individual_rewards = self.reward_metric.calculate_rewards(
                    batch_images_array,
                    self.ref_images,  # Need reference images for MI
                    gamma=None
                )
        else:
            # Extract visual features for MMD/other feature-based metrics
            batch_features = []
            
            for trajectory in trajectories:
                # Extract ResNet-18 features from final image
                # CRITICAL FIX: Use detached features to prevent computation graph contamination
                with torch.no_grad():
                    detached_image = trajectory.final_image.clone().detach()
                    from ..core.trajectory import DiffusionTrajectory
                    detached_trajectory = DiffusionTrajectory(
                        steps=[],
                        final_image=detached_image,
                        condition=trajectory.condition.clone().detach() if trajectory.condition is not None else None
                    )
                    features = self.feature_extractor.extract_trajectory_features(detached_trajectory)
                    batch_features.append(features.reshape(1, -1))

            # Stack into single np array (batch_size x feature_dim)
            batch_features_array = np.vstack(batch_features)
            
            # Log generated features for t-SNE visualization
            from ..utils.logging import log_generated_features
            # We need episode and prompt info - let's get it from the caller context
            # For now, we'll add this logging at the agent level instead
            
            # Calculate rewards using selected metric
            individual_rewards = self.reward_metric.calculate_rewards(
                batch_features_array,
                self.ref_features,
                gamma=None  # Auto-set gamma
            )

        # PHASE 4: Log reward computation
        print(f"🎯 [PHASE 4] Computing {self.reward_metric_name} rewards...")

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