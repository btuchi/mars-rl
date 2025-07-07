"""Reward functions for diffusion training"""

import numpy as np
import torch
from typing import List
from ..core.trajectory import DiffusionTrajectory
from ..core.features import FeatureExtractor
from ..diversity_reward import calculate_individual_diversity_rewards

class DiffusionRewardFunction:
    """
    Calculate diversity reward for a batch of trajectories
    Args:
        feature_extractor: clip model to extract features
        ref_features: Reference features for diversity calculation
        buffer_size: Size of the reward history buffer for normalization
        
    Returns:
        Average diversity reward across the batch
    """
    def __init__(self, ref_features: np.ndarray, feature_extractor: FeatureExtractor, buffer_size: int = 50):
        self.ref_features = ref_features
        self.buffer_size = buffer_size
        self.reward_history = []
        self.feature_extractor = feature_extractor
    
    def calculate_batch_rewards(self, trajectories: List[DiffusionTrajectory], prompt: str) -> np.ndarray:
        """
        Calculate individual diversity rewards for a batch of trajectories
        Uses your efficient calculate_individual_diversity_rewards function
        a hybrid reward: quality + diversity
        Args:
            trajectories: List of diffusion trajectories
        Returns:
            individual_rewards: Array of rewards for each trajectory
        """
        # Extract features from all trajectories
        batch_features = []
        content_scores = []

        text_features = self.feature_extractor.extract_text_features(prompt)

        # Encode the prompt using the feature extractor's method
        # with torch.no_grad():
            # Convert to tensor for similarity calculation
            # text_features_tensor = torch.from_numpy(text_features).to(self.feature_extractor.device).unsqueeze(0)
            # # Normalize
            # text_features_tensor = text_features_tensor / text_features_tensor.norm(dim=-1, keepdim=True)

        for trajectory in trajectories:
            # Extract features using the same CLIP model (no duplication)
            features = self.feature_extractor.extract_trajectory_features(trajectory)
            batch_features.append(features.reshape(1, -1))
            
            # Convert features back to tensor for similarity calculation
            image_features = torch.from_numpy(features).to(self.feature_extractor.device).unsqueeze(0)
            
            similarity_score = self.feature_extractor.calculate_similarity(image_features, text_features)
            content_scores.append(similarity_score)

        # batch_features = [traj_feat1, traj_feat2, traj_feat3, traj_feat4, ... traj_featM]
        # traj_feat_i = [f1, f2, f3, .., fn] -> one final image
        # Stack into single np array (batch_size x feature_dim)
        batch_features_array = np.vstack(batch_features)

        # Calculate Individual Rewards
        individual_rewards = calculate_individual_diversity_rewards(
            batch_features_array,
            self.ref_features,
            gamma=None  # Auto-set gamma
        )

        # Combine content quality + diversity
        hybrid_rewards = []

        # Calculate a reasonable penalty based on diversity reward scale
        diversity_std = np.std(individual_rewards)
        diversity_mean = np.mean(individual_rewards)
        penalty = min(-0.1, diversity_mean - 2 * diversity_std)  # Penalty 2 std devs below mean, max -0.1
        
        for i, (content_score, diversity_reward) in enumerate(zip(content_scores, individual_rewards)):
            # QUALITY GATE: Penalize noise, but keep scale reasonable
            # if content_score < 0.27:  # Threshold for "this is basically noise"
            #     final_reward = penalty  # Scale-appropriate penalty
            #     print(f"  🚨 Image {i}: NOISE DETECTED (CLIP={content_score:.3f}) -> Penalty {penalty:.3f}")
            # else:
                
            #     final_reward = diversity_reward
                
            #     print(f"  ✅ Image {i}: diversity_score={diversity_reward:.3f}")

            final_reward = 1.0 * diversity_reward + 0.0 * content_score
            
            hybrid_rewards.append(final_reward)

            normalized_rewards = self.normalize_batch_rewards(np.array(hybrid_rewards))
        
        return np.array(hybrid_rewards)
    
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
        Normalize a batch of rewards using running statistics
        This ensures all rewards (penalties + diversity) are on the same scale
        """
        # Add all rewards to history for running statistics
        for reward in rewards:
            self.reward_history.append(reward)
            if len(self.reward_history) > self.buffer_size:
                self.reward_history.pop(0)
        
        # If we don't have enough history, use simple scaling
        if len(self.reward_history) < 10:
            # Simple clipping and scaling for early training
            clipped_rewards = np.clip(rewards, -2.0, 2.0)
            return clipped_rewards * 0.8
        
        # Use running statistics for normalization
        mean_reward = np.mean(self.reward_history)
        std_reward = np.std(self.reward_history)
        
        # Normalize the current batch
        normalized_rewards = (rewards - mean_reward) / (std_reward + 1e-8)
        
        # Optional: clip extreme values to prevent PPO instability
        normalized_rewards = np.clip(normalized_rewards, -5.0, 5.0)
        
        # DEBUG: Print raw vs normalized rewards to check if normalization is masking progress
        print(f"🔍 RAW rewards: [{rewards.min():.6f}, {rewards.max():.6f}], mean={rewards.mean():.6f}")
        print(f"🔍 NORMALIZED rewards: [{normalized_rewards.min():.6f}, {normalized_rewards.max():.6f}], mean={normalized_rewards.mean():.6f}")
        print(f"🔍 Running mean={mean_reward:.6f}, std={std_reward:.6f}")
        
        return normalized_rewards