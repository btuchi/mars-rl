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

        # Encode the prompt once and reuse
        with torch.no_grad():
            text_features = self.feature_extractor.model.encode_text(
                torch.cat([torch.zeros(1, 77, dtype=torch.long), 
                          torch.tensor([self.feature_extractor.model.token_embedding.weight.shape[0]-1])]).to(self.feature_extractor.device)
            )
            # Proper text encoding
            import clip
            text_features = self.feature_extractor.model.encode_text(clip.tokenize([prompt]).to(self.feature_extractor.device))
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        for trajectory in trajectories:
            # Extract features using the same CLIP model (no duplication)
            features = self.feature_extractor.extract_trajectory_features(trajectory)
            batch_features.append(features.reshape(1, -1))
            
            # Convert features back to tensor for similarity calculation
            image_features = torch.from_numpy(features).to(self.feature_extractor.device).unsqueeze(0)
            
            similarity_score = self.feature_extractor.calculate_clip_similarity(image_features, text_features)
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

        for i, (content_score, diversity_reward) in enumerate(zip(content_scores, individual_rewards)):
            # QUALITY GATE: Penalize noise heavily
            if content_score < 0.30:  # Threshold for "this is basically noise"
                final_reward = -2.0  # Heavy penalty
                print(f"  🚨 Image {i}: NOISE DETECTED (CLIP={content_score:.3f}) -> Penalty")
            else:
                # Weighted combination of content quality and diversity
                content_weight = 1.0  # Prioritize content quality
                diversity_weight = 0.0
                
                final_reward = (content_weight * content_score + 
                              diversity_weight * self.normalize_reward(diversity_reward))
                
                print(f"  ✅ Image {i}: similarity_score={content_score:.3f}, diversity_score={diversity_reward:.3f} -> final_score={final_reward:.3f}")
            
            hybrid_rewards.append(final_reward)
        
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