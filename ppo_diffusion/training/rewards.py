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
        
        # Multi-component reward setup
        from ..utils.constants import (
            USE_MULTI_COMPONENT_REWARD, SEQUENTIAL_DIVERSITY_WEIGHT, 
            SPATIAL_DIVERSITY_WEIGHT, ENTROPY_REWARD_WEIGHT, SEQUENTIAL_THRESHOLD
        )
        self.use_multi_component = USE_MULTI_COMPONENT_REWARD
        self.sequential_weight = SEQUENTIAL_DIVERSITY_WEIGHT
        self.spatial_weight = SPATIAL_DIVERSITY_WEIGHT
        self.entropy_weight = ENTROPY_REWARD_WEIGHT
        self.sequential_threshold = SEQUENTIAL_THRESHOLD
        
        # Store recent features for sequential diversity
        self.recent_features = []  # Store last few batches of features
        
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
        if self.use_multi_component:
            print(f"🎯 Multi-component rewards enabled:")
            print(f"   Sequential diversity weight: {self.sequential_weight}")
            print(f"   Spatial diversity weight: {self.spatial_weight}")
            print(f"   Entropy reward weight: {self.entropy_weight}")
    
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
                # CRITICAL FIX: Keep gradients flowing for scheduler policy training
                image_tensor = trajectory.final_image.clone().squeeze(0)  # Remove batch dim but keep gradients
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
                # CRITICAL FIX: Keep gradients flowing for scheduler policy training
                # Check if we're in scheduler policy mode and need gradient flow
                from ..utils.constants import DEFAULT_TRAINING_MODE
                if DEFAULT_TRAINING_MODE == "SCHEDULER_POLICY":
                    # Use gradient-enabled feature extraction
                    features = self.feature_extractor.extract_trajectory_features_with_gradients(trajectory)
                    # Keep as tensor and add batch dimension
                    batch_features.append(features.unsqueeze(0))  # [1, 512]
                else:
                    # Use standard numpy feature extraction for other modes
                    features = self.feature_extractor.extract_trajectory_features(trajectory)
                    batch_features.append(features.reshape(1, -1))

            # Stack features - handle both tensor and numpy cases
            from ..utils.constants import DEFAULT_TRAINING_MODE
            if DEFAULT_TRAINING_MODE == "SCHEDULER_POLICY":
                # Stack tensors to preserve gradients
                batch_features_tensor = torch.cat(batch_features, dim=0)  # [batch_size, 512]
                # For scheduler policy mode, we need to modify reward calculation to work with tensors
                batch_features_array = batch_features_tensor
            else:
                # Stack into numpy array for other modes
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
            
            # Convert tensor rewards to numpy for downstream processing (except in scheduler mode)
            if DEFAULT_TRAINING_MODE == "SCHEDULER_POLICY" and isinstance(individual_rewards, torch.Tensor):
                # Keep as tensor for gradient flow in scheduler policy mode
                pass
            elif isinstance(individual_rewards, torch.Tensor):
                # Convert to numpy for other modes
                individual_rewards = individual_rewards.detach().cpu().numpy()

        # PHASE 4: Log reward computation
        print(f"🎯 [PHASE 4] Computing {self.reward_metric_name} rewards...")

        # Use diversity rewards from selected metric as base
        base_rewards = individual_rewards
        
        # Handle tensor vs numpy for logging
        if isinstance(base_rewards, torch.Tensor):
            print(f"🎯 [PHASE 4] {self.reward_metric_name} rewards: [{base_rewards.min().item():.6f}, {base_rewards.max().item():.6f}]")
        else:
            print(f"🎯 [PHASE 4] {self.reward_metric_name} rewards: [{base_rewards.min():.6f}, {base_rewards.max():.6f}]")
        
        # Add multi-component bonuses if enabled
        # Temporarily disable multi-component rewards in scheduler policy mode to avoid tensor/numpy mixing
        if self.use_multi_component and DEFAULT_TRAINING_MODE != "SCHEDULER_POLICY":
            # Extract features for bonus calculations (use features for all components)
            if needs_images and self.reward_metric_name != "MMD_MI":
                # Need to extract features for non-MMD_MI image-based metrics
                batch_features = []
                for trajectory in trajectories:
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
                batch_features_array = np.vstack(batch_features)
            else:
                # Use already extracted features
                if 'batch_features_array' in locals():
                    pass  # Already have features
                else:
                    # This shouldn't happen, but fallback
                    batch_features_array = np.zeros((len(trajectories), 512))
                    
            # Calculate bonus components
            sequential_bonus = self._calculate_sequential_diversity(batch_features_array)
            spatial_bonus = self._calculate_spatial_diversity(trajectories)
            entropy_bonus = self._calculate_entropy_reward(batch_features_array)
            
            # Combine rewards
            total_rewards = (base_rewards + 
                           self.sequential_weight * sequential_bonus +
                           self.spatial_weight * spatial_bonus +
                           self.entropy_weight * entropy_bonus)
            
            print(f"🎯 Multi-component breakdown:")
            print(f"   Base ({self.reward_metric_name}): {base_rewards.mean():.4f}")
            print(f"   Sequential bonus: {sequential_bonus.mean():.4f}")
            print(f"   Spatial bonus: {spatial_bonus.mean():.4f}")
            print(f"   Entropy bonus: {entropy_bonus.mean():.4f}")
            print(f"   Total: {total_rewards.mean():.4f}")
        else:
            total_rewards = base_rewards
        
        # Apply z-score normalization with extended warm-up
        normalized_rewards = self.normalize_batch_rewards(total_rewards)
        
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

    def normalize_batch_rewards(self, rewards):
        """
        Gentle reward normalization to improve critic learning without causing negative reward shock
        Handles both numpy arrays and PyTorch tensors
        """
        # Handle tensor/numpy conversion for history tracking
        if isinstance(rewards, torch.Tensor):
            rewards_for_history = rewards.detach().cpu().numpy()
        else:
            rewards_for_history = rewards
            
        # Add all rewards to history for running statistics
        for reward in rewards_for_history:
            self.reward_history.append(reward)
            if len(self.reward_history) > self.buffer_size:
                self.reward_history.pop(0)
        
        # Extended warm-up period - no normalization for first 100 episodes
        if len(self.reward_history) < 100:
            # Just return raw rewards during warm-up
            if isinstance(rewards, torch.Tensor):
                print(f"🔍 WARM-UP ({len(self.reward_history)}/100): Raw rewards: [{rewards.min().item():.6f}, {rewards.max().item():.6f}]")
            else:
                print(f"🔍 WARM-UP ({len(self.reward_history)}/100): Raw rewards: [{rewards.min():.6f}, {rewards.max():.6f}]")
            return rewards
        
        # Use z-score normalization after warm-up
        mean_reward = np.mean(self.reward_history)
        std_reward = np.std(self.reward_history)
        
        # Avoid division by zero
        if std_reward < 1e-8:
            print(f"🔍 WARNING: Very low std ({std_reward:.8f}), returning raw rewards")
            return rewards
        
        # Z-score normalization - handle tensor vs numpy
        if isinstance(rewards, torch.Tensor):
            # PyTorch tensor operations
            mean_tensor = torch.tensor(mean_reward, device=rewards.device, dtype=rewards.dtype)
            std_tensor = torch.tensor(std_reward, device=rewards.device, dtype=rewards.dtype)
            normalized_rewards = (rewards - mean_tensor) / std_tensor
            
            # Clip extreme values to prevent instability
            normalized_rewards = torch.clamp(normalized_rewards, -3.0, 3.0)
            
            # DEBUG: Print diagnostics
            print(f"🔍 RAW rewards: [{rewards.min().item():.6f}, {rewards.max().item():.6f}], mean={rewards.mean().item():.6f}")
            print(f"🔍 Z-SCORE normalized: [{normalized_rewards.min().item():.6f}, {normalized_rewards.max().item():.6f}], mean={normalized_rewards.mean().item():.6f}")
        else:
            # Numpy array operations
            normalized_rewards = (rewards - mean_reward) / std_reward
            
            # Clip extreme values to prevent instability
            normalized_rewards = np.clip(normalized_rewards, -3.0, 3.0)
            
            # DEBUG: Print diagnostics
            print(f"🔍 RAW rewards: [{rewards.min():.6f}, {rewards.max():.6f}], mean={rewards.mean():.6f}")
            print(f"🔍 Z-SCORE normalized: [{normalized_rewards.min():.6f}, {normalized_rewards.max():.6f}], mean={normalized_rewards.mean():.6f}")
        
        print(f"🔍 Running stats: mean={mean_reward:.6f}, std={std_reward:.6f}")
        
        return normalized_rewards
    
    def _calculate_sequential_diversity(self, batch_features: np.ndarray) -> np.ndarray:
        """
        Calculate sequential diversity bonus - reward being different from recent images
        Args:
            batch_features: Current batch features (batch_size, 512)
        Returns:
            sequential_bonuses: Array of bonuses for each image
        """
        if len(self.recent_features) == 0:
            # First batch - no previous images to compare with
            self.recent_features.append(batch_features)
            return np.zeros(batch_features.shape[0])
        
        # Compare current batch with recent batches
        sequential_bonuses = []
        
        for i, current_feature in enumerate(batch_features):
            max_similarity = 0.0
            
            # Check similarity with features from recent batches
            for recent_batch in self.recent_features[-3:]:  # Last 3 batches
                for recent_feature in recent_batch:
                    # Cosine similarity
                    similarity = np.dot(current_feature, recent_feature) / (
                        np.linalg.norm(current_feature) * np.linalg.norm(recent_feature) + 1e-8
                    )
                    max_similarity = max(max_similarity, similarity)
            
            # Bonus if similarity below threshold (sufficiently different)
            if max_similarity < self.sequential_threshold:
                bonus = (self.sequential_threshold - max_similarity)
            else:
                bonus = 0.0
                
            sequential_bonuses.append(bonus)
        
        # Store current batch for future comparisons
        self.recent_features.append(batch_features)
        if len(self.recent_features) > 5:  # Keep last 5 batches
            self.recent_features.pop(0)
        
        return np.array(sequential_bonuses)
    
    def _calculate_spatial_diversity(self, trajectories: List[DiffusionTrajectory]) -> np.ndarray:
        """
        Calculate spatial diversity bonus - reward diversity within each image
        Args:
            trajectories: List of trajectories with final images
        Returns:
            spatial_bonuses: Array of bonuses for each image
        """
        spatial_bonuses = []
        
        for trajectory in trajectories:
            try:
                # Get image tensor (1, 3, 512, 512) -> (3, 512, 512)
                image = trajectory.final_image.squeeze(0).detach()
                
                # Divide image into patches (e.g., 4x4 = 16 patches)
                patch_size = 128  # 512/4 = 128
                patches = []
                
                for i in range(0, 512, patch_size):
                    for j in range(0, 512, patch_size):
                        patch = image[:, i:i+patch_size, j:j+patch_size]
                        patches.append(patch)
                
                # Extract features from each patch
                patch_features = []
                for patch in patches:
                    # Add batch dimension and extract features
                    patch_batch = patch.unsqueeze(0)  # (1, 3, 128, 128)
                    
                    with torch.no_grad():
                        # Create temporary trajectory for feature extraction
                        temp_trajectory = DiffusionTrajectory(
                            steps=[],
                            final_image=patch_batch,
                            condition=None
                        )
                        features = self.feature_extractor.extract_trajectory_features(temp_trajectory)
                        # features is already numpy array, just flatten it
                        patch_features.append(features.flatten())
                
                # Calculate pairwise distances between patches
                patch_features = np.array(patch_features)
                distances = []
                
                for i in range(len(patch_features)):
                    for j in range(i+1, len(patch_features)):
                        # Cosine distance
                        similarity = np.dot(patch_features[i], patch_features[j]) / (
                            np.linalg.norm(patch_features[i]) * np.linalg.norm(patch_features[j]) + 1e-8
                        )
                        distance = 1 - similarity
                        distances.append(distance)
                
                # Average distance as spatial diversity measure
                spatial_bonus = np.mean(distances) if distances else 0.0
                
            except Exception as e:
                print(f"⚠️ Error calculating spatial diversity: {e}")
                spatial_bonus = 0.0
                
            spatial_bonuses.append(spatial_bonus)
        
        return np.array(spatial_bonuses)
    
    def _calculate_entropy_reward(self, batch_features: np.ndarray) -> np.ndarray:
        """
        Calculate entropy bonus - reward complex feature distributions
        Args:
            batch_features: Features for each image (batch_size, 512)
        Returns:
            entropy_bonuses: Array of entropy bonuses for each image
        """
        entropy_bonuses = []
        
        for features in batch_features:
            try:
                # Normalize features to [0, 1] for entropy calculation
                features_norm = features - features.min()
                features_norm = features_norm / (features_norm.max() + 1e-8)
                
                # Create histogram bins
                hist, _ = np.histogram(features_norm, bins=50, density=True)
                hist = hist + 1e-8  # Avoid log(0)
                hist = hist / hist.sum()  # Normalize to probabilities
                
                # Calculate entropy
                entropy = -np.sum(hist * np.log(hist))
                entropy_bonuses.append(entropy)
                
            except Exception as e:
                print(f"⚠️ Error calculating entropy: {e}")
                entropy_bonuses.append(0.0)
        
        return np.array(entropy_bonuses)