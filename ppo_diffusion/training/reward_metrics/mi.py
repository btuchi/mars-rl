"""
Mutual Information reward implementation for diffusion training
"""

import numpy as np
import torch
from sklearn.metrics.pairwise import rbf_kernel
from typing import Optional

def calculate_mutual_information_reward(generated_features: np.ndarray, 
                                       reference_features: np.ndarray, 
                                       gamma: Optional[float] = None) -> float:
    """
    Calculate Mutual Information-based diversity reward using Gaussian Process modeling.
    
    Args:
        generated_features: Feature matrix for generated images (M x feature_dim)
        reference_features: Feature matrix for reference images (N x feature_dim)
        gamma: RBF kernel bandwidth parameter
    
    Returns:
        mi_reward: Mutual information-based diversity reward
    """
    # Ensure inputs are numpy arrays and normalize
    if isinstance(generated_features, torch.Tensor):
        generated_features = generated_features.cpu().numpy()
    if isinstance(reference_features, torch.Tensor):
        reference_features = reference_features.cpu().numpy()
    
    # Normalize features
    gen_norm = np.linalg.norm(generated_features, axis=1, keepdims=True)
    ref_norm = np.linalg.norm(reference_features, axis=1, keepdims=True)
    gen_norm = np.maximum(gen_norm, 1e-10)
    ref_norm = np.maximum(ref_norm, 1e-10)
    generated_features = generated_features / gen_norm
    reference_features = reference_features / ref_norm
    
    # Auto-set gamma if not provided
    if gamma is None:
        X = np.vstack([generated_features, reference_features])
        pairwise_dists = np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=2)
        gamma = 1.0 / np.median(pairwise_dists[pairwise_dists > 0])
    
    # Calculate covariance matrices using RBF kernel
    try:
        # Covariance of generated features
        K_gg = rbf_kernel(generated_features, generated_features, gamma=gamma)
        
        # Covariance of reference features  
        K_rr = rbf_kernel(reference_features, reference_features, gamma=gamma)
        
        # Combined covariance matrix
        K_gr = rbf_kernel(generated_features, reference_features, gamma=gamma)
        K_combined = np.block([[K_gg, K_gr], 
                               [K_gr.T, K_rr]])
        
        # Add regularization for numerical stability
        reg_term = 1e-6
        K_gg += reg_term * np.eye(K_gg.shape[0])
        K_rr += reg_term * np.eye(K_rr.shape[0])
        K_combined += reg_term * np.eye(K_combined.shape[0])
        
        # Calculate log determinants (more numerically stable)
        sign_gg, logdet_gg = np.linalg.slogdet(K_gg)
        sign_rr, logdet_rr = np.linalg.slogdet(K_rr)
        sign_combined, logdet_combined = np.linalg.slogdet(K_combined)
        
        # Check for positive definiteness
        if sign_gg <= 0 or sign_rr <= 0 or sign_combined <= 0:
            print("Warning: Non-positive definite matrix detected, using fallback")
            return 0.0
        
        # Mutual Information calculation
        # MI = 0.5 * (log|K_gg| + log|K_rr| - log|K_combined|)
        mi_reward = 0.5 * (logdet_gg + logdet_rr - logdet_combined)
        
        # Clip extreme values
        mi_reward = np.clip(mi_reward, -10.0, 10.0)
        
        return mi_reward
        
    except (np.linalg.LinAlgError, ValueError) as e:
        print(f"Linear algebra error in MI calculation: {e}")
        return 0.0

def calculate_individual_mi_rewards(generated_features: np.ndarray, 
                                    reference_features: np.ndarray, 
                                    gamma: Optional[float] = None) -> np.ndarray:
    """
    Calculate individual MI-based diversity rewards using marginal utility.
    
    Args:
        generated_features: Feature matrix for generated images (M x feature_dim)
        reference_features: Feature matrix for reference images (N x feature_dim)
        gamma: RBF kernel bandwidth parameter
    
    Returns:
        individual_rewards: Array of MI-based diversity rewards for each generated image
    """
    M = len(generated_features)
    
    # Calculate overall MI reward
    overall_reward = calculate_mutual_information_reward(
        generated_features, reference_features, gamma
    )
    
    if M <= 1:
        return np.array([overall_reward])
    
    # Calculate individual contributions using marginal utility
    individual_rewards = []
    
    for i in range(M):
        # Create subset without image i
        subset_indices = np.concatenate([np.arange(i), np.arange(i+1, M)])
        subset_features = generated_features[subset_indices]
        
        # Calculate MI reward without image i
        reward_without_i = calculate_mutual_information_reward(
            subset_features, reference_features, gamma
        )
        
        # Marginal contribution of image i
        marginal_contribution = overall_reward - reward_without_i
        individual_rewards.append(marginal_contribution)
    
    individual_rewards = np.array(individual_rewards)
    
    # Normalize to ensure non-negative rewards
    if np.min(individual_rewards) < 0:
        individual_rewards -= np.min(individual_rewards)
    
    return individual_rewards