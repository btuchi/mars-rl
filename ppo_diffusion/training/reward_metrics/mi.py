"""
Mutual Information reward implementation for diffusion training
MI should work on image-level data, not compressed features
"""

import numpy as np
import torch
from sklearn.metrics.pairwise import rbf_kernel
from typing import Optional
import torch.nn.functional as F

def calculate_image_based_mi_reward(generated_images: np.ndarray, 
                                   reference_images: np.ndarray, 
                                   gamma: Optional[float] = None) -> float:
    """
    Calculate Mutual Information-based diversity reward using image-level data.
    
    Args:
        generated_images: Generated images (M x H x W x C) or (M x C x H x W) 
        reference_images: Reference images (N x H x W x C) or (N x C x H x W)
        gamma: RBF kernel bandwidth parameter
    
    Returns:
        mi_reward: Mutual information-based diversity reward
    """
    # Flatten images to vectors for MI calculation
    if generated_images.ndim == 4:
        # Convert (B, C, H, W) or (B, H, W, C) to (B, features)
        gen_flat = generated_images.reshape(generated_images.shape[0], -1)
        ref_flat = reference_images.reshape(reference_images.shape[0], -1)
    else:
        gen_flat = generated_images
        ref_flat = reference_images
    
    # Downsample for computational efficiency (MI on full 512x512x3 is expensive)
    # Take every 16th pixel to reduce from ~786k features to ~49k features
    downsample_factor = 16
    gen_flat = gen_flat[:, ::downsample_factor]
    ref_flat = ref_flat[:, ::downsample_factor]
    
    # Ensure inputs are numpy arrays and normalize
    if isinstance(gen_flat, torch.Tensor):
        gen_flat = gen_flat.cpu().numpy()
    if isinstance(ref_flat, torch.Tensor):
        ref_flat = ref_flat.cpu().numpy()
    
    # Normalize pixel values to [0,1] if needed and then to unit vectors
    gen_flat = gen_flat.astype(np.float32) / 255.0 if gen_flat.max() > 1.0 else gen_flat.astype(np.float32)
    ref_flat = ref_flat.astype(np.float32) / 255.0 if ref_flat.max() > 1.0 else ref_flat.astype(np.float32)
    
    # Normalize to unit vectors
    gen_norm = np.linalg.norm(gen_flat, axis=1, keepdims=True)
    ref_norm = np.linalg.norm(ref_flat, axis=1, keepdims=True)
    gen_norm = np.maximum(gen_norm, 1e-10)
    ref_norm = np.maximum(ref_norm, 1e-10)
    gen_flat = gen_flat / gen_norm
    ref_flat = ref_flat / ref_norm
    
    # Auto-set gamma if not provided
    if gamma is None:
        X = np.vstack([gen_flat, ref_flat])
        pairwise_dists = np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=2)
        gamma = 1.0 / np.median(pairwise_dists[pairwise_dists > 0])
    
    # Calculate covariance matrices using RBF kernel
    try:
        # Covariance of generated images
        K_gg = rbf_kernel(gen_flat, gen_flat, gamma=gamma)
        
        # Covariance of reference images  
        K_rr = rbf_kernel(ref_flat, ref_flat, gamma=gamma)
        
        # Combined covariance matrix
        K_gr = rbf_kernel(gen_flat, ref_flat, gamma=gamma)
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
    BACKWARD COMPATIBILITY WRAPPER: This now works with image data, not features!
    
    Args:
        generated_features: This should now be image data (M x H x W x C), not ResNet features!
        reference_features: This should now be image data (N x H x W x C), not ResNet features!
        gamma: RBF kernel bandwidth parameter
    
    Returns:
        individual_rewards: Array of MI-based diversity rewards for each generated image
    """
    # Print warning about the change
    print("🔄 MI now uses image-level data instead of ResNet features for better information content")
    
    M = len(generated_features)
    
    # Calculate overall MI reward using image data
    overall_reward = calculate_image_based_mi_reward(
        generated_features, reference_features, gamma
    )
    
    if M <= 1:
        return np.array([overall_reward])
    
    # Calculate individual contributions using marginal utility
    individual_rewards = []
    
    for i in range(M):
        # Create subset without image i
        subset_indices = np.concatenate([np.arange(i), np.arange(i+1, M)])
        subset_images = generated_features[subset_indices]
        
        # Calculate MI reward without image i
        reward_without_i = calculate_image_based_mi_reward(
            subset_images, reference_features, gamma
        )
        
        # Marginal contribution of image i
        marginal_contribution = overall_reward - reward_without_i
        individual_rewards.append(marginal_contribution)
    
    individual_rewards = np.array(individual_rewards)
    
    # Normalize to ensure non-negative rewards
    if np.min(individual_rewards) < 0:
        individual_rewards -= np.min(individual_rewards)
    
    return individual_rewards