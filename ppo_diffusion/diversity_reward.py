import numpy as np
import torch
from sklearn.metrics.pairwise import rbf_kernel

def calculate_mmd_reward(generated_features, reference_features, gamma=None):
    """Helper function to calculate MMD reward from pre-computed features"""
    # This is just the core of your existing function
    
    K_XX = rbf_kernel(generated_features, generated_features, gamma=gamma)
    K_YY = rbf_kernel(reference_features, reference_features, gamma=gamma)
    K_XY = rbf_kernel(generated_features, reference_features, gamma=gamma)
    
    m = generated_features.shape[0]
    n = reference_features.shape[0]
    
    mmd_XX = (np.sum(K_XX) - np.trace(K_XX)) / (m * (m - 1)) if m > 1 else 0
    mmd_YY = (np.sum(K_YY) - np.trace(K_YY)) / (n * (n - 1)) if n > 1 else 0
    mmd_XY = np.sum(K_XY) / (m * n)
    
    mmd = mmd_XX + mmd_YY - 2 * mmd_XY
    return np.exp(-0.5 * mmd)

def calculate_individual_diversity_rewards(generated_features, reference_features, gamma=None):
    """
    Efficient batched calculation of individual diversity rewards.
    Pre-computes kernel matrices to avoid redundant calculations.
    
    Args:
        generated_features: Feature matrix for generated images (M x feature_dim)
        reference_features: Feature matrix for reference images (N x feature_dim)
        gamma: RBF kernel bandwidth parameter
    
    Returns:
        individual_rewards: Array of diversity rewards for each generated image
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
    
    # For small sets, use simple averaging
    M = len(generated_features)
    if M <= 2:
        overall_reward = calculate_mmd_reward(generated_features, reference_features, gamma)
        return np.ones(M) * overall_reward / M
    
    # PRE-COMPUTE ALL KERNEL MATRICES ONCE
    
    
    K_GG = rbf_kernel(generated_features, generated_features, gamma=gamma)  # M x M
    K_RR = rbf_kernel(reference_features, reference_features, gamma=gamma)  # N x N  
    K_GR = rbf_kernel(generated_features, reference_features, gamma=gamma)  # M x N
    
    N = len(reference_features)
    
    # Pre-compute reference terms (these don't change)
    ref_ref_sum = np.sum(K_RR) - np.trace(K_RR)
    ref_ref_term = ref_ref_sum / (N * (N - 1)) if N > 1 else 0
    
    # Calculate overall reward with all generated images
    gen_gen_sum = np.sum(K_GG) - np.trace(K_GG)
    gen_gen_term = gen_gen_sum / (M * (M - 1))
    gen_ref_term = np.sum(K_GR) / (M * N)
    overall_mmd = gen_gen_term + ref_ref_term - 2 * gen_ref_term
    overall_reward = np.exp(-0.5 * overall_mmd)
    
    # Calculate individual rewards efficiently using vectorized operations
    individual_rewards = []
    
    for i in range(M):
        # Calculate terms without image i
        # For gen-gen term: subtract row i and column i from the sum
        gen_gen_sum_without_i = (gen_gen_sum 
                                - np.sum(K_GG[i, :])    # subtract row i
                                - np.sum(K_GG[:, i])    # subtract column i  
                                + K_GG[i, i])          # add back diagonal (subtracted twice)
        
        gen_gen_term_without_i = gen_gen_sum_without_i / ((M-1) * (M-2)) if M > 2 else 0
        
        # For gen-ref term: subtract row i
        gen_ref_sum_without_i = np.sum(K_GR) - np.sum(K_GR[i, :])
        gen_ref_term_without_i = gen_ref_sum_without_i / ((M-1) * N)
        
        # Calculate MMD without image i
        mmd_without_i = gen_gen_term_without_i + ref_ref_term - 2 * gen_ref_term_without_i
        reward_without_i = np.exp(-0.5 * mmd_without_i)
        
        # Individual contribution
        marginal_contribution = overall_reward - reward_without_i
        individual_rewards.append(marginal_contribution)
    
    # Convert to array and normalize if needed
    individual_rewards = np.array(individual_rewards)
    if np.min(individual_rewards) < 0:
        individual_rewards -= np.min(individual_rewards)
    
    return individual_rewards