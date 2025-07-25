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

def torch_rbf_kernel(X, Y, gamma=None):
    """PyTorch implementation of RBF kernel that preserves gradients"""
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    
    # Compute squared Euclidean distances
    X_norm = (X**2).sum(dim=1, keepdim=True)
    Y_norm = (Y**2).sum(dim=1, keepdim=True)
    dist_sq = X_norm + Y_norm.T - 2 * torch.mm(X, Y.T)
    
    # Apply RBF kernel
    return torch.exp(-gamma * dist_sq)

def calculate_individual_mmd_rewards(generated_features, reference_features, gamma=None):
    """
    Efficient batched calculation of individual diversity rewards.
    Handles both PyTorch tensors (for gradient flow) and numpy arrays.
    
    Args:
        generated_features: Feature matrix for generated images (M x feature_dim)
        reference_features: Feature matrix for reference images (N x feature_dim)
        gamma: RBF kernel bandwidth parameter
    
    Returns:
        individual_rewards: Array/tensor of diversity rewards for each generated image
    """
    # Check if we're working with tensors (for gradient flow) or numpy arrays
    is_tensor_mode = isinstance(generated_features, torch.Tensor)
    
    if is_tensor_mode:
        # PyTorch path - preserve gradients for scheduler policy training
        return _calculate_mmd_rewards_torch(generated_features, reference_features, gamma)
    else:
        # Numpy path - for other training modes
        return _calculate_mmd_rewards_numpy(generated_features, reference_features, gamma)

def _calculate_mmd_rewards_numpy(generated_features, reference_features, gamma=None):
    """Original numpy implementation"""
    # Ensure inputs are numpy arrays
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
    
    M = len(generated_features)
    N = len(reference_features)
    
    # Pre-compute all kernel matrices
    K_GG = rbf_kernel(generated_features, generated_features, gamma=gamma)  # M x M
    K_RR = rbf_kernel(reference_features, reference_features, gamma=gamma)  # N x N  
    K_GR = rbf_kernel(generated_features, reference_features, gamma=gamma)  # M x N
    
    # Pre-compute reference terms (these don't change)
    ref_ref_sum = np.sum(K_RR) - np.trace(K_RR)
    ref_ref_term = ref_ref_sum / (N * (N - 1)) if N > 1 else 0
    
    # Calculate overall reward with all generated images
    gen_gen_sum = np.sum(K_GG) - np.trace(K_GG)
    gen_gen_term = gen_gen_sum / (M * (M - 1)) if M > 1 else 0
    gen_ref_term = np.sum(K_GR) / (M * N)
    overall_mmd = gen_gen_term + ref_ref_term - 2 * gen_ref_term
    overall_reward = np.exp(-0.5 * overall_mmd)
    
    # Calculate individual rewards efficiently using vectorized operations
    individual_rewards = []
    
    for i in range(M):
        # Calculate terms without image i
        gen_gen_sum_without_i = (gen_gen_sum 
                                - np.sum(K_GG[i, :])    # subtract row i
                                - np.sum(K_GG[:, i])    # subtract column i  
                                + K_GG[i, i])          # add back diagonal (subtracted twice)
        
        gen_gen_term_without_i = gen_gen_sum_without_i / ((M-1) * (M-2)) if M > 2 else 0
        
        # For gen-ref term: subtract row i
        gen_ref_sum_without_i = np.sum(K_GR) - np.sum(K_GR[i, :])
        gen_ref_term_without_i = gen_ref_sum_without_i / ((M-1) * N) if M > 1 else 0
        
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

def _calculate_mmd_rewards_torch(generated_features, reference_features, gamma=None):
    """PyTorch implementation that preserves gradients"""
    device = generated_features.device
    
    # Convert reference features to tensor if needed
    if isinstance(reference_features, np.ndarray):
        reference_features = torch.from_numpy(reference_features).to(device).float()
    
    # Normalize features
    gen_norm = torch.norm(generated_features, dim=1, keepdim=True)
    ref_norm = torch.norm(reference_features, dim=1, keepdim=True)
    gen_norm = torch.clamp(gen_norm, min=1e-10)
    ref_norm = torch.clamp(ref_norm, min=1e-10)
    generated_features = generated_features / gen_norm
    reference_features = reference_features / ref_norm
    
    # Auto-set gamma if not provided
    if gamma is None:
        X = torch.cat([generated_features, reference_features], dim=0)
        pairwise_dists = torch.cdist(X, X) ** 2
        median_dist = torch.median(pairwise_dists[pairwise_dists > 0])
        gamma = 1.0 / median_dist
    
    M = generated_features.shape[0]
    N = reference_features.shape[0]
    
    # Pre-compute all kernel matrices using PyTorch
    K_GG = torch_rbf_kernel(generated_features, generated_features, gamma)  # M x M
    K_RR = torch_rbf_kernel(reference_features, reference_features, gamma)  # N x N  
    K_GR = torch_rbf_kernel(generated_features, reference_features, gamma)  # M x N
    
    # Pre-compute reference terms (these don't change)
    ref_ref_sum = torch.sum(K_RR) - torch.trace(K_RR)
    ref_ref_term = ref_ref_sum / (N * (N - 1)) if N > 1 else torch.tensor(0.0, device=device)
    
    # Calculate overall reward with all generated images
    gen_gen_sum = torch.sum(K_GG) - torch.trace(K_GG)
    gen_gen_term = gen_gen_sum / (M * (M - 1)) if M > 1 else torch.tensor(0.0, device=device)
    gen_ref_term = torch.sum(K_GR) / (M * N)
    overall_mmd = gen_gen_term + ref_ref_term - 2 * gen_ref_term
    overall_reward = torch.exp(-0.5 * overall_mmd)
    
    # Calculate individual rewards efficiently using vectorized operations
    individual_rewards = []
    
    for i in range(M):
        # Calculate terms without image i
        gen_gen_sum_without_i = (gen_gen_sum 
                                - torch.sum(K_GG[i, :])    # subtract row i
                                - torch.sum(K_GG[:, i])    # subtract column i  
                                + K_GG[i, i])             # add back diagonal (subtracted twice)
        
        gen_gen_term_without_i = gen_gen_sum_without_i / ((M-1) * (M-2)) if M > 2 else torch.tensor(0.0, device=device)
        
        # For gen-ref term: subtract row i
        gen_ref_sum_without_i = torch.sum(K_GR) - torch.sum(K_GR[i, :])
        gen_ref_term_without_i = gen_ref_sum_without_i / ((M-1) * N) if M > 1 else torch.tensor(0.0, device=device)
        
        # Calculate MMD without image i
        mmd_without_i = gen_gen_term_without_i + ref_ref_term - 2 * gen_ref_term_without_i
        reward_without_i = torch.exp(-0.5 * mmd_without_i)
        
        # Individual contribution
        marginal_contribution = overall_reward - reward_without_i
        individual_rewards.append(marginal_contribution)
    
    # Stack into tensor and normalize if needed
    individual_rewards = torch.stack(individual_rewards)
    min_reward = torch.min(individual_rewards)
    if min_reward < 0:
        individual_rewards = individual_rewards - min_reward
    
    return individual_rewards