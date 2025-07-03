#!/usr/bin/env python3
"""Test different MMD scaling factors to find optimal reward signal strength"""

import numpy as np
import sys
import os
sys.path.append(os.path.abspath('.'))

from sklearn.metrics.pairwise import rbf_kernel

def calculate_mmd_with_scaling(generated_features, reference_features, scaling_factor=1.0, gamma=None):
    """Calculate MMD reward with custom scaling factor"""
    K_XX = rbf_kernel(generated_features, generated_features, gamma=gamma)
    K_YY = rbf_kernel(reference_features, reference_features, gamma=gamma)
    K_XY = rbf_kernel(generated_features, reference_features, gamma=gamma)
    
    m = generated_features.shape[0]
    n = reference_features.shape[0]
    
    mmd_XX = (np.sum(K_XX) - np.trace(K_XX)) / (m * (m - 1)) if m > 1 else 0
    mmd_YY = (np.sum(K_YY) - np.trace(K_YY)) / (n * (n - 1)) if n > 1 else 0
    mmd_XY = np.sum(K_XY) / (m * n)
    
    mmd = mmd_XX + mmd_YY - 2 * mmd_XY
    return np.exp(-scaling_factor * mmd)

def test_scaling_factors():
    """Test different scaling factors to find optimal reward signal"""
    print("🔍 Testing Different MMD Scaling Factors")
    print("=" * 60)
    
    # Load reference features
    data = np.load('reference_crater_features.npz')
    reference_features = np.vstack([data[filename] for filename in data.files])
    print(f"Loaded {len(reference_features)} reference features")
    
    # Test scenarios
    np.random.seed(42)
    identical_features = reference_features[:4].copy()
    slightly_different = reference_features[:4].copy() + np.random.normal(0, 0.1, (4, reference_features.shape[1]))
    very_different = np.random.normal(0, 1.0, (4, reference_features.shape[1]))
    
    scaling_factors = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    print(f"\n{'Scale':<8} {'Identical':<10} {'Slightly':<10} {'Very Diff':<10} {'Range':<10} {'Rel %':<8}")
    print("-" * 60)
    
    best_scaling = None
    best_range = 0
    
    for scale in scaling_factors:
        reward_identical = calculate_mmd_with_scaling(identical_features, reference_features, scale)
        reward_slight = calculate_mmd_with_scaling(slightly_different, reference_features, scale)
        reward_very = calculate_mmd_with_scaling(very_different, reference_features, scale)
        
        reward_range = reward_very - reward_identical
        relative_improvement = (reward_very/reward_identical - 1) * 100
        
        print(f"{scale:<8.1f} {reward_identical:<10.6f} {reward_slight:<10.6f} {reward_very:<10.6f} {reward_range:<10.6f} {relative_improvement:<8.1f}%")
        
        # Track best range (most negative means best signal)
        if abs(reward_range) > abs(best_range):
            best_range = reward_range
            best_scaling = scale
    
    print("-" * 60)
    print(f"✅ Best scaling factor: {best_scaling} (range: {best_range:.6f})")
    
    # Recommendation
    if abs(best_range) > 0.5:
        print("💡 Excellent signal strength! Use this scaling.")
    elif abs(best_range) > 0.2:
        print("💡 Good signal strength. Should work well for training.")
    elif abs(best_range) > 0.1:
        print("💡 Moderate signal strength. May work but consider lower scaling.")
    else:
        print("⚠️  Still weak signal. Consider scaling < 0.1 or different approach.")
    
    return best_scaling

if __name__ == "__main__":
    best = test_scaling_factors()