#!/usr/bin/env python3
"""Quick test to measure current reward signal strength"""

import numpy as np
import sys
import os
sys.path.append(os.path.abspath('.'))

from ppo_diffusion.training.reward_metrics.mmd import calculate_individual_diversity_rewards, calculate_mmd_reward

def test_reward_signal_strength():
    """Test how strong/weak the current reward signals are"""
    print("🔍 Testing Current Reward Signal Strength")
    print("=" * 50)
    
    # Load reference features
    try:
        data = np.load('reference_crater_features.npz')
        reference_features = np.vstack([data[filename] for filename in data.files])
        print(f"Loaded {len(reference_features)} reference features")
    except Exception as e:
        print(f"❌ Could not load reference features: {e}")
        return
    
    # Generate some dummy "generated" features with different diversity levels
    np.random.seed(42)
    
    print("\n📊 Testing Different Scenarios:")
    
    # Scenario 1: Features identical to references (should get low reward)
    identical_features = reference_features[:4].copy()
    reward_identical = calculate_mmd_reward(identical_features, reference_features)
    individual_identical = calculate_individual_diversity_rewards(identical_features, reference_features)
    
    print(f"\n1. Identical to references:")
    print(f"   Overall reward: {reward_identical:.6f}")
    print(f"   Individual range: [{individual_identical.min():.6f}, {individual_identical.max():.6f}]")
    print(f"   Individual std: {individual_identical.std():.6f}")
    
    # Scenario 2: Slightly different features
    slightly_different = reference_features[:4].copy()
    slightly_different += np.random.normal(0, 0.1, slightly_different.shape)
    reward_slight = calculate_mmd_reward(slightly_different, reference_features)
    individual_slight = calculate_individual_diversity_rewards(slightly_different, reference_features)
    
    print(f"\n2. Slightly different (noise=0.1):")
    print(f"   Overall reward: {reward_slight:.6f}")
    print(f"   Individual range: [{individual_slight.min():.6f}, {individual_slight.max():.6f}]")
    print(f"   Individual std: {individual_slight.std():.6f}")
    
    # Scenario 3: Very different features
    very_different = np.random.normal(0, 1.0, (4, reference_features.shape[1]))
    reward_very = calculate_mmd_reward(very_different, reference_features)
    individual_very = calculate_individual_diversity_rewards(very_different, reference_features)
    
    print(f"\n3. Very different (random):")
    print(f"   Overall reward: {reward_very:.6f}")
    print(f"   Individual range: [{individual_very.min():.6f}, {individual_very.max():.6f}]")
    print(f"   Individual std: {individual_very.std():.6f}")
    
    # Calculate reward signal strength
    reward_range = reward_very - reward_identical
    print(f"\n📈 SIGNAL STRENGTH ANALYSIS:")
    print(f"   Total reward range: {reward_range:.6f}")
    print(f"   Relative improvement: {(reward_very/reward_identical - 1)*100:.1f}%")
    
    if reward_range < 0.01:
        print("   ⚠️  WEAK SIGNAL: Very small reward differences!")
        print("   💡 Recommendation: Reduce MMD scaling (change exp(-5.0*mmd) to exp(-1.0*mmd))")
    elif reward_range < 0.1:
        print("   ⚠️  MODERATE SIGNAL: Could be stronger")
        print("   💡 Recommendation: Consider reducing MMD scaling")
    else:
        print("   ✅ STRONG SIGNAL: Good reward differentiation")
    
    return reward_range

if __name__ == "__main__":
    test_reward_signal_strength()