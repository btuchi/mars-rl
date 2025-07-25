import torch
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from diffusion_ppo.trajectory_recording import DiffusionSampler, extract_features_from_trajectory

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    print("Running on CPU only")

def quick_test():
    print("🚀 Quick Test: Trajectory Recording")
    
    # Test 1: Can we load the model?
    try:
        print("Loading model...")
        sampler = DiffusionSampler()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False
    
    # Test 2: Can we generate a trajectory?
    try:
        print("Generating trajectory...")
        trajectory = sampler.sample_with_trajectory_recording(
            prompt="a photo of a mars crater",
            num_inference_steps=5  # Very fast
        )
        print(f"✅ Trajectory generated: {len(trajectory.steps)} steps")
    except Exception as e:
        print(f"❌ Trajectory generation failed: {e}")
        return False
    
    # Test 3: Can we extract features?
    try:
        print("Extracting features...")
        features = extract_features_from_trajectory(trajectory, None)
        print(f"✅ Features extracted: shape {features.shape}")
    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
        return False
    
    # Test 4: Can we calculate diversity reward?
    try:
        print("Testing diversity reward...")
        from diffusion_ppo.diversity_reward import calculate_mmd_reward
        
        # Create dummy reference features for testing
        dummy_ref_features = np.random.randn(10, 512)
        dummy_ref_features = dummy_ref_features / np.linalg.norm(dummy_ref_features, axis=1, keepdims=True)
        
        # Calculate gamma
        X = np.vstack([features.reshape(1, -1), dummy_ref_features])
        pairwise_dists = np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=2)
        gamma = 1.0 / np.median(pairwise_dists[pairwise_dists > 0])
        
        reward = calculate_mmd_reward(features.reshape(1, -1), dummy_ref_features, gamma)
        print(f"✅ Diversity reward calculated: {reward:.4f}")
    except Exception as e:
        print(f"❌ Diversity reward failed: {e}")
        return False
    
    print("🎉 All quick tests passed!")
    return True

if __name__ == "__main__":
    success = quick_test()
    if success:
        print("Ready to run full test suite!")
    else:
        print("Please fix issues before running full tests.")