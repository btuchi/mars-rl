import torch, gc
torch.cuda.empty_cache()
gc.collect()
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from diffusion_ppo.trajectory_recording import DiffusionSampler, extract_features_from_trajectory
import time

def test_basic_sampling():
    """Test basic trajectory recording functionality."""
    print("=" * 50)
    print("Test 1: Basic Sampling with Trajectory Recording")
    print("=" * 50)
    
    # Initialize sampler
    print("Loading Stable Diffusion model...")
    start_time = time.time()
    sampler = DiffusionSampler()
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds")
    
    # Test single trajectory
    print("\nGenerating single trajectory...")
    start_time = time.time()
    trajectory = sampler.sample_with_trajectory_recording(
        prompt="an aerial image of a mars crater",
        num_inference_steps=10,  # Reduced for faster testing
        height=256,
        width=256
    )
    generation_time = time.time() - start_time
    print(f"Generation completed in {generation_time:.2f} seconds")
    
    # Verify trajectory structure
    print(f"\nTrajectory Analysis:")
    print(f"- Number of steps: {len(trajectory.steps)}")
    print(f"- Final image shape: {trajectory.final_image.shape}")
    print(f"- Condition shape: {trajectory.condition.shape}")
    
    # Check first few steps
    for i in range(min(3, len(trajectory.steps))):
        step = trajectory.steps[i]
        print(f"- Step {i}: timestep={step.timestep}, state_shape={step.state.shape}")
        print(f"  - action_shape={step.action.shape}, requires_grad={step.action.requires_grad}")
        print(f"  - log_prob={step.log_prob.item():.4f}")
    
    return trajectory, sampler

def test_gradient_flow(trajectory):
    """Test that gradients are preserved through the trajectory."""
    print("\n" + "=" * 50)
    print("Test 2: Gradient Flow Verification")
    print("=" * 50)
    
    # Check if tensors have gradients enabled
    gradient_preserved = []
    
    for i, step in enumerate(trajectory.steps[:5]):  # Check first 5 steps
        has_grad = (step.action.requires_grad and 
                   step.log_prob.requires_grad and 
                   step.noise_pred.requires_grad)
        gradient_preserved.append(has_grad)
        print(f"Step {i}: Gradients preserved = {has_grad}")
    
    all_gradients_ok = all(gradient_preserved)
    print(f"\nGradient Flow Test: {'PASSED' if all_gradients_ok else 'FAILED'}")
    
    # Try a simple backward pass to verify gradients work
    try:
        test_loss = trajectory.steps[0].log_prob
        test_loss.backward(retain_graph=True)
        print("Backward pass test: PASSED")
        return True
    except Exception as e:
        print(f"Backward pass test: FAILED - {e}")
        return False

def test_feature_extraction(trajectory):
    """Test feature extraction from trajectory."""
    print("\n" + "=" * 50)
    print("Test 3: Feature Extraction")
    print("=" * 50)
    
    try:
        # Extract features from final image
        print("Extracting CLIP features...")
        with torch.no_grad():
            features = extract_features_from_trajectory(trajectory, None)
        print(f"Feature extraction successful!")
        print(f"- Feature shape: {features.shape}")
        print(f"- Feature dtype: {features.dtype}")
        print(f"- Feature range: [{features.min():.3f}, {features.max():.3f}]")
        print(f"- Feature norm: {np.linalg.norm(features):.3f}")
        
        return features
    except Exception as e:
        print(f"Feature extraction failed: {e}")
        return None

def test_batch_generation(sampler):
    """Test batch trajectory generation."""
    print("\n" + "=" * 50)
    print("Test 4: Batch Generation")
    print("=" * 50)
    
    prompts = [
        "a photo of a mars crater",
        "a large crater on mars surface", 
        "martian terrain with craters"
    ]
    
    print(f"Generating {len(prompts)} trajectories...")
    start_time = time.time()
    
    trajectories = sampler.sample_batch_with_trajectories(
        prompts=prompts,
        num_inference_steps=10  # Very fast for testing
    )
    
    batch_time = time.time() - start_time
    print(f"Batch generation completed in {batch_time:.2f} seconds")
    print(f"Average time per image: {batch_time/len(prompts):.2f} seconds")
    
    # Verify all trajectories
    for i, traj in enumerate(trajectories):
        print(f"Trajectory {i}: {len(traj.steps)} steps, final_image_shape={traj.final_image.shape}")
    
    return trajectories

def test_image_saving(trajectory, save_dir="test_outputs"):
    """Test saving generated images."""
    print("\n" + "=" * 50)
    print("Test 5: Image Saving")
    print("=" * 50)
    
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # Convert tensor to PIL Image
        final_image = trajectory.final_image.squeeze(0).cpu()
        final_image = torch.clamp(final_image, 0, 1)
        
        # Convert to PIL
        from torchvision.transforms import ToPILImage
        to_pil = ToPILImage()
        pil_image = to_pil(final_image)
        
        # Save image
        save_path = os.path.join(save_dir, "test_generated_crater.png")
        pil_image.save(save_path)
        print(f"Image saved to: {save_path}")
        print(f"Image size: {pil_image.size}")
        
        return save_path
    except Exception as e:
        print(f"Image saving failed: {e}")
        return None

def test_diversity_reward_integration(trajectories):
    """Test integration with your diversity reward system."""
    print("\n" + "=" * 50)
    print("Test 6: Diversity Reward Integration")
    print("=" * 50)
    
    try:
        # Load your reference features
        print("Loading reference features...")
        data = np.load('reference_clip_features.npz')
        reference_filenames = list(data.files)
        reference_features = np.vstack([data[filename] for filename in reference_filenames])
        
        # Extract features from generated trajectories
        print("Extracting features from generated images...")
        generated_features = []
        for i, traj in enumerate(trajectories):
            features = extract_features_from_trajectory(traj, None)
            if features is not None:
                generated_features.append(features)
                print(f"  Trajectory {i}: Features extracted")
        
        generated_features = np.vstack(generated_features)
        
        # Calculate diversity reward using your MMD function
        print("Calculating diversity reward...")
        from diffusion_ppo.diversity_reward import calculate_mmd_reward  # Adjust import as needed
        
        diversity_reward = calculate_mmd_reward(generated_features, reference_features)
        print(f"Diversity reward: {diversity_reward:.4f}")
        
        # Calculate individual rewards
        from diffusion_ppo.diversity_reward import calculate_individual_diversity_rewards
        individual_rewards = calculate_individual_diversity_rewards(
            generated_features, reference_features
        )
        print(f"Individual rewards: {individual_rewards}")
        print(f"Reward range: [{individual_rewards.min():.4f}, {individual_rewards.max():.4f}]")
        
        return diversity_reward, individual_rewards
        
    except Exception as e:
        print(f"Diversity reward integration failed: {e}")
        print("Make sure you have 'reference_clip_features.npz' and diversity_reward.py")
        return None, None

def test_memory_usage():
    """Test memory usage during trajectory generation."""
    print("\n" + "=" * 50)
    print("Test 7: Memory Usage")
    print("=" * 50)
    
    if torch.cuda.is_available():
        # Clear cache and measure
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated() / 1024**2  # MB
        
        print(f"Initial GPU memory: {initial_memory:.1f} MB")
        
        # Generate trajectory and measure
        sampler = DiffusionSampler()
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        trajectory = sampler.sample_with_trajectory_recording(
            "a photo of a mars crater",
            num_inference_steps=10
        )
        
        final_memory = torch.cuda.memory_allocated() / 1024**2  # MB
        
        print(f"Peak GPU memory: {peak_memory:.1f} MB")
        print(f"Final GPU memory: {final_memory:.1f} MB")
        print(f"Memory increase: {final_memory - initial_memory:.1f} MB")
        
        return sampler
    else:
        print("CUDA not available, skipping memory test")
        return DiffusionSampler()

def run_all_tests():
    """Run all tests in sequence."""
    print("🚀 Starting Trajectory Recording Tests")
    print("=" * 60)
    
    # Test 1: Basic functionality
    trajectory, sampler = test_basic_sampling()
    
    # Test 2: Gradient flow
    gradient_ok = test_gradient_flow(trajectory)
    
    # Test 3: Feature extraction
    features = test_feature_extraction(trajectory)
    
    # Test 4: Batch generation
    trajectories = test_batch_generation(sampler)
    
    # Test 5: Image saving
    save_path = test_image_saving(trajectory)
    
    # Test 6: Diversity reward integration
    diversity_reward, individual_rewards = test_diversity_reward_integration(trajectories)
    
    # Test 7: Memory usage
    # sampler = test_memory_usage()  # Uncomment if you want to test memory
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 TEST SUMMARY")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 6
    
    if trajectory is not None and len(trajectory.steps) > 0:
        print("✅ Basic sampling: PASSED")
        tests_passed += 1
    else:
        print("❌ Basic sampling: FAILED")
    
    if gradient_ok:
        print("✅ Gradient flow: PASSED")
        tests_passed += 1
    else:
        print("❌ Gradient flow: FAILED")
    
    if features is not None:
        print("✅ Feature extraction: PASSED")
        tests_passed += 1
    else:
        print("❌ Feature extraction: FAILED")
    
    if trajectories and len(trajectories) > 0:
        print("✅ Batch generation: PASSED")
        tests_passed += 1
    else:
        print("❌ Batch generation: FAILED")
    
    if save_path:
        print("✅ Image saving: PASSED")
        tests_passed += 1
    else:
        print("❌ Image saving: FAILED")
    
    if diversity_reward is not None:
        print("✅ Diversity reward integration: PASSED")
        tests_passed += 1
    else:
        print("❌ Diversity reward integration: FAILED")
    
    print(f"\nOverall: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Ready to implement policy gradients.")
    else:
        print("⚠️  Some tests failed. Please fix issues before proceeding.")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    success = run_all_tests()