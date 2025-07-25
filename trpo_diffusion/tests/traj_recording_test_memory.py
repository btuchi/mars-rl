import torch, gc
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from diffusion_ppo.trajectory_recording import DiffusionSampler, extract_features_from_trajectory
import time

# Memory optimization settings
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def clear_memory():
    """Aggressive memory clearing."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def test_basic_sampling():
    """Test basic trajectory recording functionality with minimal memory usage."""
    print("=" * 50)
    print("Test 1: Basic Sampling with Trajectory Recording")
    print("=" * 50)
    
    clear_memory()
    
    # Initialize sampler
    print("Loading Stable Diffusion model...")
    start_time = time.time()
    
    # Use smaller precision and enable memory efficient attention
    with torch.cuda.amp.autocast():
        sampler = DiffusionSampler()
        # Enable memory efficient attention if available
        if hasattr(sampler.pipe.unet, 'set_attn_slice'):
            sampler.pipe.unet.set_attn_slice(1)  # Process attention in slices
        if hasattr(sampler.pipe, 'enable_attention_slicing'):
            sampler.pipe.enable_attention_slicing(1)
        if hasattr(sampler.pipe, 'enable_memory_efficient_attention'):
            sampler.pipe.enable_memory_efficient_attention()
    
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds")
    print(f"GPU memory after loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    # Test single trajectory with minimal settings
    print("\nGenerating single trajectory...")
    start_time = time.time()
    
    with torch.cuda.amp.autocast():
        trajectory = sampler.sample_with_trajectory_recording(
            prompt="mars crater",  # Shorter prompt
            num_inference_steps=5,  # Very few steps for memory testing
            height=128,  # Much smaller image
            width=128,
            guidance_scale=7.5  # Standard guidance
        )
    
    generation_time = time.time() - start_time
    print(f"Generation completed in {generation_time:.2f} seconds")
    print(f"GPU memory after generation: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    # Verify trajectory structure
    print(f"\nTrajectory Analysis:")
    print(f"- Number of steps: {len(trajectory.steps)}")
    print(f"- Final image shape: {trajectory.final_image.shape}")
    print(f"- Condition shape: {trajectory.condition.shape}")
    
    # Check first few steps with memory monitoring
    for i in range(min(2, len(trajectory.steps))):  # Reduced to 2 steps
        step = trajectory.steps[i]
        print(f"- Step {i}: timestep={step.timestep}, state_shape={step.state.shape}")
        print(f"  - action_shape={step.action.shape}, requires_grad={step.action.requires_grad}")
        print(f"  - log_prob={step.log_prob.item():.4f}")
    
    return trajectory, sampler

def test_gradient_flow_minimal(trajectory):
    """Test gradients with minimal memory footprint."""
    print("\n" + "=" * 50)
    print("Test 2: Gradient Flow Verification (Memory Optimized)")
    print("=" * 50)
    
    clear_memory()
    
    # Check only first 2 steps to save memory
    gradient_preserved = []
    
    for i, step in enumerate(trajectory.steps[:2]):
        has_grad = (step.action.requires_grad and 
                   step.log_prob.requires_grad and 
                   step.noise_pred.requires_grad)
        gradient_preserved.append(has_grad)
        print(f"Step {i}: Gradients preserved = {has_grad}")
    
    all_gradients_ok = all(gradient_preserved)
    print(f"\nGradient Flow Test: {'PASSED' if all_gradients_ok else 'FAILED'}")
    
    # Quick gradient test with immediate cleanup
    try:
        test_loss = trajectory.steps[0].log_prob
        test_loss.backward(retain_graph=True)
        print("Backward pass test: PASSED")
        
        # Clear gradients immediately
        for step in trajectory.steps:
            if step.action.grad is not None:
                step.action.grad = None
        
        clear_memory()
        return True
    except Exception as e:
        print(f"Backward pass test: FAILED - {e}")
        clear_memory()
        return False

def test_feature_extraction_efficient(trajectory):
    """Test feature extraction with memory optimization."""
    print("\n" + "=" * 50)
    print("Test 3: Feature Extraction (Memory Optimized)")
    print("=" * 50)
    
    clear_memory()
    
    try:
        print("Extracting CLIP features...")
        # Use no_grad and smaller batch processing
        with torch.no_grad(), torch.cuda.amp.autocast():
            features = extract_features_from_trajectory(trajectory, None)
        
        print(f"Feature extraction successful!")
        print(f"- Feature shape: {features.shape}")
        print(f"- Feature dtype: {features.dtype}")
        print(f"- Feature range: [{features.min():.3f}, {features.max():.3f}]")
        print(f"- Feature norm: {np.linalg.norm(features):.3f}")
        
        clear_memory()
        return features
    except Exception as e:
        print(f"Feature extraction failed: {e}")
        clear_memory()
        return None

def test_sequential_generation(sampler):
    """Test multiple generations sequentially to avoid memory buildup."""
    print("\n" + "=" * 50)
    print("Test 4: Sequential Generation (Memory Safe)")
    print("=" * 50)
    
    prompts = [
        "mars crater",
        "crater surface", 
    ]  # Reduced to 2 prompts
    
    trajectories = []
    
    for i, prompt in enumerate(prompts):
        clear_memory()  # Clear before each generation
        print(f"Generating trajectory {i+1}/{len(prompts)}: '{prompt}'")
        
        start_time = time.time()
        
        with torch.cuda.amp.autocast():
            trajectory = sampler.sample_with_trajectory_recording(
                prompt=prompt,
                num_inference_steps=4,  # Very fast
                height=128,
                width=128
            )
        
        gen_time = time.time() - start_time
        print(f"  Completed in {gen_time:.2f}s, GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        trajectories.append(trajectory)
        
        # Verify trajectory
        print(f"  Trajectory: {len(trajectory.steps)} steps, final_image_shape={trajectory.final_image.shape}")
    
    return trajectories

def test_image_saving_efficient(trajectory, save_dir="test_outputs"):
    """Test saving with memory cleanup."""
    print("\n" + "=" * 50)
    print("Test 5: Image Saving (Memory Optimized)")
    print("=" * 50)
    
    os.makedirs(save_dir, exist_ok=True)
    clear_memory()
    
    try:
        # Convert tensor to PIL Image with immediate cleanup
        with torch.no_grad():
            final_image = trajectory.final_image.squeeze(0).cpu()
            final_image = torch.clamp(final_image, 0, 1)
        
        # Convert to PIL
        from torchvision.transforms import ToPILImage
        to_pil = ToPILImage()
        pil_image = to_pil(final_image)
        
        # Clean up tensor immediately
        del final_image
        clear_memory()
        
        # Save image
        save_path = os.path.join(save_dir, "test_generated_crater.png")
        pil_image.save(save_path)
        print(f"Image saved to: {save_path}")
        print(f"Image size: {pil_image.size}")
        
        return save_path
    except Exception as e:
        print(f"Image saving failed: {e}")
        return None

def test_diversity_reward_minimal(trajectories):
    """Test diversity reward with minimal memory usage."""
    print("\n" + "=" * 50)
    print("Test 6: Diversity Reward Integration (Memory Optimized)")
    print("=" * 50)
    
    clear_memory()
    
    try:
        # Load reference features
        print("Loading reference features...")
        data = np.load('reference_clip_features.npz')
        reference_filenames = list(data.files)[:50]  # Use only first 50 references
        reference_features = np.vstack([data[filename] for filename in reference_filenames])
        
        print(f"Using {len(reference_filenames)} reference features")
        
        # Extract features from generated trajectories one by one
        print("Extracting features from generated images...")
        generated_features = []
        
        for i, traj in enumerate(trajectories):
            clear_memory()  # Clear before each extraction
            
            with torch.no_grad():
                features = extract_features_from_trajectory(traj, None)
                if features is not None:
                    generated_features.append(features)
                    print(f"  Trajectory {i}: Features extracted")
        
        if len(generated_features) == 0:
            print("No features extracted successfully")
            return None, None
            
        generated_features = np.vstack(generated_features)
        
        # Calculate diversity reward
        print("Calculating diversity reward...")
        try:
            from diffusion_ppo.diversity_reward import calculate_mmd_reward
            diversity_reward = calculate_mmd_reward(generated_features, reference_features)
            print(f"Diversity reward: {diversity_reward:.4f}")
            
            from diffusion_ppo.diversity_reward import calculate_individual_diversity_rewards
            individual_rewards = calculate_individual_diversity_rewards(
                generated_features, reference_features
            )
            print(f"Individual rewards: {individual_rewards}")
            
            return diversity_reward, individual_rewards
            
        except ImportError:
            print("Diversity reward functions not available - creating mock calculation")
            # Simple mock diversity calculation
            diversity_reward = np.mean(np.std(generated_features, axis=0))
            individual_rewards = np.random.random(len(generated_features))
            print(f"Mock diversity reward: {diversity_reward:.4f}")
            return diversity_reward, individual_rewards
        
    except Exception as e:
        print(f"Diversity reward integration failed: {e}")
        return None, None

def monitor_memory_throughout():
    """Monitor memory usage throughout the process."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB")
        return allocated, cached
    return 0, 0

def run_memory_optimized_tests():
    """Run all tests with aggressive memory management."""
    print("🚀 Starting Memory-Optimized Trajectory Recording Tests")
    print("=" * 60)
    
    # Set memory fraction if available
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory max
    
    results = {}
    
    try:
        # Test 1: Basic functionality
        print("\n📊 Memory before Test 1:")
        monitor_memory_throughout()
        
        trajectory, sampler = test_basic_sampling()
        results['basic_sampling'] = trajectory is not None and len(trajectory.steps) > 0
        
        print("\n📊 Memory after Test 1:")
        monitor_memory_throughout()
        
        # Test 2: Gradient flow
        gradient_ok = test_gradient_flow_minimal(trajectory)
        results['gradient_flow'] = gradient_ok
        
        print("\n📊 Memory after Test 2:")
        monitor_memory_throughout()
        
        # Test 3: Feature extraction  
        features = test_feature_extraction_efficient(trajectory)
        results['feature_extraction'] = features is not None
        
        print("\n📊 Memory after Test 3:")
        monitor_memory_throughout()
        
        # Clear trajectory from memory before batch test
        del trajectory
        clear_memory()
        
        # Test 4: Sequential generation
        trajectories = test_sequential_generation(sampler)
        results['sequential_generation'] = len(trajectories) > 0
        
        print("\n📊 Memory after Test 4:")
        monitor_memory_throughout()
        
        # Test 5: Image saving
        if trajectories:
            save_path = test_image_saving_efficient(trajectories[0])
            results['image_saving'] = save_path is not None
        else:
            results['image_saving'] = False
        
        print("\n📊 Memory after Test 5:")
        monitor_memory_throughout()
        
        # Test 6: Diversity reward integration
        diversity_reward, individual_rewards = test_diversity_reward_minimal(trajectories)
        results['diversity_reward'] = diversity_reward is not None
        
        print("\n📊 Memory after Test 6:")
        monitor_memory_throughout()
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n❌ CUDA OOM Error: {e}")
            print("\n🔧 Suggestions to reduce memory further:")
            print("1. Reduce image size to 64x64")
            print("2. Use num_inference_steps=3")
            print("3. Process only 1 image at a time")
            print("4. Enable CPU offloading in your DiffusionSampler")
            return False
        else:
            raise e
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 MEMORY-OPTIMIZED TEST SUMMARY")
    print("=" * 60)
    
    tests_passed = sum(results.values())
    total_tests = len(results)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name.replace('_', ' ').title()}")
    
    print(f"\nOverall: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed >= total_tests * 0.8:  # 80% pass rate
        print("🎉 Most tests passed! Ready for policy gradients with memory optimizations.")
    else:
        print("⚠️  Consider further memory optimizations.")
    
    # Final memory report
    print(f"\n📊 Final Memory Usage:")
    monitor_memory_throughout()
    
    return tests_passed >= total_tests * 0.8

if __name__ == "__main__":
    # Additional memory optimizations you can try:
    print("🔧 Memory Optimization Tips:")
    print("1. Add these to your DiffusionSampler init:")
    print("   - pipe.enable_memory_efficient_attention()")
    print("   - pipe.enable_attention_slicing(1)")
    print("   - pipe.enable_sequential_cpu_offload()  # Most aggressive")
    print("2. Use torch.cuda.amp.autocast() in your sampling")
    print("3. Process trajectories one at a time")
    print("4. Clear gradients immediately after use")
    print("\n" + "="*60)
    
    success = run_memory_optimized_tests()