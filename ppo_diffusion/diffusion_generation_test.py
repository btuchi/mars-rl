"""
Test script to generate images using the trained Diffusion PPO model.
Compares trained model vs original pretrained model and evaluates diversity rewards.
"""

import torch
import numpy as np
import os
import time
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from pathlib import Path

# Import your modules
from trajectory_recording import DiffusionSampler, extract_features_from_trajectory
from diffusion_ppo_agent import DiffusionPPOAgent
from diversity_reward import calculate_individual_diversity_rewards

class DiffusionModelTester:
    """Test trained diffusion models and compare with baseline"""
    
    def __init__(self, device: str = "cuda", use_fp16: bool = False, training_timestamp: str = None):
        self.device = device
        self.use_fp16 = use_fp16
        self.training_timestamp = training_timestamp
        
        # ppo_diffusion/
        self.current_dir = Path(os.path.abspath(__file__)).parent
        
        # Use proper path construction
        self.test_dir = self.current_dir / "images" / "after_training" / self.training_timestamp
        self.baseline_dir = self.current_dir / "images" / "no_training" / self.training_timestamp
        self.comparison_dir = self.current_dir / "plots" / "comparison" / self.training_timestamp
        
        # Create ALL directories
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        self.comparison_dir.mkdir(parents=True, exist_ok=True)
        
        # Rest of init
        self.ref_features = self.load_reference_features()
        print(f"📁 Test images: {self.test_dir}")
        print(f"📁 Baseline images: {self.baseline_dir}")
        self.create_test_metadata()
    
    def create_test_metadata(self):
        """Create metadata file for the test session"""
        import time
        
        metadata_path = self.test_dir / "test_info.txt"
        
        with open(metadata_path, 'w') as f:
            f.write(f"Test Session for Training: {self.training_timestamp}\n")
            f.write(f"Test Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"FP16: {self.use_fp16}\n")
            f.write("="*50 + "\n")
            f.write("Test Results:\n")
        
        print(f"📝 Created test metadata: {metadata_path}")
    
    def log_test_result(self, prompt: str, trained_reward: float, baseline_reward: float = None):
        """Log test results to metadata file"""
        metadata_path = self.test_dir / "test_info.txt"
        
        try:
            with open(metadata_path, 'a') as f:
                if baseline_reward is not None:
                    improvement = trained_reward - baseline_reward
                    f.write(f"'{prompt}' | Trained: {trained_reward:.4f} | Baseline: {baseline_reward:.4f} | Improvement: {improvement:+.4f}\n")
                else:
                    f.write(f"'{prompt}' | Trained: {trained_reward:.4f}\n")
        except Exception as e:
            print(f"⚠️ Could not log test result: {e}")
    
    def load_reference_features(self):
        """Load reference features for diversity evaluation"""

        # Use the RL directory to find reference features
        ref_features_path = self.current_dir / "reference_crater_features.npz"
        
        npz_data = np.load(ref_features_path)
        array_keys = list(npz_data.keys())
        
        ref_features_list = []
        for key in array_keys:
            ref_features_list.append(npz_data[key])
        
        ref_features = np.stack(ref_features_list)
        print(f"📊 Loaded reference features: {ref_features.shape}")
        npz_data.close()
        return ref_features

    def load_baseline_model(self):
        """Load baseline (untrained) model for comparison"""
        print("🔄 Loading baseline (pretrained) model...")
        
        sampler = DiffusionSampler(device=self.device, use_fp16=self.use_fp16)
        feature_dim = self.ref_features.shape[1] if self.ref_features is not None else 512
        
        baseline_agent = DiffusionPPOAgent(
            sampler=sampler,
            ref_features=self.ref_features if self.ref_features is not None else np.zeros((10, 512)),
            batch_size=1,
            feature_dim=feature_dim,
            num_inference_steps=20,
            images_per_prompt=1,
            save_samples=False,
            training_start=self.training_timestamp
        )

        print("✅ Baseline model loaded!")
        return baseline_agent
    

    def load_trained_model(self):
        """Load the trained diffusion model for the specific timestamp"""
        print(f"🔄 Loading trained model for timestamp: {self.training_timestamp}")
        
        # Look for model file with timestamp
        models_dir = self.current_dir / "models"
        model_path = models_dir / f"diffusion_ppo_policy_{self.training_timestamp}.pth"

        # Check if file exists first
        if not model_path.exists():
            print(f"❌ Model file not found: {model_path}")
            # Try without timestamp
            fallback_path = models_dir / "diffusion_ppo_policy.pth"
            if fallback_path.exists():
                model_path = fallback_path
                print(f"🔄 Using fallback model: {model_path}")
            else:
                raise FileNotFoundError(f"No model found for timestamp {self.training_timestamp}")
        
        sampler = DiffusionSampler(device=self.device, use_fp16=self.use_fp16)
        feature_dim = self.ref_features.shape[1] if self.ref_features is not None else 512
        
        agent = DiffusionPPOAgent(
            sampler=sampler,
            ref_features=self.ref_features if self.ref_features is not None else np.zeros((10, 512)),
            batch_size=1,
            feature_dim=feature_dim,
            num_inference_steps=20,
            images_per_prompt=1,
            save_samples=False,
            training_start=self.training_timestamp
        )

        if hasattr(agent.actor.unet, 'module'):
            agent.actor.unet.module.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            agent.actor.unet.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"✅ Loaded trained model from: {model_path}")

        return agent

    def generate_single_image(self, agent, prompt: str, save_path: str = None):
        """Generate a single image and calculate its diversity reward"""
        print(f"🎨 Generating: '{prompt}'")
        
        # Generate trajectory
        trajectory, _ = agent.actor.select_trajectory(prompt)
        
        # Convert to PIL image and resize to 64x64 (same as training samples)
        final_image = trajectory.final_image.squeeze(0).cpu()
        final_image = torch.clamp(final_image, 0, 1)
        
        to_pil = transforms.ToPILImage()
        pil_image = to_pil(final_image)
        
        # Resize to 64x64 to match training samples
        pil_image = pil_image.resize((64, 64), Image.Resampling.LANCZOS)
        
        # Calculate diversity reward if reference features available
        diversity_reward = None
        if self.ref_features is not None:
            try:
                from trajectory_recording import extract_features_from_trajectory
                from diversity_reward import calculate_individual_diversity_rewards
                
                features = extract_features_from_trajectory(trajectory, None)
                features = features.reshape(1, -1)
                
                individual_rewards = calculate_individual_diversity_rewards(
                    features, self.ref_features, gamma=None
                )
                diversity_reward = individual_rewards[0]
            except Exception as e:
                print(f"⚠️ Could not calculate diversity reward: {e}")
                diversity_reward = 0.0
        
        # Save image if path provided
        if save_path:
            pil_image.save(save_path, format='JPEG', quality=85, optimize=True)
            print(f"💾 Saved: {save_path}")
        
        return pil_image, diversity_reward
    
    def generate_test_batch(self, agent, prompts: list, prefix: str = "test"):
        """Generate a batch of test images"""
        import time
        
        results = []
        test_timestamp = time.strftime("%H%M%S")  # Time when test was run
        
        print(f"\n🚀 Generating {len(prompts)} images with {prefix} model...")
        
        for i, prompt in enumerate(prompts):
            # Generate image
            image, diversity_reward = self.generate_single_image(agent, prompt)
            
            # Save with descriptive filename
            safe_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_prompt = safe_prompt[:30]  # Limit length
            
            reward_str = f"_r{diversity_reward:.3f}" if diversity_reward is not None else ""
            filename = f"{prefix}_{i+1:02d}_{safe_prompt}{reward_str}_{test_timestamp}.jpg"
            
            # Fix: Proper directory selection
            if prefix == 'trained':
                save_path = self.test_dir / filename
            else:
                save_path = self.baseline_dir / filename

            # Save the image
            image.save(save_path, format='JPEG', quality=85, optimize=True)
            results.append((prompt, image, diversity_reward))
            
            # Log the result
            self.log_test_result(prompt, diversity_reward)
            
            reward_str = f"Diversity: {diversity_reward:.4f}" if diversity_reward is not None else "No reward"
            print(f"  ✅ {i+1}/{len(prompts)}: {reward_str}")
        
        return results
    
        
    def compare_models(self, prompts: list):
        """Compare trained vs baseline models on the same prompts"""
        print("\n🔬 === MODEL COMPARISON ===")
        
        # Load trained model, generate results
        trained_agent = self.load_trained_model()
        trained_results = self.generate_test_batch(trained_agent, prompts, "trained")

        print("🧹 Cleaning up trained model memory...")
        del trained_agent  # Delete the agent
        torch.cuda.empty_cache()  # Clear CUDA cache
        import gc
        gc.collect()  # Python garbage collection

        # Small delay to ensure cleanup
        time.sleep(2)
        
         # Load baseline model, generate results
        baseline_agent = self.load_baseline_model()
        baseline_results = self.generate_test_batch(baseline_agent, prompts, "baseline")

        # Clean up baseline model too
        print("🧹 Cleaning up baseline model memory...")
        del baseline_agent
        torch.cuda.empty_cache()
        gc.collect()
                
        # Create comparison visualizations
        self.create_comparison_plots(trained_results, baseline_results)
        
        # Calculate statistics
        if self.ref_features is not None:
            trained_rewards = [r[2] for r in trained_results if r[2] is not None]
            baseline_rewards = [r[2] for r in baseline_results if r[2] is not None]
            
            comparison_stats = {
                'trained_avg_reward': np.mean(trained_rewards) if trained_rewards else 0,
                'baseline_avg_reward': np.mean(baseline_rewards) if baseline_rewards else 0,
                'trained_std_reward': np.std(trained_rewards) if trained_rewards else 0,
                'baseline_std_reward': np.std(baseline_rewards) if baseline_rewards else 0,
                'improvement': (np.mean(trained_rewards) - np.mean(baseline_rewards)) if trained_rewards and baseline_rewards else 0
            }
            
            print(f"\n📊 === DIVERSITY COMPARISON ===")
            print(f"Trained Model   - Avg Reward: {comparison_stats['trained_avg_reward']:.4f} ± {comparison_stats['trained_std_reward']:.4f}")
            print(f"Baseline Model  - Avg Reward: {comparison_stats['baseline_avg_reward']:.4f} ± {comparison_stats['baseline_std_reward']:.4f}")
            print(f"Improvement: {comparison_stats['improvement']:+.4f}")
            
            # Log comparison summary
            metadata_path = self.test_dir / "test_info.txt"
            with open(metadata_path, 'a') as f:
                f.write("="*50 + "\n")
                f.write("COMPARISON SUMMARY:\n")
                f.write(f"Trained Avg: {comparison_stats['trained_avg_reward']:.4f} ± {comparison_stats['trained_std_reward']:.4f}\n")
                f.write(f"Baseline Avg: {comparison_stats['baseline_avg_reward']:.4f} ± {comparison_stats['baseline_std_reward']:.4f}\n")
                f.write(f"Improvement: {comparison_stats['improvement']:+.4f}\n")
            
            return comparison_stats
        
        return {}
    
    def create_comparison_plots(self, trained_results: list, baseline_results: list):
        """Create side-by-side comparison plots"""
        
        n_images = len(trained_results)
        fig, axes = plt.subplots(2, n_images, figsize=(4*n_images, 8))
        
        if n_images == 1:
            axes = axes.reshape(2, 1)
        
        for i, (trained, baseline) in enumerate(zip(trained_results, baseline_results)):
            # Trained model (top row)
            axes[0, i].imshow(trained[1])
            reward_str = f"R: {trained[2]:.3f}" if trained[2] is not None else "No reward"
            axes[0, i].set_title(f"Trained\n{trained[0][:30]}...\n{reward_str}")
            axes[0, i].axis('off')
            
            # Baseline model (bottom row)
            axes[1, i].imshow(baseline[1])
            reward_str = f"R: {baseline[2]:.3f}" if baseline[2] is not None else "No reward"
            axes[1, i].set_title(f"Baseline\n{baseline[0][:30]}...\n{reward_str}")
            axes[1, i].axis('off')
        
        plt.suptitle(f"Model Comparison - Training: {self.training_timestamp}", fontsize=16)
        plt.tight_layout()
        
        plot_timestamp = time.strftime("%H%M%S")
        comparison_path = self.comparison_dir / f"model_comparison_{plot_timestamp}.png"
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"💾 Comparison plot saved: {comparison_path}")
    
    def load_test_prompts(self, category: str):
        # Load prompts from prompts folder
        test_prompts_file = os.path.join(self.current_dir, "prompts", "test", category)
        
        # Read prompts from file
        test_prompts = []

        with open(test_prompts_file, 'r') as f:
            test_prompts = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(test_prompts)} training prompts from {test_prompts_file}")


def get_latest_timestamp():
        """Get the latest timestamp from model files"""
        models_dir = Path(__file__).parent / "models"
        
        if not models_dir.exists():
            print(f"❌ Models directory not found: {models_dir}")
            return None
        
        # Find all model files with timestamps
        model_files = list(models_dir.glob("diffusion_ppo_policy_*.pth"))
        
        if not model_files:
            print("⚠️ No timestamped model files found")
            # Check for default model
            default_model = models_dir / "diffusion_ppo_policy.pth"
            if default_model.exists():
                print("📝 Found default model, please enter timestamp manually:")
                return input("Enter training timestamp (YYYYMMDDHHMISS): ").strip()
            else:
                print("❌ No model files found!")
                return None
            
        # Extract timestamps and find the latest
        timestamps = []
        for model_file in model_files:
            try:
                # Extract timestamp from filename: diffusion_ppo_policy_TIMESTAMP.pth
                timestamp = model_file.stem.split('_')[-1]
                if len(timestamp) == 14 and timestamp.isdigit():
                    timestamps.append(timestamp)
            except:
                continue
        
        if not timestamps:
            print("⚠️ No valid timestamps found in model files")
            return None
        
        # Return the latest (largest number)
        latest_timestamp = max(timestamps)
        print(f"🕐 Found latest timestamp: {latest_timestamp}")
        return latest_timestamp

    



def main():
    """Test a trained model"""

    timestamp = get_latest_timestamp()

    if timestamp is None:
        print("❌ Cannot proceed without timestamp")
        return

    tester = DiffusionModelTester(device="cuda", use_fp16=False, training_timestamp=timestamp)
    
    category = "crater"
    
    # Test prompts
    test_prompts = tester.load_test_prompts(category)    
    print("🔧 Choose testing mode:")
    print("1. Test trained model only")
    print("2. Compare trained vs baseline models")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        # Full comparison
        stats = tester.compare_models(test_prompts)
    else:
        # Just test trained model
        trained_agent = tester.load_trained_model()
        results = tester.generate_test_batch(trained_agent, test_prompts, "trained")
        
        if tester.ref_features is not None:
            rewards = [r[2] for r in results if r[2] is not None]
            if rewards:
                print(f"\n📊 Average Diversity Reward: {np.mean(rewards):.4f} ± {np.std(rewards):.4f}")
    
    print(f"\n✅ Testing complete! Check {tester.test_dir} for generated images.")

if __name__ == "__main__":
    main()
    

