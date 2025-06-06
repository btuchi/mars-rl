import torch
import numpy as np
import time
import os.path
from tqdm import tqdm
import matplotlib.pyplot as plt

from trajectory_recording import DiffusionSampler
from diffusion_ppo_agent import DiffusionPPOAgent
from log_utils import ACTOR_LOSS_LOG, CRITIC_LOSS_LOG, BEST_REWARD_LOG, REWARD_LOG, VALUE_PREDICTION_LOG, RETURN_LOG

# Diffusion PPO Training Parameters (equivalent to vanilla PPO structure)
NUM_EPISODE = 1000              # Total number of "episodes" (trajectory generations)
NUM_STEP = 1                    # Each "episode" generates 1 trajectory (unlike Pendulum's 200 steps)
BATCH_SIZE = 16                 # Mini-batch size for PPO updates
TRAJECTORY_LENGTH = 32          # Collect this many trajectories before updating
EPISODES_PER_UPDATE = 32        # Same as TRAJECTORY_LENGTH for diffusion

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Computing device: {device}")

# Directory setup
current_path = os.path.dirname(os.path.realpath(__file__))
model_dir = os.path.join(current_path, "models")
os.makedirs(model_dir, exist_ok=True)
timestamp = time.strftime("%Y%m%d%H%M%S")

def main():
    """Main training loop - mirrors vanilla PPO trainer structure exactly"""
    
    print("=== DIFFUSION PPO TRAINING ===")
    
    # Load reference features (equivalent to environment setup)
    try:
        ref_features = np.load("reference_crater_features.npy")
        print(f"Loaded reference features: {ref_features.shape}")
    except FileNotFoundError:
        print("Error: reference_crater_features.npy not found!")
        print("Please ensure you have pre-computed reference features.")
        return
    
    # Initialize diffusion sampler (equivalent to gym.make())
    print("Initializing diffusion sampler...")
    sampler = DiffusionSampler(device=device)
    
    # Initialize PPO agent (same as vanilla PPO)
    feature_dim = ref_features.shape[1] if len(ref_features.shape) > 1 else 512
    agent = DiffusionPPOAgent(
        sampler=sampler,
        ref_features=ref_features,
        batch_size=BATCH_SIZE,
        feature_dim=feature_dim,
        num_inference_steps=15  # Reduced for faster training
    )
    
    # Define crater prompts (equivalent to different environment scenarios)
    crater_prompts = [
        "a photo of a mars crater",
        "a detailed mars crater with shadows", 
        "a large mars crater on red terrain",
        "a small mars crater with rocks",
        "an ancient mars crater with erosion",
        "a fresh mars crater with sharp edges",
        "a deep mars crater with visible layers",
        "a mars crater with debris inside",
        "a circular mars crater on rocky surface",
        "a weathered mars crater with smooth edges"
    ]
    
    # Training tracking (same as vanilla PPO)
    REWARD_BUFFER = np.empty(shape=NUM_EPISODE)
    best_reward = -float('inf')
    
    # Trajectory collection tracking (same logic as vanilla PPO)
    trajectories_collected = 0
    episodes_since_update = 0
    recent_rewards = []
    
    print("Starting Diffusion PPO training...")
    print(f"Target trajectory length: {TRAJECTORY_LENGTH}")
    print(f"Episodes per update: {EPISODES_PER_UPDATE}")
    print(f"Crater prompts: {len(crater_prompts)}")
    
    # Main training loop (mirrors vanilla PPO structure exactly)
    for episode_i in range(NUM_EPISODE):
        
        # "Reset environment" - sample a random prompt
        prompt = np.random.choice(crater_prompts)
        episode_reward = 0
        
        # "Run episode" - for diffusion, this is just 1 step (generate 1 trajectory)
        for step_i in range(NUM_STEP):
            
            # Get "action" (generate trajectory) and value
            trajectory, log_prob, value, prompt_features = agent.get_action(prompt)
            
            # Calculate "reward" from "environment" (diversity function)
            reward = agent.reward_function.calculate_reward(trajectory)
            episode_reward += reward
            
            # "Step" is always "done" for diffusion (complete generation)
            done = True
            
            # Store experience (same as vanilla PPO)
            agent.replay_buffer.add_memo(prompt_features, trajectory, reward, value, log_prob)
            trajectories_collected += 1
        
        # Track episode (same as vanilla PPO)
        recent_rewards.append(episode_reward)
        episodes_since_update += 1
        REWARD_BUFFER[episode_i] = episode_reward
        
        # Update policy when we have enough data (same logic as vanilla PPO)
        if episodes_since_update >= EPISODES_PER_UPDATE or trajectories_collected >= TRAJECTORY_LENGTH:
            print(f"\nUpdating after {episodes_since_update} episodes ({trajectories_collected} trajectories)")
            
            # Perform PPO update
            agent.update()
            
            # Reset counters
            trajectories_collected = 0
            episodes_since_update = 0
            
            # Print progress (same format as vanilla PPO)
            if len(ACTOR_LOSS_LOG) > 0:
                avg_recent_reward = np.mean(recent_rewards[-20:]) if len(recent_rewards) >= 20 else np.mean(recent_rewards)
                print(f"Ep {episode_i}: Actor Loss: {ACTOR_LOSS_LOG[-1]:.4f}, "
                      f"Critic Loss: {CRITIC_LOSS_LOG[-1]:.4f}, "
                      f"Avg Reward(20): {avg_recent_reward:.4f}")
        
        # Save best model (same as vanilla PPO)
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save_policy()
            print(f"Episode {episode_i}: New best reward: {best_reward:.4f}")
        
        BEST_REWARD_LOG.append(best_reward)
        
        # Progress logging (same as vanilla PPO)
        if episode_i % 50 == 0:
            avg_reward = np.mean(recent_rewards[-50:]) if len(recent_rewards) >= 50 else np.mean(recent_rewards)
            print(f"Episode {episode_i}: Current: {episode_reward:.4f}, Avg(50): {avg_reward:.4f}, Best: {best_reward:.4f}")
        
        # Early stopping if solved (adapted for diversity rewards)
        if len(recent_rewards) >= 100 and np.mean(recent_rewards[-100:]) > 0.5:  # Adjust threshold as needed
            print(f"High diversity achieved at episode {episode_i}!")
            break
    
    # Training completed
    final_episode = episode_i
    print(f"\nTraining completed after {final_episode + 1} episodes")
    
    # Plot training results (same function as vanilla PPO)
    plot_diffusion_training(REWARD_BUFFER, ACTOR_LOSS_LOG, CRITIC_LOSS_LOG, BEST_REWARD_LOG, final_episode)

def plot_diffusion_training(REWARD_BUFFER, ACTOR_LOSS_LOG, CRITIC_LOSS_LOG, BEST_REWARD_LOG, final_episode=None):
    """
    Plotting function adapted from vanilla PPO but for diffusion metrics
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Determine the actual number of episodes run
    if final_episode is not None:
        num_episodes = final_episode + 1
    else:
        num_episodes = len(REWARD_BUFFER)
    
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    
    # Episode Rewards (Diversity Scores)
    episode_range = range(num_episodes)
    axs[0, 0].plot(episode_range, REWARD_BUFFER[:num_episodes], label="Diversity Reward", alpha=0.6)
    if num_episodes > 20:
        smoothed = np.convolve(REWARD_BUFFER[:num_episodes], np.ones(20)/20, mode='valid')
        smoothed_range = range(19, num_episodes)
        axs[0, 0].plot(smoothed_range, smoothed, label="Smoothed (MA-20)", linewidth=2)
    axs[0, 0].set_title("Diversity Reward per Episode")
    axs[0, 0].set_xlabel("Episode")
    axs[0, 0].set_ylabel("Diversity Reward")
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)

    # Loss Curves
    if len(ACTOR_LOSS_LOG) > 0:
        loss_x_points = np.linspace(0, num_episodes, len(ACTOR_LOSS_LOG))
        
        axs[0, 1].plot(loss_x_points, ACTOR_LOSS_LOG, label="UNet Loss", linewidth=2, marker='o', markersize=3)
        axs[0, 1].plot(loss_x_points, CRITIC_LOSS_LOG, label="Value Loss", linewidth=2, marker='s', markersize=3)
        axs[0, 1].set_title(f"Loss per Update ({len(ACTOR_LOSS_LOG)} updates)")
        axs[0, 1].set_xlabel("Episode")
        axs[0, 1].set_ylabel("Loss")
        axs[0, 1].legend()
        axs[0, 1].grid(True, alpha=0.3)
        
        if len(ACTOR_LOSS_LOG) > 1:
            avg_episodes_per_update = num_episodes / len(ACTOR_LOSS_LOG)
            axs[0, 1].text(0.02, 0.98, f"~{avg_episodes_per_update:.1f} episodes/update", 
                          transform=axs[0, 1].transAxes, fontsize=9, 
                          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        axs[0, 1].text(0.5, 0.5, "No Loss Data\n(No updates performed yet)", 
                      ha='center', va='center', transform=axs[0, 1].transAxes, fontsize=12)
        axs[0, 1].set_title("Loss per Update")

    # Best Reward Progress
    axs[0, 2].plot(episode_range, BEST_REWARD_LOG[:num_episodes], label="Best Diversity Reward", linewidth=2)
    axs[0, 2].set_title("Best Diversity Reward Progress")
    axs[0, 2].set_xlabel("Episode")
    axs[0, 2].set_ylabel("Best Diversity Reward")
    axs[0, 2].legend()
    axs[0, 2].grid(True, alpha=0.3)

    # Value Predictions vs Returns
    if len(VALUE_PREDICTION_LOG) > 0 and len(RETURN_LOG) > 0:
        sample_size = min(500, len(VALUE_PREDICTION_LOG))
        if len(VALUE_PREDICTION_LOG) > sample_size:
            indices = np.random.choice(len(VALUE_PREDICTION_LOG), sample_size, replace=False)
            pred_sample = [VALUE_PREDICTION_LOG[i] for i in indices]
            ret_sample = [RETURN_LOG[i] for i in indices]
        else:
            pred_sample = VALUE_PREDICTION_LOG
            ret_sample = RETURN_LOG
        
        axs[1, 0].scatter(pred_sample, ret_sample, alpha=0.5, s=10)
        
        all_values = pred_sample + ret_sample
        min_val, max_val = min(all_values), max(all_values)
        axs[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        axs[1, 0].set_title("Value Predictions vs Returns")
        axs[1, 0].set_xlabel("Predicted V(prompt)")
        axs[1, 0].set_ylabel("Actual Return")
        axs[1, 0].legend()
        axs[1, 0].grid(True, alpha=0.3)
        
        # Calculate and display correlation
        correlation = np.corrcoef(pred_sample, ret_sample)[0, 1]
        axs[1, 0].text(0.02, 0.98, f"Correlation: {correlation:.3f}", 
                      transform=axs[1, 0].transAxes, fontsize=9,
                      verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        axs[1, 0].text(0.5, 0.5, "No Value Prediction Data\n(No updates performed yet)", 
                      ha='center', va='center', transform=axs[1, 0].transAxes, fontsize=12)
        axs[1, 0].set_title("Value Predictions vs Returns")

    # Diversity Reward Distribution
    if num_episodes > 10:
        axs[1, 1].hist(REWARD_BUFFER[:num_episodes], bins=min(30, num_episodes//2), 
                      alpha=0.7, edgecolor='black', density=True)
        mean_reward = np.mean(REWARD_BUFFER[:num_episodes])
        axs[1, 1].axvline(mean_reward, color='red', linestyle='--', linewidth=2, 
                         label=f'Mean: {mean_reward:.3f}')
        axs[1, 1].set_title("Diversity Reward Distribution")
        axs[1, 1].set_xlabel("Diversity Reward")
        axs[1, 1].set_ylabel("Density")
        axs[1, 1].legend()
        axs[1, 1].grid(True, alpha=0.3)
    else:
        axs[1, 1].text(0.5, 0.5, "Not enough data\nfor distribution", 
                      ha='center', va='center', transform=axs[1, 1].transAxes, fontsize=12)
        axs[1, 1].set_title("Diversity Reward Distribution")

    # Learning Progress (rolling average)
    if num_episodes > 10:
        window = min(50, num_episodes // 4)
        rolling_avg = []
        for i in range(num_episodes):
            start_idx = max(0, i - window + 1)
            rolling_avg.append(np.mean(REWARD_BUFFER[start_idx:i+1]))
        
        axs[1, 2].plot(episode_range, rolling_avg, linewidth=2, label=f'MA-{window}')
        axs[1, 2].set_title(f"Diversity Learning Progress")
        axs[1, 2].set_xlabel("Episode")
        axs[1, 2].set_ylabel("Average Diversity Reward")
        axs[1, 2].legend()
        axs[1, 2].grid(True, alpha=0.3)
        
        # Show improvement
        if len(rolling_avg) > window:
            early_avg = np.mean(rolling_avg[:window])
            recent_avg = np.mean(rolling_avg[-window:])
            improvement = recent_avg - early_avg
            axs[1, 2].text(0.02, 0.98, f"Improvement: {improvement:+.3f}", 
                          transform=axs[1, 2].transAxes, fontsize=9,
                          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        axs[1, 2].text(0.5, 0.5, "Not enough data\nfor moving average", 
                      ha='center', va='center', transform=axs[1, 2].transAxes, fontsize=12)
        axs[1, 2].set_title("Diversity Learning Progress")

    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(model_dir, f"diffusion_ppo_training_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics (adapted for diffusion)
    print(f"\n=== DIFFUSION PPO TRAINING SUMMARY ===")
    print(f"Episodes completed: {num_episodes}")
    print(f"Policy updates: {len(ACTOR_LOSS_LOG)}")
    if num_episodes > 0:
        print(f"Final diversity reward: {REWARD_BUFFER[num_episodes-1]:.4f}")
        print(f"Best diversity reward: {max(BEST_REWARD_LOG[:num_episodes]):.4f}")
        print(f"Average diversity reward: {np.mean(REWARD_BUFFER[:num_episodes]):.4f}")
    if len(ACTOR_LOSS_LOG) > 0:
        print(f"Final UNet loss: {ACTOR_LOSS_LOG[-1]:.6f}")
        print(f"Final value loss: {CRITIC_LOSS_LOG[-1]:.6f}")

def test_trained_model(num_test_images: int = 5):
    """
    Test the trained diffusion model
    Generate diverse crater images and evaluate them
    """
    print(f"\n=== TESTING TRAINED DIFFUSION MODEL ===")
    
    # Load reference features
    try:
        ref_features = np.load("reference_crater_features.npy")
        print(f"Loaded reference features for testing: {ref_features.shape}")
    except FileNotFoundError:
        print("Error: Cannot test without reference features!")
        return
    
    # Initialize components
    sampler = DiffusionSampler(device=device)
    feature_dim = ref_features.shape[1] if len(ref_features.shape) > 1 else 512
    
    agent = DiffusionPPOAgent(
        sampler=sampler,
        ref_features=ref_features,
        batch_size=1,  # Single trajectory for testing
        feature_dim=feature_dim,
        num_inference_steps=20  # More steps for higher quality
    )
    
    # Load trained policy
    try:
        agent.actor.unet.load_state_dict(torch.load("diffusion_ppo_policy.pth"))
        print("Loaded trained policy successfully!")
    except FileNotFoundError:
        print("Warning: No trained policy found. Using pre-trained Stable Diffusion.")
    
    # Test prompts
    test_prompts = [
        "a photo of a mars crater",
        "a large ancient mars crater with erosion",
        "a small fresh mars crater with sharp edges",
        "a deep mars crater with visible rock layers",
        "a crater on mars with interesting geological features"
    ]
    
    print(f"Generating {num_test_images} test images...")
    test_rewards = []
    
    for i in range(num_test_images):
        prompt = test_prompts[i % len(test_prompts)]
        print(f"\nTest {i+1}: '{prompt}'")
        
        # Generate image
        trajectory, log_prob, value, prompt_features = agent.get_action(prompt)
        
        # Calculate diversity reward
        reward = agent.reward_function.calculate_reward(trajectory)
        test_rewards.append(reward)
        
        print(f"  Diversity reward: {reward:.4f}")
        print(f"  Value prediction: {value:.4f}")
        print(f"  Log probability: {log_prob:.4f}")
        
        # Save test image
        try:
            from PIL import Image
            import torchvision.transforms as transforms
            
            final_image = trajectory.final_image.squeeze(0).cpu()
            final_image = torch.clamp(final_image, 0, 1)
            
            to_pil = transforms.ToPILImage()
            pil_image = to_pil(final_image)
            
            # Create test images directory
            test_dir = os.path.join(current_path, "test_images")
            os.makedirs(test_dir, exist_ok=True)
            
            image_path = os.path.join(test_dir, f"diffusion_ppo_test_{i+1}_{timestamp}.png")
            pil_image.save(image_path)
            print(f"  Test image saved: {image_path}")
            
        except Exception as e:
            print(f"  Could not save image: {e}")
    
    # Test summary
    print(f"\n=== TEST RESULTS SUMMARY ===")
    print(f"Average test diversity reward: {np.mean(test_rewards):.4f}")
    print(f"Test reward std: {np.std(test_rewards):.4f}")
    print(f"Best test reward: {max(test_rewards):.4f}")
    print(f"Test images saved to: test_images/")

if __name__ == "__main__":
    # Run training
    main()
    
    # Optionally run testing
    print("\nWould you like to test the trained model? (y/n)")
    user_input = input().lower().strip()
    if user_input in ['y', 'yes']:
        test_trained_model(num_test_images=3)
    
    print("\nDiffusion PPO training and testing completed!")