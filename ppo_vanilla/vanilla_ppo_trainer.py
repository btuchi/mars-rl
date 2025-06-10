import gymnasium as gym
import torch
import os.path
import time
import numpy as np

from vanilla_ppo_agent import PPOAgent
from log_utils import ACTOR_LOSS_LOG, CRITIC_LOSS_LOG, BEST_REWARD_LOG, REWARD_LOG, VALUE_PREDICTION_LOG, RETURN_LOG

import matplotlib.pyplot as plt

scenario = "Pendulum-v1"
env = gym.make(scenario)

# Training parameters
NUM_EPISODE = 3000
NUM_STEP = 200
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]
BATCH_SIZE = 32
TRAJECTORY_LENGTH = 2048  # Collect this many steps before updating
EPISODES_PER_UPDATE = 10  # Approximately TRAJECTORY_LENGTH / NUM_STEP

REWARD_BUFFER = np.empty(shape=NUM_EPISODE)
agent = PPOAgent(STATE_DIM, ACTION_DIM, BATCH_SIZE)
best_reward = -2000

# Tracking
steps_collected = 0
episodes_since_update = 0
recent_rewards = []

print("Starting PPO training...")
print(f"Target trajectory length: {TRAJECTORY_LENGTH}")
print(f"Episodes per update: {EPISODES_PER_UPDATE}")

for episode_i in range(NUM_EPISODE):
    state, _ = env.reset()
    episode_reward = 0
    
    # Run episode
    for step_i in range(NUM_STEP):
        action, value = agent.get_action(state)
        next_state, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        
        # For Pendulum, episodes are fixed length
        done = (step_i + 1) == NUM_STEP
        
        # Store experience
        agent.replay_buffer.add_memo(state, action, reward, value, done, next_state)
        state = next_state
        steps_collected += 1
    
    # Track episode
    recent_rewards.append(episode_reward)
    episodes_since_update += 1
    REWARD_BUFFER[episode_i] = episode_reward
    
    # Update policy when we have enough data
    if episodes_since_update >= EPISODES_PER_UPDATE or steps_collected >= TRAJECTORY_LENGTH:
        print(f"\nUpdating after {episodes_since_update} episodes ({steps_collected} steps)")
        
        # Perform update
        agent.update()
        
        # Reset counters
        steps_collected = 0
        episodes_since_update = 0
        
        # Print progress
        if len(ACTOR_LOSS_LOG) > 0:
            avg_recent_reward = np.mean(recent_rewards[-20:]) if len(recent_rewards) >= 20 else np.mean(recent_rewards)
            print(f"Ep {episode_i}: Actor Loss: {ACTOR_LOSS_LOG[-1]:.4f}, "
                  f"Critic Loss: {CRITIC_LOSS_LOG[-1]:.4f}, "
                  f"Avg Reward(20): {avg_recent_reward:.2f}")
    
    # Save best model
    if episode_reward > best_reward:
        best_reward = episode_reward
        agent.save_policy()
        print(f"Episode {episode_i}: New best reward: {best_reward:.2f}")
    
    BEST_REWARD_LOG.append(best_reward)
    
    # Progress logging
    if episode_i % 100 == 0:
        avg_reward = np.mean(recent_rewards[-100:]) if len(recent_rewards) >= 100 else np.mean(recent_rewards)
        print(f"Episode {episode_i}: Current: {episode_reward:.2f}, Avg(100): {avg_reward:.2f}, Best: {best_reward:.2f}")
    
    # Early stopping if solved
    if len(recent_rewards) >= 100 and np.mean(recent_rewards[-100:]) > -150:
        print(f"Environment solved at episode {episode_i}!")
        break


def plot_training(REWARD_BUFFER, ACTOR_LOSS_LOG, CRITIC_LOSS_LOG, BEST_REWARD_LOG, final_episode=None):
    """
    Enhanced plotting function that handles mismatched array lengths
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Determine the actual number of episodes run
    if final_episode is not None:
        num_episodes = final_episode + 1
    else:
        num_episodes = len(REWARD_BUFFER)
    
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    
    # Episode Rewards
    episode_range = range(num_episodes)
    axs[0, 0].plot(episode_range, REWARD_BUFFER[:num_episodes], label="Episode Reward", alpha=0.6)
    if num_episodes > 50:
        smoothed = np.convolve(REWARD_BUFFER[:num_episodes], np.ones(50)/50, mode='valid')
        smoothed_range = range(49, num_episodes)
        axs[0, 0].plot(smoothed_range, smoothed, label="Smoothed (MA-50)", linewidth=2)
    axs[0, 0].set_title("Reward per Episode")
    axs[0, 0].set_xlabel("Episode")
    axs[0, 0].set_ylabel("Reward")
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)

    # Loss Curves
    if len(ACTOR_LOSS_LOG) > 0:
        # Create x-axis points that match the actual number of loss entries
        loss_x_points = np.linspace(0, num_episodes, len(ACTOR_LOSS_LOG))
        
        axs[0, 1].plot(loss_x_points, ACTOR_LOSS_LOG, label="Actor Loss", linewidth=2, marker='o', markersize=3)
        axs[0, 1].plot(loss_x_points, CRITIC_LOSS_LOG, label="Critic Loss", linewidth=2, marker='s', markersize=3)
        axs[0, 1].set_title(f"Loss per Update ({len(ACTOR_LOSS_LOG)} updates)")
        axs[0, 1].set_xlabel("Episode")
        axs[0, 1].set_ylabel("Loss")
        axs[0, 1].legend()
        axs[0, 1].grid(True, alpha=0.3)
        
        # Add text showing update frequency
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
    axs[0, 2].plot(episode_range, BEST_REWARD_LOG[:num_episodes], label="Best Reward", linewidth=2)
    axs[0, 2].set_title("Best Reward Progress")
    axs[0, 2].set_xlabel("Episode")
    axs[0, 2].set_ylabel("Best Reward")
    axs[0, 2].legend()
    axs[0, 2].grid(True, alpha=0.3)

    # Value Predictions vs Returns
    if len(VALUE_PREDICTION_LOG) > 0 and len(RETURN_LOG) > 0:
        # Sample data if we have too many points
        sample_size = min(1000, len(VALUE_PREDICTION_LOG))
        if len(VALUE_PREDICTION_LOG) > sample_size:
            indices = np.random.choice(len(VALUE_PREDICTION_LOG), sample_size, replace=False)
            pred_sample = [VALUE_PREDICTION_LOG[i] for i in indices]
            ret_sample = [RETURN_LOG[i] for i in indices]
        else:
            pred_sample = VALUE_PREDICTION_LOG
            ret_sample = RETURN_LOG
        
        axs[1, 0].scatter(pred_sample, ret_sample, alpha=0.5, s=10)
        
        # Perfect prediction line
        all_values = pred_sample + ret_sample
        min_val, max_val = min(all_values), max(all_values)
        axs[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        axs[1, 0].set_title("Value Predictions vs Returns")
        axs[1, 0].set_xlabel("Predicted V(s)")
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

    # Reward distribution
    if num_episodes > 10:
        axs[1, 1].hist(REWARD_BUFFER[:num_episodes], bins=min(50, num_episodes//2), 
                      alpha=0.7, edgecolor='black', density=True)
        mean_reward = np.mean(REWARD_BUFFER[:num_episodes])
        axs[1, 1].axvline(mean_reward, color='red', linestyle='--', linewidth=2, 
                         label=f'Mean: {mean_reward:.1f}')
        axs[1, 1].set_title("Reward Distribution")
        axs[1, 1].set_xlabel("Reward")
        axs[1, 1].set_ylabel("Density")
        axs[1, 1].legend()
        axs[1, 1].grid(True, alpha=0.3)
    else:
        axs[1, 1].text(0.5, 0.5, "Not enough data\nfor distribution", 
                      ha='center', va='center', transform=axs[1, 1].transAxes, fontsize=12)
        axs[1, 1].set_title("Reward Distribution")

    # Learning progress (rolling average)
    if num_episodes > 20:
        window = min(100, num_episodes // 4)  # Adaptive window size
        rolling_avg = []
        for i in range(num_episodes):
            start_idx = max(0, i - window + 1)
            rolling_avg.append(np.mean(REWARD_BUFFER[start_idx:i+1]))
        
        axs[1, 2].plot(episode_range, rolling_avg, linewidth=2, label=f'MA-{window}')
        axs[1, 2].set_title(f"Learning Progress (Moving Average)")
        axs[1, 2].set_xlabel("Episode")
        axs[1, 2].set_ylabel("Average Reward")
        axs[1, 2].legend()
        axs[1, 2].grid(True, alpha=0.3)
        
        # Show improvement
        if len(rolling_avg) > window:
            early_avg = np.mean(rolling_avg[:window])
            recent_avg = np.mean(rolling_avg[-window:])
            improvement = recent_avg - early_avg
            axs[1, 2].text(0.02, 0.98, f"Improvement: {improvement:+.1f}", 
                          transform=axs[1, 2].transAxes, fontsize=9,
                          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        axs[1, 2].text(0.5, 0.5, "Not enough data\nfor moving average", 
                      ha='center', va='center', transform=axs[1, 2].transAxes, fontsize=12)
        axs[1, 2].set_title("Learning Progress")

    plt.tight_layout()
    plt.savefig("analysis_plots/ppo_training_enhanced.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print(f"\n=== TRAINING SUMMARY ===")
    print(f"Episodes completed: {num_episodes}")
    print(f"Policy updates: {len(ACTOR_LOSS_LOG)}")
    if num_episodes > 0:
        print(f"Final reward: {REWARD_BUFFER[num_episodes-1]:.2f}")
        print(f"Best reward: {max(BEST_REWARD_LOG[:num_episodes]):.2f}")
        print(f"Average reward: {np.mean(REWARD_BUFFER[:num_episodes]):.2f}")
    if len(ACTOR_LOSS_LOG) > 0:
        print(f"Final actor loss: {ACTOR_LOSS_LOG[-1]:.4f}")
        print(f"Final critic loss: {CRITIC_LOSS_LOG[-1]:.4f}")

print(f"\nTraining completed after {episode_i+1} episodes")
plot_training(REWARD_BUFFER, ACTOR_LOSS_LOG, CRITIC_LOSS_LOG, BEST_REWARD_LOG)
env.close()