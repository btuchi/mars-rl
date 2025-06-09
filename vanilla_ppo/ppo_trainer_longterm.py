import gymnasium as gym
import torch
import os.path
import time
import numpy as np
import json
import argparse
from datetime import datetime

from ppo_agent_longterm import LongTermPPOAgent
from log_utils import ACTOR_LOSS_LOG, CRITIC_LOSS_LOG, BEST_REWARD_LOG, REWARD_LOG, VALUE_PREDICTION_LOG, RETURN_LOG

import matplotlib.pyplot as plt

def clear_logs():
    """Clear all logging lists for fresh start"""
    global ACTOR_LOSS_LOG, CRITIC_LOSS_LOG, BEST_REWARD_LOG, REWARD_LOG, VALUE_PREDICTION_LOG, RETURN_LOG
    ACTOR_LOSS_LOG.clear()
    CRITIC_LOSS_LOG.clear()
    BEST_REWARD_LOG.clear()
    REWARD_LOG.clear()
    VALUE_PREDICTION_LOG.clear()
    RETURN_LOG.clear()

def plot_long_term_training(REWARD_BUFFER, training_stats, final_episode=None, start_episode=0):
    """Fixed 7-plot layout for comprehensive PPO training analysis"""
    
    # Determine episode count and range
    num_episodes = final_episode + 1 if final_episode else len(REWARD_BUFFER)
    actual_episodes_trained = num_episodes - start_episode
    episode_range = range(start_episode, num_episodes)
    
    # Handle array size mismatches for resumed training
    reward_data = REWARD_BUFFER[start_episode:num_episodes]
    
    # Handle BEST_REWARD_LOG size mismatch
    if len(BEST_REWARD_LOG) < num_episodes:
        best_reward_data = BEST_REWARD_LOG
        best_reward_episodes = range(len(BEST_REWARD_LOG))
    else:
        best_reward_data = BEST_REWARD_LOG[start_episode:num_episodes]
        best_reward_episodes = episode_range
    
    # Create fixed 3x3 layout (7 plots + 2 empty or filled)
    fig, axs = plt.subplots(3, 3, figsize=(20, 15))
    axs = axs.flatten()  # Convert to 1D array for easy indexing
    
    # Plot 1: Episode Rewards (Top Left)
    axs[0].plot(episode_range, reward_data, alpha=0.3, label="Raw Rewards", color='lightblue')
    
    if actual_episodes_trained > 100:
        smooth_100 = np.convolve(reward_data, np.ones(100)/100, mode='valid')
        smooth_episodes = range(start_episode + 99, num_episodes)
        axs[0].plot(smooth_episodes, smooth_100, linewidth=2, label="MA-100", color='orange')
    
    if actual_episodes_trained > 500:
        smooth_500 = np.convolve(reward_data, np.ones(500)/500, mode='valid')
        smooth_episodes = range(start_episode + 499, num_episodes)
        axs[0].plot(smooth_episodes, smooth_500, linewidth=3, label="MA-500", color='green')
    
    axs[0].set_title("Episode Rewards (Long-term View)", fontsize=12, fontweight='bold')
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Reward")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    
    # Plot 2: Combined Training Losses (Top Middle)
    if len(ACTOR_LOSS_LOG) > 0 and len(CRITIC_LOSS_LOG) > 0:
        loss_episodes = np.linspace(start_episode, num_episodes, len(ACTOR_LOSS_LOG))
        axs[1].semilogy(loss_episodes, ACTOR_LOSS_LOG, label="Actor Loss", linewidth=2, color='blue')
        axs[1].semilogy(loss_episodes, CRITIC_LOSS_LOG, label="Critic Loss", linewidth=2, color='red')
        axs[1].set_title("Training Losses (Log Scale)", fontsize=12, fontweight='bold')
        axs[1].set_xlabel("Episode")
        axs[1].set_ylabel("Loss (log scale)")
        axs[1].legend()
        axs[1].grid(True, alpha=0.3)
    else:
        axs[1].text(0.5, 0.5, "No Loss Data Available", ha='center', va='center', 
                   transform=axs[1].transAxes, fontsize=14)
        axs[1].set_title("Training Losses", fontsize=12, fontweight='bold')
    
    # Plot 3: Best Reward Progress (Top Right)
    if len(best_reward_data) > 0:
        axs[2].plot(best_reward_episodes, best_reward_data, linewidth=3, color='green', marker='o', markersize=2)
        axs[2].set_title("Best Reward Progress", fontsize=12, fontweight='bold')
        axs[2].set_xlabel("Episode")
        axs[2].set_ylabel("Best Reward")
        axs[2].grid(True, alpha=0.3)
    else:
        axs[2].text(0.5, 0.5, "No Best Reward Data", ha='center', va='center', 
                   transform=axs[2].transAxes, fontsize=14)
        axs[2].set_title("Best Reward Progress", fontsize=12, fontweight='bold')
    
    # Plot 4: Evaluation vs Training Performance (Middle Left)
    if 'episodes' in training_stats and 'eval_rewards' in training_stats and len(training_stats['eval_rewards']) > 0:
        eval_episodes = training_stats['episodes']
        eval_rewards = training_stats['eval_rewards']
        training_rewards = training_stats.get('rewards', [])
        
        axs[3].plot(eval_episodes, eval_rewards, 'o-', linewidth=2, label="Evaluation", 
                   markersize=4, color='blue')
        if training_rewards:
            axs[3].plot(eval_episodes, training_rewards, 's-', linewidth=2, 
                       label="Training (avg)", markersize=3, color='orange')
        
        axs[3].set_title("Evaluation vs Training Performance", fontsize=12, fontweight='bold')
        axs[3].set_xlabel("Episode")
        axs[3].set_ylabel("Reward")
        axs[3].legend()
        axs[3].grid(True, alpha=0.3)
    else:
        axs[3].text(0.5, 0.5, "No Evaluation Data", ha='center', va='center', 
                   transform=axs[3].transAxes, fontsize=14)
        axs[3].set_title("Evaluation vs Training Performance", fontsize=12, fontweight='bold')
    
    # Plot 5: Critic Loss Convergence (Middle Center)
    if len(CRITIC_LOSS_LOG) > 10:
        loss_episodes = np.linspace(start_episode, num_episodes, len(CRITIC_LOSS_LOG))
        axs[4].plot(loss_episodes, CRITIC_LOSS_LOG, linewidth=1, alpha=0.7, color='lightcoral')
        
        # Add moving average for critic loss
        if len(CRITIC_LOSS_LOG) > 50:
            critic_smooth = np.convolve(CRITIC_LOSS_LOG, np.ones(50)/50, mode='valid')
            loss_smooth_episodes = np.linspace(start_episode, num_episodes, len(critic_smooth))
            axs[4].plot(loss_smooth_episodes, critic_smooth, linewidth=3, color='red', label='MA-50')
            axs[4].legend()
        
        axs[4].set_title("Critic Loss Convergence", fontsize=12, fontweight='bold')
        axs[4].set_xlabel("Episode")
        axs[4].set_ylabel("Critic Loss")
        axs[4].grid(True, alpha=0.3)
    else:
        axs[4].text(0.5, 0.5, "Insufficient Critic Data", ha='center', va='center', 
                   transform=axs[4].transAxes, fontsize=14)
        axs[4].set_title("Critic Loss Convergence", fontsize=12, fontweight='bold')
    
    # Plot 6: Value Predictions vs Returns (Middle Right)
    if len(VALUE_PREDICTION_LOG) > 100 and len(RETURN_LOG) > 100:
        # Sample data for plotting (to avoid overcrowding)
        sample_size = min(2000, len(VALUE_PREDICTION_LOG))
        indices = np.random.choice(len(VALUE_PREDICTION_LOG), sample_size, replace=False)
        pred_sample = [VALUE_PREDICTION_LOG[i] for i in indices]
        ret_sample = [RETURN_LOG[i] for i in indices]
        
        axs[5].scatter(pred_sample, ret_sample, alpha=0.5, s=2, color='blue')
        
        # Perfect prediction line
        all_values = pred_sample + ret_sample
        min_val, max_val = min(all_values), max(all_values)
        axs[5].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        # Calculate and display correlation
        correlation = np.corrcoef(pred_sample, ret_sample)[0, 1]
        axs[5].set_title(f"Value Predictions vs Returns (r={correlation:.3f})", fontsize=12, fontweight='bold')
        axs[5].set_xlabel("Predicted V(s)")
        axs[5].set_ylabel("Actual Return")
        axs[5].legend()
        axs[5].grid(True, alpha=0.3)
    else:
        axs[5].text(0.5, 0.5, "Insufficient Value Data", ha='center', va='center', 
                   transform=axs[5].transAxes, fontsize=14)
        axs[5].set_title("Value Predictions vs Returns", fontsize=12, fontweight='bold')
    
    # Plot 7: Actor Loss Only (Bottom Left)
    if len(ACTOR_LOSS_LOG) > 0:
        loss_episodes = np.linspace(start_episode, num_episodes, len(ACTOR_LOSS_LOG))
        axs[6].semilogy(loss_episodes, ACTOR_LOSS_LOG, linewidth=2, color='blue', alpha=0.7)
        
        # Add moving average for actor loss
        if len(ACTOR_LOSS_LOG) > 50:
            actor_smooth = np.convolve(ACTOR_LOSS_LOG, np.ones(50)/50, mode='valid')
            loss_smooth_episodes = np.linspace(start_episode, num_episodes, len(actor_smooth))
            axs[6].semilogy(loss_smooth_episodes, actor_smooth, linewidth=3, color='darkblue', label='MA-50')
            axs[6].legend()
        
        axs[6].set_title("Actor Loss Convergence", fontsize=12, fontweight='bold')
        axs[6].set_xlabel("Episode")
        axs[6].set_ylabel("Actor Loss (log scale)")
        axs[6].grid(True, alpha=0.3)
    else:
        axs[6].text(0.5, 0.5, "No Actor Loss Data", ha='center', va='center', 
                   transform=axs[6].transAxes, fontsize=14)
        axs[6].set_title("Actor Loss Convergence", fontsize=12, fontweight='bold')
    
    # Plot 8: Training Summary (Bottom Center)
    axs[7].axis('off')
    
    # Calculate comprehensive statistics
    final_reward = REWARD_BUFFER[final_episode] if final_episode < len(REWARD_BUFFER) else 0
    training_data = REWARD_BUFFER[start_episode:num_episodes]
    avg_final_1000 = np.mean(training_data[-1000:]) if len(training_data) >= 1000 else np.mean(training_data)
    
    available_best_rewards = BEST_REWARD_LOG if len(BEST_REWARD_LOG) > 0 else [0]
    best_reward = max(available_best_rewards)
    
    # Performance improvement
    if len(training_data) > 1000:
        early_avg = np.mean(training_data[:1000])
        improvement = avg_final_1000 - early_avg
    else:
        improvement = 0
    
    # Training efficiency metrics
    total_updates = len(ACTOR_LOSS_LOG)
    episodes_per_update = num_episodes / total_updates if total_updates > 0 else 0
    
    # Loss statistics
    final_actor_loss = ACTOR_LOSS_LOG[-1] if len(ACTOR_LOSS_LOG) > 0 else 0
    final_critic_loss = CRITIC_LOSS_LOG[-1] if len(CRITIC_LOSS_LOG) > 0 else 0
    
    if len(CRITIC_LOSS_LOG) > 100:
        recent_critic_loss = np.mean(CRITIC_LOSS_LOG[-50:])
        early_critic_loss = np.mean(CRITIC_LOSS_LOG[:50]) if len(CRITIC_LOSS_LOG) > 50 else recent_critic_loss
        critic_improvement = ((early_critic_loss - recent_critic_loss) / early_critic_loss) * 100 if early_critic_loss != 0 else 0
    else:
        recent_critic_loss = final_critic_loss
        critic_improvement = 0
    
    # Performance assessment
    convergence_status = 'Excellent' if recent_critic_loss < 0.1 else 'Partial' if recent_critic_loss < 0.5 else 'Poor'
    performance_status = 'Solved' if avg_final_1000 > -150 else 'Good' if avg_final_1000 > -300 else 'Needs Work'
    stability_status = 'Stable' if abs(improvement) < 50 else 'Variable'
    
    summary_text = f"""
        TRAINING SUMMARY
        ================
        Episodes: {num_episodes:,}
        Updates: {total_updates:,}
        Episodes/Update: {episodes_per_update:.1f}

        PERFORMANCE
        ===========
        Final: {final_reward:.1f}
        Best: {best_reward:.1f}
        Avg Last 1000: {avg_final_1000:.1f}
        Improvement: {improvement:+.1f}

        LOSSES
        ======
        Actor: {final_actor_loss:.4f}
        Critic: {final_critic_loss:.4f}
        Recent Critic: {recent_critic_loss:.4f}
        Critic Δ: {critic_improvement:.1f}%

        ASSESSMENT
        ==========
        Convergence: {convergence_status}
        Performance: {performance_status}
        Stability: {stability_status}
    """
    
    axs[7].text(0.05, 0.95, summary_text, transform=axs[7].transAxes, 
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Plot 9: Hide this one (Bottom Right)
    axs[8].axis('off')
    
    # Add a title or additional info in the empty space
    axs[8].text(0.5, 0.5, f"""
        PPO TRAINING ANALYSIS
        ===================
        Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

        Configuration:
        • Environment: Pendulum-v1
        • Episodes: {start_episode:,} → {num_episodes:,}
        • Total Training Time: {actual_episodes_trained:,} episodes

        Status: Training {'Completed' if final_episode else 'In Progress'}
    """, transform=axs[8].transAxes, ha='center', va='center',
               fontsize=10, fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout(pad=2.0)
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ppo_training_analysis_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    print(f"📊 7-plot analysis saved as: {filename}")
    plt.show()
    
    # Print summary to console
    print(f"\n" + "="*60)
    print(f"TRAINING ANALYSIS COMPLETED")
    print(f"="*60)
    print(f"Episodes: {num_episodes:,} | Updates: {total_updates:,}")
    
    if len(training_data) > 0:
        print(f"Final Performance: {avg_final_1000:.1f}")
        print(f"Best Performance: {best_reward:.1f}")
    
    if len(CRITIC_LOSS_LOG) > 50:
        print(f"Recent Critic Loss: {recent_critic_loss:.4f}")
    
    if len(training_data) > 1000:
        print(f"Total Improvement: {improvement:+.1f}")
    
    return filename


def save_training_summary(training_stats, num_episodes, save_path="training_summary.json"):
    """Save comprehensive training summary"""
    summary = {
        'training_completed_at': datetime.now().isoformat(),
        'total_episodes': num_episodes,
        'total_updates': len(ACTOR_LOSS_LOG),
        'final_actor_loss': ACTOR_LOSS_LOG[-1] if ACTOR_LOSS_LOG else None,
        'final_critic_loss': CRITIC_LOSS_LOG[-1] if CRITIC_LOSS_LOG else None,
        'best_reward': max(BEST_REWARD_LOG) if BEST_REWARD_LOG else None,
        'training_stats': training_stats,
        'hyperparameters': {
            'max_episodes': num_episodes,
            'batch_size': 64,
            'episodes_per_update': 10
        }
    }
    
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Training summary saved to {save_path}")

def long_term_training_loop(max_episodes=20000, resume_from=None, save_interval=500):
    """
    Main training loop optimized for long-term training (10k-20k episodes)
    
    Args:
        max_episodes (int): Maximum number of episodes to train
        resume_from (str): Path to checkpoint to resume from
        save_interval (int): Save checkpoint every N episodes
    """
    
    print(f"🚀 STARTING LONG-TERM PPO TRAINING")
    print(f"Target episodes: {max_episodes:,}")
    print(f"Save interval: {save_interval}")
    print(f"Resume from: {resume_from or 'Fresh start'}")
    print("="*60)
    
    # Environment setup
    env = gym.make("Pendulum-v1")
    eval_env = gym.make("Pendulum-v1")  # Separate environment for evaluation
    
    # Training parameters
    STATE_DIM = env.observation_space.shape[0]
    ACTION_DIM = env.action_space.shape[0]
    BATCH_SIZE = 64  # Larger batch size for stability
    EPISODES_PER_UPDATE = 10
    
    # Initialize agent
    agent = LongTermPPOAgent(STATE_DIM, ACTION_DIM, BATCH_SIZE)
    
    # Resume from checkpoint if specified
    start_episode = 0
    if resume_from and os.path.exists(resume_from):
        agent.load_checkpoint(resume_from)
        start_episode = agent.episode_count
        print(f"Resumed from episode {start_episode}")
        
        # Important: Don't clear logs when resuming, just extend them
        print(f"Current log lengths - Actor: {len(ACTOR_LOSS_LOG)}, Critic: {len(CRITIC_LOSS_LOG)}")
    else:
        clear_logs()  # Clear logs for fresh start
    
    # Training metrics tracking
    training_stats = {
        'episodes': [],
        'rewards': [],
        'eval_rewards': [],
        'actor_losses': [],
        'critic_losses': [],
        'learning_rates': [],
        'clip_ranges': []
    }
    
    # Main training variables
    REWARD_BUFFER = np.empty(shape=max_episodes)
    episode_rewards = []
    episodes_since_update = 0
    
    print(f"Training from episode {start_episode} to {max_episodes}")
    training_start_time = time.time()
    
    # Main training loop
    for episode_i in range(start_episode, max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        agent.episode_count = episode_i
        
        # Run episode
        for step_i in range(200):  # Pendulum episode length
            action, value = agent.get_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            done = (step_i + 1) == 200
            
            agent.replay_buffer.add_memo(state, action, reward, value, done, next_state)
            state = next_state
        
        episode_rewards.append(episode_reward)
        episodes_since_update += 1
        REWARD_BUFFER[episode_i] = episode_reward
        BEST_REWARD_LOG.append(max(BEST_REWARD_LOG[-1] if BEST_REWARD_LOG else -2000, episode_reward))
        
        # Update policy
        if episodes_since_update >= EPISODES_PER_UPDATE:
            agent.update()
            episodes_since_update = 0
        
        # Periodic evaluation
        if episode_i % agent.EVAL_INTERVAL == 0 and episode_i > start_episode:
            eval_mean, eval_std = agent.evaluate_policy(eval_env)
            training_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            
            print(f"\nEpisode {episode_i:,} Evaluation:")
            print(f"  Eval Reward: {eval_mean:.2f} ± {eval_std:.2f}")
            print(f"  Training Reward (last 100): {training_reward:.2f}")
            
            # Early stopping check
            if eval_mean > agent.best_eval_reward:
                agent.best_eval_reward = eval_mean
                agent.patience_counter = 0
                print(f"  🎉 New best evaluation reward!")
            else:
                agent.patience_counter += agent.EVAL_INTERVAL
            
            # Store statistics
            training_stats['episodes'].append(episode_i)
            training_stats['eval_rewards'].append(eval_mean)
            training_stats['rewards'].append(training_reward)
            
            if len(ACTOR_LOSS_LOG) > 0:
                training_stats['actor_losses'].append(ACTOR_LOSS_LOG[-1])
                training_stats['critic_losses'].append(CRITIC_LOSS_LOG[-1])
            
            hyperparams = agent.get_current_hyperparams()
            training_stats['learning_rates'].append(hyperparams['actor_lr'])
            training_stats['clip_ranges'].append(hyperparams['clip_range'])
        
        # Checkpointing
        if episode_i % save_interval == 0 and episode_i > start_episode:
            avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            agent.save_checkpoint(episode_i, avg_reward)
            
            # Save training statistics
            with open(f"{agent.checkpoint_dir}/training_stats.json", 'w') as f:
                json.dump(training_stats, f, indent=2)
            
            # Estimate remaining time
            elapsed_time = time.time() - training_start_time
            episodes_done = episode_i - start_episode + 1
            episodes_remaining = max_episodes - episode_i - 1
            estimated_time_remaining = (elapsed_time / episodes_done) * episodes_remaining
            
            print(f"\n📊 CHECKPOINT: Episode {episode_i:,}/{max_episodes:,}")
            print(f"Elapsed: {elapsed_time/3600:.1f}h | Remaining: {estimated_time_remaining/3600:.1f}h")
            print(f"Recent performance: {avg_reward:.1f}")
        
        # # Early stopping
        # if agent.patience_counter >= agent.max_patience:
        #     print(f"\n⏹️  Early stopping at episode {episode_i:,}")
        #     print(f"No improvement for {agent.max_patience} episodes")
        #     break
        
        # Memory management for very long runs
        if episode_i % 2000 == 0 and episode_i > 0:
            if len(ACTOR_LOSS_LOG) > 10000:
                # Keep last 5000 entries to prevent memory issues
                ACTOR_LOSS_LOG[:] = ACTOR_LOSS_LOG[-5000:]
                CRITIC_LOSS_LOG[:] = CRITIC_LOSS_LOG[-5000:]
                VALUE_PREDICTION_LOG[:] = VALUE_PREDICTION_LOG[-10000:]
                RETURN_LOG[:] = RETURN_LOG[-10000:]
                print(f"  🧹 Cleaned up logs to prevent memory issues")
        
        # Progress reporting
        if episode_i % 1000 == 0 and episode_i > start_episode:
            hyperparams = agent.get_current_hyperparams()
            recent_performance = np.mean(episode_rewards[-1000:]) if len(episode_rewards) >= 1000 else np.mean(episode_rewards)
            
            print(f"\n📈 PROGRESS REPORT: Episode {episode_i:,}")
            print(f"Performance (last 1000): {recent_performance:.1f}")
            print(f"Best eval reward: {agent.best_eval_reward:.1f}")
            print(f"Learning rates: Actor {hyperparams['actor_lr']:.6f}, Critic {hyperparams['critic_lr']:.6f}")
            print(f"Clip range: {hyperparams['clip_range']:.3f}")
            print(f"Progress: {hyperparams['progress']:.1%}")
            if len(CRITIC_LOSS_LOG) > 10:
                recent_critic_loss = np.mean(CRITIC_LOSS_LOG[-10:])
                print(f"Recent critic loss: {recent_critic_loss:.4f}")
    
    # Training completed
    final_episode = episode_i
    total_time = time.time() - training_start_time
    
    print(f"\n🏁 TRAINING COMPLETED")
    print(f"Episodes: {final_episode - start_episode + 1:,}")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Best evaluation reward: {agent.best_eval_reward:.2f}")
    
    # Final evaluation
    print(f"\n🧪 FINAL EVALUATION...")
    final_eval_mean, final_eval_std = agent.evaluate_policy(eval_env, num_episodes=20)
    print(f"Final evaluation (20 episodes): {final_eval_mean:.2f} ± {final_eval_std:.2f}")
    
    # Save final checkpoint
    agent.save_checkpoint(final_episode, final_eval_mean)
    
    # Save comprehensive training summary
    save_training_summary(training_stats, final_episode + 1)
    
    # Generate final plots
    print(f"\n📊 Generating training analysis plots...")
    plot_long_term_training(REWARD_BUFFER, training_stats, final_episode, start_episode)
    
    # Clean up
    env.close()
    eval_env.close()
    
    return agent, training_stats, REWARD_BUFFER[:final_episode+1]

def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description='Long-term PPO Training')
    parser.add_argument('--episodes', type=int, default=20000, help='Maximum episodes to train')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint path to resume from')
    parser.add_argument('--save-interval', type=int, default=500, help='Save checkpoint every N episodes')
    parser.add_argument('--eval-only', type=str, default=None, help='Only evaluate a checkpoint')
    
    args = parser.parse_args()
    
    if args.eval_only:
        # Evaluation only mode
        print(f"🧪 EVALUATION MODE")
        print(f"Loading checkpoint: {args.eval_only}")
        
        env = gym.make("Pendulum-v1")
        agent = LongTermPPOAgent(3, 1, 64)  # Pendulum dimensions
        agent.load_checkpoint(args.eval_only)
        
        eval_mean, eval_std = agent.evaluate_policy(env, num_episodes=50)
        print(f"Evaluation result (50 episodes): {eval_mean:.2f} ± {eval_std:.2f}")
        
        env.close()
    else:
        # Training mode
        agent, stats, rewards = long_term_training_loop(
            max_episodes=args.episodes,
            resume_from=args.resume,
            save_interval=args.save_interval
        )
        
        print(f"\n✅ Training pipeline completed successfully!")
        print(f"Final model saved in: {agent.checkpoint_dir}")

if __name__ == "__main__":
    main()