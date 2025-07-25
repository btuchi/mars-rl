"""Plotting and visualization utilities (extracted from trainer.py)"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from .constants import DEFAULT_CATEGORY, DEFAULT_TRAINING_MODE

def plot_diffusion_training(reward_buffer, actor_loss_log, critic_loss_log, best_reward_log, 
                          value_prediction_log, return_log, log_prob_log=None, final_episode=None, 
                          category=DEFAULT_CATEGORY, timestamp=None, training_mode=None):
    """Plotting function adapted from vanilla PPO but for diffusion metrics"""
    
    # Determine the actual number of episodes run
    if final_episode is not None:
        num_episodes = final_episode + 1
    else:
        num_episodes = len(reward_buffer)
    
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    
    # Episode Rewards (Diversity Scores)
    episode_range = range(num_episodes)
    axs[0, 0].plot(episode_range, reward_buffer[:num_episodes], label="Diversity Reward", alpha=0.6)
    if num_episodes > 20:
        smoothed = np.convolve(reward_buffer[:num_episodes], np.ones(20)/20, mode='valid')
        smoothed_range = range(19, num_episodes)
        axs[0, 0].plot(smoothed_range, smoothed, label="Smoothed (MA-20)", linewidth=2)
    axs[0, 0].set_title("Diversity Reward per Episode")
    axs[0, 0].set_xlabel("Episode")
    axs[0, 0].set_ylabel("Diversity Reward")
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)

    # Loss Curves
    if len(actor_loss_log) > 0:
        loss_x_points = np.linspace(0, num_episodes, len(actor_loss_log))
        
        # Mode-aware actor loss label
        actor_label = "LoRA Loss" if training_mode == "LORA_UNET" else "Diversity Policy Loss"
        if training_mode is None:
            actor_label = "Actor Loss"  # Fallback for backward compatibility
            
        axs[0, 1].plot(loss_x_points, actor_loss_log, label=actor_label, linewidth=2, marker='o', markersize=3)
        axs[0, 1].plot(loss_x_points, critic_loss_log, label="Value Loss", linewidth=2, marker='s', markersize=3)
        # Mode-aware title
        mode_suffix = f" ({training_mode})" if training_mode else ""
        axs[0, 1].set_title(f"Loss per Update ({len(actor_loss_log)} updates){mode_suffix}")
        axs[0, 1].set_xlabel("Episode")
        axs[0, 1].set_ylabel("Loss")
        axs[0, 1].legend()
        axs[0, 1].grid(True, alpha=0.3)
        
        if len(actor_loss_log) > 1:
            avg_episodes_per_update = num_episodes / len(actor_loss_log)
            axs[0, 1].text(0.02, 0.98, f"~{avg_episodes_per_update:.1f} episodes/update", 
                          transform=axs[0, 1].transAxes, fontsize=9, 
                          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        axs[0, 1].text(0.5, 0.5, "No Loss Data\n(No updates performed yet)", 
                      ha='center', va='center', transform=axs[0, 1].transAxes, fontsize=12)
        mode_suffix = f" ({training_mode})" if training_mode else ""
        axs[0, 1].set_title(f"Loss per Update{mode_suffix}")

    # Log Probability Progress (if available)
    if log_prob_log and len(log_prob_log) > 0:
        # Show log probability trend over time
        log_prob_episodes = np.linspace(0, num_episodes, len(log_prob_log))
        # Mode-aware log probability label
        log_prob_label = "Denoising Log Prob" if training_mode == "LORA_UNET" else "Policy Log Prob"
        if training_mode is None:
            log_prob_label = "Log Probability"  # Fallback
            
        axs[0, 2].plot(log_prob_episodes, log_prob_log, label=log_prob_label, alpha=0.6)
        
        # Add smoothed version if enough data
        if len(log_prob_log) > 20:
            window = 20
            smoothed_log_prob = np.convolve(log_prob_log, np.ones(window)/window, mode='valid')
            smoothed_episodes = np.linspace(window-1, num_episodes, len(smoothed_log_prob))
            axs[0, 2].plot(smoothed_episodes, smoothed_log_prob, label="Smoothed Log Prob", linewidth=2)
        
        # Mode-aware log probability title
        log_prob_title = "Denoising Log Probability" if training_mode == "LORA_UNET" else "Policy Log Probability"
        if training_mode is None:
            log_prob_title = "Log Probability Progress"  # Fallback
            
        axs[0, 2].set_title(log_prob_title)
        axs[0, 2].set_xlabel("Episode")
        axs[0, 2].set_ylabel("Log Probability")
        axs[0, 2].legend()
        axs[0, 2].grid(True, alpha=0.3)
        
        # Add statistics
        mean_log_prob = np.mean(log_prob_log)
        std_log_prob = np.std(log_prob_log)
        axs[0, 2].text(0.02, 0.98, f"Mean: {mean_log_prob:.3f}\nStd: {std_log_prob:.3f}", 
                      transform=axs[0, 2].transAxes, fontsize=9,
                      verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        # Fallback to best reward progress
        axs[0, 2].plot(episode_range, best_reward_log[:num_episodes], label="Best Diversity Reward", linewidth=2)
        # Mode-aware fallback title  
        fallback_title = "Best Reward Progress" if training_mode == "LORA_UNET" else "Best Diversity Reward Progress"
        if training_mode is None:
            fallback_title = "Best Diversity Reward Progress"  # Fallback
            
        axs[0, 2].set_title(fallback_title)
        axs[0, 2].set_xlabel("Episode")
        axs[0, 2].set_ylabel("Best Diversity Reward")
        axs[0, 2].legend()
        axs[0, 2].grid(True, alpha=0.3)

    # Value Predictions vs Returns
    if len(value_prediction_log) > 0 and len(return_log) > 0:
        sample_size = min(500, len(value_prediction_log))
        if len(value_prediction_log) > sample_size:
            indices = np.random.choice(len(value_prediction_log), sample_size, replace=False)
            pred_sample = [value_prediction_log[i] for i in indices]
            ret_sample = [return_log[i] for i in indices]
        else:
            pred_sample = value_prediction_log
            ret_sample = return_log
        
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
        axs[1, 1].hist(reward_buffer[:num_episodes], bins=min(30, num_episodes//2), 
                      alpha=0.7, edgecolor='black', density=True)
        mean_reward = np.mean(reward_buffer[:num_episodes])
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
            rolling_avg.append(np.mean(reward_buffer[start_idx:i+1]))
        
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
    current_path = Path(__file__).parent.parent
    plots_dir = current_path / "outputs" / "plots" / "training"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    if timestamp:
        plot_path = plots_dir / f"{category}_diffusion_ppo_training_{timestamp}.png"
    else:
        plot_path = plots_dir / f"{category}_diffusion_ppo_training.png"
        
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Plot saved: {plot_path}")


def plot_from_csv(training_timestamp: str, category: str = DEFAULT_CATEGORY):
    """Create plots from saved CSV data with automatic training mode detection"""
    
    data = load_training_data(training_timestamp, category)
    if not data:
        return
    
    episodes_df = data.get('episodes')
    losses_df = data.get('losses')
    value_predictions_df = data.get('value_predictions')
    returns_df = data.get('returns')
    gradients_df = data.get('gradients')
    log_prob_df = data.get('log_probabilities')
    
    # Try to detect training mode from metadata or filename
    training_mode = None
    metadata_df = data.get('metadata')
    
    # Handle both dict and DataFrame cases for metadata
    if metadata_df is not None:
        if isinstance(metadata_df, dict):
            training_mode = metadata_df.get('training_mode')
        elif hasattr(metadata_df, 'columns') and 'training_mode' in metadata_df.columns:
            training_mode = metadata_df['training_mode'].iloc[0] if len(metadata_df) > 0 else None
    
    # Fallback: detect from log directory name
    training_mode = DEFAULT_TRAINING_MODE
    
    print(f"📊 Detected training mode: {training_mode}")
    
    if episodes_df is None:
        print("❌ No episode data to plot")
        return
    
    # Create 3x3 subplot layout to include log probabilities (mode-aware)
    mode_suffix = f" ({training_mode})" if training_mode else ""
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle(f'Training Progress from CSV{mode_suffix}', fontsize=16, y=0.98)
    
    # Episode rewards
    axes[0, 0].plot(episodes_df['episode'], episodes_df['avg_reward'], alpha=0.6, label='Avg Reward')
    axes[0, 0].plot(episodes_df['episode'], episodes_df['best_reward'], label='Best Reward')
    axes[0, 0].set_title('Training Progress')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss curves (if available)
    if losses_df is not None:
        # Use separate y-axes for actor and critic loss due to scale differences
        ax1 = axes[0, 1]
        ax2 = ax1.twinx()
        
        # Actor loss (primary y-axis, often large)
        line1 = ax1.plot(losses_df['update'], losses_df['actor_loss'], 'b-', label='Actor Loss')
        ax1.set_xlabel('Update')
        ax1.set_ylabel('Actor Loss', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.set_yscale('symlog')  # Symmetric log scale to handle negative values
        
        # Critic loss (secondary y-axis, often small)  
        line2 = ax2.plot(losses_df['update'], losses_df['critic_loss'], 'r-', label='Critic Loss')
        ax2.set_ylabel('Critic Loss', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
        
        ax1.set_title('Loss Curves (Dual Scale)')
        ax1.grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'No Loss Data', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Loss Curves')
    
    # Log probability plot over time
    if log_prob_df is not None and len(log_prob_df) > 0:
        # Group by episode and get mean log probability
        episode_log_probs = log_prob_df.groupby('episode')['log_probability'].mean()
        axes[0, 2].plot(episode_log_probs.index, episode_log_probs.values, 
                       linewidth=2, alpha=0.7, label='Mean Log Probability')
        
        # Add smoothed version if enough data
        if len(episode_log_probs) > 10:
            window = min(20, len(episode_log_probs) // 4)
            smoothed = episode_log_probs.rolling(window=window, center=True).mean()
            axes[0, 2].plot(smoothed.index, smoothed.values, 
                           linewidth=2, color='red', label=f'Smoothed (MA-{window})')
        
        axes[0, 2].set_title('Log Probability Over Time')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Log Probability')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Add statistics
        mean_log_prob = episode_log_probs.mean()
        std_log_prob = episode_log_probs.std()
        axes[0, 2].text(0.02, 0.98, f'Mean: {mean_log_prob:.3f}\nStd: {std_log_prob:.3f}', 
                        transform=axes[0, 2].transAxes, fontsize=9,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        axes[0, 2].text(0.5, 0.5, 'No Log Probability Data', ha='center', va='center', transform=axes[0, 2].transAxes)
        axes[0, 2].set_title('Log Probability Over Time')
    
    # Reward distribution
    axes[1, 0].hist(episodes_df['avg_reward'], bins=30, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(episodes_df['avg_reward'].mean(), color='red', linestyle='--', 
                      label=f'Mean: {episodes_df["avg_reward"].mean():.3f}')
    axes[1, 0].set_title('Reward Distribution')
    axes[1, 0].set_xlabel('Avg Reward')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
        
    # Value predictions vs returns
    min_len = min(len(value_predictions_df), len(returns_df)) if value_predictions_df is not None and returns_df is not None else 0
    if min_len > 0:
        pred_values = value_predictions_df['value_prediction'].iloc[:min_len]
        return_values = returns_df['return_value'].iloc[:min_len]
        
        # Create scatter plot
        axes[1, 0].scatter(pred_values, return_values, alpha=0.6, s=10)
        
        # Add perfect prediction line
        all_values = list(pred_values) + list(return_values)
        min_val, max_val = min(all_values), max(all_values)
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        1
        # Calculate and display correlation
        correlation = pred_values.corr(return_values)
        axes[1, 0].text(0.02, 0.98, f"Correlation: {correlation:.3f}", 
                        transform=axes[1, 0].transAxes, fontsize=9,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        axes[1, 0].set_title('Value Predictions vs Returns')
        axes[1, 0].set_xlabel('Predicted V(prompt)')
        axes[1, 0].set_ylabel('Actual Return')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
            
        print(f"📊 Plotted {min_len} value prediction vs return pairs")
    else:
        axes[1, 0].text(0.5, 0.5, 'No matching data found', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Value Predictions vs Returns')
    
    # Gradient plots
    if gradients_df is not None and len(gradients_df) > 0:
        # Plot gradient norms over time
        axes[0, 2].plot(gradients_df['update'], gradients_df['actor_grad_before'], 
                       label='Actor Grad (Before Clip)', linewidth=2, marker='o', markersize=3)
        axes[0, 2].plot(gradients_df['update'], gradients_df['actor_grad_after'], 
                       label='Actor Grad (After Clip)', linewidth=2, marker='s', markersize=3)
        axes[0, 2].plot(gradients_df['update'], gradients_df['critic_grad'], 
                       label='Critic Grad', linewidth=2, marker='^', markersize=3)
        axes[0, 2].set_title('Gradient Norms')
        axes[0, 2].set_xlabel('Update')
        axes[0, 2].set_ylabel('Gradient Norm')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_yscale('log')  # Log scale for better visualization
        
        # Log probability plot over time (moved from [0,2] to [1,1])
        if log_prob_df is not None and len(log_prob_df) > 0:
            # Group by episode and get mean log probability
            episode_log_probs = log_prob_df.groupby('episode')['log_probability'].mean()
            axes[1, 1].plot(episode_log_probs.index, episode_log_probs.values, 
                           linewidth=2, alpha=0.7, label='Mean Log Probability')
            
            # Add smoothed version if enough data
            if len(episode_log_probs) > 10:
                window = min(20, len(episode_log_probs) // 4)
                smoothed = episode_log_probs.rolling(window=window, center=True).mean()
                axes[1, 1].plot(smoothed.index, smoothed.values, 
                               linewidth=2, color='red', label=f'Smoothed (MA-{window})')
            
            axes[1, 1].set_title('Log Probability Over Time')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Log Probability')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add statistics
            mean_log_prob = episode_log_probs.mean()
            std_log_prob = episode_log_probs.std()
            axes[1, 1].text(0.02, 0.98, f'Mean: {mean_log_prob:.3f}\\nStd: {std_log_prob:.3f}', 
                            transform=axes[1, 1].transAxes, fontsize=9,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            axes[1, 1].text(0.5, 0.5, 'No Log Probability Data', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Log Probability Over Time')
        
        print(f"📊 Plotted gradient data for {len(gradients_df)} updates")
    else:
        axes[0, 2].text(0.5, 0.5, 'No Gradient Data', ha='center', va='center', transform=axes[0, 2].transAxes)
        axes[0, 2].set_title('Gradient Norms')
        axes[1, 1].text(0.5, 0.5, 'No Log Probability Data', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Log Probability Over Time')
    
    # Gradient vs Loss correlation (if both available)
    if gradients_df is not None and losses_df is not None and len(gradients_df) > 0 and len(losses_df) > 0:
        # Merge on update number
        merged_df = pd.merge(gradients_df, losses_df, on='update', how='inner')
        if len(merged_df) > 0:
            axes[1, 2].scatter(merged_df['actor_grad_before'], merged_df['actor_loss'], 
                             alpha=0.6, s=20, label='Before Clipping')
            axes[1, 2].scatter(merged_df['actor_grad_after'], merged_df['actor_loss'], 
                             alpha=0.6, s=20, label='After Clipping')
            axes[1, 2].set_title('Gradient vs Loss Correlation')
            axes[1, 2].set_xlabel('Gradient Norm')
            axes[1, 2].set_ylabel('Actor Loss')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
            axes[1, 2].set_xscale('log')
            
            # Calculate correlation
            corr_before = merged_df['actor_grad_before'].corr(merged_df['actor_loss'])
            corr_after = merged_df['actor_grad_after'].corr(merged_df['actor_loss'])
            axes[1, 2].text(0.02, 0.98, f"Corr (before): {corr_before:.3f}\nCorr (after): {corr_after:.3f}", 
                           transform=axes[1, 2].transAxes, fontsize=8,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            axes[1, 2].text(0.5, 0.5, 'No matching gradient/loss data', ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Gradient vs Loss Correlation')
    else:
        axes[1, 2].text(0.5, 0.5, 'No Gradient or Loss Data', ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Gradient vs Loss Correlation')
    
    # Log probability distribution and statistics
    if log_prob_df is not None and len(log_prob_df) > 0:
        axes[2, 0].hist(log_prob_df['log_probability'], bins=50, alpha=0.7, edgecolor='black')
        mean_log_prob = log_prob_df['log_probability'].mean()
        axes[2, 0].axvline(mean_log_prob, color='red', linestyle='--', 
                          label=f'Mean: {mean_log_prob:.3f}')
        axes[2, 0].set_title('Log Probability Distribution')
        axes[2, 0].set_xlabel('Log Probability')
        axes[2, 0].set_ylabel('Frequency')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
    else:
        axes[2, 0].text(0.5, 0.5, 'No Log Probability Data', ha='center', va='center', transform=axes[2, 0].transAxes)
        axes[2, 0].set_title('Log Probability Distribution')
    
    # Log probability vs reward correlation
    if log_prob_df is not None and len(log_prob_df) > 0:
        # Merge log prob data with episode rewards
        episode_rewards = episodes_df.set_index('episode')['avg_reward']
        log_prob_episode_mean = log_prob_df.groupby('episode')['log_probability'].mean()
        
        # Find common episodes
        common_episodes = episode_rewards.index.intersection(log_prob_episode_mean.index)
        if len(common_episodes) > 0:
            reward_data = episode_rewards.loc[common_episodes]
            log_prob_data = log_prob_episode_mean.loc[common_episodes]
            
            axes[2, 1].scatter(log_prob_data, reward_data, alpha=0.6, s=20)
            axes[2, 1].set_title('Log Probability vs Reward')
            axes[2, 1].set_xlabel('Mean Log Probability')
            axes[2, 1].set_ylabel('Episode Reward')
            axes[2, 1].grid(True, alpha=0.3)
            
            # Calculate correlation
            correlation = log_prob_data.corr(reward_data)
            axes[2, 1].text(0.02, 0.98, f'Correlation: {correlation:.3f}', 
                           transform=axes[2, 1].transAxes, fontsize=9,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            axes[2, 1].text(0.5, 0.5, 'No matching episodes', ha='center', va='center', transform=axes[2, 1].transAxes)
            axes[2, 1].set_title('Log Probability vs Reward')
    else:
        axes[2, 1].text(0.5, 0.5, 'No Log Probability Data', ha='center', va='center', transform=axes[2, 1].transAxes)
        axes[2, 1].set_title('Log Probability vs Reward')
    
    # Log probability variance over time (policy exploration indicator)
    if log_prob_df is not None and len(log_prob_df) > 0:
        episode_log_prob_std = log_prob_df.groupby('episode')['log_probability'].std()
        axes[2, 2].plot(episode_log_prob_std.index, episode_log_prob_std.values, 
                       linewidth=2, alpha=0.7, label='Log Prob Std Dev')
        
        # Add smoothed version
        if len(episode_log_prob_std) > 10:
            window = min(20, len(episode_log_prob_std) // 4)
            smoothed_std = episode_log_prob_std.rolling(window=window, center=True).mean()
            axes[2, 2].plot(smoothed_std.index, smoothed_std.values, 
                           linewidth=2, color='red', label=f'Smoothed (MA-{window})')
        
        axes[2, 2].set_title('Policy Exploration (Log Prob Variance)')
        axes[2, 2].set_xlabel('Episode')
        axes[2, 2].set_ylabel('Log Probability Std Dev')
        axes[2, 2].legend()
        axes[2, 2].grid(True, alpha=0.3)
        
        # Add interpretation text
        mean_std = episode_log_prob_std.mean()
        if mean_std > 1.0:
            exploration_text = "High exploration"
        elif mean_std > 0.1:
            exploration_text = "Moderate exploration"
        else:
            exploration_text = "Low exploration"
        
        axes[2, 2].text(0.02, 0.98, f'Mean Std: {mean_std:.3f}\n{exploration_text}', 
                        transform=axes[2, 2].transAxes, fontsize=9,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        axes[2, 2].text(0.5, 0.5, 'No Log Probability Data', ha='center', va='center', transform=axes[2, 2].transAxes)
        axes[2, 2].set_title('Policy Exploration (Log Prob Variance)')
    
    plt.tight_layout()
    
    # Save plot
    current_path = Path(__file__).parent.parent
    plots_dir = current_path / "outputs" / "plots" / "training"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plots_dir / f"training_plot_{category}_{training_timestamp}_from_csv.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Plot saved: {plot_path}")


def load_training_data(training_timestamp: str, category: str = DEFAULT_CATEGORY):
    """Load training data from CSV files for plotting"""
    current_path = Path(__file__).parent.parent
    logs_dir = current_path / "outputs" / "logs" / f"{category}_{training_timestamp}"
    
    data = {}
    
    try:
        # Load episode data
        episode_csv = logs_dir / "episode_log.csv"
        if episode_csv.exists():
            data['episodes'] = pd.read_csv(episode_csv)
            print(f"📊 Loaded {len(data['episodes'])} episodes")
        
        # Load loss data
        loss_csv = logs_dir / "loss_log.csv"
        if loss_csv.exists():
            data['losses'] = pd.read_csv(loss_csv)
            print(f"📊 Loaded {len(data['losses'])} updates")
        
        # Load metadata
        metadata_csv = logs_dir / "metadata.csv"
        if metadata_csv.exists():
            data['metadata'] = pd.read_csv(metadata_csv).iloc[0].to_dict()
            print(f"📊 Loaded metadata")
        
        # Load value predictions
        value_csv = logs_dir / "value_predictions.csv"
        if value_csv.exists():
            data['value_predictions'] = pd.read_csv(value_csv)
            print(f"📊 Loaded {len(data['value_predictions'])} value predictions")
        
        # Load returns
        returns_csv = logs_dir / "returns.csv"
        if returns_csv.exists():
            data['returns'] = pd.read_csv(returns_csv)
            print(f"📊 Loaded {len(data['returns'])} returns")
        
        # Load gradients
        gradient_csv = logs_dir / "gradient_log.csv"
        if gradient_csv.exists():
            data['gradients'] = pd.read_csv(gradient_csv)
            print(f"📊 Loaded {len(data['gradients'])} gradient entries")
        
        # Load log probabilities
        log_prob_csv = logs_dir / "log_probabilities.csv"
        if log_prob_csv.exists():
            data['log_probabilities'] = pd.read_csv(log_prob_csv)
            print(f"📊 Loaded {len(data['log_probabilities'])} log probability entries")
        
        return data
        
    except Exception as e:
        print(f"❌ Error loading training data: {e}")
        return None


# def plot_feature_distributions(training_timestamp: str, category: str = DEFAULT_CATEGORY):
#     """
#     Create t-SNE plots for reference vs generated feature distributions
#     Saves side-by-side comparison plots to outputs/plots/feature_distribution/{timestamp}/
#     """
#     try:
#         from sklearn.manifold import TSNE
#         from sklearn.preprocessing import StandardScaler
#     except ImportError:
#         print("❌ Error: scikit-learn required for t-SNE plots")
#         print("💡 Install with: pip install scikit-learn")
#         return False
    
#     current_path = Path(__file__).parent.parent
    
#     # Load generated features from CSV
#     logs_dir = current_path / "outputs" / "logs" / f"{category}_{training_timestamp}"
#     features_csv = logs_dir / "generated_features.csv"
    
#     if not features_csv.exists():
#         print(f"❌ Generated features not found: {features_csv}")
#         print("💡 Features are logged during training - run a training session first")
#         return False
    
#     # Load reference features from NPZ
#     ref_features_npz = current_path / "reference_features" / f"reference_{category}_features_v2.npz"
#     if not ref_features_npz.exists():
#         print(f"❌ Reference features not found: {ref_features_npz}")
#         return False
    
#     print(f"📊 Loading feature data for t-SNE visualization...")
    
#     # Load generated features
#     gen_df = pd.read_csv(features_csv)
#     feature_cols = [col for col in gen_df.columns if col.startswith('feature_')]
    
#     if not feature_cols:
#         print(f"❌ No feature columns found in {features_csv}")
#         return False
    
#     gen_features = gen_df[feature_cols].values
#     print(f"📊 Loaded {len(gen_features)} generated feature vectors")
    
#     # Load reference features  
#     ref_npz = np.load(ref_features_npz)
#     ref_features_list = []
#     for key in ref_npz.keys():
#         ref_features_list.append(ref_npz[key])
#     ref_features = np.stack(ref_features_list)
#     ref_npz.close()
#     print(f"📊 Loaded {len(ref_features)} reference feature vectors")
    
#     # Combine features for t-SNE
#     all_features = np.vstack([ref_features, gen_features])
#     labels = ['Reference'] * len(ref_features) + ['Generated'] * len(gen_features)
    
#     # Standardize features
#     scaler = StandardScaler()
#     all_features_scaled = scaler.fit_transform(all_features)
    
#     print(f"🔄 Running t-SNE on {len(all_features)} feature vectors...")
    
#     # Run t-SNE
#     tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_features)//4))
#     features_2d = tsne.fit_transform(all_features_scaled)
    
#     # Split back into reference and generated
#     ref_2d = features_2d[:len(ref_features)]
#     gen_2d = features_2d[len(ref_features):]
    
#     # Create plots directory
#     plots_dir = current_path / "outputs" / "plots" / "feature_distribution" / training_timestamp
#     plots_dir.mkdir(parents=True, exist_ok=True)
    
#     # Create side-by-side plots
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
#     # Plot 1: Reference features
#     ax1.scatter(ref_2d[:, 0], ref_2d[:, 1], c='blue', alpha=0.6, s=20, label=f'Reference (n={len(ref_features)})')
#     ax1.set_title(f'Reference Feature Distribution\n{category.title()} Category')
#     ax1.set_xlabel('t-SNE Dimension 1')
#     ax1.set_ylabel('t-SNE Dimension 2')
#     ax1.legend()
#     ax1.grid(True, alpha=0.3)
    
#     # Plot 2: Generated features
#     ax2.scatter(gen_2d[:, 0], gen_2d[:, 1], c='red', alpha=0.6, s=20, label=f'Generated (n={len(gen_features)})')
#     ax2.set_title(f'Generated Feature Distribution\nTraining: {training_timestamp}')
#     ax2.set_xlabel('t-SNE Dimension 1')
#     ax2.set_ylabel('t-SNE Dimension 2')
#     ax2.legend()
#     ax2.grid(True, alpha=0.3)
    
#     plt.tight_layout()
    
#     # Save plot
#     plot_path = plots_dir / f"feature_distribution_comparison_{category}_{training_timestamp}.png"
#     plt.savefig(plot_path, dpi=300, bbox_inches='tight')
#     plt.show()
    
#     print(f"✅ Feature distribution plot saved: {plot_path}")
    
#     # Also create combined overlay plot
#     fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
#     ax.scatter(ref_2d[:, 0], ref_2d[:, 1], c='blue', alpha=0.6, s=30, label=f'Reference (n={len(ref_features)})')
#     ax.scatter(gen_2d[:, 0], gen_2d[:, 1], c='red', alpha=0.6, s=30, label=f'Generated (n={len(gen_features)})')
    
#     ax.set_title(f'Feature Distribution Comparison\n{category.title()} - {training_timestamp}')
#     ax.set_xlabel('t-SNE Dimension 1')
#     ax.set_ylabel('t-SNE Dimension 2')
#     ax.legend()
#     ax.grid(True, alpha=0.3)
    
#     plt.tight_layout()
    
#     # Save overlay plot
#     overlay_path = plots_dir / f"feature_distribution_overlay_{category}_{training_timestamp}.png"
#     plt.savefig(overlay_path, dpi=300, bbox_inches='tight')
#     plt.show()
    
#     print(f"✅ Feature distribution overlay saved: {overlay_path}")
    
#     return True
