"""Plotting and visualization utilities (extracted from trainer.py)"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from .constants import DEFAULT_CATEGORY

def plot_diffusion_training(reward_buffer, actor_loss_log, critic_loss_log, best_reward_log, 
                          value_prediction_log, return_log, final_episode=None, 
                          category=DEFAULT_CATEGORY, timestamp=None):
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
        
        axs[0, 1].plot(loss_x_points, actor_loss_log, label="UNet Loss", linewidth=2, marker='o', markersize=3)
        axs[0, 1].plot(loss_x_points, critic_loss_log, label="Value Loss", linewidth=2, marker='s', markersize=3)
        axs[0, 1].set_title(f"Loss per Update ({len(actor_loss_log)} updates)")
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
        axs[0, 1].set_title("Loss per Update")

    # Best Reward Progress
    axs[0, 2].plot(episode_range, best_reward_log[:num_episodes], label="Best Diversity Reward", linewidth=2)
    axs[0, 2].set_title("Best Diversity Reward Progress")
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
    """Create plots from saved CSV data"""
    
    data = load_training_data(training_timestamp, category)
    if not data:
        return
    
    episodes_df = data.get('episodes')
    losses_df = data.get('losses')
    value_predictions_df = data.get('value_predictions')
    returns_df = data.get('returns')
    
    if episodes_df is None:
        print("❌ No episode data to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
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
        axes[0, 1].plot(losses_df['update'], losses_df['actor_loss'], label='Actor Loss')
        axes[0, 1].plot(losses_df['update'], losses_df['critic_loss'], label='Critic Loss')
        axes[0, 1].set_title('Loss Curves')
        axes[0, 1].set_xlabel('Update')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'No Loss Data', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Loss Curves')
    
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
        axes[1, 1].scatter(pred_values, return_values, alpha=0.6, s=10)
        
        # Add perfect prediction line
        all_values = list(pred_values) + list(return_values)
        min_val, max_val = min(all_values), max(all_values)
        axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        # Calculate and display correlation
        correlation = pred_values.corr(return_values)
        axes[1, 1].text(0.02, 0.98, f"Correlation: {correlation:.3f}", 
                        transform=axes[1, 1].transAxes, fontsize=9,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        axes[1, 1].set_title('Value Predictions vs Returns')
        axes[1, 1].set_xlabel('Predicted V(prompt)')
        axes[1, 1].set_ylabel('Actual Return')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
            
        print(f"📊 Plotted {min_len} value prediction vs return pairs")
    else:
        axes[1, 1].text(0.5, 0.5, 'No matching data found', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Value Predictions vs Returns')
    
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
        
        return data
        
    except Exception as e:
        print(f"❌ Error loading training data: {e}")
        return None
