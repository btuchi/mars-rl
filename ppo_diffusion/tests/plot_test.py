#!/usr/bin/env python"""Standalone test script to plot the most recent training logs with gradient visualization"""

from pathlib import Path
import sys
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Default category
DEFAULT_CATEGORY = "crater"

def find_most_recent_logs(category: str = DEFAULT_CATEGORY):
    """Find the most recent training logs for a given category"""
    current_path = Path(__file__).parent.parent
    logs_dir = current_path / "outputs" / "logs"
    
    if not logs_dir.exists():
        print(f"❌ Logs directory not found: {logs_dir}")
        return None
    
    # Find all log directories for this category
    pattern = f"{category}_*"
    log_dirs = [d for d in logs_dir.glob(pattern) if d.is_dir()]
    
    if not log_dirs:
        print(f"❌ No log directories found for category '{category}'")
        return None
    
    # Sort by timestamp (directory name contains timestamp)
    log_dirs.sort(key=lambda x: x.name.split('_')[-1], reverse=True)
    
    # Check which ones have actual data
    for log_dir in log_dirs:
        episode_log = log_dir / "episode_log.csv"
        if episode_log.exists():
            timestamp = log_dir.name.split('_')[-1]
            print(f"📊 Found most recent logs: {category}_{timestamp}")
            return timestamp
    
    print(f"❌ No log directories with episode data found for category '{category}'")
    return None

def list_available_logs(category: str = DEFAULT_CATEGORY):
    """List all available training logs for a category"""
    current_path = Path(__file__).parent.parent
    logs_dir = current_path / "outputs" / "logs"
    
    if not logs_dir.exists():
        print(f"❌ Logs directory not found: {logs_dir}")
        return []
    
    pattern = f"{category}_*"
    log_dirs = [d for d in logs_dir.glob(pattern) if d.is_dir()]
    
    timestamps = []
    for log_dir in log_dirs:
        episode_log = log_dir / "episode_log.csv"
        if episode_log.exists():
            timestamp = log_dir.name.split('_')[-1]
            
            # Convert timestamp to readable format
            try:
                dt = datetime.strptime(timestamp, "%Y%m%d%H%M%S")
                readable_time = dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                readable_time = timestamp
            
            # Check what files exist
            files = [f.name for f in log_dir.glob("*.csv")]
            timestamps.append((timestamp, readable_time, files))
    
    # Sort by timestamp (most recent first)
    timestamps.sort(key=lambda x: x[0], reverse=True)
    
    print(f"📋 Available training logs for '{category}':")
    for i, (timestamp, readable_time, files) in enumerate(timestamps):
        print(f"  {i+1}. {timestamp} ({readable_time})")
        print(f"     Files: {', '.join(files)}")
    
    return timestamps

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
        
        return data
        
    except Exception as e:
        print(f"❌ Error loading training data: {e}")
        return None

def plot_from_csv(training_timestamp: str, category: str = DEFAULT_CATEGORY):
    """Create plots from saved CSV data"""
    
    data = load_training_data(training_timestamp, category)
    if not data:
        return
    
    episodes_df = data.get('episodes')
    losses_df = data.get('losses')
    value_predictions_df = data.get('value_predictions')
    returns_df = data.get('returns')
    gradients_df = data.get('gradients')
    
    if episodes_df is None:
        print("❌ No episode data to plot")
        return
    
    # Create 2x3 subplot layout to include gradients
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
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
        axes[1, 0].scatter(pred_values, return_values, alpha=0.6, s=10)
        
        # Add perfect prediction line
        all_values = list(pred_values) + list(return_values)
        min_val, max_val = min(all_values), max(all_values)
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
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
        
        # Plot gradient clipping frequency
        clipping_rate = gradients_df['grad_clipped'].rolling(window=10, min_periods=1).mean()
        axes[1, 1].plot(gradients_df['update'], clipping_rate * 100, 
                       linewidth=2, color='red', label='Clipping Rate (%)')
        axes[1, 1].set_title('Gradient Clipping Rate (10-update rolling avg)')
        axes[1, 1].set_xlabel('Update')
        axes[1, 1].set_ylabel('Clipping Rate (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(0, 100)
        
        # Add statistics text
        total_clipped = gradients_df['grad_clipped'].sum()
        total_updates = len(gradients_df)
        clip_percentage = (total_clipped / total_updates) * 100
        axes[1, 1].text(0.02, 0.98, f"Overall: {clip_percentage:.1f}% clipped", 
                        transform=axes[1, 1].transAxes, fontsize=9,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        print(f"📊 Plotted gradient data for {len(gradients_df)} updates")
    else:
        axes[0, 2].text(0.5, 0.5, 'No Gradient Data', ha='center', va='center', transform=axes[0, 2].transAxes)
        axes[0, 2].set_title('Gradient Norms')
        axes[1, 1].text(0.5, 0.5, 'No Gradient Data', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Gradient Clipping Rate')
    
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
    
    plt.tight_layout()
    
    # Save plot
    current_path = Path(__file__).parent.parent
    plots_dir = current_path / "outputs" / "plots" / "training"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plots_dir / f"training_plot_{category}_{training_timestamp}_from_csv.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Plot saved: {plot_path}")

def main():
    """Main function to plot the most recent training logs"""
    category = DEFAULT_CATEGORY
    
    print(f"🔍 Looking for most recent training logs for category: '{category}'")
    
    # First, list all available logs
    available_logs = list_available_logs(category)
    
    if not available_logs:
        print(f"❌ No training logs found for category '{category}'")
        return
    
    # Find the most recent one with data
    most_recent_timestamp = find_most_recent_logs(category)
    
    if not most_recent_timestamp:
        print("❌ No valid training logs found")
        return
    
    print(f"📈 Plotting training data for: {category}_{most_recent_timestamp}")
    
    try:
        # Call the visualization function with gradient plots
        plot_from_csv(most_recent_timestamp, category)
        print("✅ Plots generated successfully!")
        
    except Exception as e:
        print(f"❌ Error generating plots: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()