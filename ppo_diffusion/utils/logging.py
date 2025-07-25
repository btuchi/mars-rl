"""Centralized logging functionality (cleaned up from diffusion_log_utils.py)"""

import pandas as pd
import numpy as np
import os
import time
import signal
import sys
from pathlib import Path
import atexit
import matplotlib.pyplot as plt
from .constants import DEFAULT_CATEGORY, LOG_SAVE_FREQUENCY

# Global lists for backward compatibility
ACTOR_LOSS_LOG = []
CRITIC_LOSS_LOG = []
REWARD_LOG = []
VALUE_PREDICTION_LOG = []
RETURN_LOG = []
BEST_REWARD_LOG = []
LOG_PROB_LOG = []


class TrainingLogger:
    """Robust CSV logger that saves periodically and on interruption"""

    def __init__(self, training_timestamp: str, category: str = DEFAULT_CATEGORY, training_mode: str = None):
        self.training_timestamp = training_timestamp
        self.category = category
        self.training_mode = training_mode
        
        # Setup directory
        current_path = Path(__file__).parent.parent
        self.logs_dir = current_path / "outputs" / "logs" / f"{category}_{training_timestamp}"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # CSV file paths
        self.episode_csv = self.logs_dir / "episode_log.csv"
        self.loss_csv = self.logs_dir / "loss_log.csv"
        self.value_csv = self.logs_dir / "value_predictions.csv"
        self.return_csv = self.logs_dir / "returns.csv"
        self.gradient_csv = self.logs_dir / "gradient_log.csv"
        self.metadata_csv = self.logs_dir / "metadata.csv"
        self.log_prob_csv = self.logs_dir / "log_probabilities.csv"
        self.features_csv = self.logs_dir / "generated_features.csv"
        
        # Initialize metadata
        self.metadata = {
            'training_timestamp': training_timestamp,
            'category': category,
            'training_mode': training_mode or 'UNKNOWN',
            'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': None,
            'total_episodes': 0,
            'total_updates': 0,
            'best_reward': -float('inf'),
            'final_avg_reward': 0,
            'completed': False,
            'interruption_reason': None
        }
        
        # Episode tracking
        self.episode_data = []
        self.loss_data = []
        self.return_data = []
        self.value_prediction_data = []
        self.gradient_data = []
        self.log_prob_data = []
        self.feature_data = []
        
        # Auto-save frequency
        self.save_frequency = LOG_SAVE_FREQUENCY
        self.last_save_episode = 0
        
        # Setup graceful shutdown handlers
        self.setup_signal_handlers()
        self.setup_exit_handler()
        
        print(f"📊 Logger initialized - Logs will be saved to: {self.logs_dir}")
    
    def setup_signal_handlers(self):
        """Setup handlers for SIGINT (Ctrl+C) and SIGTERM"""
        def signal_handler(signum, frame):
            signal_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
            print(f"\n🛑 Received {signal_name} - Saving logs before exit...")
            self.metadata['interruption_reason'] = f"Interrupted by {signal_name}"
            self.save_all_logs(final=True)
            print("✅ Logs saved successfully!")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def setup_exit_handler(self):
        """Setup handler for normal program exit"""
        def exit_handler():
            if not self.metadata['completed']:
                print("\n📊 Saving final logs on exit...")
                self.metadata['interruption_reason'] = "Unexpected exit"
                self.save_all_logs(final=True)
        
        atexit.register(exit_handler)

    def log_value_prediction(self, value_prediction: float):
        """Log value prediction data"""
        value_entry = {
            'value_prediction': value_prediction,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        # Add to global list (for compatibility)
        VALUE_PREDICTION_LOG.append(value_prediction)
        
        # Also save to dedicated list for CSV
        self.value_prediction_data.append(value_entry)

        # Auto-save every 10 entries
        if len(self.value_prediction_data) % 10 == 0:
            self.save_value_predictions()
    
    def log_return(self, return_value: float):
        """Log return value data"""
        return_entry = {
            'return_value': return_value,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Add to global list (for compatibility)
        RETURN_LOG.append(return_value)
        
        # Also save to dedicated list for CSV
        self.return_data.append(return_entry)

        # Auto-save every 10 entries
        if len(self.return_data) % 10 == 0:
            self.save_returns()
    
    def log_log_probability(self, log_prob: float, episode: int, trajectory_idx: int = 0):
        """Log log probability data"""
        log_prob_entry = {
            'episode': episode,
            'trajectory_idx': trajectory_idx,
            'log_probability': log_prob,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Add to global list (for compatibility)
        LOG_PROB_LOG.append(log_prob)
        
        # Also save to dedicated list for CSV
        self.log_prob_data.append(log_prob_entry)

        # Auto-save every 10 entries
        if len(self.log_prob_data) % 10 == 0:
            self.save_log_probabilities()
    
    def log_generated_features(self, features: np.ndarray, episode: int, prompt: str):
        """Log generated features for t-SNE visualization"""
        
        # features shape: (batch_size, feature_dim)
        for i, feature_vector in enumerate(features):
            feature_entry = {
                'episode': episode,
                'trajectory_idx': i,
                'prompt': prompt,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Add each feature dimension as a separate column
            for j, feature_val in enumerate(feature_vector):
                feature_entry[f'feature_{j:03d}'] = float(feature_val)
            
            self.feature_data.append(feature_entry)
        
        # Auto-save every 20 entries (since each episode generates multiple features)
        if len(self.feature_data) % 20 == 0:
            self.save_generated_features()
    
    def log_gradient_info(self, update_num: int, episode: int, gradient_info: dict):
        """Log gradient information"""
        gradient_entry = {
            'update': update_num,
            'episode': episode,
            'actor_grad_before': gradient_info.get('actor_grad_before', 0.0),
            'actor_grad_after': gradient_info.get('actor_grad_after', 0.0),
            'critic_grad': gradient_info.get('critic_grad', 0.0),
            'grad_clipped': gradient_info.get('grad_clipped', False),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        self.gradient_data.append(gradient_entry)
        
        # Save gradient log
        self.save_gradient_log()
        
    def log_episode(self, episode: int, prompt: str, individual_rewards: list, 
                   avg_reward: float, best_reward: float):
        """Log episode-level data"""
        episode_entry = {
            'episode': episode,
            'prompt': prompt,
            'avg_reward': avg_reward,
            'best_reward': best_reward,
            'reward_1': individual_rewards[0] if len(individual_rewards) > 0 else None,
            'reward_2': individual_rewards[1] if len(individual_rewards) > 1 else None,
            'reward_3': individual_rewards[2] if len(individual_rewards) > 2 else None,
            'reward_4': individual_rewards[3] if len(individual_rewards) > 3 else None,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        self.episode_data.append(episode_entry)
        
        # Update metadata
        self.metadata['total_episodes'] = episode
        self.metadata['best_reward'] = max(self.metadata['best_reward'], best_reward)
        
        # Auto-save periodically
        if episode - self.last_save_episode >= self.save_frequency:
            self.save_episode_log()
            self.last_save_episode = episode
        
    def log_update(self, update_num: int, actor_loss: float, critic_loss: float, episode: int, gradient_info: dict = None):
        """Log PPO update data"""
        loss_entry = {
            'update': update_num,
            'episode': episode,
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        self.loss_data.append(loss_entry)
        
        # Log gradient information if provided
        if gradient_info:
            self.log_gradient_info(update_num, episode, gradient_info)
        
        # Update metadata
        self.metadata['total_updates'] = update_num
        
        # Save loss log
        self.save_loss_log()
    
    def save_value_predictions(self):
        """Save value predictions to CSV"""
        try:
            if self.value_prediction_data:
                df = pd.DataFrame(self.value_prediction_data)
                df.to_csv(self.value_csv, index=False)
                print(f"💾 Value predictions saved: {len(self.value_prediction_data)} entries")
        except Exception as e:
            print(f"⚠️ Error saving value predictions: {e}")
    
    def save_returns(self):
        """Save returns to CSV"""
        try:                
            if self.return_data:
                df = pd.DataFrame(self.return_data)
                df.to_csv(self.return_csv, index=False)
                print(f"💾 Returns saved: {len(self.return_data)} entries")
        except Exception as e:
            print(f"⚠️ Error saving returns: {e}")
    
    def save_log_probabilities(self):
        """Save log probabilities to CSV"""
        try:
            if self.log_prob_data:
                df = pd.DataFrame(self.log_prob_data)
                df.to_csv(self.log_prob_csv, index=False)
                print(f"💾 Log probabilities saved: {len(self.log_prob_data)} entries")
        except Exception as e:
            print(f"⚠️ Error saving log probabilities: {e}")
    
    def save_generated_features(self):
        """Save generated features to CSV"""
        try:
            if self.feature_data:
                df = pd.DataFrame(self.feature_data)
                df.to_csv(self.features_csv, index=False)
                print(f"💾 Generated features saved: {len(self.feature_data)} entries")
        except Exception as e:
            print(f"⚠️ Error saving generated features: {e}")
    
    def save_episode_log(self):
        """Save episode data to CSV"""
        try:
            if self.episode_data:
                df = pd.DataFrame(self.episode_data)
                df.to_csv(self.episode_csv, index=False)
                print(f"💾 Episode log saved: {len(self.episode_data)} episodes")
        except Exception as e:
            print(f"⚠️ Error saving episode log: {e}")
        
    def save_loss_log(self):
        """Save loss data to CSV"""
        try:
            if self.loss_data:
                df = pd.DataFrame(self.loss_data)
                df.to_csv(self.loss_csv, index=False)
        except Exception as e:
            print(f"⚠️ Error saving loss log: {e}")
    
    def save_gradient_log(self):
        """Save gradient data to CSV"""
        try:
            if self.gradient_data:
                df = pd.DataFrame(self.gradient_data)
                df.to_csv(self.gradient_csv, index=False)
                print(f"💾 Gradient log saved: {len(self.gradient_data)} entries")
        except Exception as e:
            print(f"⚠️ Error saving gradient log: {e}")

    def save_metadata(self):
        """Save training metadata"""
        try:
            # Calculate final average reward
            if self.episode_data:
                recent_rewards = [ep['avg_reward'] for ep in self.episode_data[-20:]]
                self.metadata['final_avg_reward'] = sum(recent_rewards) / len(recent_rewards)
            
            df = pd.DataFrame([self.metadata])
            df.to_csv(self.metadata_csv, index=False)
        except Exception as e:
            print(f"⚠️ Error saving metadata: {e}")
        
    def save_all_logs(self, final: bool = False):
        """Save all logs at once"""
        if final:
            self.metadata['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
            self.metadata['completed'] = True
        
        self.save_episode_log()
        self.save_loss_log()
        self.save_gradient_log()
        self.save_metadata()
        self.save_value_predictions()
        self.save_returns()
        self.save_log_probabilities()
        self.save_generated_features()
        
        if final:
            print(f"📊 Final logs saved to: {self.logs_dir}")
            self.create_summary_report()

    def create_summary_report(self):
        """Create a human-readable summary report"""
        try:
            summary_path = self.logs_dir / "training_summary.txt"
            
            with open(summary_path, 'w') as f:
                f.write(f"=== DIFFUSION PPO TRAINING SUMMARY ===\n")
                f.write(f"Category: {self.metadata['category']}\n")
                f.write(f"Timestamp: {self.metadata['training_timestamp']}\n")
                f.write(f"Started: {self.metadata['start_time']}\n")
                f.write(f"Ended: {self.metadata['end_time']}\n")
                f.write(f"Completed: {self.metadata['completed']}\n")
                if self.metadata['interruption_reason']:
                    f.write(f"Interruption: {self.metadata['interruption_reason']}\n")
                f.write(f"Total Episodes: {self.metadata['total_episodes']}\n")
                f.write(f"Total Updates: {self.metadata['total_updates']}\n")
                f.write(f"Best Reward: {self.metadata['best_reward']:.4f}\n")
                f.write(f"Final Avg Reward: {self.metadata['final_avg_reward']:.4f}\n")
                f.write("="*50 + "\n")
                
                # Episode statistics
                if self.episode_data:
                    rewards = [ep['avg_reward'] for ep in self.episode_data]
                    f.write(f"Reward Statistics:\n")
                    f.write(f"  Mean: {sum(rewards)/len(rewards):.4f}\n")
                    f.write(f"  Min: {min(rewards):.4f}\n")
                    f.write(f"  Max: {max(rewards):.4f}\n")
                    f.write(f"  Std: {pd.Series(rewards).std():.4f}\n")
            
            print(f"📋 Summary report created: {summary_path}")
            
        except Exception as e:
            print(f"⚠️ Error creating summary report: {e}")


# Global logger instance
_logger = None

def initialize_logger(training_timestamp: str, category: str = DEFAULT_CATEGORY, training_mode: str = None):
    """Initialize the global logger"""
    global _logger
    _logger = TrainingLogger(training_timestamp, category, training_mode)
    return _logger

# Wrapper functions for backward compatibility
def log_value_prediction(value_prediction: float):
    """Log value prediction (convenient wrapper)"""
    if _logger:
        _logger.log_value_prediction(value_prediction)

def log_return(return_value: float):
    """Log return value (convenient wrapper)"""
    if _logger:
        _logger.log_return(return_value)

def log_episode(episode: int, prompt: str, individual_rewards: list, 
               avg_reward: float, best_reward: float):
    """Log episode data (convenient wrapper)"""
    if _logger:
        _logger.log_episode(episode, prompt, individual_rewards, avg_reward, best_reward)

def log_update(update_num: int, actor_loss: float, critic_loss: float, episode: int, gradient_info: dict = None):
    """Log update data (convenient wrapper)"""
    if _logger:
        _logger.log_update(update_num, actor_loss, critic_loss, episode, gradient_info)

def log_log_probability(log_prob: float, episode: int, trajectory_idx: int = 0):
    """Log log probability (convenient wrapper)"""
    if _logger:
        _logger.log_log_probability(log_prob, episode, trajectory_idx)

def log_generated_features(features, episode: int, prompt: str):
    """Log generated features (convenient wrapper)"""
    if _logger:
        _logger.log_generated_features(features, episode, prompt)

def plot_per_image_feature_distribution(features: np.ndarray, episode: int, prompt: str, training_timestamp: str):
    """
    Plot the distribution of features WITHIN each image (what your prof wants)
    Args:
        features: (batch_size, 512) - each row is one image's 512 features
        episode: episode number
        prompt: prompt used
        training_timestamp: timestamp for folder naming
    """
    # Create plots directory
    current_path = Path(__file__).parent.parent
    plots_dir = current_path / "outputs" / "plots" / "feature_distribution" / training_timestamp
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot feature distribution for first image in batch
    feature_vector = features[0]  # Take first image's 512 features
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Main histogram
    plt.subplot(2, 2, 1)
    plt.hist(feature_vector, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    plt.title(f'Feature Distribution - Episode {episode}')
    plt.xlabel('Feature Value')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    
    # Box plot
    plt.subplot(2, 2, 2)
    plt.boxplot(feature_vector, vert=True)
    plt.title('Feature Distribution (Box Plot)')
    plt.ylabel('Feature Value')
    plt.grid(True, alpha=0.3)
    
    # Feature values over indices
    plt.subplot(2, 2, 3)
    plt.plot(feature_vector, alpha=0.7, color='darkgreen')
    plt.title('Feature Values by Index')
    plt.xlabel('Feature Index (0-511)')
    plt.ylabel('Feature Value')
    plt.grid(True, alpha=0.3)
    
    # Statistics text
    plt.subplot(2, 2, 4)
    plt.axis('off')
    stats_text = f"""
    Episode: {episode}
    Prompt: {prompt[:50]}...
    
    Statistics:
    Mean: {np.mean(feature_vector):.4f}
    Std: {np.std(feature_vector):.4f}
    Min: {np.min(feature_vector):.4f}
    Max: {np.max(feature_vector):.4f}
    Range: {np.max(feature_vector) - np.min(feature_vector):.4f}
    Skew: {np.mean(((feature_vector - np.mean(feature_vector)) / np.std(feature_vector)) ** 3):.4f}
    """
    plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', fontsize=10, fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save plot
    safe_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_')).strip()[:20]
    filename = f"ep{episode:04d}_feature_dist_{safe_prompt}.png"
    save_path = plots_dir / filename
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Feature distribution plot saved: {filename}")
    
    return str(save_path)

def plot_tsne_feature_space(features: np.ndarray, episode: int, prompt: str, training_timestamp: str):
    """
    Plot t-SNE visualization of 512 features within the first image
    Args:
        features: (batch_size, 512) - each row is one image's 512 features
        episode: episode number
        prompt: prompt used
        training_timestamp: timestamp for folder naming
    """
    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        
        # Create plots directory
        current_path = Path(__file__).parent.parent
        plots_dir = current_path / "outputs" / "plots" / "tsne" / training_timestamp
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Take first image's 512 features
        first_image_features = features[0]  # Shape: (512,)
        
        # Reshape for t-SNE: each feature becomes a "data point"
        # We need to create a 2D array where each row is a feature
        # But t-SNE needs multiple dimensions, so we'll use feature index as additional info
        feature_data = np.column_stack([
            first_image_features,  # Feature values
            np.arange(len(first_image_features))  # Feature indices
        ])
        
        # Run t-SNE on the 512 features
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(first_image_features)-1))
        tsne_features = tsne.fit_transform(feature_data)
        
        # Create the plot
        plt.figure(figsize=(10, 8))
        
        # Scatter plot - each point is one of the 512 features
        scatter = plt.scatter(tsne_features[:, 0], tsne_features[:, 1], 
                             c=first_image_features, cmap='viridis', 
                             s=30, alpha=0.7, edgecolors='black')
        
        plt.title(f't-SNE of 512 Features within Image - Episode {episode}\nPrompt: {prompt[:60]}...')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.grid(True, alpha=0.3)
        plt.colorbar(scatter, label='Feature Value')
        
        # Add statistics about feature clustering
        distances = []
        for i in range(len(tsne_features)):
            for j in range(i+1, len(tsne_features)):
                dist = np.linalg.norm(tsne_features[i] - tsne_features[j])
                distances.append(dist)
        
        if distances:
            plt.figtext(0.02, 0.02, 
                       f'Feature Clustering Stats:\nMean distance: {np.mean(distances):.3f}\nStd distance: {np.std(distances):.3f}\nTotal features: {len(first_image_features)}',
                       fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        
        # Save plot
        safe_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_')).strip()[:20]
        filename = f"ep{episode:04d}_feature_tsne_{safe_prompt}.png"
        save_path = plots_dir / filename
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Per-image feature t-SNE plot saved: {filename}")
        return str(save_path)
        
    except ImportError:
        print("⚠️ scikit-learn not installed - skipping t-SNE plot")
        return None
    except Exception as e:
        print(f"⚠️ Error creating t-SNE plot: {e}")
        return None

def finalize_logging():
    """Complete logging and save final files"""
    if _logger:
        _logger.save_all_logs(final=True)

