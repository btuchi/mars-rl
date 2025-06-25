import pandas as pd
import os
import time
import signal
import sys
from pathlib import Path
import atexit
import matplotlib.pyplot as plt

ACTOR_LOSS_LOG = []
CRITIC_LOSS_LOG = []
REWARD_LOG = []
VALUE_PREDICTION_LOG = []
RETURN_LOG = []
BEST_REWARD_LOG = []

# Episode-level logging for CSV
EPISODE_LOG = []
TRAINING_METADATA = {}

# Training category
CATEGORY = "crater"

class TrainingLogger:
    """Robust CSV logger that saves periodically and on interruption"""

    def __init__(self, training_timestamp: str, category: str = "mars_craters"):
            self.training_timestamp = training_timestamp
            self.category = category
            
            # Setup directory
            current_path = Path(__file__).parent
            self.logs_dir = current_path / "logs" / f"{category}_{training_timestamp}"
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            
            
            # CSV file paths
            self.episode_csv = self.logs_dir / "episode_log.csv"
            self.loss_csv = self.logs_dir / "loss_log.csv"
            self.value_csv = self.logs_dir / "value_predictions.csv"
            self.return_csv = self.logs_dir / "returns.csv"
            self.metadata_csv = self.logs_dir / "metadata.csv"
            
            # Initialize metadata
            self.metadata = {
                'training_timestamp': training_timestamp,
                'category': category,
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
            
            # Auto-save every N episodes
            self.save_frequency = 10
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
        
        def graceful_exit(signum, frame):
            print("🛑 Job being killed - saving logs...")
            self.save_all_logs(final=True)
            sys.exit(0)
        
        atexit.register(exit_handler)
        signal.signal(signal.SIGTERM, graceful_exit)

    def log_value_prediction(self, value_prediction: float):
        """Log value prediction data"""
        value_entry = {
            'value_prediction': value_prediction,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        # Add to global list (for compatibility with existing code)
        VALUE_PREDICTION_LOG.append(value_prediction)
        
        # Also save to dedicated list for CSV
        self.value_prediction_data.append(value_entry)
    
    def log_return(self, return_value: float):
        """Log return value data"""
        return_entry = {
            'return_value': return_value,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Add to global list (for compatibility with existing code)
        RETURN_LOG.append(return_value)
        
        # Also save to dedicated list for CSV
        self.return_data.append(return_entry)
        
        
    def log_episode(self, episode: int, prompt: str, individual_rewards: list, 
                   avg_reward: float, best_reward: float):
        
        """Log episode-level data"""
        episode_entry = {
            'episode': episode,
            'prompt': prompt,
            'avg_reward': avg_reward,
            'best_reward': best_reward,
            # TODO: Handle individual rewards dynamically
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
        
    def log_update(self, update_num: int, actor_loss: float, critic_loss: float, episode: int):
        """Log PPO update data"""
        loss_entry = {
            'update': update_num,
            'episode': episode,
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        self.loss_data.append(loss_entry)
        
        # Update metadata
        self.metadata['total_updates'] = update_num
        
        # Save loss log (smaller, save more frequently)
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
        self.save_metadata()
        self.save_value_predictions()
        self.save_returns()
        
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

def initialize_logger(training_timestamp: str, category: str = "mars_craters"):
    """Initialize the global logger"""
    global _logger
    _logger = TrainingLogger(training_timestamp, category)
    return _logger

# Add these wrapper functions at the module level:
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

def log_update(update_num: int, actor_loss: float, critic_loss: float, episode: int):
    """Log update data (convenient wrapper)"""
    if _logger:
        _logger.log_update(update_num, actor_loss, critic_loss, episode)

def finalize_logging():
    """Complete logging and save final files"""
    if _logger:
        _logger.save_all_logs(final=True)

# Function to load and plot saved data
def load_training_data(training_timestamp: str, category: str = "crater"):
    """Load training data from CSV files for plotting"""
    logs_dir = Path(__file__).parent / "logs" / f"{category}_{training_timestamp}"
    
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

def plot_from_csv(training_timestamp: str, category: str = "crater"):
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
        
    # Ensure we have the same number of predictions and returns
    min_len = min(len(value_predictions_df), len(returns_df)) if value_predictions_df and returns_df else 0
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
        raise ValueError("No matching data found")
    
    plt.tight_layout()
    
    # Save plot
    plot_path = Path(__file__).parent / "plots" / "training" / f"training_plot_{category}_{training_timestamp}_from_csv.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Plot saved: {plot_path}")


