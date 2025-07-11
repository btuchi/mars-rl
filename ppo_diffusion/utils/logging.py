"""Centralized logging functionality (cleaned up from diffusion_log_utils.py)"""

import pandas as pd
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

def finalize_logging():
    """Complete logging and save final files"""
    if _logger:
        _logger.save_all_logs(final=True)

