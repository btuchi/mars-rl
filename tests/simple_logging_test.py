import os
import time
import pandas as pd
import sys
from pathlib import Path
# Add the project root directory to sys.path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from ppo_diffusion.diffusion_log_utils import initialize_logger, log_episode, log_update, log_value_prediction, log_return, finalize_logging

# Test parameters
NUM_EPISODES = 10  # Small number of episodes for testing
CATEGORY = "test_category"
TIMESTAMP = time.strftime("%Y%m%d%H%M%S")

def test_training():
    """Simple test script to verify logging and saving"""
    print("🚀 Starting test training...")
    
    # Initialize logger
    initialize_logger(training_timestamp=TIMESTAMP, category=CATEGORY)
    
    try:
        for episode in range(1, NUM_EPISODES + 1):
            # Simulate episode data
            prompt = f"Test prompt {episode}"
            individual_rewards = [1.0, 2.0, 3.0, 4.0]
            avg_reward = sum(individual_rewards) / len(individual_rewards)
            best_reward = max(individual_rewards)
            
            # Log episode
            log_episode(episode, prompt, individual_rewards, avg_reward, best_reward)
            
            print(f"✅ Episode {episode} logged successfully.")
            
            # Simulate PPO update every few episodes
            if episode % 2 == 0:  # Perform PPO update every 2 episodes
                actor_loss = 0.01 * episode
                critic_loss = 0.02 * episode
                value_prediction = avg_reward + 0.5
                return_value = avg_reward * 0.9
                
                log_update(update_num=episode // 2, actor_loss=actor_loss, critic_loss=critic_loss, episode=episode)
                log_value_prediction(value_prediction=value_prediction)
                log_return(return_value=return_value)
                
                print(f"✅ PPO update logged for episode {episode}.")
        
        print("🎉 Test training completed successfully.")
    
    except Exception as e:
        print(f"❌ Test training interrupted by error: {e}")
    
    finally:
        # Finalize logging
        finalize_logging()
        print("💾 Logs finalized.")
        
        # Validate value predictions and returns
        validate_logs()

def validate_logs():
    """Validate the value predictions and returns logging"""
    logs_dir = Path(__file__).parent.parent / "ppo_diffusion" / "logs" / f"{CATEGORY}_{TIMESTAMP}"
    
    # Validate value predictions
    value_csv = logs_dir / "value_predictions.csv"
    if value_csv.exists():
        value_data = pd.read_csv(value_csv)
        print(f"✅ Value predictions logged: {len(value_data)} entries")
        print(value_data.head())
    else:
        print(f"❌ Value predictions CSV not found: {value_csv}")
    
    # Validate returns
    return_csv = logs_dir / "returns.csv"
    if return_csv.exists():
        return_data = pd.read_csv(return_csv)
        print(f"✅ Returns logged: {len(return_data)} entries")
        print(return_data.head())
    else:
        print(f"❌ Returns CSV not found: {return_csv}")

if __name__ == "__main__":
    test_training()