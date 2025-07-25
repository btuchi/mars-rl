"""Main training script for diffusion PPO (cleaned up and modular)"""

import torch
import numpy as np
import time
import os
from pathlib import Path
import traceback

from .core.trajectory import DiffusionSampler
from .training.agent import DiffusionPPOAgent
from .utils.device import setup_h100_optimizations, get_device_info, clear_gpu_cache
from .utils.logging import initialize_logger, log_episode, log_update, finalize_logging, ACTOR_LOSS_LOG, CRITIC_LOSS_LOG, BEST_REWARD_LOG, VALUE_PREDICTION_LOG, RETURN_LOG, LOG_PROB_LOG
from .utils.visualization import plot_diffusion_training, plot_from_csv
from .utils.constants import *

def main(category: str = DEFAULT_CATEGORY):
    """Main training loop with robust CSV logging"""
    
    # Setup optimizations and get device info
    setup_h100_optimizations()
    device = get_device_info()
    
    # Create training timestamp
    timestamp = time.strftime("%Y%m%d%H%M%S")
    
    print(f"=== DIFFUSION PPO TRAINING - {timestamp} ===")
    print(f"🎯 Training Category: {category}")

    # Get training mode from constants for logging
    from .utils.constants import DEFAULT_TRAINING_MODE
    
    # Initialize CSV logger with training mode
    logger = initialize_logger(timestamp, category, DEFAULT_TRAINING_MODE)
    
    try:
        # Load reference features
        try:
            current_path = Path(__file__).parent
            npz_data = np.load(current_path / "reference_features" / f"reference_{category}_features_v2.npz")
            array_keys = list(npz_data.keys())
            
            # Stack all individual feature vectors into a single array
            ref_features_list = []
            for key in array_keys:
                ref_features_list.append(npz_data[key])
            
            ref_features = np.stack(ref_features_list)
            print(f"Loaded reference features: {ref_features.shape}")
            npz_data.close()
        except FileNotFoundError:
            print("Error: Cannot train without reference features!")
            return
        except Exception as e:
            print(f"Error loading reference features: {e}")
            return
        
        # Load reference images for MI calculation
        ref_images = None
        if DEFAULT_REWARD_METRIC in ["MI", "MMD_MI"]:
            try:
                images_npz_data = np.load(current_path / "reference_features" / f"reference_{category}_images.npz")
                images_keys = list(images_npz_data.keys())
                
                # Stack all individual images into a single array
                ref_images_list = []
                for key in images_keys:
                    ref_images_list.append(images_npz_data[key])
                
                ref_images = np.stack(ref_images_list)
                print(f"Loaded reference images: {ref_images.shape}")
                images_npz_data.close()
            except FileNotFoundError:
                print(f"⚠️ Warning: Reference images not found for {DEFAULT_REWARD_METRIC} metric!")
                print(f"⚠️ Expected: {current_path}/reference_features/reference_{category}_images.npz")
                print(f"⚠️ Run: python create_reference_images_npz.py --category {category}")
                print(f"⚠️ MI component will be disabled for MMD_MI metric")
            except Exception as e:
                print(f"⚠️ Warning: Error loading reference images: {e}")
                print(f"⚠️ MI component will be disabled for MMD_MI metric")
        
        # Initialize diffusion sampler
        print("Initializing diffusion sampler...")
        sampler = DiffusionSampler(device=device, use_fp16=USE_FP16)
        print(f"Sampler dtype: {sampler.dtype}")
        
        # Initialize PPO agent
        feature_dim = ref_features.shape[1] if len(ref_features.shape) > 1 else 512
        agent = DiffusionPPOAgent(
            sampler=sampler,
            ref_features=ref_features,
            batch_size=DEFAULT_BATCH_SIZE,
            feature_dim=feature_dim,
            num_inference_steps=DEFAULT_NUM_INFERENCE_STEPS,
            images_per_prompt=DEFAULT_IMAGES_PER_PROMPT,
            training_start=timestamp,
            ref_images=ref_images
        )
        
        # Load prompts from prompts folder
        train_prompts_file = current_path / "prompts" / "train" / f"{category}.txt"
        
        # Read prompts from file
        train_prompts = []
        if train_prompts_file.exists():
            with open(train_prompts_file, 'r') as f:
                train_prompts = [line.strip() for line in f if line.strip()]
        else:
            print(f"❌ Training prompts file not found: {train_prompts_file}")
            return

        if not train_prompts:
            print(f"❌ No training prompts found in file: {train_prompts_file}")
            return

        print(f"Loaded {len(train_prompts)} training prompts from {train_prompts_file}")

        # Training tracking
        reward_buffer = np.empty(shape=DEFAULT_NUM_EPISODES)
        best_reward = -float('inf')
        episodes_since_update = 0
        recent_rewards = []
        update_counter = 0
        
        print("🚀 Starting Diffusion PPO training...")
        # print(f"📊 CSV logs will be saved every {logger.save_frequency} episodes")
        # print(f"Episodes per update: {DEFAULT_EPISODES_PER_UPDATE}")
        print(f"Images per episode: {DEFAULT_BATCH_SIZE} (batch generation)")
        print(f"Total episodes: {DEFAULT_NUM_EPISODES}")
        
        # MAIN TRAINING LOOP
        for episode_i in range(DEFAULT_NUM_EPISODES):
            print(f"\n=== Episode {episode_i+1}/{DEFAULT_NUM_EPISODES} ===")
            
            # Sample random prompt
            prompt = np.random.choice(train_prompts)
            print(f"Prompt: '{prompt}'")
            
            # Generate batch of images for this prompt
            trajectories, individual_rewards, avg_reward, prompt_features = agent.generate_batch_for_prompt(
                prompt=prompt, episode=episode_i
            )
            
            # Track episode reward (average of batch)
            episode_reward = avg_reward
            recent_rewards.append(episode_reward)
            episodes_since_update += 1
            reward_buffer[episode_i] = episode_reward
            
            print(f"Episode reward: {episode_reward:.4f}")
            print(f"Individual rewards: {individual_rewards}")

            # Track best reward
            if episode_reward > best_reward:
                best_reward = episode_reward
                agent.save_policy()
                print(f"🎉 New best reward: {best_reward:.4f}")
            
            BEST_REWARD_LOG.append(best_reward)
            
            # LOG EPISODE DATA TO CSV
            log_episode(
                episode=episode_i+1,
                prompt=prompt,
                individual_rewards=individual_rewards.tolist(),
                avg_reward=episode_reward,
                best_reward=best_reward
            )

            # Update policy when we have enough episodes
            if episodes_since_update >= DEFAULT_EPISODES_PER_UPDATE:
                print(f"\n🔄 Performing PPO update after {episodes_since_update} episodes...")
                print(f"Replay buffer size: {len(agent.replay_buffer.trajectories)} trajectories")
                
                # Perform PPO update
                actor_loss, critic_loss, values, returns, gradient_info = agent.update()
                episodes_since_update = 0
                update_counter += 1
                
                # Log update data if we got valid losses
                if actor_loss is not None and critic_loss is not None:
                    ACTOR_LOSS_LOG.append(actor_loss)
                    CRITIC_LOSS_LOG.append(critic_loss)
                    
                    # Log values and returns to CSV
                    for value in values.flatten():
                        logger.log_value_prediction(float(value))
                    
                    for return_val in returns.flatten():
                        logger.log_return(float(return_val))
                    
                    # LOG UPDATE DATA TO CSV
                    log_update(
                        update_num=update_counter,
                        actor_loss=actor_loss,
                        critic_loss=critic_loss,
                        episode=episode_i+1,
                        gradient_info=gradient_info
                    )

                    print(f"  ✅ Actor Loss: {actor_loss:.4f}")
                    print(f"  ✅ Critic Loss: {critic_loss:.4f}")
                    
                    avg_recent = np.mean(recent_rewards[-20:]) if len(recent_rewards) >= 20 else np.mean(recent_rewards)
                    print(f"  ✅ Avg Reward (last 20): {avg_recent:.4f}")
            
            # Progress logging every 10 episodes
            if episode_i % 10 == 0 or episode_i == DEFAULT_NUM_EPISODES - 1:
                avg_reward_recent = np.mean(recent_rewards[-10:]) if len(recent_rewards) >= 10 else np.mean(recent_rewards)
                print(f"📊 Progress - Episode {episode_i}: Current: {episode_reward:.4f}, Avg(10): {avg_reward_recent:.4f}, Best: {best_reward:.4f}")
            
            # Clear GPU memory periodically
            if episode_i % 10 == 0:
                clear_gpu_cache()
        
        # TRAINING COMPLETED
        print(f"\n🏁 Training completed after {DEFAULT_NUM_EPISODES} episodes!")
        final_avg = np.mean(recent_rewards[-20:]) if len(recent_rewards) >= 20 else np.mean(recent_rewards)
        print(f"Final average reward: {final_avg:.4f}")
        print(f"Best reward achieved: {best_reward:.4f}")
        print(f"Total PPO updates: {len(ACTOR_LOSS_LOG)}")

        
        plot_diffusion_training(
            reward_buffer, ACTOR_LOSS_LOG, CRITIC_LOSS_LOG, BEST_REWARD_LOG,
            VALUE_PREDICTION_LOG, RETURN_LOG, LOG_PROB_LOG, DEFAULT_NUM_EPISODES-1, 
            category=category, timestamp=timestamp, training_mode=agent.actor.training_mode
        )

    except KeyboardInterrupt:
        print(f"\n⚠️ Training interrupted by user (Ctrl+C)")
        
    except Exception as e:
        tb = traceback.format_exc()
        print(f"\n❌ Training interrupted by error: {e}")
        print(f"Traceback:\n{tb}")
        
    finally:
        # This will ALWAYS run, even if training is interrupted
        print(f"\n💾 Finalizing logs...")
        finalize_logging()
        plot_from_csv(timestamp, category)
        
        # Generate feature distribution plots
        print(f"\n🎨 Generating feature distribution plots...")
        try:
            from .utils.visualization import plot_feature_distributions
            success = plot_feature_distributions(timestamp, category)
            if success:
                print(f"✅ Feature distribution plots saved to: outputs/plots/feature_distribution/{timestamp}/")
            else:
                print(f"⚠️ Feature distribution plots failed - may need more training data or scikit-learn")
        except Exception as e:
            print(f"⚠️ Could not generate feature distribution plots: {e}")


if __name__ == "__main__":
    category = DEFAULT_CATEGORY
    main(category)
    print("\nDiffusion PPO training completed!")
