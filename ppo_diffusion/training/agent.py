# File: training/agent.py
"""Main PPO agent for diffusion training (cleaned up)"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from pathlib import Path
import time
from PIL import Image
import torchvision.transforms as transforms
from typing import List, Tuple
import gc

from ..core.trajectory import DiffusionSampler, DiffusionTrajectory
from ..models.policy import DiffusionPolicyNetwork
from ..models.value import DiffusionValueNetwork
from ..core.features import FeatureExtractor
from .memory import DiffusionReplayMemory
from .rewards import DiffusionRewardFunction
from ..utils.device import clear_gpu_cache
from ..utils.constants import *


class DiffusionPPOAgent:
    """PPO Agent for Diffusion Models - cleaned up and modular"""
    
    def __init__(self, sampler: DiffusionSampler, ref_features: np.ndarray, batch_size: int, 
                 feature_dim: int = 512, num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
                 images_per_prompt: int = DEFAULT_IMAGES_PER_PROMPT, 
                 save_samples: bool = True, training_start: str = None):
        
        self.dtype = sampler.dtype if hasattr(sampler, 'dtype') else torch.float32
        self.device = sampler.device
        self.training_start = training_start

        print(f"PPO Agent using dtype: {self.dtype}")
        
        # PPO hyperparameters
        self.LR_ACTOR = DEFAULT_LR_ACTOR
        self.LR_CRITIC = DEFAULT_LR_CRITIC
        self.GAMMA = DEFAULT_GAMMA
        self.LAMBDA = DEFAULT_LAMBDA
        self.EPOCH = 1
        self.EPSILON_CLIP = DEFAULT_EPSILON_CLIP
        self.images_per_prompt = images_per_prompt
        
        # Initialize centralized feature extractor (single CLIP instance)
        self.feature_extractor = FeatureExtractor(device=self.device)
        
        # Initialize networks
        self.actor = DiffusionPolicyNetwork(sampler, num_inference_steps)
        self.old_actor = DiffusionPolicyNetwork(sampler, num_inference_steps)
        self.critic = DiffusionValueNetwork(feature_dim).to(self.device).to(self.dtype)

        # Handle DataParallel case for optimizers
        # if hasattr(self.actor.unet, 'module'):
        #     actor_params = self.actor.unet.module.parameters()
        # else:
        #     actor_params = self.actor.unet.parameters()
        
        # Optimizers
        self.actor_optimizer = optim.AdamW(
            self.actor.diversity_policy.parameters(),
            lr=self.LR_ACTOR,
            weight_decay=1e-4
        )
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.LR_CRITIC)
        print(f"🎯 Only training {sum(p.numel() for p in self.actor.diversity_policy.parameters())} policy parameters")
        
        # Components
        self.replay_buffer = DiffusionReplayMemory(batch_size)
        self.reward_function = DiffusionRewardFunction(ref_features, self.feature_extractor)

        # Image saving parameters
        self.save_size = (64, 64)
        self.save_quality = 85
        self.save_samples = save_samples
        
        if self.save_samples:
            self.setup_sample_saving()
    
    def setup_sample_saving(self):
        """Set up directories and tracking for sample image saving"""
        current_path = Path(__file__).parent.parent
        self.samples_dir = current_path / f"outputs/images/while_training/{self.training_start}"
        os.makedirs(self.samples_dir, exist_ok=True)
        
        self.create_training_metadata()
        
        print(f"📁 Sample images will be saved to: {self.samples_dir}")
        print(f"🕐 Training timestamp: {self.training_start}")

    def create_training_metadata(self):
        """Create a metadata file with training information"""
        metadata_path = self.samples_dir / "training_info.txt"
        
        with open(metadata_path, 'w') as f:
            f.write(f"Training Session: {self.training_start}\n")
            f.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Images per prompt: {self.images_per_prompt}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Dtype: {self.dtype}\n")
            f.write("="*50 + "\n")
            f.write("Episode Log:\n")
        
        print(f"📝 Created training metadata: {metadata_path}")
    
    def log_episode_info(self, episode: int, prompt: str, avg_reward: float, individual_rewards: list):
        """Log episode information to metadata file"""
        metadata_path = self.samples_dir / "training_info.txt"
        
        try:
            with open(metadata_path, 'a') as f:
                f.write(f"Ep {episode:04d}: '{prompt}' | Avg: {avg_reward:.4f} | Rewards: {individual_rewards}\n")
        except Exception as e:
            print(f"⚠️ Could not log episode info: {e}")

    def save_trajectory_image(self, trajectory, prompt: str, episode: int, image_idx: int, reward: float = None):
        """Save a single trajectory image with metadata"""
        try:
            # Convert trajectory to PIL image
            final_image = trajectory.final_image.squeeze(0).cpu()
            final_image = torch.clamp(final_image, 0, 1)
            
            to_pil = transforms.ToPILImage()
            pil_image = to_pil(final_image)

            # Resize to exactly 64x64
            resized_image = pil_image.resize(
                self.save_size,
                Image.Resampling.LANCZOS
            )
            
            # Create descriptive filename
            safe_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_prompt = safe_prompt[:30]
            
            reward_str = f"_r{reward:.3f}" if reward is not None else ""
            filename = f"ep{episode:04d}_img{image_idx}_{safe_prompt}{reward_str}.png"
            
            save_path = self.samples_dir / filename

            # Save as JPEG with fixed quality
            resized_image.save(
                save_path, 
                format='JPEG',
                quality=self.save_quality,
                optimize=True
            )
            
            print(f"  💾 Saved sample: {filename}")
            return str(save_path)
            
        except Exception as e:
            print(f"  ⚠️ Could not save sample image: {e}")
            return None
    
    def generate_batch_for_prompt(self, prompt: str, episode: int = None, save_samples: bool = None) -> Tuple[List[DiffusionTrajectory], np.ndarray, float, np.ndarray]:
        """Generate a batch of images for a single prompt and calculate individual rewards"""
        clear_gpu_cache()

        # Determine if we should save samples this episode
        should_save = False
        if save_samples is not None:
            should_save = save_samples
        elif self.save_samples and episode is not None:
            should_save = (episode % 5 == 0)  # Save every 5 episodes

        # Get prompt features and value using centralized feature extractor
        prompt_features = self.feature_extractor.extract_text_features(prompt)
        clear_gpu_cache()

        prompt_features_tensor = torch.from_numpy(prompt_features).to(
            device=self.device,
            dtype=self.dtype
        ).unsqueeze(0)

        value = self.critic(prompt_features_tensor).detach().cpu().numpy()[0][0]

        # Clean up prompt tensor
        del prompt_features_tensor
        clear_gpu_cache()
            
        # Generate batch of trajectories
        trajectories = []
        log_probs = []
        log_prob_tensors = []
        
        print(f"Generating {self.images_per_prompt} images for prompt: '{prompt}'")
        for i in range(self.images_per_prompt):
            clear_gpu_cache()

            trajectory, log_prob_tensor = self.actor.select_trajectory(prompt)

            # Store and then immediately clear the trajectory's computation graph
            log_prob_value = log_prob_tensor.item()
            log_prob_detached = log_prob_tensor.detach()

            # Clear the trajectory's gradient connection after storing what we need
            if hasattr(trajectory, 'total_log_prob') and trajectory.total_log_prob is not None:
                trajectory.total_log_prob = trajectory.total_log_prob.detach().requires_grad_(True)
            elif hasattr(trajectory, 'policy_log_prob') and trajectory.policy_log_prob is not None:
                trajectory.policy_log_prob = trajectory.policy_log_prob.detach().requires_grad_(True)

            trajectories.append(trajectory)
            log_probs.append(log_prob_value)
            log_prob_tensors.append(log_prob_detached.requires_grad_(True))
            
            print(f"  Image {i+1}/{self.images_per_prompt} generated")

            # Save sample image if needed
            if should_save and i == 0:  # Save first image of batch
                try:
                    features = self.feature_extractor.extract_trajectory_features(trajectory)
                    features = features.reshape(1, -1)
                    from ..diversity_reward import calculate_individual_diversity_rewards
                    individual_reward = calculate_individual_diversity_rewards(
                        features, self.reward_function.ref_features, gamma=None
                    )[0]
                    individual_reward = self.reward_function.normalize_reward(individual_reward)
                except:
                    individual_reward = None
                
                self.save_trajectory_image(trajectory, prompt, episode, i+1, individual_reward)

            # Clean up trajectory steps to save memory
            if hasattr(trajectory, 'steps'):
                for step in trajectory.steps:
                    if hasattr(step, 'state'):
                        step.state = None
                    if hasattr(step, 'noise_pred'):
                        step.noise_pred = None
            
            del trajectory, log_prob_tensor
            clear_gpu_cache()
        
        # Calculate individual diversity rewards
        individual_rewards = self.reward_function.calculate_batch_rewards(trajectories, prompt)
        avg_reward = np.mean(individual_rewards)
        
        print(f"  Individual rewards: {individual_rewards}")
        print(f"  Average reward: {avg_reward:.4f}")

        # Log episode information
        if episode is not None:
            self.log_episode_info(episode, prompt, avg_reward, individual_rewards.tolist())
        
        # Store each trajectory with its individual reward
        for i, (trajectory, log_prob, log_prob_tensor, reward) in enumerate(zip(trajectories, log_probs, log_prob_tensors, individual_rewards)):
            self.replay_buffer.add_memo(
                prompt_features,
                prompt,
                trajectory,
                reward,
                value,
                log_prob,
                log_prob_tensor
            )
        
        return trajectories, individual_rewards, avg_reward, prompt_features

    def compute_gae(self, rewards, values):
        """GAE computation for diffusion models"""
        advantages = []
        gae = 0
        
        # For diffusion, each trajectory is independent, so simplified GAE
        for step in reversed(range(len(rewards))):
            next_value = 0  # Always terminal
            next_non_terminal = 0  # Always terminal
            
            delta = rewards[step] + self.GAMMA * next_value * next_non_terminal - values[step]
            gae = delta + self.GAMMA * self.LAMBDA * next_non_terminal * gae
            advantages.insert(0, gae)
        
        return np.array(advantages)

    def update(self):
        """PPO update for diffusion models"""
        print("Checking replay buffer...")
        if len(self.replay_buffer.trajectories) == 0:
            print("Skipping update: empty replay buffer")
            return
        
        print("Starting PPO update...")
        clear_gpu_cache()

        # Copy current actor to old_actor
        # if hasattr(self.actor.unet, 'module') and hasattr(self.old_actor.unet, 'module'):
        #     self.old_actor.unet.module.load_state_dict(self.actor.unet.module.state_dict())
        # elif hasattr(self.actor.unet, 'module'):
        #     self.old_actor.unet.load_state_dict(self.actor.unet.module.state_dict())
        # else:
        #     self.old_actor.unet.load_state_dict(self.actor.unet.state_dict())
        self.old_actor.diversity_policy.load_state_dict(self.actor.diversity_policy.state_dict())
        
        # Get trajectory data
        memo_features, memo_prompts, memo_trajectories, memo_rewards, memo_values, memo_log_probs, memo_log_prob_tensors, batches = self.replay_buffer.sample()
        
        print(f"🔍 Rewards: {memo_rewards}")
        print(f"🔍 Values: {memo_values}")
        print(f"🔍 Raw advantages (R-V): {memo_rewards - memo_values}")

        # Compute advantages using GAE
        memo_advantages = self.compute_gae(memo_rewards, memo_values)
        memo_returns = memo_advantages + memo_values

        # Convert to tensors
        memo_features_tensor = torch.from_numpy(np.array(memo_features)).to(
            device=self.device, 
            dtype=self.dtype
        )
        memo_advantages_tensor = torch.tensor(
            memo_advantages, 
            dtype=self.dtype, 
            device=self.device
        )
        memo_returns_tensor = torch.tensor(
            memo_returns, 
            dtype=self.dtype, 
            device=self.device
        )
        
        # Accumulate losses
        all_actor_losses = []
        all_critic_losses = []

        # Train for multiple epochs
        for epoch_i in range(self.EPOCH):
            for batch in batches:
                if len(batch) == 0:
                    continue
                
                # Get current log probs
                current_log_probs = []
                for idx in batch:
                    trajectory = memo_trajectories[idx]
                    prompt = memo_prompts[idx]

                    # Generate FRESH trajectory with current policy
                    fresh_trajectory, fresh_log_prob = self.actor.select_trajectory(prompt)
                    current_log_probs.append(fresh_log_prob)

                current_log_probs_tensor = torch.stack(current_log_probs)
                old_log_probs_batch = torch.tensor([memo_log_probs[idx] for idx in batch], 
                                          dtype=self.dtype, device=self.device)
    
                # Calculate ratio
                ratio = torch.exp(current_log_probs_tensor - old_log_probs_batch)

                # Analyze the ratio distribution
                ratio_mean = ratio.mean().item()
                ratio_max = ratio.max().item()
                ratio_min = ratio.min().item()
                
                print(f"🔍 Ratio stats: mean={ratio_mean:.3f}, min={ratio_min:.3f}, max={ratio_max:.3f}")
                
                # # Check if we're hitting PPO clipping frequently
                # clipped_ratios = torch.clamp(ratio, 1 - self.EPSILON_CLIP, 1 + self.EPSILON_CLIP)
                # clipping_rate = (ratio != clipped_ratios).float().mean().item()
                
                # print(f"🔍 PPO clipping rate: {clipping_rate:.2%}")
                
                # if clipping_rate > 0.5:
                #     print("⚠️ High clipping rate - policy changing too fast!")
                
                # Batch advantages
                batch_advantages = memo_advantages_tensor[batch]
                print("Batch advantages mean:", batch_advantages.mean().item())
                
                # PPO clipped objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.EPSILON_CLIP, 1 + self.EPSILON_CLIP) * batch_advantages

                # print("surr1 mean:", surr1.mean().item())
                # print("surr2 mean:", surr2.mean().item())
                
                # Actor loss
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Critic loss
                batch_values = self.critic(memo_features_tensor[batch]).squeeze()
                batch_returns = memo_returns_tensor[batch]

                batch_values = batch_values.to(self.dtype)
                batch_returns = batch_returns.to(self.dtype)
                
                critic_loss = nn.MSELoss()(batch_values, batch_returns)
                
                # Store losses
                all_actor_losses.append(actor_loss.item())
                all_critic_losses.append(critic_loss.item())
                
                # Update actor (UNet)
                self.actor_optimizer.zero_grad()

                print(f"Actor loss requires_grad: {actor_loss.requires_grad}")
                print(f"Actor loss value: {actor_loss.item()}")

                actor_loss.backward()

                # total_grad_norm = 0
                # param_count = 0
                # for param in self.actor.unet.parameters():
                #     if param.grad is not None:
                #         total_grad_norm += param.grad.norm().item()
                #         param_count += 1
                # print(f"Total gradient norm: {total_grad_norm}, Params with gradients: {param_count}")

                # Handle gradient clipping for diversity policy (not UNet)
                torch.nn.utils.clip_grad_norm_(self.actor.diversity_policy.parameters(), max_norm=0.5)
                
                self.actor_optimizer.step()

                # Update critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()

                total_norm = 0
                for name, param in self.critic.named_parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                print(f"🔍 Critic gradient norm: {total_norm:.6f}")

                actor_grad_norm = 0
                for param in self.actor.diversity_policy.parameters():
                    if param.grad is not None:
                        actor_grad_norm += param.grad.data.norm(2).item() ** 2
                actor_grad_norm = actor_grad_norm ** 0.5
                print(f"🔍 Actor (diversity policy) gradient norm: {actor_grad_norm:.6f}")

                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.critic_optimizer.step()

                # Clear gradients and memory after each batch
                del current_log_probs, fresh_trajectory, fresh_log_prob
                clear_gpu_cache()
                    
        # Return losses for logging
        if all_actor_losses:
            avg_actor_loss = np.mean(all_actor_losses)
            avg_critic_loss = np.mean(all_critic_losses)
            
            print(f"Update - Actor Loss: {avg_actor_loss:.4f}, "
                  f"Critic Loss: {avg_critic_loss:.4f}, "
                  f"Avg Advantage: {memo_advantages.mean():.4f}")
            
            # Clear buffer
            self.replay_buffer.clear_memo()
            
            return avg_actor_loss, avg_critic_loss, memo_values, memo_returns
        
        # Clear buffer even if no losses
        self.replay_buffer.clear_memo()
        

        # # Weight monitoring
        # unet = self.actor.unet.module if hasattr(self.actor.unet, 'module') else self.actor.unet
        # max_weight = max(p.abs().max().item() for p in unet.parameters())
        # weight_norm = sum(p.norm().item() for p in unet.parameters())
        
        # print(f"🔍 Max UNet weight: {max_weight:.6f}")
        # print(f"🔍 Total weight norm: {weight_norm:.6f}")
        
        # # Emergency reset if weights explode
        # if max_weight > 100.0 or torch.isnan(torch.tensor(max_weight)):
        #     print("🚨 UNet weights exploded! Consider resetting or lower LR")
        return None, None, None, None
    
    def save_policy(self):
        """Save the trained diversity policy"""
        models_dir = Path(__file__).parent.parent / "outputs" / "models"
        models_dir.mkdir(exist_ok=True)
        policy_path = models_dir / f"{DEFAULT_CATEGORY}_diversity_policy_{self.training_start}.pth"

        # Save diversity policy state dict
        state_dict = self.actor.diversity_policy.state_dict()
        torch.save(state_dict, policy_path)
        print(f"Diversity policy saved to: {policy_path}")