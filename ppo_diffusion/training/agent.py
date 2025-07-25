# File: training/agent.py
"""Main PPO agent for diffusion training (cleaned up)"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from pathlib import Path
import time

# PHASE 1: Enable gradient anomaly detection for debugging
torch.autograd.set_detect_anomaly(True)
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
from ..utils.logging import log_log_probability, log_generated_features


class DiffusionPPOAgent:
    """PPO Agent for Diffusion Models - cleaned up and modular"""
    
    def __init__(self, sampler: DiffusionSampler, ref_features: np.ndarray, batch_size: int, 
                 feature_dim: int = 512, num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
                 images_per_prompt: int = DEFAULT_IMAGES_PER_PROMPT, 
                 save_samples: bool = True, training_start: str = None, ref_images: np.ndarray = None):
        
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
        self.ENTROPY_COEFF = DEFAULT_ENTROPY_COEFF
        self.images_per_prompt = images_per_prompt
        
        # Initialize centralized feature extractor (ResNet-18)
        self.feature_extractor = FeatureExtractor(device=self.device)
        
        # Initialize networks
        self.actor = DiffusionPolicyNetwork(sampler, num_inference_steps)
        self.old_actor = DiffusionPolicyNetwork(sampler, num_inference_steps)
        self.critic = DiffusionValueNetwork(feature_dim).to(self.device).to(self.dtype)

        # Optimizers - use switchable parameter selection
        trainable_params = list(self.actor.get_trainable_parameters())
        self.actor_optimizer = optim.AdamW(
            trainable_params,  # Mode-aware parameter selection
            lr=self.LR_ACTOR,
            weight_decay=1e-4
        )
        # PHASE 2: Switch critic from Adam to AdamW for better weight decay handling
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=self.LR_CRITIC, weight_decay=1e-3)
        # Mode-aware parameter counting and reporting
        trainable_params = sum(p.numel() for p in self.actor.get_trainable_parameters())
        print(f"🎯 Training {trainable_params:,} parameters in {self.actor.training_mode} mode")
        
        from ..utils.constants import DEFAULT_REWARD_METRIC
        print(f"🔍 Using {DEFAULT_REWARD_METRIC} reward metric (configurable in constants.py)")
        
        # Components
        self.replay_buffer = DiffusionReplayMemory(batch_size)
        # Use configurable reward metric from constants
        self.reward_function = DiffusionRewardFunction(
            ref_features, 
            self.feature_extractor, 
            reward_metric=DEFAULT_REWARD_METRIC,
            ref_images=ref_images
        )

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

        # Generate batch of trajectories
        trajectories = []
        log_probs = []
        log_prob_tensors = []
        first_image_features = None
        
        print(f"Generating {self.images_per_prompt} images for prompt: '{prompt}'")
        for i in range(self.images_per_prompt):
            clear_gpu_cache()

            trajectory, log_prob_tensor = self.actor.select_trajectory(prompt)

            # Store log prob value for memory buffer, but keep gradient connection
            log_prob_value = log_prob_tensor.item()
            
            # Log the log probability for monitoring
            log_log_probability(log_prob_value, episode if episode is not None else 0, i)

            # Keep gradient connection intact for policy training
            # Note: We need gradients to flow back to the diversity policy
            if hasattr(trajectory, 'total_log_prob') and trajectory.total_log_prob is not None:
                # Keep the gradient connection for policy learning
                pass
            elif hasattr(trajectory, 'policy_log_prob') and trajectory.policy_log_prob is not None:
                # Keep the gradient connection for policy learning  
                pass

            trajectories.append(trajectory)
            log_probs.append(log_prob_value)
            log_prob_tensors.append(log_prob_tensor)  # Keep gradient connection intact
            
            # Extract features from first trajectory for critic
            # CRITICAL FIX: Detach features from actor computation graph to prevent critic contamination
            if i == 0:
                with torch.no_grad():  # Ensure no gradients flow through feature extraction
                    # Create a detached copy of the image for feature extraction
                    detached_image = trajectory.final_image.clone().detach()
                    # Create a temporary trajectory with detached image
                    from ..core.trajectory import DiffusionTrajectory
                    detached_trajectory = DiffusionTrajectory(
                        steps=[],  # Empty steps - we only need the final image
                        final_image=detached_image,
                        condition=trajectory.condition.clone().detach() if trajectory.condition is not None else None
                    )
                    first_image_features = self.feature_extractor.extract_trajectory_features(detached_trajectory)
            
            print(f"  Image {i+1}/{self.images_per_prompt} generated (log_prob: {log_prob_value:.6f})")

            # Save sample image if needed
            if should_save and i == 0:  # Save first image of batch
                try:
                    # CRITICAL FIX: Use detached features for sample evaluation too
                    with torch.no_grad():
                        detached_image = trajectory.final_image.clone().detach()
                        from ..core.trajectory import DiffusionTrajectory
                        detached_trajectory = DiffusionTrajectory(
                            steps=[],
                            final_image=detached_image,
                            condition=trajectory.condition.clone().detach() if trajectory.condition is not None else None
                        )
                        features = self.feature_extractor.extract_trajectory_features(detached_trajectory)
                    features = features.reshape(1, -1)
                    # Use the reward function's current metric for sample evaluation
                    individual_reward = self.reward_function.reward_metric.calculate_rewards(
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
            
            # Note: Keep trajectory and log_prob_tensor in memory for gradient flow
            clear_gpu_cache()
        
        # Calculate value from first image features
        image_features_tensor = torch.from_numpy(first_image_features).to(
            device=self.device,
            dtype=self.dtype
        ).unsqueeze(0)
        value = self.critic(image_features_tensor).detach().cpu().numpy()[0][0]
        del image_features_tensor
        clear_gpu_cache()
        
        # Calculate individual diversity rewards
        individual_rewards = self.reward_function.calculate_batch_rewards(trajectories, prompt)
        
        # Handle tensor vs numpy for average calculation
        if isinstance(individual_rewards, torch.Tensor):
            avg_reward = individual_rewards.mean().item()  # Convert to Python float
        else:
            avg_reward = np.mean(individual_rewards)
        
        # Log generated features for t-SNE visualization  
        if episode is not None:
            # Extract features from all trajectories for logging
            # CRITICAL FIX: Use detached features for logging to prevent computation graph contamination
            batch_features = []
            for trajectory in trajectories:
                with torch.no_grad():
                    detached_image = trajectory.final_image.clone().detach()
                    from ..core.trajectory import DiffusionTrajectory
                    detached_trajectory = DiffusionTrajectory(
                        steps=[],
                        final_image=detached_image,
                        condition=trajectory.condition.clone().detach() if trajectory.condition is not None else None
                    )
                    features = self.feature_extractor.extract_trajectory_features(detached_trajectory)
                    batch_features.append(features)
            
            batch_features_array = np.vstack([f.reshape(1, -1) for f in batch_features])
            log_generated_features(batch_features_array, episode, prompt)
            
            # Plot per-image feature distribution every 5 episodes
            if episode % 5 == 0:
                from ..utils.logging import plot_per_image_feature_distribution, plot_tsne_feature_space
                plot_per_image_feature_distribution(batch_features_array, episode, prompt, self.training_start)
                plot_tsne_feature_space(batch_features_array, episode, prompt, self.training_start)
        
        print(f"  Individual rewards: {individual_rewards}")
        print(f"  Average reward: {avg_reward:.4f}")
        
        # Diagnostic: Check reward signal quality
        if isinstance(individual_rewards, torch.Tensor):
            reward_std = individual_rewards.std().item()
            reward_range = (individual_rewards.max() - individual_rewards.min()).item()
        else:
            reward_std = np.std(individual_rewards)
            reward_range = np.max(individual_rewards) - np.min(individual_rewards)
        print(f"  🔍 Reward std: {reward_std:.6f}, range: {reward_range:.6f}")
        if reward_std < 0.01:
            print("  ⚠️ WARNING: Very low reward variance - weak learning signal!")

        # Log episode information
        if episode is not None:
            if isinstance(individual_rewards, torch.Tensor):
                rewards_list = individual_rewards.detach().cpu().numpy().tolist()
            else:
                rewards_list = individual_rewards.tolist()
            self.log_episode_info(episode, prompt, avg_reward, rewards_list)
        
        # Store each trajectory with its individual reward (use first image features for consistency)
        # Convert tensor rewards to numpy for storage in replay buffer
        if isinstance(individual_rewards, torch.Tensor):
            individual_rewards_for_storage = individual_rewards.detach().cpu().numpy()
        else:
            individual_rewards_for_storage = individual_rewards
            
        for i, (trajectory, log_prob, log_prob_tensor, reward) in enumerate(zip(trajectories, log_probs, log_prob_tensors, individual_rewards_for_storage)):
            self.replay_buffer.add_memo(
                first_image_features,  # Use image features instead of prompt features
                prompt,
                trajectory,
                reward,
                value,
                log_prob,
                log_prob_tensor
            )
        
        return trajectories, individual_rewards, avg_reward, first_image_features

    def compute_gae(self, rewards, values):
        """Simplified advantage computation for episodic tasks"""
        # For episodic tasks, advantage is simply reward - value
        advantages = rewards - values
        return advantages

    def update(self):
        """PPO update for diffusion models
        
        PHASE 1 DEBUGGING: Enhanced gradient anomaly detection active
        - torch.autograd.set_detect_anomaly(True) enabled globally
        - Detailed gradient anomaly monitoring for actor and critic
        - Checks for NaN, Inf, and zero gradients
        - Monitors gradient explosion/vanishing with thresholds
        """
        print("Checking replay buffer...")
        if len(self.replay_buffer.trajectories) == 0:
            print("Skipping update: empty replay buffer")
            return
        
        print("Starting PPO update...")
        print("🔍 [PHASE 1] Gradient anomaly detection ACTIVE")
        clear_gpu_cache()

        # Copy current actor to old_actor (mode-aware)
        if self.actor.training_mode == "DIVERSITY_POLICY":
            self.old_actor.diversity_policy.load_state_dict(self.actor.diversity_policy.state_dict())
        elif self.actor.training_mode == "LORA_UNET":
            # For LoRA mode, we need to copy the LoRA adapter weights
            try:
                from peft import get_peft_model_state_dict, set_peft_model_state_dict
                lora_state_dict = get_peft_model_state_dict(self.actor.unet)
                set_peft_model_state_dict(self.old_actor.unet, lora_state_dict)
                print("🔄 LoRA adapter weights copied to old_actor")
            except Exception as e:
                print(f"⚠️ Could not copy LoRA state: {e}")
                # Fallback: create fresh old_actor (less efficient but safe)
                from ..models.policy import DiffusionPolicyNetwork
                self.old_actor = DiffusionPolicyNetwork(self.actor.sampler, self.actor.num_inference_steps)
                print("🔄 Created fresh old_actor as fallback")
        
        # Get trajectory data
        memo_features, memo_prompts, memo_trajectories, memo_rewards, memo_values, memo_log_probs, memo_log_prob_tensors, batches = self.replay_buffer.sample()
        
        print(f"🔍 Rewards: {memo_rewards}")
        print(f"🔍 Values: {memo_values}")
        print(f"🔍 Raw advantages (R-V): {memo_rewards - memo_values}")

        # Compute advantages using GAE
        # Debug: Check for NaN values before GAE calculation
        print(f"🔍 DEBUG GAE inputs:")
        print(f"   memo_rewards: {memo_rewards} (has NaN: {np.isnan(memo_rewards).any()})")
        print(f"   memo_values: {memo_values} (has NaN: {np.isnan(memo_values).any()})")
        
        memo_advantages = self.compute_gae(memo_rewards, memo_values)
        memo_returns = memo_advantages + memo_values
        
        # Debug: Check for NaN values after GAE calculation
        print(f"🔍 DEBUG GAE outputs:")
        print(f"   memo_advantages: {memo_advantages} (has NaN: {np.isnan(memo_advantages).any()})")
        print(f"   memo_returns: {memo_returns} (has NaN: {np.isnan(memo_returns).any()})")

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
        
        # Initialize gradient info variables
        actor_grad_norm_before = 0.0
        actor_grad_norm_after = 0.0
        total_norm = 0.0

        # Train for multiple epochs
        for _ in range(self.EPOCH):
            for batch in batches:
                if len(batch) == 0:
                    continue
                
                # Get current log probs - different approach based on training mode
                current_log_probs = []
                for idx in batch:
                    if self.actor.training_mode == "SCHEDULER_POLICY":
                        # For scheduler policy, use stored log prob tensors that maintain gradients
                        # because the policy parameters affect the guidance scales used in the original trajectories
                        stored_log_prob_tensor = memo_log_prob_tensors[idx]
                        current_log_probs.append(stored_log_prob_tensor)
                    else:
                        # For other policies (diversity, LoRA), generate fresh trajectories as before
                        # because the policy directly modifies the generation process
                        prompt = memo_prompts[idx]
                        _, fresh_log_prob = self.actor.select_trajectory(prompt)
                        current_log_probs.append(fresh_log_prob)

                current_log_probs_tensor = torch.stack(current_log_probs)
                old_log_probs_batch = torch.tensor([memo_log_probs[idx] for idx in batch], 
                                          dtype=self.dtype, device=self.device)
    
                # Calculate ratio (clamped)
                ratio = torch.exp(torch.clamp(current_log_probs_tensor - old_log_probs_batch, -2, 2))

                # Analyze the ratio distribution
                ratio_mean = ratio.mean().item()
                ratio_max = ratio.max().item()
                ratio_min = ratio.min().item()
                
                # Batch advantages with normalization
                batch_advantages = memo_advantages_tensor[batch]
                
                # Normalize advantages to improve training stability
                if len(batch_advantages) > 1:
                    advantage_mean = batch_advantages.mean()
                    advantage_std = batch_advantages.std()
                    if advantage_std > 1e-8:  # Avoid division by zero
                        batch_advantages = (batch_advantages - advantage_mean) / (advantage_std + 1e-8)
                
                # PPO clipped objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.EPSILON_CLIP, 1 + self.EPSILON_CLIP) * batch_advantages

                # Calculate policy entropy for exploration bonus
                policy_entropy = self._calculate_policy_entropy(batch)
                
                # Actor loss with entropy bonus
                actor_loss = -torch.min(surr1, surr2).mean() - self.ENTROPY_COEFF * policy_entropy
                
                # Critic loss
                batch_values = self.critic(memo_features_tensor[batch])
                batch_returns = memo_returns_tensor[batch]

                # Ensure consistent shapes without squeezing away batch dimension
                if batch_values.dim() > 1:
                    batch_values = batch_values.squeeze(-1)  # Only squeeze last dimension
                if batch_returns.dim() == 0:
                    batch_returns = batch_returns.unsqueeze(0)  # Add batch dimension if missing
                
                batch_values = batch_values.to(self.dtype)
                batch_returns = batch_returns.to(self.dtype)
                
                # Debug critic inputs before loss calculation
                print(f"🔍 CRITIC DEBUG:")
                print(f"   batch_values: {batch_values} (has NaN: {torch.isnan(batch_values).any()})")
                print(f"   batch_returns: {batch_returns} (has NaN: {torch.isnan(batch_returns).any()})")
                print(f"   batch_values shape: {batch_values.shape}, batch_returns shape: {batch_returns.shape}")
                
                critic_loss = nn.MSELoss()(batch_values, batch_returns)
                print(f"🔍 CRITIC LOSS: {critic_loss} (has NaN: {torch.isnan(critic_loss)})")
                
                # Store losses for tracking
                all_actor_losses.append(actor_loss.item())
                all_critic_losses.append(critic_loss.item())
                
                # For scheduler policy, accumulate losses instead of immediate updates
                # to avoid in-place operation conflicts
                if self.actor.training_mode == "SCHEDULER_POLICY":
                    if not hasattr(self, '_accumulated_actor_loss'):
                        self._accumulated_actor_loss = actor_loss
                    else:
                        self._accumulated_actor_loss = self._accumulated_actor_loss + actor_loss
                else:
                    # For other policies, update immediately as before
                    self.actor_optimizer.zero_grad()

                    print(f"Actor loss requires_grad: {actor_loss.requires_grad}")
                    print(f"Actor loss value: {actor_loss.item()}")

                    # PHASE 1: Enhanced gradient anomaly detection during backward pass
                    print("🔍 [PHASE 1] Starting actor backward pass with anomaly detection...")
                    try:
                        actor_loss.backward()
                        print("🔍 [PHASE 1] ✅ Actor backward pass completed successfully")
                    except RuntimeError as e:
                        print(f"🔍 [PHASE 1] ❌ GRADIENT ANOMALY DETECTED in actor backward pass: {e}")
                        print(f"🔍 [PHASE 1] Actor loss value: {actor_loss.item()}")
                        print(f"🔍 [PHASE 1] Actor loss shape: {actor_loss.shape}")
                        raise e
                    # Calculate gradient norm BEFORE clipping for non-scheduler policies
                    actor_grad_norm_before = 0
                    nan_gradients = 0
                    inf_gradients = 0
                    zero_gradients = 0
                    
                    # Get trainable parameters based on mode
                    if self.actor.training_mode == "DIVERSITY_POLICY":
                        trainable_params = self.actor.diversity_policy.parameters()
                    elif self.actor.training_mode == "LORA_UNET":
                        trainable_params = list(filter(lambda p: p.requires_grad, self.actor.unet.parameters()))
                    else:
                        trainable_params = []
                    
                    for param in trainable_params:
                        if param.grad is not None:
                            # PHASE 1: Check for gradient anomalies
                            if torch.isnan(param.grad).any():
                                nan_gradients += 1
                            if torch.isinf(param.grad).any():
                                inf_gradients += 1
                            if param.grad.norm() == 0:
                                zero_gradients += 1
                                
                            actor_grad_norm_before += param.grad.data.norm(2).item() ** 2
                            
                    actor_grad_norm_before = actor_grad_norm_before ** 0.5
                    
                    # PHASE 1: Report gradient anomalies
                    if nan_gradients > 0 or inf_gradients > 0:
                        print(f"🔍 [PHASE 1] ⚠️ GRADIENT ANOMALIES DETECTED:")
                        print(f"🔍 [PHASE 1]   - NaN gradients: {nan_gradients} parameters")
                        print(f"🔍 [PHASE 1]   - Inf gradients: {inf_gradients} parameters")
                        print(f"🔍 [PHASE 1]   - Zero gradients: {zero_gradients} parameters")
                        print(f"🔍 [PHASE 1]   - Gradient norm: {actor_grad_norm_before:.6f}")
                    
                    if zero_gradients > 10:  # Threshold for concern
                        print(f"🔍 [PHASE 1] ⚠️ HIGH ZERO GRADIENT COUNT: {zero_gradients} parameters have zero gradients (potential vanishing gradient)")
                    
                    if actor_grad_norm_before > 100:  # Threshold for exploding gradients
                        print(f"🔍 [PHASE 1] ⚠️ EXPLODING GRADIENT DETECTED: norm={actor_grad_norm_before:.6f}")
                        
                    if actor_grad_norm_before < 1e-6:  # Threshold for vanishing gradients
                        print(f"🔍 [PHASE 1] ⚠️ VANISHING GRADIENT DETECTED: norm={actor_grad_norm_before:.6f}")
                    
                    # No gradient clipping - let natural gradients flow
                    actor_grad_norm_after = actor_grad_norm_before  # No clipping applied
                    print(f"🔍 {self.actor.training_mode} gradient norm: {actor_grad_norm_before:.6f} (no clipping)")
                    
                    self.actor_optimizer.step()

                # Update critic - completely isolated from actor
                self.critic_optimizer.zero_grad()
                
                # TEMP FIX: Restore normal critic training to test gradient flow
                # Use regular critic loss computation (not isolated) to restore gradients
                critic_loss_for_backward = nn.MSELoss()(batch_values, batch_returns)
                
                print(f"🔍 CRITIC LOSS FOR BACKWARD: {critic_loss_for_backward} (has NaN: {torch.isnan(critic_loss_for_backward)})")
                
                # Update the stored critic loss
                all_critic_losses[-1] = critic_loss_for_backward.item()
                
                # PHASE 1: Enhanced gradient anomaly detection for critic
                print("🔍 [PHASE 1] Starting critic backward pass with anomaly detection...")
                try:
                    critic_loss_for_backward.backward()
                    print("🔍 [PHASE 1] ✅ Critic backward pass completed successfully")
                except RuntimeError as e:
                    print(f"🔍 [PHASE 1] ❌ GRADIENT ANOMALY DETECTED in critic backward pass: {e}")
                    print(f"🔍 [PHASE 1] Critic loss value: {critic_loss_for_backward.item()}")
                    print(f"🔍 [PHASE 1] Critic loss shape: {critic_loss_for_backward.shape}")
                    raise e

                total_norm = 0
                critic_nan_gradients = 0
                critic_inf_gradients = 0
                critic_zero_gradients = 0
                
                for param in self.critic.parameters():
                    if param.grad is not None:
                        # PHASE 1: Check for critic gradient anomalies
                        if torch.isnan(param.grad).any():
                            critic_nan_gradients += 1
                        if torch.isinf(param.grad).any():
                            critic_inf_gradients += 1
                        if param.grad.norm() == 0:
                            critic_zero_gradients += 1
                            
                        param_norm = param.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                        
                total_norm = total_norm ** (1. / 2)
                
                # PHASE 1: Report critic gradient anomalies
                if critic_nan_gradients > 0 or critic_inf_gradients > 0:
                    print(f"🔍 [PHASE 1] ⚠️ CRITIC GRADIENT ANOMALIES DETECTED:")
                    print(f"🔍 [PHASE 1]   - NaN gradients: {critic_nan_gradients} parameters")
                    print(f"🔍 [PHASE 1]   - Inf gradients: {critic_inf_gradients} parameters")
                    print(f"🔍 [PHASE 1]   - Zero gradients: {critic_zero_gradients} parameters")
                    print(f"🔍 [PHASE 1]   - Gradient norm: {total_norm:.6f}")
                
                if critic_zero_gradients > 5:  # Lower threshold for critic (smaller network)
                    print(f"🔍 [PHASE 1] ⚠️ CRITIC HIGH ZERO GRADIENT COUNT: {critic_zero_gradients} parameters have zero gradients")
                
                if total_norm > 100:  # Threshold for exploding gradients
                    print(f"🔍 [PHASE 1] ⚠️ CRITIC EXPLODING GRADIENT DETECTED: norm={total_norm:.6f}")
                    
                if total_norm < 1e-6:  # Threshold for vanishing gradients
                    print(f"🔍 [PHASE 1] ⚠️ CRITIC VANISHING GRADIENT DETECTED: norm={total_norm:.6f}")
                
                print(f"🔍 Critic gradient norm: {total_norm:.6f}")


                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.critic_optimizer.step()

                # Clear gradients and memory after each batch
                # Note: Keep fresh_trajectory and fresh_log_prob alive for gradient flow
                del current_log_probs
                clear_gpu_cache()
        
        # Process accumulated actor loss for scheduler policy (after all batches)
        if self.actor.training_mode == "SCHEDULER_POLICY" and hasattr(self, '_accumulated_actor_loss'):
            self.actor_optimizer.zero_grad()
            
            print(f"Accumulated actor loss requires_grad: {self._accumulated_actor_loss.requires_grad}")
            print(f"Accumulated actor loss value: {self._accumulated_actor_loss.item()}")
            
            # PHASE 1: Enhanced gradient anomaly detection during backward pass
            print("🔍 [PHASE 1] Starting accumulated actor backward pass with anomaly detection...")
            try:
                self._accumulated_actor_loss.backward()
                print("🔍 [PHASE 1] ✅ Accumulated actor backward pass completed successfully")
            except RuntimeError as e:
                print(f"🔍 [PHASE 1] ❌ GRADIENT ANOMALY DETECTED in accumulated actor backward pass: {e}")
                print(f"🔍 [PHASE 1] Accumulated actor loss value: {self._accumulated_actor_loss.item()}")
                print(f"🔍 [PHASE 1] Accumulated actor loss shape: {self._accumulated_actor_loss.shape}")
                raise e
            
            # Calculate gradient norm for scheduler policy
            actor_grad_norm_before = 0
            trainable_params = self.actor.scheduler_policy.parameters()
            for param in trainable_params:
                if param.grad is not None:
                    actor_grad_norm_before += param.grad.data.norm(2).item() ** 2
            actor_grad_norm_before = actor_grad_norm_before ** 0.5
            actor_grad_norm_after = actor_grad_norm_before  # No clipping applied
            
            print(f"🔍 SCHEDULER_POLICY accumulated gradient norm: {actor_grad_norm_before:.6f} (no clipping)")
            
            self.actor_optimizer.step()
            
            # Clear accumulated loss
            del self._accumulated_actor_loss
                    
        # Return losses for logging
        if all_actor_losses:
            avg_actor_loss = np.mean(all_actor_losses)
            avg_critic_loss = np.mean(all_critic_losses)
            
            print(f"Update - Actor Loss: {avg_actor_loss:.4f}, "
                  f"Critic Loss: {avg_critic_loss:.4f}, "
                  f"Avg Advantage: {memo_advantages.mean():.4f}")
            
            # Clear buffer
            self.replay_buffer.clear_memo()
            
            # IMPORTANT: Clear gradients after PPO update to prevent memory accumulation
            # This is especially critical for LoRA training
            # self.actor.clear_gradients()
            # if hasattr(self.critic, 'zero_grad'):
            #     self.critic.zero_grad()
            
            # Clear GPU cache after gradient clearing
            # clear_gpu_cache()
            
            # Return additional gradient info for logging (you can enhance this)
            gradient_info = {
                'actor_grad_before': actor_grad_norm_before,
                'actor_grad_after': actor_grad_norm_after, 
                'critic_grad': total_norm,
                'grad_clipped': actor_grad_norm_before > 1.0
            }
            
            return avg_actor_loss, avg_critic_loss, memo_values, memo_returns, gradient_info
        
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
        return None, None, None, None, None
    
    def _calculate_policy_entropy(self, batch_indices):
        """
        Calculate policy entropy for exploration bonus based on current training mode
        Args:
            batch_indices: List of trajectory indices in current batch
        Returns:
            entropy: Policy entropy value for PPO entropy bonus
        """
        if self.actor.training_mode == "SCHEDULER_POLICY":
            # For scheduler policy, calculate entropy from the Gaussian distributions
            # This is the entropy of the policy distribution itself
            beta_std = torch.exp(self.actor.scheduler_policy.beta_log_std)
            guidance_std = torch.exp(self.actor.scheduler_policy.guidance_log_std)
            
            # Gaussian entropy: 0.5 * log(2πe * σ²) = 0.5 * log(2πe) + log(σ)
            log_2pi_e = torch.log(torch.tensor(2 * torch.pi * torch.e, device=beta_std.device))
            beta_entropy = 0.5 * log_2pi_e + torch.log(beta_std + 1e-8)
            guidance_entropy = 0.5 * log_2pi_e + torch.log(guidance_std + 1e-8)
            
            # Sum entropies across all dimensions (this is constant per policy state)
            total_entropy = torch.sum(beta_entropy) + torch.sum(guidance_entropy)
            
            return total_entropy
                
        elif self.actor.training_mode == "DIVERSITY_POLICY":
            # For diversity policy, calculate entropy from the current standard deviations
            std = torch.exp(self.actor.diversity_policy.log_std)
            
            # Gaussian entropy: 0.5 * log(2πe * σ²)
            log_2pi_e = torch.log(torch.tensor(2 * torch.pi * torch.e, device=std.device))
            entropy = 0.5 * log_2pi_e + torch.log(std + 1e-8)
            
            return torch.sum(entropy)
                
        else:
            # For other modes (like LORA_UNET), return zero entropy for now
            return torch.tensor(0.0, device=self.device)
    
    def save_policy(self):
        """Save the trained policy (mode-aware)"""
        models_dir = Path(__file__).parent.parent / "outputs" / "models"
        models_dir.mkdir(exist_ok=True)
        
        if self.actor.training_mode == "DIVERSITY_POLICY":
            policy_path = models_dir / f"{DEFAULT_CATEGORY}_diversity_policy_{self.training_start}.pth"
            # Save diversity policy state dict
            state_dict = self.actor.diversity_policy.state_dict()
            torch.save(state_dict, policy_path)
            print(f"Diversity policy saved to: {policy_path}")
            
        elif self.actor.training_mode == "LORA_UNET":
            policy_path = models_dir / f"{DEFAULT_CATEGORY}_lora_unet_{self.training_start}.pth"
            # Save LoRA adapter weights
            try:
                from peft import get_peft_model_state_dict
                lora_state_dict = get_peft_model_state_dict(self.actor.unet)
                torch.save(lora_state_dict, policy_path)
                print(f"LoRA adapter weights saved to: {policy_path}")
            except Exception as e:
                print(f"⚠️ Could not save LoRA weights: {e}")
                # Fallback: save entire UNet state (less efficient)
                fallback_path = models_dir / f"{DEFAULT_CATEGORY}_full_unet_{self.training_start}.pth"
                torch.save(self.actor.unet.state_dict(), fallback_path)
                print(f"Full UNet state saved to: {fallback_path}")