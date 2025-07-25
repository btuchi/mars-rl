# File: training/agent.py
"""Main TRPO agent for diffusion training"""

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


class DiffusionTRPOAgent:
    """TRPO Agent for Diffusion Models"""
    
    def __init__(self, sampler: DiffusionSampler, ref_features: np.ndarray, batch_size: int, 
                 feature_dim: int = 512, num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
                 images_per_prompt: int = DEFAULT_IMAGES_PER_PROMPT, 
                 save_samples: bool = True, training_start: str = None, ref_images: np.ndarray = None):
        
        self.dtype = sampler.dtype if hasattr(sampler, 'dtype') else torch.float32
        self.device = sampler.device
        self.training_start = training_start

        print(f"TRPO Agent using dtype: {self.dtype}")
        
        # TRPO hyperparameters
        self.LR_CRITIC = DEFAULT_LR_CRITIC  # Only critic uses optimizer
        self.GAMMA = DEFAULT_GAMMA
        self.LAMBDA = DEFAULT_LAMBDA
        
        # TRPO-specific parameters
        self.kl_target = DEFAULT_KL_TARGET
        self.damping = DEFAULT_DAMPING
        self.cg_iters = DEFAULT_CG_ITERS
        self.backtrack_iters = DEFAULT_BACKTRACK_ITERS
        self.backtrack_coeff = DEFAULT_BACKTRACK_COEFF
        self.accept_ratio = DEFAULT_ACCEPT_RATIO
        self.images_per_prompt = images_per_prompt
        
        # Initialize centralized feature extractor (ResNet-18)
        self.feature_extractor = FeatureExtractor(device=self.device)
        
        # Initialize networks
        self.actor = DiffusionPolicyNetwork(sampler, num_inference_steps)
        self.old_actor = DiffusionPolicyNetwork(sampler, num_inference_steps)
        self.critic = DiffusionValueNetwork(feature_dim).to(self.device).to(self.dtype)

        # TRPO only uses optimizer for critic (actor updated via natural gradients)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=self.LR_CRITIC, weight_decay=1e-3)
        
        # Mode-aware parameter counting and reporting
        trainable_params = sum(p.numel() for p in self.actor.get_trainable_parameters())
        print(f"🎯 Training {trainable_params:,} parameters in {self.actor.training_mode} mode (TRPO)")
        print(f"🎯 TRPO settings: KL target={self.kl_target}, CG iters={self.cg_iters}")
        
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
            log_prob_tensors.append(log_prob_tensor)  # ✅ GRADIENT FLOW: Preserves connection to policy parameters
            
            # Extract features from first trajectory for critic
            # CRITIC FEATURES: Detach from actor computation graph to prevent critic contamination
            # This is correct - critic should not affect policy gradients
            if i == 0:
                with torch.no_grad():  # Ensure no gradients flow to critic
                    # Create a detached copy of the image for critic feature extraction
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
                    # SAMPLE EVALUATION: Detach for evaluation only (not affecting training gradients)
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
            # LOGGING FEATURES: Detach for visualization only (reward calculation already done)
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
        """TRPO update for diffusion models
        
        TRPO Algorithm:
        1. Compute policy gradient 
        2. Solve for natural gradient using conjugate gradient
        3. Line search with KL divergence constraint
        4. Update critic with standard gradient descent
        """
        print("Checking replay buffer...")
        if len(self.replay_buffer.trajectories) == 0:
            print("Skipping update: empty replay buffer")
            return
        
        print("Starting TRPO update...")
        clear_gpu_cache()

        # Save current policy parameters (TRPO needs this for rollback)
        self._save_current_policy()
        
        # Get trajectory data
        memo_features, memo_prompts, memo_trajectories, memo_rewards, memo_values, memo_log_probs, memo_log_prob_tensors, batches = self.replay_buffer.sample()
        
        # Compute advantages using GAE (same as PPO)
        memo_advantages = self.compute_gae(memo_rewards, memo_values)
        memo_returns = memo_advantages + memo_values
        
        print(f"🔍 Advantages mean: {memo_advantages.mean():.4f}, std: {memo_advantages.std():.4f}")

        # Convert to tensors
        memo_features_tensor = torch.from_numpy(np.array(memo_features)).to(
            device=self.device, dtype=self.dtype
        )
        memo_advantages_tensor = torch.tensor(
            memo_advantages, dtype=self.dtype, device=self.device
        )
        memo_returns_tensor = torch.tensor(
            memo_returns, dtype=self.dtype, device=self.device
        )
        
        # Normalize advantages for better training stability
        if memo_advantages_tensor.std() > 1e-8:
            memo_advantages_tensor = (memo_advantages_tensor - memo_advantages_tensor.mean()) / (memo_advantages_tensor.std() + 1e-8)
        
        # ============ TRPO ALGORITHM ============
        
        # Step 1: Compute policy gradient
        print("Step 1: Computing policy gradient...")
        policy_gradient = self._compute_policy_gradient(
            memo_prompts, memo_log_prob_tensors, memo_advantages_tensor
        )
        
        # Step 2: Compute natural gradient using conjugate gradient
        print("Step 2: Computing natural gradient with conjugate gradient...")
        search_direction = self._conjugate_gradient(policy_gradient, memo_prompts, memo_log_prob_tensors)
        
        # Step 3: Line search with KL constraint
        print("Step 3: Line search with KL constraint...")
        step_size, improvement = self._line_search(
            search_direction, policy_gradient, memo_prompts, 
            memo_log_prob_tensors, memo_advantages_tensor
        )
        
        # Step 4: Update policy parameters
        if improvement > 0:
            print(f"Step 4: Applying TRPO update (step_size={step_size:.6f}, improvement={improvement:.6f})")
            self._apply_update(search_direction, step_size)
            actor_loss = -improvement  # Negative because we're maximizing
        else:
            print("Step 4: No improvement found, keeping current policy")
            actor_loss = 0.0
        
        # Step 5: Update critic with standard gradient descent
        print("Step 5: Updating critic...")
        critic_loss = self._update_critic(memo_features_tensor, memo_returns_tensor)
        
        # Cleanup
        self.replay_buffer.clear_memo()
        
        # Return loss info for logging
        gradient_info = {
            'actor_grad_before': 0.0,  # TRPO doesn't use standard gradients
            'actor_grad_after': 0.0,
            'critic_grad': 0.0,
            'grad_clipped': False,
            'step_size': step_size,
            'kl_divergence': 0.0,  # Will be computed in line search
            'improvement': improvement
        }
        
        print(f"TRPO Update - Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}")
        print(f"Step size: {step_size:.6f}, Improvement: {improvement:.6f}")
        
        return actor_loss, critic_loss, memo_values, memo_returns, gradient_info
    
    # ============ TRPO HELPER METHODS ============
    
    def _save_current_policy(self):
        """Synchronize old_actor with current actor state for TRPO comparison"""
        print("🔄 Synchronizing old policy with current policy...")
        
        # Set old_actor to same training mode as current actor
        self.old_actor.set_training_mode(self.actor.training_mode)
        
        if self.actor.training_mode == "SCHEDULER_POLICY":
            # Copy scheduler policy parameters to old_actor
            self.old_actor.scheduler_policy.load_state_dict(
                self.actor.scheduler_policy.state_dict()
            )
        elif self.actor.training_mode == "DIVERSITY_POLICY":
            # Copy diversity policy parameters to old_actor
            self.old_actor.diversity_policy.load_state_dict(
                self.actor.diversity_policy.state_dict()
            )
        elif self.actor.training_mode == "LORA_UNET":
            # For LoRA, we need to sync the entire UNet state
            # This is more complex due to LoRA adapter weights
            try:
                # Copy UNet state including LoRA adapters
                self.old_actor.unet.load_state_dict(self.actor.unet.state_dict())
            except Exception as e:
                print(f"⚠️ Failed to sync LoRA UNet state: {e}")
        
        print("✅ Old policy synchronized")
    
    def _compute_policy_gradient(self, memo_prompts, memo_log_prob_tensors, memo_advantages_tensor):
        """Compute policy gradient vector"""
        # For scheduler policy, we use the stored log prob tensors that preserve gradients
        if self.actor.training_mode == "SCHEDULER_POLICY":
            # Use stored log probabilities that maintain gradient connections
            policy_loss = -torch.mean(torch.stack(memo_log_prob_tensors) * memo_advantages_tensor)
        else:
            # For other modes, compute fresh log probabilities
            current_log_probs = []
            for prompt in memo_prompts:
                _, log_prob = self.actor.select_trajectory(prompt)
                current_log_probs.append(log_prob)
            policy_loss = -torch.mean(torch.stack(current_log_probs) * memo_advantages_tensor)
        
        # Compute gradients
        grad_params = list(self.actor.get_trainable_parameters())
        policy_gradient = torch.autograd.grad(policy_loss, grad_params, create_graph=True)
        
        # Flatten gradient vector
        flat_grad = torch.cat([grad.flatten() for grad in policy_gradient])
        return flat_grad
    
    def _conjugate_gradient(self, policy_gradient, memo_prompts, memo_log_prob_tensors):
        """Solve Ax = b using conjugate gradient, where A is Fisher Information Matrix"""
        print("🔄 Running conjugate gradient solver...")
        
        def fisher_vector_product(vector):
            """Compute Fisher Information Matrix times vector using proper definition"""
            print("  🔄 Computing Fisher-vector product...")
            
            try:
                # Get trainable parameters
                trainable_params = list(self.actor.get_trainable_parameters())
                
                # Compute average KL divergence between current and old policy
                kl_divs = []
                for prompt in memo_prompts:
                    if self.actor.training_mode == "SCHEDULER_POLICY":
                        # For scheduler policy, compare Gaussian distributions
                        kl_div = self._compute_scheduler_kl_divergence_between_policies(prompt)
                    else:
                        # For other modes, use log probability difference
                        _, current_log_prob = self.actor.select_trajectory(prompt)
                        with torch.no_grad():
                            _, old_log_prob = self.old_actor.select_trajectory(prompt)
                        
                        # KL divergence approximation using importance sampling
                        log_ratio = current_log_prob - old_log_prob.detach()
                        kl_div = 0.5 * log_ratio ** 2  # Second-order approximation
                    
                    kl_divs.append(kl_div)
                
                # Average KL divergence across batch
                avg_kl = torch.mean(torch.stack(kl_divs))
                
                # Compute gradients of KL w.r.t. parameters
                kl_grads = torch.autograd.grad(avg_kl, trainable_params, create_graph=True)
                
                # Reshape vector to match parameter shapes
                start_idx = 0
                vector_params = []
                for param in trainable_params:
                    param_size = param.numel()
                    param_vector = vector[start_idx:start_idx + param_size].view(param.shape)
                    vector_params.append(param_vector)
                    start_idx += param_size
                
                # Compute gradient-vector product
                grad_vector_product = torch.sum(torch.stack([
                    torch.sum(grad * vec_param) 
                    for grad, vec_param in zip(kl_grads, vector_params)
                ]))
                
                # Compute Hessian-vector product (second derivative)
                hvp = torch.autograd.grad(
                    grad_vector_product, trainable_params, retain_graph=False
                )
                
                # Flatten and add damping
                flat_hvp = torch.cat([h.flatten() for h in hvp])
                return flat_hvp + self.damping * vector
                
            except Exception as e:
                print(f"⚠️ Fisher matrix computation failed: {e}")
                # Fallback: Use identity matrix with damping
                return self.damping * vector
        
        # Simple conjugate gradient implementation (simplified for demo)
        x = torch.zeros_like(policy_gradient)
        r = policy_gradient.clone()
        p = r.clone()
        rsold = torch.dot(r, r)
        
        for i in range(self.cg_iters):
            Ap = fisher_vector_product(p)
            alpha = rsold / (torch.dot(p, Ap) + 1e-8)
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = torch.dot(r, r)
            
            if torch.sqrt(rsnew) < 1e-10:
                break
                
            beta = rsnew / rsold
            p = r + beta * p
            rsold = rsnew
        
        return x
    
    def _line_search(self, search_direction, policy_gradient, memo_prompts, memo_log_prob_tensors, memo_advantages_tensor):
        """Line search with KL divergence constraint"""
        print("🔄 Running line search...")
        
        # Save current policy parameters for rollback
        current_params = self._get_flat_params()
        
        # Compute initial policy performance
        initial_loss = self._compute_policy_loss(memo_prompts, memo_log_prob_tensors, memo_advantages_tensor)
        
        # Start with a reasonable step size
        step_size = 1.0
        improvement = 0.0
        
        for i in range(self.backtrack_iters):
            print(f"  🔄 Line search iteration {i+1}, step_size={step_size:.6f}")
            
            # Apply candidate update
            candidate_params = current_params + step_size * search_direction
            self._set_flat_params(candidate_params)
            
            # Compute KL divergence between current (modified) policy and old policy
            # Note: current policy has been modified with candidate_params
            kl_divergence = self._compute_kl_divergence(memo_prompts, memo_log_prob_tensors)
            
            # Check KL constraint (trust region constraint)
            if kl_divergence.item() > self.kl_target:
                print(f"    ⚠️ KL constraint violated: {kl_divergence.item():.6f} > {self.kl_target:.6f}")
                # Restore original parameters before trying smaller step
                self._set_flat_params(current_params)
                step_size *= self.backtrack_coeff
                continue
            
            print(f"    ✅ KL constraint satisfied: {kl_divergence.item():.6f} ≤ {self.kl_target:.6f}")
            
            # Compute new policy performance
            new_loss = self._compute_policy_loss(memo_prompts, memo_log_prob_tensors, memo_advantages_tensor)
            actual_improvement = initial_loss - new_loss
            
            # Compute expected improvement (linear approximation)
            expected_improvement = torch.dot(policy_gradient, search_direction) * step_size
            
            # Check acceptance ratio
            if expected_improvement.item() > 0:
                improvement_ratio = actual_improvement.item() / expected_improvement.item()
                
                if improvement_ratio > self.accept_ratio:
                    improvement = actual_improvement.item()
                    print(f"    ✅ Step accepted: improvement={improvement:.6f}, KL={kl_divergence.item():.6f}")
                    break
                else:
                    print(f"    ❌ Insufficient improvement ratio: {improvement_ratio:.4f} < {self.accept_ratio}")
            
            # Reduce step size for next iteration
            step_size *= self.backtrack_coeff
        else:
            # No acceptable step found, rollback to original parameters
            print("    ⚠️ No acceptable step found, keeping original policy")
            self._set_flat_params(current_params)
            step_size = 0.0
            improvement = 0.0
        
        return step_size, improvement
    
    def _apply_update(self, search_direction, step_size):
        """Apply parameter update using natural gradient direction"""
        print(f"🔄 Applying TRPO update with step_size={step_size:.6f}")
        
        # Get current parameters
        current_params = self._get_flat_params()
        
        # Apply update: θ_new = θ_old + step_size * search_direction
        new_params = current_params + step_size * search_direction
        
        # Set new parameters
        self._set_flat_params(new_params)
        
        # Verify update was applied
        updated_params = self._get_flat_params()
        param_change = torch.norm(updated_params - current_params).item()
        print(f"✅ Parameter update applied, change magnitude: {param_change:.6f}")
    
    def _update_critic(self, memo_features_tensor, memo_returns_tensor):
        """Update critic using standard gradient descent"""
        self.critic_optimizer.zero_grad()
        
        predicted_values = self.critic(memo_features_tensor).squeeze(-1)
        critic_loss = nn.MSELoss()(predicted_values, memo_returns_tensor)
        
        critic_loss.backward()
        self.critic_optimizer.step()
        
        return critic_loss.item()
    
    def _compute_scheduler_kl_divergence_between_policies(self, prompt):
        """Compute KL divergence between current and old scheduler policies"""
        # Get current policy parameters
        curr_beta_mean = self.actor.scheduler_policy.beta_mean
        curr_beta_log_std = self.actor.scheduler_policy.beta_log_std
        curr_guidance_mean = self.actor.scheduler_policy.guidance_mean
        curr_guidance_log_std = self.actor.scheduler_policy.guidance_log_std
        
        # Get old policy parameters
        old_beta_mean = self.old_actor.scheduler_policy.beta_mean
        old_beta_log_std = self.old_actor.scheduler_policy.beta_log_std
        old_guidance_mean = self.old_actor.scheduler_policy.guidance_mean
        old_guidance_log_std = self.old_actor.scheduler_policy.guidance_log_std
        
        # Convert log_std to std
        curr_beta_std = torch.exp(curr_beta_log_std)
        curr_guidance_std = torch.exp(curr_guidance_log_std)
        old_beta_std = torch.exp(old_beta_log_std.detach())
        old_guidance_std = torch.exp(old_guidance_log_std.detach())
        
        # KL divergence between two multivariate Gaussians:
        # KL(N(μ1,Σ1) || N(μ2,Σ2)) = 0.5 * [tr(Σ2^-1 Σ1) + (μ2-μ1)^T Σ2^-1 (μ2-μ1) - k + ln(det(Σ2)/det(Σ1))]
        # For diagonal covariance: simplifies to sum over dimensions
        
        # Beta parameters KL
        beta_var_ratio = (curr_beta_std ** 2) / (old_beta_std ** 2 + 1e-8)
        beta_mean_diff = curr_beta_mean - old_beta_mean.detach()
        beta_mean_term = (beta_mean_diff ** 2) / (old_beta_std ** 2 + 1e-8)
        beta_log_det_term = 2 * (old_beta_log_std.detach() - curr_beta_log_std)
        beta_kl = 0.5 * torch.sum(beta_var_ratio + beta_mean_term + beta_log_det_term - 1)
        
        # Guidance parameters KL  
        guidance_var_ratio = (curr_guidance_std ** 2) / (old_guidance_std ** 2 + 1e-8)
        guidance_mean_diff = curr_guidance_mean - old_guidance_mean.detach()
        guidance_mean_term = (guidance_mean_diff ** 2) / (old_guidance_std ** 2 + 1e-8)
        guidance_log_det_term = 2 * (old_guidance_log_std.detach() - curr_guidance_log_std)
        guidance_kl = 0.5 * torch.sum(guidance_var_ratio + guidance_mean_term + guidance_log_det_term - 1)
        
        return beta_kl + guidance_kl
    
    def _get_flat_params(self):
        """Get flattened parameters from trainable policy"""
        # Use the same parameter selection as get_trainable_parameters for consistency
        trainable_params = list(self.actor.get_trainable_parameters())
        
        if not trainable_params:
            print("⚠️ No trainable parameters found!")
            return torch.tensor([])
        
        # Flatten and concatenate all trainable parameters
        flat_params = []
        for param in trainable_params:
            if param.requires_grad:
                flat_params.append(param.data.flatten())
            else:
                print(f"⚠️ Parameter not requiring gradients found: {param.shape}")
        
        if not flat_params:
            print("⚠️ No parameters requiring gradients found!")
            return torch.tensor([])
            
        return torch.cat(flat_params)
    
    def _set_flat_params(self, flat_params):
        """Set flattened parameters to trainable policy"""
        trainable_params = list(self.actor.get_trainable_parameters())
        
        if not trainable_params:
            print("⚠️ No trainable parameters found!")
            return
        
        if flat_params.numel() == 0:
            print("⚠️ Empty flat_params provided!")
            return
        
        start_idx = 0
        with torch.no_grad():
            for param in trainable_params:
                if not param.requires_grad:
                    continue
                    
                param_size = param.numel()
                
                # Check bounds
                if start_idx + param_size > flat_params.numel():
                    print(f"⚠️ Parameter size mismatch! Expected {param_size}, available {flat_params.numel() - start_idx}")
                    break
                
                # Set parameter data
                param.data.copy_(
                    flat_params[start_idx:start_idx + param_size].view(param.shape)
                )
                start_idx += param_size
        
        # Verify all parameters were set
        if start_idx != flat_params.numel():
            print(f"⚠️ Parameter count mismatch! Used {start_idx}, provided {flat_params.numel()}")
    
    def _compute_policy_loss(self, memo_prompts, memo_log_prob_tensors, memo_advantages_tensor):
        """Compute policy loss for line search"""
        if self.actor.training_mode == "SCHEDULER_POLICY":
            # Use stored log probabilities for scheduler policy
            policy_loss = -torch.mean(torch.stack(memo_log_prob_tensors) * memo_advantages_tensor)
        else:
            # Compute fresh log probabilities for other modes
            current_log_probs = []
            for prompt in memo_prompts:
                _, log_prob = self.actor.select_trajectory(prompt)
                current_log_probs.append(log_prob)
            policy_loss = -torch.mean(torch.stack(current_log_probs) * memo_advantages_tensor)
        
        return policy_loss
    
    def _compute_kl_divergence(self, memo_prompts, memo_log_prob_tensors):
        """Compute KL divergence between old and new policy"""
        kl_divs = []
        
        for i, prompt in enumerate(memo_prompts):
            if self.actor.training_mode == "SCHEDULER_POLICY":
                # For scheduler policy, compute exact KL between Gaussian distributions
                kl_div = self._compute_scheduler_kl_divergence_between_policies(prompt)
            else:
                # For other modes, compute KL using log probability differences
                # Get new policy log probability
                _, new_log_prob = self.actor.select_trajectory(prompt)
                
                # Get old policy log probability (stored)
                old_log_prob = memo_log_prob_tensors[i]
                
                # KL divergence using importance sampling ratio
                # KL(π_new || π_old) ≈ E[π_new/π_old * log(π_new/π_old)]
                # For single sample: (π_new/π_old) * log(π_new/π_old)
                log_ratio = new_log_prob - old_log_prob.detach()
                
                # Use second-order approximation for stability
                # KL ≈ 0.5 * (log_ratio)^2 when log_ratio is small
                if torch.abs(log_ratio) < 0.1:
                    kl_div = 0.5 * log_ratio ** 2
                else:
                    # Use full formula for larger ratios
                    ratio = torch.exp(log_ratio)
                    kl_div = ratio * log_ratio
            
            kl_divs.append(kl_div)
        
        # Return average KL divergence
        return torch.mean(torch.stack(kl_divs))
    
    def save_policy(self):
        """Save the trained policy (mode-aware)"""
        models_dir = Path(__file__).parent.parent / "outputs" / "models"
        models_dir.mkdir(exist_ok=True)
        
        if self.actor.training_mode == "SCHEDULER_POLICY":
            policy_path = models_dir / f"{DEFAULT_CATEGORY}_scheduler_policy_{self.training_start}.pth"
            # Save scheduler policy state dict
            state_dict = self.actor.scheduler_policy.state_dict()
            torch.save(state_dict, policy_path)
            print(f"Scheduler policy saved to: {policy_path}")
            
        elif self.actor.training_mode == "DIVERSITY_POLICY":
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