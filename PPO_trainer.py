import torch
import torch.nn as nn
import torch.optim as optim
from trajectory_recording import DiffusionSampler, DiffusionTrajectory, extract_features_from_trajectory
from diversity_reward import calculate_mmd_reward
import numpy as np
from typing import List, Optional
from tqdm import tqdm

class RolloutBuffer:
    """Clean buffer implementation for storing PPO rollouts"""
    """Each rollout is a complete image-generation event plus its metadata."""
    def __init__(self):
        self.trajectories = []
        self.log_probs = []
        self.rewards = []
        self.advantages = []
        self.returns = []
        
    def add(self, trajectory: DiffusionTrajectory, log_prob: torch.Tensor, reward: float):
        """Add a single trajectory, log_prob, reward to buffer"""
        self.trajectories.append(trajectory)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
    
    def clear(self):
        """Clear all stored data"""
        self.__init__()
    
    def size(self):
        return len(self.trajectories)

class RewardNormalizer:
    """Reward normalization following DDPO approach"""
    """Prevents reward scale drift from destabilizing PPO"""
    def __init__(self, buffer_size: int = 50):
        self.buffer_size = buffer_size
        self.reward_history = []
    
    def normalize(self, reward: float) -> float:
        """Normalize reward based on running statistics"""
        self.reward_history.append(reward)
        if len(self.reward_history) > self.buffer_size:
            self.reward_history.pop(0)
        
        if len(self.reward_history) < 2:
            return reward
        
        mean_reward = np.mean(self.reward_history)
        std_reward = np.std(self.reward_history)
        return (reward - mean_reward) / (std_reward + 1e-8)

class PPOTrainer:
    
    def __init__(self, 
                 sampler: DiffusionSampler, 
                 ref_features: np.ndarray,
                 lr: float = 1e-5,
                 gamma: float = 1.0,  # For diffusion, usually 1.0 since reward only at end
                 eps_clip: float = 0.2,
                 epochs: int = 4,
                 max_grad_norm: float = 1.0,
                 num_inference_steps: int = 20):
        
        self.sampler = sampler
        self.ref_features = ref_features
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.epochs = epochs
        self.max_grad_norm = max_grad_norm
        self.num_inference_steps = num_inference_steps
        
        # Set up optimizer for UNet parameters
        self.optimizer = optim.AdamW(self.sampler.unet.parameters(), lr=self.lr)
        
        # Initialize components
        self.buffer = RolloutBuffer()
        self.reward_normalizer = RewardNormalizer()
        
        # Training statistics
        self.step_count = 0
        self.training_stats = {
            'avg_rewards': [],
            'policy_losses': [],
            'reward_stds': []
        }
    
    def collect_trajectory(self, prompt: str) -> tuple[DiffusionTrajectory, torch.Tensor, float]:
        """Collect a single trajectory and compute its reward"""
        
        # Sample trajectory with recording
        trajectory = self.sampler.sample_with_trajectory_recording(
            prompt=prompt,
            num_inference_steps=self.num_inference_steps
        )
        
        # Calculate log probability of entire trajectory
        log_prob = self.calculate_trajectory_log_prob(trajectory)
        
        # Extract features and compute reward
        features = extract_features_from_trajectory(trajectory, None)
        features = features.reshape(1, -1)
        
        # Calculate diversity reward
        raw_reward = calculate_mmd_reward(features, self.ref_features)
        normalized_reward = self.reward_normalizer.normalize(raw_reward)
        
        return trajectory, log_prob, normalized_reward
    
    def calculate_trajectory_log_prob(self, trajectory: DiffusionTrajectory) -> torch.Tensor:
        """Calculate log probability of trajectory (sum of step log probs)"""
        log_probs = []
        for step in trajectory.steps:
            log_probs.append(step.log_prob)
        return torch.stack(log_probs).sum()
    
    def recalculate_log_probs(self, trajectories: List[DiffusionTrajectory]) -> List[torch.Tensor]:
        """
        CRITICAL: Recalculate log probs with current policy parameters
        In PPO, we need to compare old and new log-probabilities for stability.
        """
        new_log_probs = []
        for trajectory in trajectories:
            # This requires re-running forward pass with current parameters
            # For now, we'll use the stored log_probs (this is a simplification)
            # In full implementation, re-run the UNet forward pass
            log_prob = self.calculate_trajectory_log_prob(trajectory)
            new_log_probs.append(log_prob)
        return new_log_probs
    
    def compute_advantages_and_returns(self, rewards: List[float]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute returns and advantages for PPO"""
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        
        # For diffusion models, returns are often just the rewards (gamma=1.0, single-step reward)
        returns = rewards_tensor.clone()
        
        # Simple advantage estimation (can be improved with GAE)
        advantages = returns - returns.mean()
        if len(advantages) > 1:
            # Normalizes advantages to help with stability and balance
            advantages = advantages / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def ppo_update(self):
        """Perform PPO update using collected trajectories"""
        if self.buffer.size() == 0:
            return 0.0
        
        # Get old log probs (frozen)
        old_log_probs = torch.stack(self.buffer.log_probs).detach()
        
        # Compute advantages and returns
        advantages, returns = self.compute_advantages_and_returns(self.buffer.rewards)
        
        # PPO epochs
        policy_losses = []
        for epoch in range(self.epochs):
            # Recalculate log probs with current policy
            new_log_probs = self.recalculate_log_probs(self.buffer.trajectories)
            new_log_probs_tensor = torch.stack(new_log_probs)
            
            # Calculate ratio
            ratio = torch.exp(new_log_probs_tensor - old_log_probs)
            
            # PPO clipped objective
            clipped_ratio = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip)
            
            # Policy loss
            policy_loss1 = ratio * advantages
            policy_loss2 = clipped_ratio * advantages
            policy_loss = -torch.min(policy_loss1, policy_loss2).mean()
            
            # Backward pass
            self.optimizer.zero_grad()
            policy_loss.backward()
            
            # Gradient clipping (IMPORTANT!)
            torch.nn.utils.clip_grad_norm_(self.sampler.unet.parameters(), self.max_grad_norm)
            
            self.optimizer.step()
            
            policy_losses.append(policy_loss.item())
        
        avg_policy_loss = np.mean(policy_losses)
        return avg_policy_loss
    
    def train_step(self, prompt_batch: List[str]) -> dict:
        """Execute one training step with a batch of prompts"""
        
        print(f"Step {self.step_count}: Collecting {len(prompt_batch)} trajectories...")
        
        # Collect trajectories
        for prompt in tqdm(prompt_batch, desc="Sampling"):
            trajectory, log_prob, reward = self.collect_trajectory(prompt)
            self.buffer.add(trajectory, log_prob, reward)
        
        # Compute statistics
        avg_reward = np.mean(self.buffer.rewards)
        reward_std = np.std(self.buffer.rewards)
        
        # PPO update
        policy_loss = self.ppo_update()
        
        # Clear buffer
        self.buffer.clear()
        
        # Update statistics
        self.training_stats['avg_rewards'].append(avg_reward)
        self.training_stats['policy_losses'].append(policy_loss)
        self.training_stats['reward_stds'].append(reward_std)
        
        self.step_count += 1
        
        return {
            'step': self.step_count,
            'avg_reward': avg_reward,
            'policy_loss': policy_loss,
            'reward_std': reward_std
        }
    
    def train(self, prompts: List[str], num_steps: int, batch_size: int = 4):
        """Main training loop"""
        print(f"Starting PPO training for {num_steps} steps...")
        
        for step in range(num_steps):
            # Sample batch of prompts
            batch_prompts = np.random.choice(prompts, size=batch_size, replace=True).tolist()
            
            # Execute training step
            stats = self.train_step(batch_prompts)
            
            # Log progress
            print(f"Step {stats['step']}: "
                  f"Reward={stats['avg_reward']:.4f}, "
                  f"Loss={stats['policy_loss']:.4f}, "
                  f"Std={stats['reward_std']:.4f}")
        
        print("Training completed!")
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        torch.save({
            'unet_state_dict': self.sampler.unet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats,
            'step_count': self.step_count
        }, filepath)
        print(f"Model saved to {filepath}")


# Example usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load your pre-computed reference features
    ref_features = np.load("reference_crater_features.npy")
    print(f"Loaded reference features: {ref_features.shape}")
    
    # Initialize sampler and trainer
    sampler = DiffusionSampler(device=device)
    trainer = PPOTrainer(
        sampler=sampler, 
        ref_features=ref_features,
        lr=1e-5,
        eps_clip=0.2,
        epochs=4,
        num_inference_steps=15  # Reduced for faster training
    )
    
    # Define crater prompts
    crater_prompts = [
        "a photo of a mars crater",
        "a detailed mars crater with shadows", 
        "a large mars crater on red terrain",
        "a small mars crater with rocks",
        "an ancient mars crater with erosion",
        "a fresh mars crater with sharp edges"
    ]
    
    # Train the model
    trainer.train(
        prompts=crater_prompts,
        num_steps=50,  # Start small for testing
        batch_size=3   # Small batch for memory efficiency
    )
    
    # Save the trained model
    trainer.save_model("ppo_trained_diffusion_model.pt")