import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Optional, Tuple
from trajectory_recording import DiffusionSampler, DiffusionTrajectory, extract_features_from_trajectory
from diversity_reward import calculate_mmd_reward
from diffusion_log_utils import ACTOR_LOSS_LOG, CRITIC_LOSS_LOG, VALUE_PREDICTION_LOG, RETURN_LOG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Computing device: {device}")

class DiffusionPolicyNetwork(nn.Module):
    """
    Policy Network for Diffusion Models:
        - Wraps the UNet to act like vanilla PPO's Actor
        - Instead of outputting action distribution, it IS the policy that generates images
    Input:
        - Text prompt (converted to embeddings internally)
    Output:
        - Complete trajectory with log probabilities --> The Action we are trying to optimize
    """
    def __init__(self, sampler: DiffusionSampler, num_inference_steps: int = 20):
        super(DiffusionPolicyNetwork, self).__init__()
        self.sampler = sampler
        self.unet = sampler.unet  # This is our "policy network"
        self.num_inference_steps = num_inference_steps
    
    def forward(self, prompt: str) -> DiffusionTrajectory:
        """
        Generate trajectory for given prompt (equivalent to actor forward pass)
        Input:
            - Prompt
        Output:
            - complete trajectory (20 denoising actions)
            - instead of an action distribution (vanilla)
        """
        return self.sampler.sample_with_trajectory_recording(
            prompt=prompt,
            num_inference_steps=self.num_inference_steps
        )
    
    def calculate_log_prob(self, trajectory: DiffusionTrajectory) -> torch.Tensor:
        """
        Calculate log probability of trajectory (equivalent to action log prob)
        """
        log_probs = []

        # Each step has its own log probability
        for step in trajectory.steps:
            log_probs.append(step.log_prob)
        
        # Total log probability = sum of all steps
        return torch.stack(log_probs).sum()
    
    def select_trajectory(self, prompt: str) -> Tuple[DiffusionTrajectory, torch.Tensor]:
        """Sample trajectory with log prob (equivalent to select_action)"""
        with torch.no_grad():
            trajectory = self.forward(prompt)
            log_prob = self.calculate_log_prob(trajectory)
        return trajectory, log_prob


class DiffusionValueNetwork(nn.Module):
    """
    Value Function for Diffusion Models:
        - Estimates how good a prompt/context is for generating diverse images
        - Takes text embeddings as input (like state in vanilla PPO)
    Input:
        - Text embedding or features
    Output:
        - Scalar value V(prompt): expected diversity reward for this prompt
    """
    def __init__(self, feature_dim: int = 512, hidden_dim: int = 256):
        super(DiffusionValueNetwork, self).__init__()
        
        # Simple MLP for value estimation
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        self.relu = nn.ReLU()
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Estimate value from prompt features"""
        x = self.relu(self.fc1(features))
        x = self.relu(self.fc2(x))
        value = self.fc3(x)
        return value


class DiffusionReplayMemory:
    """
    Stores diffusion trajectories collected during rollout
    - Similar structure to vanilla PPO's ReplayMemory
    - Each "transition" is a complete diffusion trajectory
    """
    def __init__(self, batch_size):
        self.prompt_features = []      # "States" - text prompt features
        self.trajectories = []         # "Actions" - complete diffusion trajectories  
        self.rewards = []             # Diversity rewards
        self.values = []              # Value predictions
        self.log_probs = []          # Log probabilities of trajectories
        self.done_cap = []           # Always True for diffusion (complete generations)
        self.BATCH_SIZE = batch_size
    
    def add_memo(self, prompt_features, trajectory, reward, value, log_prob):
        """Add a trajectory experience (equivalent to add_memo in vanilla PPO)"""
        self.prompt_features.append(prompt_features)
        self.trajectories.append(trajectory)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.done_cap.append(True)  # Always done for diffusion trajectories
    
    def sample(self):
        """
        Prepare batched trajectory data for PPO training
        Returns data in same format as vanilla PPO
        """
        num_trajectories = len(self.trajectories)
        
        # Generate mini-batches (same logic as vanilla PPO)
        batch_start_points = np.arange(0, num_trajectories, self.BATCH_SIZE)
        sample_indices = np.arange(num_trajectories, dtype=np.int32)
        np.random.shuffle(sample_indices)
        batches = [sample_indices[i:i+self.BATCH_SIZE] for i in batch_start_points]
        
        return (np.array(self.prompt_features),     # "states"
                self.trajectories,                  # "actions" 
                np.array(self.rewards),            # rewards
                np.array(self.values),             # values
                np.array(self.done_cap),           # dones (always True)
                np.array(self.log_probs),          # for next_state equivalent
                batches)
    
    def clear_memo(self):
        """Clear all stored trajectories"""
        self.prompt_features = []
        self.trajectories = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.done_cap = []

class DiffusionRewardFunction:
    """
    Diversity reward function with normalization
    Replaces the environment reward in vanilla PPO
    """
    def __init__(self, ref_features: np.ndarray, buffer_size: int = 50):
        self.ref_features = ref_features
        self.buffer_size = buffer_size
        self.reward_history = []
    
    def calculate_reward(self, trajectory: DiffusionTrajectory) -> float:
        """Calculate diversity reward for trajectory"""
        features = extract_features_from_trajectory(trajectory, None)
        features = features.reshape(1, -1)
        
        raw_reward = calculate_mmd_reward(features, self.ref_features)
        normalized_reward = self.normalize_reward(raw_reward)
        return normalized_reward
    
    def normalize_reward(self, reward: float) -> float:
        """Normalize reward based on running statistics"""
        self.reward_history.append(reward)
        if len(self.reward_history) > self.buffer_size:
            self.reward_history.pop(0)
        
        if len(self.reward_history) < 2:
            return reward
        
        mean_reward = np.mean(self.reward_history)
        std_reward = np.std(self.reward_history)
        return (reward - mean_reward) / (std_reward + 1e-8)

class DiffusionPPOAgent:
    """
    PPO Agent for Diffusion Models
    - Same structure and hyperparameters as vanilla PPO
    - Adapted for diffusion trajectory generation
    """
    def __init__(self, sampler: DiffusionSampler, ref_features: np.ndarray, batch_size: int, 
                 feature_dim: int = 512, num_inference_steps: int = 20):
        
        # PPO hyperparameters (same as vanilla PPO)
        self.VALUE_CLIP = False
        self.VALUE_CLIP_RANGE = 0.2
        
        self.LR_ACTOR = 1e-5       # Lower for diffusion models
        self.LR_CRITIC = 1e-4      # Lower for diffusion models
        
        self.GAMMA = 1.0           # Usually 1.0 for diffusion (reward only at end)
        self.LAMBDA = 0.95         # Same GAE parameter
        
        self.EPOCH = 4
        self.EPSILON_CLIP = 0.2
        
        # Initialize networks
        self.actor = DiffusionPolicyNetwork(sampler, num_inference_steps).to(device)
        self.old_actor = DiffusionPolicyNetwork(sampler, num_inference_steps).to(device)
        self.critic = DiffusionValueNetwork(feature_dim).to(device)
        
        # Optimizers (only optimize UNet, not the whole diffusion pipeline)
        self.actor_optimizer = optim.AdamW(self.actor.unet.parameters(), lr=self.LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.LR_CRITIC)
        
        # Components
        self.replay_buffer = DiffusionReplayMemory(batch_size)
        self.reward_function = DiffusionRewardFunction(ref_features)
        
        # For text embeddings (simple approach - could be improved)
        import clip
        self.clip_model, _ = clip.load("ViT-B/32", device=device)
        self.clip_model.eval()
    
    def get_prompt_features(self, prompt: str) -> np.ndarray:
        """Convert prompt to features (equivalent to state representation)"""
        with torch.no_grad():
            text_tokens = clip.tokenize([prompt]).to(device)
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy().flatten()
    
    def get_action(self, prompt: str) -> Tuple[DiffusionTrajectory, float]:
        """
        Generate trajectory and value for prompt
        Equivalent to get_action in vanilla PPO
        """
        # Get prompt features for value estimation
        prompt_features = self.get_prompt_features(prompt)
        prompt_features_tensor = torch.FloatTensor(prompt_features).unsqueeze(0).to(device)
        
        # Generate trajectory (equivalent to action sampling)
        trajectory, log_prob = self.actor.select_trajectory(prompt)
        
        # Get value estimate
        value = self.critic(prompt_features_tensor)
        
        return trajectory, log_prob.item(), value.detach().cpu().numpy()[0][0], prompt_features
    
    def compute_gae(self, rewards, values, dones):
        """
        GAE computation for diffusion models
        Simplified since each trajectory is independent
        """
        advantages = []
        gae = 0
        
        # For diffusion, each trajectory is independent, so simplified GAE
        for step in reversed(range(len(rewards))):
            # No temporal dependencies between trajectories in diffusion
            next_value = 0  # Always terminal
            next_non_terminal = 0  # Always terminal
            
            delta = rewards[step] + self.GAMMA * next_value * next_non_terminal - values[step]
            gae = delta + self.GAMMA * self.LAMBDA * next_non_terminal * gae
            advantages.insert(0, gae)
        
        return np.array(advantages)
    
    def update(self):
        """
        PPO update for diffusion models
        Same structure as vanilla PPO but adapted for trajectories
        """
        print("Checking replay buffer...")
        if len(self.replay_buffer.trajectories) == 0:
            print("Skipping update: empty replay buffer")
            return
        
        print("Starting PPO update...")
        
        # Copy current actor to old_actor (copy UNet weights)
        self.old_actor.unet.load_state_dict(self.actor.unet.state_dict())
        
        # Get trajectory data
        memo_features, memo_trajectories, memo_rewards, memo_values, memo_dones, memo_log_probs, batches = self.replay_buffer.sample()
        
        # Convert log_probs to numpy array
        memo_log_probs_array = np.array([lp.item() if torch.is_tensor(lp) else lp for lp in memo_log_probs])
        
        # Compute advantages using GAE
        memo_advantages = self.compute_gae(memo_rewards, memo_values, memo_dones)
        
        # Normalize advantages
        memo_advantages = (memo_advantages - memo_advantages.mean()) / (memo_advantages.std() + 1e-8)
        
        # Compute returns
        memo_returns = memo_advantages + memo_values
        
        # Convert to tensors
        memo_features_tensor = torch.FloatTensor(memo_features).to(device)
        memo_advantages_tensor = torch.tensor(memo_advantages, dtype=torch.float32).to(device)
        memo_returns_tensor = torch.tensor(memo_returns, dtype=torch.float32).to(device)
        memo_old_log_probs_tensor = torch.tensor(memo_log_probs_array, dtype=torch.float32).to(device)
        
        # Get old policy log probabilities (frozen)
        with torch.no_grad():
            old_log_probs = []
            for trajectory in memo_trajectories:
                old_log_prob = self.old_actor.calculate_log_prob(trajectory)
                old_log_probs.append(old_log_prob.item())
            old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32).to(device)
        
        # Accumulate losses
        all_actor_losses = []
        all_critic_losses = []
        
        # Train for multiple epochs
        for epoch_i in range(self.EPOCH):
            for batch in batches:
                if len(batch) == 0:
                    continue
                
                # Current policy log probabilities
                current_log_probs = []
                for idx in batch:
                    traj = memo_trajectories[idx]
                    current_log_prob = self.actor.calculate_log_prob(traj)
                    current_log_probs.append(current_log_prob)
                
                current_log_probs_tensor = torch.stack(current_log_probs)
                
                # Calculate ratio
                ratio = torch.exp(current_log_probs_tensor - old_log_probs[batch])
                
                # Batch advantages
                batch_advantages = memo_advantages_tensor[batch]
                
                # PPO clipped objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.EPSILON_CLIP, 1 + self.EPSILON_CLIP) * batch_advantages
                
                # Actor loss (no entropy for diffusion models - they have inherent stochasticity)
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Critic loss
                batch_values = self.critic(memo_features_tensor[batch]).squeeze()
                batch_returns = memo_returns_tensor[batch]
                
                if self.VALUE_CLIP:
                    with torch.no_grad():
                        old_values = self.critic(memo_features_tensor).squeeze()
                    
                    batch_old_values = old_values[batch]
                    value_loss_unclipped = (batch_values - batch_returns) ** 2
                    
                    values_clipped = batch_old_values + torch.clamp(
                        batch_values - batch_old_values, -self.VALUE_CLIP_RANGE, self.VALUE_CLIP_RANGE
                    )
                    value_loss_clipped = (values_clipped - batch_returns) ** 2
                    critic_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()
                else:
                    critic_loss = nn.MSELoss()(batch_values, batch_returns)
                
                # Store losses
                all_actor_losses.append(actor_loss.item())
                all_critic_losses.append(critic_loss.item())
                
                # Update actor (UNet)
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.unet.parameters(), max_norm=0.5)
                self.actor_optimizer.step()
                
                # Update critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.critic_optimizer.step()
        
        # Log losses
        if all_actor_losses:
            ACTOR_LOSS_LOG.append(np.mean(all_actor_losses))
            CRITIC_LOSS_LOG.append(np.mean(all_critic_losses))
            
            # Store data for plotting
            VALUE_PREDICTION_LOG.extend(memo_values.flatten().tolist())
            RETURN_LOG.extend(memo_returns.flatten().tolist())
            
            # Debug info
            print(f"Update - Actor Loss: {np.mean(all_actor_losses):.4f}, "
                  f"Critic Loss: {np.mean(all_critic_losses):.4f}, "
                  f"Avg Advantage: {memo_advantages.mean():.4f}")
        
        # Clear buffer
        self.replay_buffer.clear_memo()
    
    def save_policy(self):
        """Save the trained policy (UNet weights)"""
        torch.save(self.actor.unet.state_dict(), "diffusion_ppo_policy.pth")
        print("Diffusion PPO policy saved!")