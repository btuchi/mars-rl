from torch import nn
import torch
from torch.distributions import Normal
import numpy as np
import torch.optim as optim
import os
import json
from log_utils import ACTOR_LOSS_LOG, CRITIC_LOSS_LOG, VALUE_PREDICTION_LOG, RETURN_LOG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Computing device: {device}")

class Actor(nn.Module):
    """
    Policy Network optimized for long-term training
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_std = nn.Linear(hidden_dim, action_dim)

        # Better weight initialization for long-term stability
        # self._init_weights()

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
    
    def _init_weights(self):
        """Orthogonal initialization for better long-term training"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)
        
        # Initialize policy head with smaller weights
        nn.init.orthogonal_(self.fc_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.fc_std.weight, gain=0.01)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        mean = self.tanh(self.fc_mean(x)) * 2
        std = self.softplus(self.fc_std(x)) + 1e-3
        std = std.clamp(min=1e-2, max=1.0)

        return mean, std

    def select_action(self, s):
        with torch.no_grad():
            mu, sigma = self.forward(s)
            normal_dist = Normal(mu, sigma)
            action = normal_dist.sample()
            action = action.clamp(-2.0, 2.0)
        
        return action

class Critic(nn.Module):
    """
    Value Function Estimator with improved architecture
    """
    def __init__(self, state_dim, hidden_dim=256):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        # Better initialization
        self._init_weights()

        self.relu = nn.ReLU()
    
    def _init_weights(self):
        """Orthogonal initialization for stable value learning"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        value = self.fc3(x)
        return value

class ReplayMemory:
    """
    Memory buffer optimized for long-term training
    """
    def __init__(self, batch_size):
        self.state_cap = []
        self.action_cap = []
        self.reward_cap = []
        self.value_cap = []
        self.done_cap = []
        self.next_state_cap = []
        self.BATCH_SIZE = batch_size
    
    def add_memo(self, state, action, reward, value, done, next_state):
        self.state_cap.append(state)
        self.action_cap.append(action)
        self.reward_cap.append(reward)
        self.value_cap.append(value)
        self.done_cap.append(done)
        self.next_state_cap.append(next_state)
    
    def sample(self):
        num_state = len(self.state_cap)
        batch_start_points = np.arange(0, num_state, self.BATCH_SIZE)
        sample_indices = np.arange(num_state, dtype=np.int32)
        np.random.shuffle(sample_indices)
        batches = [sample_indices[i:i+self.BATCH_SIZE] for i in batch_start_points]

        return (np.array(self.state_cap), np.array(self.action_cap), 
                np.array(self.reward_cap), np.array(self.value_cap), 
                np.array(self.done_cap), np.array(self.next_state_cap), batches)
    
    def clear_memo(self):
        self.state_cap = []
        self.action_cap = []
        self.reward_cap = []
        self.value_cap = []
        self.done_cap = []
        self.next_state_cap = []

class LongTermPPOAgent:
    """
    PPO Agent optimized for extended training (10k-20k episodes)
    """
    def __init__(self, state_dim, action_dim, batch_size):
        # Learning Rate
        self.LR_ACTOR = 3e-4
        self.LR_CRITIC = 1e-3
        
        # PPO hyperparameters
        self.GAMMA = 0.99
        self.LAMBDA = 0.95
        self.EPOCH = 4
        self.EPSILON_CLIP = 0.2
        self.ENTROPY_COEFF = 0.01 
        
        # Long-term training parameters
        self.DECAY_EPISODES = 15000
        self.SAVE_INTERVAL = 500
        self.EVAL_INTERVAL = 100
        self.LOG_INTERVAL = 50
        
        # Value function clipping (DISABLED)
        self.VALUE_CLIP = False
        self.VALUE_CLIP_RANGE = 0.2
        
        # Networks
        self.actor = Actor(state_dim, action_dim).to(device)
        self.old_actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.LR_CRITIC)
        
        self.replay_buffer = ReplayMemory(batch_size)
        
        # Long-term tracking
        self.episode_count = 0
        self.update_count = 0
        self.best_eval_reward = -float('inf')
        self.patience_counter = 0
        self.max_patience = 2000
        
        # Checkpointing
        self.checkpoint_dir = "ppo_checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def get_action(self, state):
        """Get action and value with optional monitoring"""
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.actor.select_action(state)
        value = self.critic.forward(state)
        return action.detach().cpu().numpy()[0], value.detach().cpu().numpy()[0][0]
    
    def get_current_hyperparams(self):
        """Get current hyperparameters"""
        return {
            'clip_range': self.EPSILON_CLIP,  # Always 0.2
            'actor_lr': self.actor_optimizer.param_groups[0]['lr'],  # Always 3e-4
            'critic_lr': self.critic_optimizer.param_groups[0]['lr'],  # Always 1e-3
            'progress': min(self.episode_count / self.DECAY_EPISODES, 1.0),  # Just for display
            'entropy_coeff': self.ENTROPY_COEFF,  # Always 0.01
            'epochs': self.EPOCH  # Always 4
        }
    
    def compute_gae(self, rewards, values, dones):
        """Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[step]
                next_value = 0
            else:
                next_non_terminal = 1.0 - dones[step]
                next_value = values[step + 1]
            
            delta = rewards[step] + self.GAMMA * next_value * next_non_terminal - values[step]
            gae = delta + self.GAMMA * self.LAMBDA * next_non_terminal * gae
            advantages.insert(0, gae)
        
        return np.array(advantages)

    def update(self):
        if len(self.replay_buffer.state_cap) == 0:
            print("Skipping update: empty replay buffer")
            return
        
        self.update_count += 1
        
        # Get trajectory data
        memo_states, memo_actions, memo_rewards, memo_values, memo_dones, memo_next_states, batches = self.replay_buffer.sample()
        memo_advantages = self.compute_gae(memo_rewards, memo_values, memo_dones)
        memo_advantages = (memo_advantages - memo_advantages.mean()) / (memo_advantages.std() + 1e-8)
        memo_returns = memo_advantages + memo_values
        
        # Convert to tensors
        memo_states_tensor = torch.FloatTensor(memo_states).to(device)
        memo_actions_tensor = torch.FloatTensor(memo_actions).to(device)
        memo_advantages_tensor = torch.tensor(memo_advantages, dtype=torch.float32).to(device)
        memo_returns_tensor = torch.tensor(memo_returns, dtype=torch.float32).to(device)
        
        # Store old policy
        self.old_actor.load_state_dict(self.actor.state_dict())
        with torch.no_grad():
            old_mu, old_sigma = self.old_actor(memo_states_tensor)
            old_dist = Normal(old_mu, old_sigma)
            old_log_probs = old_dist.log_prob(memo_actions_tensor).sum(axis=-1)
        
        # Training statistics
        all_actor_losses = []
        all_critic_losses = []
        all_ratios = []
        all_entropies = []
        clipped_count = 0
        total_samples = 0
        
        for epoch_i in range(self.EPOCH):
            for batch in batches:

                # Current policy forward pass
                mu, sigma = self.actor(memo_states_tensor[batch])
                dist = Normal(mu, sigma)
                log_probs = dist.log_prob(memo_actions_tensor[batch]).sum(axis=-1)
                
                # Calculate ratio
                ratio = torch.exp(log_probs - old_log_probs[batch])
                batch_advantages = memo_advantages_tensor[batch]
                
                # PPO clipped surrogate loss - FIXED clipping
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.EPSILON_CLIP, 1 + self.EPSILON_CLIP) * batch_advantages
                
                # Constant entropy coefficient
                entropy = dist.entropy().sum(axis=-1).mean()
                actor_loss = -torch.min(surr1, surr2).mean() - self.ENTROPY_COEFF * entropy
                
                # Value function loss (no clipping)
                batch_values = self.critic(memo_states_tensor[batch]).squeeze()
                batch_returns = memo_returns_tensor[batch]
                critic_loss = nn.MSELoss()(batch_values, batch_returns)
                
                # Track statistics
                all_actor_losses.append(actor_loss.item())
                all_critic_losses.append(critic_loss.item())
                all_ratios.extend(ratio.detach().cpu().numpy())
                all_entropies.append(entropy.item())
                
                # Count clipped ratios
                clipped = torch.abs(ratio - torch.clamp(ratio, 1 - self.EPSILON_CLIP, 1 + self.EPSILON_CLIP)) > 1e-6
                clipped_count += clipped.sum().item()
                total_samples += batch_advantages.size(0)
                
                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                self.actor_optimizer.step()
                
                # Update critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.critic_optimizer.step()
        
        # Log losses
        ACTOR_LOSS_LOG.append(np.mean(all_actor_losses))
        CRITIC_LOSS_LOG.append(np.mean(all_critic_losses))
        
        # Store value predictions for plotting
        VALUE_PREDICTION_LOG.extend(memo_values.flatten().tolist())
        RETURN_LOG.extend(memo_returns.flatten().tolist())
        
        # Detailed logging every N updates
        if self.update_count % 10 == 0:
            clipped_fraction = clipped_count / total_samples if total_samples > 0 else 0
            ratio_mean = np.mean(all_ratios)
            
            print(f"Update {self.update_count} (Episode {self.episode_count}):")
            print(f"  Losses: Actor {np.mean(all_actor_losses):.4f}, Critic {np.mean(all_critic_losses):.4f}")
            print(f"  LR: Actor {self.actor_optimizer.param_groups[0]['lr']:.6f}, Critic {self.critic_optimizer.param_groups[0]['lr']:.6f}")
            print(f"  Ratio: {ratio_mean:.3f}, Clipped: {clipped_fraction:.1%}")
            print(f"  Entropy: {np.mean(all_entropies):.3f}")
        
        # Clear buffer
        self.replay_buffer.clear_memo()
    
    def evaluate_policy(self, env, num_episodes=5):
        """Evaluate current policy without training"""
        eval_rewards = []
        
        for _ in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            step_count = 0
            
            while not done and step_count < 200:  # Pendulum episode length
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action = self.actor.select_action(state_tensor)
                    state, reward, done, truncated, _ = env.step(action.cpu().numpy()[0])
                    episode_reward += reward
                    step_count += 1
                    
                    if truncated or step_count >= 200:
                        done = True
            
            eval_rewards.append(episode_reward)
        
        return np.mean(eval_rewards), np.std(eval_rewards)
    
    def save_checkpoint(self, episode, avg_reward):
        """Save comprehensive model checkpoint"""
        checkpoint = {
            'episode': episode,
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'avg_reward': avg_reward,
            'update_count': self.update_count,
            'best_eval_reward': self.best_eval_reward,
            'hyperparams': self.get_current_hyperparams(),
            'config': {
                'VALUE_CLIP': self.VALUE_CLIP,
                'VALUE_CLIP_RANGE': self.VALUE_CLIP_RANGE,
                'DECAY_EPISODES': self.DECAY_EPISODES
            }
        }
        
        checkpoint_path = f"{self.checkpoint_dir}/checkpoint_episode_{episode}.pth"
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
        # Keep only last 5 checkpoints
        self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self, keep_last=5):
        """Remove old checkpoints to save disk space"""
        import glob
        checkpoints = glob.glob(f"{self.checkpoint_dir}/checkpoint_episode_*.pth")
        if len(checkpoints) > keep_last:
            # Sort by episode number and remove oldest
            checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            for checkpoint in checkpoints[:-keep_last]:
                os.remove(checkpoint)
                print(f"Removed old checkpoint: {checkpoint}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint and resume training"""
        print(f"Loading checkpoint: {checkpoint_path}")
        
        # Try multiple loading strategies for compatibility
        checkpoint = None
        loading_strategies = [
            # Strategy 1: Try with weights_only=False (for PyTorch 2.6+)
            lambda: torch.load(checkpoint_path, map_location=device, weights_only=False),
            # Strategy 2: Legacy loading (PyTorch < 2.6)
            lambda: torch.load(checkpoint_path, map_location=device),
            # Strategy 3: With safe globals for numpy compatibility
            lambda: self._load_with_safe_globals(checkpoint_path)
        ]
        
        for i, strategy in enumerate(loading_strategies):
            try:
                checkpoint = strategy()
                print(f"✅ Checkpoint loaded successfully using strategy {i+1}")
                break
            except Exception as e:
                print(f"❌ Strategy {i+1} failed: {type(e).__name__}")
                if i == len(loading_strategies) - 1:  # Last strategy
                    raise e
                continue
        
        if checkpoint is None:
            raise RuntimeError("Failed to load checkpoint with all strategies")
        
        # Load state dictionaries
        try:
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            
            # Skip scheduler loading since we removed schedulers
            if 'actor_scheduler_state_dict' in checkpoint:
                print("⚠️  Ignoring scheduler state (schedulers disabled)")
                
            if 'critic_scheduler_state_dict' in checkpoint:
                print("⚠️  Ignoring scheduler state (schedulers disabled)")
            
            # Load training progress
            self.episode_count = checkpoint.get('episode', 0)
            self.update_count = checkpoint.get('update_count', 0)
            self.best_eval_reward = checkpoint.get('best_eval_reward', -float('inf'))
            
            # Load config if available (for newer checkpoints)
            if 'config' in checkpoint:
                config = checkpoint['config']
                self.VALUE_CLIP = config.get('VALUE_CLIP', self.VALUE_CLIP)
                self.VALUE_CLIP_RANGE = config.get('VALUE_CLIP_RANGE', self.VALUE_CLIP_RANGE)
                self.DECAY_EPISODES = config.get('DECAY_EPISODES', self.DECAY_EPISODES)
            
            print(f"📊 Checkpoint Details:")
            print(f"   Episode: {self.episode_count}")
            print(f"   Update count: {self.update_count}")
            print(f"   Best eval reward: {self.best_eval_reward:.2f}")
            print(f"   Average reward: {checkpoint.get('avg_reward', 'N/A')}")
            
            # Display hyperparameters if available
            if 'hyperparams' in checkpoint:
                hyperparams = checkpoint['hyperparams']
                print(f"   Current progress: {hyperparams.get('progress', 0):.1%}")
                print(f"   Actor LR: {hyperparams.get('actor_lr', 'N/A')}")
                print(f"   Clip range: {hyperparams.get('clip_range', 'N/A')}")
            
            return checkpoint.get('avg_reward', 0)
            
        except Exception as e:
            print(f"❌ Error loading checkpoint contents: {e}")
            raise e
    
    def _load_with_safe_globals(self, checkpoint_path):
        """Load checkpoint with safe globals for numpy compatibility"""
        import torch.serialization
        
        # Add safe globals for numpy compatibility
        safe_globals = [
            'numpy._core.multiarray.scalar',
            'numpy.core.multiarray.scalar',  # For older numpy versions
            'numpy.ndarray',
            'numpy.dtype'
        ]
        
        with torch.serialization.safe_globals(safe_globals):
            return torch.load(checkpoint_path, map_location=device, weights_only=True)

    def save_policy(self):
        """Save just the policy for deployment"""
        torch.save(self.actor.state_dict(), "ppo_policy_longterm.pth")
        print("Policy saved to ppo_policy_longterm.pth")