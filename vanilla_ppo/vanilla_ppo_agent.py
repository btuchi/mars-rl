from torch import nn
import torch
from torch.distributions import Normal
import numpy as np
import torch.optim as optim
from log_utils import ACTOR_LOSS_LOG, CRITIC_LOSS_LOG, VALUE_PREDICTION_LOG, RETURN_LOG
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Computing device: {device}")

class Actor(nn.Module):
    """
    Policy Network: 
        - It decides what action to take given the current state. 
        - PPO will update this using the clipped policy loss to make it better over time.
    Input:
        - State
    Output:
        - action probabilities (discreet)
        - OR, Gaussian Distribution of actions (continuous)
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        """
        state_dim: input dimension
        action_dim: output dimension
        hidden_dim: size of hidden layers
        """
        super(Actor, self).__init__()

        # two fully connected layers 
        # (state vector -> learned representation)
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Gaussian Distribution of each action
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_std = nn.Linear(hidden_dim, action_dim)

        # Activation
        self.relu = nn.ReLU() # used in hidden layers
        self.tanh = nn.Tanh() # applied to mean to bound action range (it's a pendulum..)
        self.softplus = nn.Softplus() # make sure std > 0
    
    def forward(self, x):
        """state -> distribution of actions"""

        # encode state
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        # distribution info
        mean = self.tanh(self.fc_mean(x)) * 2 # bound raw mean
        std = self.softplus(self.fc_std(x)) + 1e-3
        std = std.clamp(min=1e-2, max=1.0)  # prevent zero or exploding std

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
    Value Function Estimator: 
        - we need the Value to compute the Advantage (A)
        - helps reduce variance in policy gradients
    Input:
        - State
    Output:
        - A scalor value V(s): the critic’s estimate of how good that state is
    """
    def __init__(self, state_dim, hidden_dim=256):
        """
        state_dim: input dimension
        action_dim: output dimension
        hidden_dim: size of hidden layers
        """
        super(Critic, self).__init__()

        # three fully connected layers 
        # (state vector -> learned representation)
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        # Activation
        self.relu = nn.ReLU() # used in hidden layers
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        value = self.fc3(x)
        return value

class ReplayMemory:
    """
    Stores transitions collected during rollout
    Use the full buffer to
        - Compute advantages
        - Compute discounted returns
        - Train the actor and critic
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
        """
        Preparing batched trajectory data for PPO training

        Output:
            - The entire stored trajectory (states, actions, rewards, values, dones)
            - a list of random minibatches (index arrays) for PPO updates
        
        """
        # total number of steps stored (e.g., NUM_EPISODE × NUM_STEP)
        num_state = len(self.state_cap)
        
        # Generates starting indices for batching.
        batch_start_points = np.arange(0, num_state, self.BATCH_SIZE)
        
        # Create a list of all sample indices from 0 to num_state - 1
        sample_indicies = np.arange(num_state, dtype=np.int32)

        # Shuffle them so the mini-batches are random (standard in SGD)
        np.random.shuffle(sample_indicies)
        batches = [sample_indicies[i:i+self.BATCH_SIZE] for i in batch_start_points]

        return np.array(self.state_cap), \
            np.array(self.action_cap), \
            np.array(self.reward_cap), \
            np.array(self.value_cap), \
            np.array(self.done_cap), \
            np.array(self.next_state_cap), \
            batches
    
    

    def clear_memo(self):
        self.state_cap = []
        self.action_cap = []
        self.reward_cap = []
        self.value_cap = []
        self.done_cap = []
        self.next_state_cap = []


class PPOAgent:
    """
    The orchestrator.
    Combines actor, critic, memory, and all the PPO logic.
    """
    def __init__(self, state_dim, action_dim, batch_size):
        # Clip
        self.VALUE_CLIP = False # Enable/disable value clipping
        self.VALUE_CLIP_RANGE = 0.2  # Same as policy clip range (common choice)

        # learning rate
        self.LR_ACTOR = 1e-4
        self.LR_CRITIC = 1e-3

        # discount factor
        self.GAMMA = 0.99
        self.LAMBDA = 0.95

        # others
        self.EPOCH = 4
        self.EPSILON_CLIP = 0.2
        
        # critical components
        self.actor = Actor(state_dim, action_dim).to(device)
        self.old_actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.LR_CRITIC)
        self.replay_buffer = ReplayMemory(batch_size)

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.actor.select_action(state)
        value = self.critic.forward(state)
        return action.detach().cpu().numpy()[0], value.detach().cpu().numpy()[0][0]
    
    def compute_gae(self, rewards, values, dones):
        """
        GAE computation -> Advantages
        """
        advantages = []
        gae = 0
        
        # We need to get the last value for bootstrapping
        # Since we don't have next_values, we'll assume terminal value is 0
        next_value = 0  # Terminal value
        
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[step]
                next_value = 0  # Terminal state
            else:
                next_non_terminal = 1.0 - dones[step]
                next_value = values[step + 1]
            
            delta = rewards[step] + self.GAMMA * next_value * next_non_terminal - values[step]
            gae = delta + self.GAMMA * self.LAMBDA * next_non_terminal * gae
            advantages.insert(0, gae)
        
        return np.array(advantages)

    def update(self):

        print("Skipping update: empty replay buffer")
        if len(self.replay_buffer.state_cap) == 0:
            return
    
        # Copy current actor to old_actor
        self.old_actor.load_state_dict(self.actor.state_dict())

        # Get trajectory info
        memo_states, memo_actions, memo_rewards, memo_values, memo_dones, memo_next_states, batches = self.replay_buffer.sample()

        # Get Advantages
        memo_advantages = self.compute_gae(memo_rewards, memo_values, memo_dones)

        # Normalize Advantages
        memo_advantages = (memo_advantages - memo_advantages.mean()) / (memo_advantages.std() + 1e-8)

        # Compute returns
        memo_returns = memo_advantages + memo_values

        # Convert to tensors
        memo_states_tensor = torch.FloatTensor(memo_states).to(device)
        memo_actions_tensor = torch.FloatTensor(memo_actions).to(device)
        memo_advantages_tensor = torch.tensor(memo_advantages, dtype=torch.float32).to(device)  # Shape: (N,)
        memo_returns_tensor = torch.tensor(memo_returns, dtype=torch.float32).to(device)        # Shape: (N,)

        # Get old policy log probabilities
        with torch.no_grad():
            old_mu, old_sigma = self.old_actor(memo_states_tensor)
            old_dist = Normal(old_mu, old_sigma)
            old_log_probs = old_dist.log_prob(memo_actions_tensor).sum(axis=-1)
        
        # Accumulate losses across ALL epochs
        all_actor_losses = []
        all_critic_losses = []

        # Train for multiple Epochs
        for epoch_i in range(self.EPOCH):
            for batch in batches:
                # Current Policy
                mu, sigma = self.actor(memo_states_tensor[batch])
                dist = Normal(mu, sigma)
                log_probs = dist.log_prob(memo_actions_tensor[batch]).sum(axis=-1)

                # Calculate ratio: r = pi(cur) / pi(old)
                ratio = torch.exp(log_probs - old_log_probs[batch])


                # advantage indexing
                batch_advantages = memo_advantages_tensor[batch]
                
                # Surrogate losses
                surr1 = ratio *  memo_advantages_tensor[batch]
                surr2 = torch.clamp(ratio, 1 - self.EPSILON_CLIP, 1 + self.EPSILON_CLIP) * memo_advantages_tensor[batch]

                # Actor loss (with entropy bonus)
                entropy = dist.entropy().sum(axis=-1).mean()
                actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy

                # Critic loss
                batch_values = self.critic(memo_states_tensor[batch]).squeeze()
                batch_returns = memo_returns_tensor[batch]

                # Clip critic loss if we want to
                if self.VALUE_CLIP:
                    # Store old values before any updates
                    with torch.no_grad():
                        old_values = self.critic(memo_states_tensor).squeeze()
                    
                    batch_old_values = old_values[batch]
                    value_loss_unclipped = (batch_values - batch_returns) ** 2

                    values_clipped = batch_old_values + torch.clamp(
                        batch_values - batch_old_values, -0.2, 0.2
                    )
                    value_loss_clipped = (values_clipped - batch_returns) ** 2

                    critic_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()
                else:
                    critic_loss = nn.MSELoss()(batch_values, batch_returns)

                # Store losses
                all_actor_losses.append(actor_loss.item())
                all_critic_losses.append(critic_loss.item())

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
        
        # Store data for plotting
        VALUE_PREDICTION_LOG.extend(memo_values.flatten().tolist())
        RETURN_LOG.extend(memo_returns.flatten().tolist()) # A = Q - V(base) -> Q = A+V

        # Debug info
        print(f"Update - Actor Loss: {np.mean(all_actor_losses):.4f}, "
              f"Critic Loss: {np.mean(all_critic_losses):.4f}, "
              f"Avg Advantage: {memo_advantages.mean():.4f}")
        
        # clear buffer
        self.replay_buffer.clear_memo()
 

    def save_policy(self):
        torch.save(self.actor.state_dict(), "ppo_policy_pendulum_v1.para")

