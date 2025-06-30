"""Replay memory for storing trajectories"""

import numpy as np
from typing import List


class DiffusionReplayMemory:
    """
    Stores diffusion trajectories collected during rollout
    - Similar structure to vanilla PPO's ReplayMemory
    - Each "transition" is a complete diffusion trajectory
    """
    def __init__(self, batch_size):
        self.prompt_features = []      # "States" - text prompt features
        self.prompts = []               # Store actual prompts for regeneration
        self.trajectories = []         # "Actions" - complete diffusion trajectories  
        self.rewards = []             # Diversity rewards
        self.values = []              # Value predictions
        self.log_probs = []          # Log probabilities of trajectories
        self.log_prob_tensors = []  # Store actual tensors with gradients
        self.BATCH_SIZE = batch_size
        
    def add_memo(self, prompt_features, prompt, trajectory, reward, value, log_prob, log_prob_tensor):
        """Add a trajectory experience (equivalent to add_memo in vanilla PPO)"""
        self.prompt_features.append(prompt_features)
        self.trajectories.append(trajectory)
        self.prompts.append(prompt)  # Store the actual prompt
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.log_prob_tensors.append(log_prob_tensor)
    
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
                self.prompts,
                self.trajectories,                  # "actions" 
                np.array(self.rewards),            # rewards
                np.array(self.values),             # values
                np.array(self.log_probs),          # for next_state equivalent
                self.log_prob_tensors,
                batches)
    
    def clear_memo(self):
        """Clear all stored trajectories"""
        self.prompt_features = []
        self.prompts = []
        self.trajectories = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.log_prob_tensors = []