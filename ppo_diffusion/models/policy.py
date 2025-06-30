"""Policy network for diffusion models"""

import torch
import torch.nn as nn
from typing import Tuple
from ..core.trajectory import DiffusionSampler, DiffusionTrajectory

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
        # TODO: ResNet? VGG16?
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
        Arg:
            - A diffusion trajectory
        Return
            - the log probability this trajectory happens to generate images
        """
        if trajectory.total_log_prob is not None:
            print("log probability from trajectory:", trajectory.total_log_prob)
            return trajectory.total_log_prob
        else:

            log_probs = []

            # Each step has its own log probability
            for step in trajectory.steps:
                # Ensure log_prob is a tensor with gradients
                # print("log probability:", step.log_prob)
                if isinstance(step.log_prob, torch.Tensor):
                    log_probs.append(step.log_prob)
                else:
                    # Convert to tensor if needed
                    log_probs.append(torch.tensor(step.log_prob, requires_grad=True))
            
            # Total log probability = sum of all steps
            return torch.stack(log_probs).sum()
    
    def select_trajectory(self, prompt: str) -> Tuple[DiffusionTrajectory, torch.Tensor]:
        """
        Sample trajectory with log prob (equivalent to select_action)
        disabled torch.no_grad() to enable gradient flow for training
        Arg:
            - prompt
        Return
            - a diffusion trajectory
            - its log probability
        
        """
        trajectory = self.forward(prompt)
        log_prob = self.calculate_log_prob(trajectory)
        return trajectory, log_prob