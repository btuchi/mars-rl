import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import clip
import os.path
from pathlib import Path
import torchvision.transforms as transforms
import time
from PIL import Image
from typing import List, Optional, Tuple
from trajectory_recording import DiffusionSampler, DiffusionTrajectory, extract_features_from_trajectory
from diversity_reward import calculate_individual_diversity_rewards
from diffusion_log_utils import ACTOR_LOSS_LOG, CRITIC_LOSS_LOG, VALUE_PREDICTION_LOG, RETURN_LOG, CATEGORY

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
    
    # TODO: Track log probabilities of each step
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
                print("log probability:", step.log_prob)
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
    # TODO: More layers (5-6)?
    # TODO: Hidden dimensions slowly scaling down (1024 -> 512 -> 256 -> 128)?
    # TODO: Change activation?
    def __init__(self, feature_dim: int = 512, hidden_dim: int = 256):
        super(DiffusionValueNetwork, self).__init__()
        
        # Simple MLP for value estimation
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        self.relu = nn.ReLU()
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Estimate value from prompt features"""
        features = features.to(dtype=torch.float32)
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

class DiffusionRewardFunction:
    """
    Calculate diversity reward for a batch of trajectories
    Args:
        trajectories: List of diffusion trajectories (or single trajectory)
        batch_size: Number of images to generate for averaging
        
    Returns:
        Average diversity reward across the batch
    """
    def __init__(self, ref_features: np.ndarray, buffer_size: int = 50):
        self.ref_features = ref_features
        self.buffer_size = buffer_size
        self.reward_history = []
    
    def calculate_batch_rewards(self, trajectories: List[DiffusionTrajectory]) -> np.ndarray:
        """
        Calculate individual diversity rewards for a batch of trajectories
        Uses your efficient calculate_individual_diversity_rewards function
        Args:
            trajectories: List of diffusion trajectories
        Returns:
            individual_rewards: Array of rewards for each trajectory
        """
        # Extract features from all trajectories
        batch_features = []
        for trajectory in trajectories:
            features = extract_features_from_trajectory(trajectory, None)
            features = features.reshape(1, -1)  # Ensure 2D
            batch_features.append(features)
        
        # batch_features = [traj_feat1, traj_feat2, traj_feat3, traj_feat4, ... traj_featM]
        # traj_feat_i = [f1, f2, f3, .., fn] -> one final image
        # Stack into single np array (batch_size x feature_dim)
        batch_features_array = np.vstack(batch_features)

        # Calculate Individual Rewards
        individual_rewards = calculate_individual_diversity_rewards(
            batch_features_array, 
            self.ref_features,
            gamma=None  # Auto-set gamma
        )

        # Normalize individual rewards
        normalized_rewards = []
        for reward in individual_rewards:
            normalized_reward = self.normalize_reward(reward)
            normalized_rewards.append(normalized_reward)
        
        return np.array(normalized_rewards)
    
    def normalize_reward(self, reward: float) -> float:
        """Normalize reward based on running statistics"""
        self.reward_history.append(reward)
        if len(self.reward_history) > self.buffer_size:
            self.reward_history.pop(0)
        
        if len(self.reward_history) < 10:
            return reward * 0.1
        
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
                 feature_dim: int = 512, num_inference_steps: int = 20,
                 images_per_prompt: int = 4, use_fp16: bool = False,
                 save_samples: bool = True, training_start: str = None):
        
        # Add dtype from sampler
        self.dtype = sampler.dtype if hasattr(sampler, 'dtype') else torch.float32
        self.device = device
        self.training_start = training_start

        print(f"PPO Agent using dtype: {self.dtype}")  # Debug print
        
        # PPO hyperparameters (same as vanilla PPO)
        
        # TODO: Try changing learning rates
        # self.LR_ACTOR = 3e-5       # Lower for diffusion models
        # self.LR_CRITIC = 3e-4     # Lower for diffusion models
        self.LR_ACTOR = 0.001       # Lower for diffusion models
        self.LR_CRITIC = 0.001     # Lower for diffusion models
        
        self.GAMMA = 0.9           # Usually 1.0 for diffusion (reward only at end)
        self.LAMBDA = 0.95         # Same GAE parameter
        
        self.EPOCH = 1
        self.EPSILON_CLIP = 0.1

        # Batch settings
        self.images_per_prompt = images_per_prompt
        
        # Initialize networks
        self.actor = DiffusionPolicyNetwork(sampler, num_inference_steps)
        self.old_actor = DiffusionPolicyNetwork(sampler, num_inference_steps)
        self.critic = DiffusionValueNetwork(feature_dim).to(device).to(self.dtype)

        # Handle DataParallel case for optimizers
        if hasattr(self.actor.unet, 'module'):
            actor_params = self.actor.unet.module.parameters()
        else:
            actor_params = self.actor.unet.parameters()
        
        # Optimizers (only optimize UNet, not the whole diffusion pipeline)
        self.actor_optimizer = optim.AdamW(actor_params, lr=self.LR_ACTOR, weight_decay=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.LR_CRITIC)
        
        # Components
        self.replay_buffer = DiffusionReplayMemory(batch_size)
        self.reward_function = DiffusionRewardFunction(ref_features)
        
        # For text embeddings (simple approach - could be improved)
        self.clip_model, _ = clip.load("ViT-B/32", device=device)
        self.clip_model.eval()

        # image saving parameters
        self.save_size = (64, 64)               # Fixed 64x64 size
        self.save_quality = 85                  # Good quality JPEG

        # sample saving setup
        self.save_samples = save_samples
        if self.save_samples:
            self.setup_sample_saving()
        
        
    
    def setup_sample_saving(self):
        """Set up directories and tracking for sample image saving"""
        # ppo_diffusion/
        current_path = Path(__file__).parent
        self.samples_dir = current_path / f"images/while_training/{self.training_start}"
        os.makedirs(self.samples_dir, exist_ok=True)
        
        # Also create a metadata file
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
                self.save_size,  # (64, 64)
                Image.Resampling.LANCZOS  # High-quality resampling
            )
            
            # Create descriptive filename
            safe_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_prompt = safe_prompt[:30]  # Limit length
            
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
        
    # TODO: Play with feature sizes
    def get_prompt_features(self, prompt: str) -> np.ndarray:
        """Convert prompt to features (equivalent to state representation)"""
        with torch.no_grad():
            text_tokens = clip.tokenize([prompt]).to(device)
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy().flatten()
    
    def generate_batch_for_prompt(self, prompt: str, episode: int = None, save_samples: bool = None) -> Tuple[List[DiffusionTrajectory], np.ndarray, float, np.ndarray]:
        """
        Generate a batch of images for a single prompt and calculate individual rewards
        
        Returns:
            trajectories: List of generated trajectories
            individual_rewards: Individual diversity reward for each trajectory
            avg_reward: Average reward for episode tracking
            prompt_features: Prompt feature representation
        """

        torch.cuda.empty_cache()
        import gc
        gc.collect()

        # Determine if we should save samples this episode
        should_save = False
        if save_samples is not None:
            should_save = save_samples
        elif self.save_samples and episode is not None:
            should_save = (episode % 5 == 0)  # Save every 5 episodes

        # Get prompt features and value (same for all images from this prompt)
        prompt_features = self.get_prompt_features(prompt)

        # Clear immediately after getting features
        torch.cuda.empty_cache()

        prompt_features_tensor = torch.from_numpy(prompt_features).to(
            device=device,
            dtype=self.dtype
        ).unsqueeze(0)

        value = self.critic(prompt_features_tensor).detach().cpu().numpy()[0][0]
        # value = 0.0

        # Clean up prompt tensor
        del prompt_features_tensor
        torch.cuda.empty_cache()
            
        # Generate batch of trajectories
        trajectories = []
        log_probs = []
        log_prob_tensors = []
        
        print(f"Generating {self.images_per_prompt} images for prompt: '{prompt}'")
        for i in range(self.images_per_prompt):

            # Clear cache before each generation
            torch.cuda.empty_cache()
            gc.collect()

            trajectory, log_prob_tensor = self.actor.select_trajectory(prompt)

            # Store and then immediately clear the trajectory's computation graph
            log_prob_value = log_prob_tensor.item()
            log_prob_detached = log_prob_tensor.detach()

            # Clear the trajectory's gradient connection after storing what we need
            if hasattr(trajectory, 'total_log_prob'):
                trajectory.total_log_prob = trajectory.total_log_prob.detach().requires_grad_(True)


            trajectories.append(trajectory)

            log_probs.append(log_prob_value)
            log_prob_tensors.append(log_prob_detached.requires_grad_(True))  # Keep tensor with gradients!

            
            print(f"  Image {i+1}/{self.images_per_prompt} generated")

            # Save sample image if needed (before cleaning up trajectory)
            if should_save and i == 0:  # Save first image of batch
                # Calculate individual reward for this image first
                try:
                    features = extract_features_from_trajectory(trajectory, None)
                    features = features.reshape(1, -1)
                    individual_reward = calculate_individual_diversity_rewards(
                        features, self.ref_features, gamma=None
                    )[0]
                    individual_reward = self.reward_function.normalize_reward(individual_reward)
                except:
                    individual_reward = None
                
                self.save_trajectory_image(trajectory, prompt, episode, i+1, individual_reward)

            # Clean up trajectory steps to save memory (but keep the trajectory object)
            if hasattr(trajectory, 'steps'):
                for step in trajectory.steps:
                    # Clear intermediate tensors but keep essential data
                    if hasattr(step, 'state'):
                        step.state = None  # Clear intermediate states
                    if hasattr(step, 'noise_pred'):
                        step.noise_pred = None  # Clear noise predictions
            
            # Force garbage collection
            del trajectory, log_prob_tensor
            torch.cuda.empty_cache()
            gc.collect()

        
        # Calculate individual diversity rewards using your efficient function
        individual_rewards = self.reward_function.calculate_batch_rewards(trajectories)
        avg_reward = np.mean(individual_rewards)
        
        print(f"  Individual rewards: {individual_rewards}")
        print(f"  Average reward: {avg_reward:.4f}")

        # Log episode information
        if episode is not None:
            self.log_episode_info(episode, prompt, avg_reward, individual_rewards.tolist())
        
        # Store each trajectory with its individual reward
        for i, (trajectory, log_prob, log_prob_tensor, reward) in enumerate(zip(trajectories, log_probs, log_prob_tensors, individual_rewards)):
            self.replay_buffer.add_memo(
                prompt_features,    # features
                prompt,            # store the actual prompt
                trajectory,        # trajectory
                reward,           # reward
                value,            # value
                log_prob,          # log_prob
                log_prob_tensor
            )
    
        
        return trajectories, individual_rewards, avg_reward, prompt_features
    
    
    def compute_gae(self, rewards, values):
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


    #########################
    #### Gradient Flow Tests####
    #########################
    # Test 1: Direct UNet Gradient Test
    def test_direct_unet_gradients(self):
        """Test if UNet can compute gradients at all"""
        print("🧪 TEST 1: Direct UNet Gradient Test")
        
        unet = self.actor.unet.module if hasattr(self.actor.unet, 'module') else self.actor.unet
        unet.train()
        
        # Create simple inputs
        latent = torch.randn(1, 4, 64, 64, device=self.device, requires_grad=True)
        timestep = torch.tensor([500], device=self.device)
        text_emb = torch.randn(1, 77, 768, device=self.device)
        
        # Forward pass
        output = unet(latent, timestep, text_emb)[0]
        loss = output.mean()
        
        # Clear and compute gradients
        unet.zero_grad()
        loss.backward()
        
        # Check results
        grad_count = sum(1 for p in unet.parameters() if p.grad is not None)
        grad_norm = sum(p.grad.norm().item() for p in unet.parameters() if p.grad is not None)
        
        print(f"   Direct UNet test - Params with grads: {grad_count}")
        print(f"   Direct UNet test - Grad norm: {grad_norm:.6f}")
        
        unet.zero_grad()
        return grad_count > 0

    # Test 2: Policy Network Gradient Test  
    def test_policy_network_gradients(self):
        """Test if DiffusionPolicyNetwork maintains gradients"""
        print("🧪 TEST 2: Policy Network Gradient Test")
        
        # Generate a trajectory
        prompt = "test crater"
        trajectory, log_prob = self.actor.select_trajectory(prompt)
        
        print(f"   Log prob value: {log_prob.item():.8f}")
        print(f"   Log prob requires_grad: {log_prob.requires_grad}")
        print(f"   Log prob grad_fn: {log_prob.grad_fn}")
        
        # Test backward through log_prob
        unet = self.actor.unet.module if hasattr(self.actor.unet, 'module') else self.actor.unet
        unet.zero_grad()
        
        loss = -log_prob
        loss.backward()
        
        # Check results
        grad_count = sum(1 for p in unet.parameters() if p.grad is not None)
        grad_norm = sum(p.grad.norm().item() for p in unet.parameters() if p.grad is not None)
        
        print(f"   Policy network test - Params with grads: {grad_count}")
        print(f"   Policy network test - Grad norm: {grad_norm:.6f}")
        
        unet.zero_grad()
        return grad_count > 0

    # Test 3: Stored vs Fresh Trajectory Test
    def test_stored_vs_fresh_trajectories(self):
        """Test difference between stored and fresh log probabilities"""
        print("🧪 TEST 3: Stored vs Fresh Trajectory Test")
        
        prompt = "test crater"
        
        # Generate fresh trajectory
        fresh_trajectory, fresh_log_prob = self.actor.select_trajectory(prompt)
        print(f"   Fresh log prob: {fresh_log_prob.item():.8f}, requires_grad: {fresh_log_prob.requires_grad}")
        
        # Test fresh backward
        unet = self.actor.unet.module if hasattr(self.actor.unet, 'module') else self.actor.unet
        unet.zero_grad()
        fresh_loss = -fresh_log_prob
        fresh_loss.backward()
        
        fresh_grad_count = sum(1 for p in unet.parameters() if p.grad is not None)
        fresh_grad_norm = sum(p.grad.norm().item() for p in unet.parameters() if p.grad is not None)
        
        print(f"   Fresh trajectory - Params with grads: {fresh_grad_count}")
        print(f"   Fresh trajectory - Grad norm: {fresh_grad_norm:.6f}")
        
        # Now test recalculated log prob (simulating your PPO update)
        unet.zero_grad()
        recalc_log_prob = self.actor.calculate_log_prob(fresh_trajectory)
        print(f"   Recalc log prob: {recalc_log_prob.item():.8f}, requires_grad: {recalc_log_prob.requires_grad}")
        
        recalc_loss = -recalc_log_prob
        recalc_loss.backward()
        
        recalc_grad_count = sum(1 for p in unet.parameters() if p.grad is not None)
        recalc_grad_norm = sum(p.grad.norm().item() for p in unet.parameters() if p.grad is not None)
        
        print(f"   Recalc trajectory - Params with grads: {recalc_grad_count}")
        print(f"   Recalc trajectory - Grad norm: {recalc_grad_norm:.6f}")
        
        unet.zero_grad()
        return fresh_grad_count > 0, recalc_grad_count > 0

    # Add this to your DiffusionPPOAgent class and call it before update():
    def run_gradient_tests(self):
        """Run all gradient flow tests"""
        print("="*60)
        print("🔬 RUNNING GRADIENT FLOW TESTS")
        print("="*60)
        
        test1_pass = self.test_direct_unet_gradients()
        test2_pass = self.test_policy_network_gradients()  
        test3_fresh, test3_recalc = self.test_stored_vs_fresh_trajectories()
        
        print("\n📊 TEST RESULTS:")
        print(f"   Direct UNet: {'✅ PASS' if test1_pass else '❌ FAIL'}")
        print(f"   Policy Network: {'✅ PASS' if test2_pass else '❌ FAIL'}")
        print(f"   Fresh Trajectory: {'✅ PASS' if test3_fresh else '❌ FAIL'}")
        print(f"   Recalc Trajectory: {'✅ PASS' if test3_recalc else '❌ FAIL'}")
        
        if test1_pass and not test2_pass:
            print("\n🎯 DIAGNOSIS: Issue in DiffusionPolicyNetwork or trajectory recording")
        elif test1_pass and test2_pass and test3_fresh and not test3_recalc:
            print("\n🎯 DIAGNOSIS: Issue with calculate_log_prob method")
        elif not test1_pass:
            print("\n🎯 DIAGNOSIS: Issue with UNet setup (DataParallel, optimizer, etc.)")
        
        print("="*60)


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

        # Run gradient flow tests before update
        self.run_gradient_tests()

        print(f"🔍 UNet training mode: {self.actor.unet.training}")

        # Copy current actor to old_actor (copy UNet weights) - Handle DataParallel
        if hasattr(self.actor.unet, 'module') and hasattr(self.old_actor.unet, 'module'):
            self.old_actor.unet.module.load_state_dict(self.actor.unet.module.state_dict())
        elif hasattr(self.actor.unet, 'module'):
            # Actor has DataParallel but old_actor doesn't - extract the module state
            self.old_actor.unet.load_state_dict(self.actor.unet.module.state_dict())
        else:
            # Neither has DataParallel
            self.old_actor.unet.load_state_dict(self.actor.unet.state_dict())
        
        # Copy current actor to old_actor (copy UNet weights)
        # self.old_actor.unet.load_state_dict(self.actor.unet.state_dict())
        
        # Get trajectory data
        memo_features, memo_prompts, memo_trajectories, memo_rewards, memo_values, memo_log_probs, memo_log_prob_tensors, batches = self.replay_buffer.sample()
        

        print(f"🔍 Rewards: {memo_rewards}")
        print(f"🔍 Values: {memo_values}")
        print(f"🔍 Raw advantages (R-V): {memo_rewards - memo_values}")

        # Convert log_probs to numpy array
        memo_log_probs_array = np.array([lp.item() if torch.is_tensor(lp) else lp for lp in memo_log_probs])

        # Compute advantages using GAE
        memo_advantages = self.compute_gae(memo_rewards, memo_values)
        # memo_advantages = np.array([1.0, -1.0, 0.5, -0.5])
        
        # Normalize advantages
        # memo_advantages = (memo_advantages - memo_advantages.mean())
        # avg_advantage = memo_advantages.mean()
        # print(f"Avg Advantage: {avg_advantage:.4f}")

        # Compute returns
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
        
        # Get old policy log probabilities
        old_log_probs = torch.tensor([lp if isinstance(lp, float) else lp.item() 
                              for lp in memo_log_probs], dtype=self.dtype).to(device)
        
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

                # NO REGENERATION! Use stored tensors with gradients
                print(f"  Using stored gradients for {len(batch)} trajectories...")
                
                # Get current log probs from stored tensors (they still have gradients!)
                current_log_probs = []
                for idx in batch:
                    # Get the trajectory and regenerate log prob with current actor (maintains gradients)
                    trajectory = memo_trajectories[idx]
                    prompt = memo_prompts[idx]

                    # Generate FRESH trajectory with current policy (not stored one)
                    fresh_trajectory, fresh_log_prob = self.actor.select_trajectory(prompt)
                    current_log_probs.append(fresh_log_prob)

                current_log_probs_tensor = torch.stack(current_log_probs)

                old_log_probs_batch = torch.tensor([memo_log_probs[idx] for idx in batch], 
                                          dtype=self.dtype, device=self.device)
                # current_log_probs_tensor = current_log_probs_tensor.to(self.dtype)

                # Check gradient requirements
                # print(f"🔍 Current log probs require grad: {[lp.requires_grad for lp in current_log_probs[:3]]}")
                
                # Calculate ratio
                ratio = torch.exp(current_log_probs_tensor - old_log_probs_batch)
                
                # Batch advantages
                # TODO: Log advantages
                batch_advantages = memo_advantages_tensor[batch]
                print("Batch advantages mean:", batch_advantages.mean().item())
                
                # PPO clipped objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.EPSILON_CLIP, 1 + self.EPSILON_CLIP) * batch_advantages

                # TODO: Print surrogates
                print("surr1 mean:", surr1.mean().item())
                print("surr2 mean:", surr2.mean().item())
                
                # Actor loss (no entropy for diffusion models - they have inherent stochasticity)
                # TODO: Check
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Critic loss
                batch_values = self.critic(memo_features_tensor[batch]).squeeze()
                batch_returns = memo_returns_tensor[batch]

                # Ensure both tensors have the same dtype
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

                total_grad_norm = 0
                param_count = 0
                for param in self.actor.unet.parameters():
                    if param.grad is not None:
                        total_grad_norm += param.grad.norm().item()
                        param_count += 1
                print(f"Total gradient norm: {total_grad_norm}, Params with gradients: {param_count}")


                # Handle gradient clipping for DataParallel
                if hasattr(self.actor.unet, 'module'):
                    torch.nn.utils.clip_grad_norm_(self.actor.unet.module.parameters(), max_norm=0.5)
                else:
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
        models_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")
        os.makedirs(models_dir, exist_ok=True)
        policy_path = os.path.join(models_dir, f"{CATEGORY}_diffusion_ppo_policy_{self.training_start}.pth")

        # Handle DataParallel case for saving
        if hasattr(self.actor.unet, 'module'):
            state_dict = self.actor.unet.module.state_dict()
        else:
            state_dict = self.actor.unet.state_dict()

        torch.save(state_dict, policy_path)
        print(f"Diffusion PPO policy saved to: {policy_path}")