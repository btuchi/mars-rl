"""Policy network for diffusion models"""
from typing import TYPE_CHECKING, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualMeanNetwork(nn.Module):
    """Residual network with skip connections for better gradient flow"""
    def __init__(self, input_dim, output_dim):
        super(ResidualMeanNetwork, self).__init__()
        
        # First layer
        self.fc1 = nn.Linear(input_dim, 512)
        self.norm1 = nn.LayerNorm(512)
        
        # Residual blocks
        self.fc2 = nn.Linear(512, 512)
        self.norm2 = nn.LayerNorm(512)
        
        self.fc3 = nn.Linear(512, 512)
        self.norm3 = nn.LayerNorm(512)
        
        # Output layer
        self.fc_out = nn.Linear(512, output_dim)
        
        # Initialize for stable gradients
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.normal_(m.bias, 0, 0.1)
    
    def forward(self, x):
        # First layer
        x1 = F.gelu(self.norm1(self.fc1(x)))
        
        # First residual block
        x2 = F.gelu(self.norm2(self.fc2(x1)))
        x2 = x1 + x2  # Skip connection
        
        # Second residual block  
        x3 = F.gelu(self.norm3(self.fc3(x2)))
        x3 = x2 + x3  # Skip connection
        
        # Output (no activation)
        return self.fc_out(x3)

# Add this new class to diffusion_ppo_agent.py
class SchedulerPolicy(nn.Module):
    """
    Policy that controls diffusion scheduler parameters for diversity
    Much more powerful than latent modifications - controls the entire denoising process
    """
    def __init__(self, text_dim=768, num_steps=20):
        super(SchedulerPolicy, self).__init__()
        self.num_steps = num_steps
        
        # Residual networks for better gradient flow
        self.beta_mean_network = ResidualMeanNetwork(text_dim, num_steps)
        self.guidance_mean_network = ResidualMeanNetwork(text_dim, num_steps)
        
        # Learnable log standard deviations for Gaussian distribution
        self.beta_log_std = nn.Parameter(torch.zeros(num_steps) - 1.0)  # Start with std = exp(-1) ≈ 0.37
        self.guidance_log_std = nn.Parameter(torch.zeros(num_steps) - 1.0)  # Start with std = exp(-1) ≈ 0.37
        
        # Default parameter ranges for mapping
        self.beta_min, self.beta_max = 0.0001, 0.05
        self.guidance_min, self.guidance_max = 3.0, 15.0
        
        # Initialize weights conservatively
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights for better gradient flow"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use Kaiming initialization for GELU activations (better gradient flow)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    # Larger random bias for symmetry breaking
                    nn.init.normal_(m.bias, 0, 0.2)
    
    def forward(self, text_features):
        """
        Gaussian policy implementation for scheduler parameters
        Args:
            text_features: [batch_size, 768] text embeddings
        Returns:
            custom_betas: [batch_size, num_steps] custom beta schedule
            custom_guidance: [batch_size, num_steps] custom guidance scales
            log_prob: log probability of the sampled parameters
        """
        batch_size = text_features.shape[0]
        
        # Generate means from networks
        beta_means = self.beta_mean_network(text_features)  # [batch_size, num_steps]
        guidance_means = self.guidance_mean_network(text_features)  # [batch_size, num_steps]
        
        # Get Gaussian standard deviation parameters (expand to match batch size)
        beta_stds = torch.exp(self.beta_log_std).expand_as(beta_means)
        guidance_stds = torch.exp(self.guidance_log_std).expand_as(guidance_means)
        
        # Sample from Gaussian distributions
        if self.training:
            # During training, sample using Gaussian reparameterization
            beta_eps = torch.randn_like(beta_means)
            guidance_eps = torch.randn_like(guidance_means)
            beta_samples = beta_means + beta_stds * beta_eps
            guidance_samples = guidance_means + guidance_stds * guidance_eps
        else:
            # During inference, use means
            beta_samples = beta_means
            guidance_samples = guidance_means
        
        # Map to valid parameter ranges
        custom_betas = self.beta_min + (self.beta_max - self.beta_min) * torch.sigmoid(beta_samples)
        custom_guidance = self.guidance_min + (self.guidance_max - self.guidance_min) * torch.sigmoid(guidance_samples)
        
        # Calculate Gaussian log probabilities
        beta_log_prob = self._calculate_gaussian_log_prob(beta_samples, beta_means, beta_stds)
        guidance_log_prob = self._calculate_gaussian_log_prob(guidance_samples, guidance_means, guidance_stds)
        
        # Combined log probability
        log_prob = beta_log_prob + guidance_log_prob
        
        # Debug output (occasionally)
        if torch.rand(1).item() < 0.05:
            print(f"🔍 Gaussian Scheduler log_prob value: {log_prob.item():.6f}")
            print(f"🔍 Beta range: [{custom_betas.min().item():.4f}, {custom_betas.max().item():.4f}]")
            print(f"🔍 Guidance range: [{custom_guidance.min().item():.2f}, {custom_guidance.max().item():.2f}]")
            print(f"🔍 Beta std: {beta_stds.mean().item():.4f}, Guidance std: {guidance_stds.mean().item():.4f}")
        
        return custom_betas, custom_guidance, log_prob
    
    def _calculate_gaussian_log_prob(self, samples, means, stds):
        """Calculate log probability of samples under Gaussian distribution"""
        # Gaussian log probability: -0.5 * ((x - μ) / σ)² - log(σ) - 0.5*log(2π)
        # Sum over all dimensions
        log_2pi = torch.log(torch.tensor(2 * torch.pi, device=samples.device, dtype=samples.dtype))
        log_prob = -0.5 * torch.sum(((samples - means) / (stds + 1e-8)) ** 2) \
                   - torch.sum(torch.log(stds + 1e-8)) \
                   - 0.5 * samples.numel() * log_2pi
        
        return log_prob

class LatentDiversityPolicy(nn.Module):
    """
    Small policy network that modifies initial latents for diversity
    Input: Text features from prompt
    Output: Latent space modifications
    """
    def __init__(self, text_dim=768, latent_dim=4, latent_size=64):
        super(LatentDiversityPolicy, self).__init__()
        
        # self.policy_net = nn.Sequential(
        #     nn.Linear(text_dim, 512),
        #     nn.GELU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(512, 256),
        #     nn.GELU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(256, latent_dim * latent_size * latent_size),  # 4*64*64 = 16384
        #     nn.Tanh()  # Bounded output [-1, +1]
        # )

        # Separate networks for mean and log_std for proper probability modeling
        # self.mean_net = nn.Sequential(
        #     nn.Linear(text_dim, 1024),
        #     # nn.GELU(),
        #     nn.ReLU(),
        #     nn.Dropout(0.1),  # PHASE 3: Dropout disabled (was 0.1)
        #     nn.Linear(1024, 512),
        #     # nn.GELU(),
        #     nn.ReLU(),
        #     nn.Dropout(0.1),  # PHASE 3: Dropout disabled (was 0.1)
        #     nn.Linear(512, latent_dim * latent_size * latent_size),
        #     nn.Tanh()
        # )

        self.mean_net = nn.Sequential(
            nn.Linear(text_dim, 2048),  # Increased width
            nn.GELU(),                  # Using GELU
            nn.Dropout(0.1),
            nn.Linear(2048, 1024),      # Added another layer
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, latent_dim * latent_size * latent_size),
            nn.Tanh()
        )

        
        # Log standard deviation (learned parameter)
        self.log_std = nn.Parameter(torch.zeros(latent_dim * latent_size * latent_size) - 1.0)  # Start with small std

        
        self.modification_scale = 0.5  # How much to modify latents (reduced for stability)
        
        # Gradient scaling parameters for better gradient-loss correlation
        # self.gradient_scale = nn.Parameter(torch.tensor(1.0))  # Learnable gradient scaling
        # self.target_grad_norm = 1e-3  # Target gradient magnitude
        
        # Initialize weights properly for stability
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights for stable training"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier normal with small gain for stability
                nn.init.xavier_normal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, text_features):
        """
        Args:
            text_features: [batch_size, 768] Stable Diffusion text features
        Returns:
            latent_modification: [batch_size, 4, 64, 64] modification to add to initial latents
            log_prob: log probability of the sampled modification
        """
        batch_size = text_features.shape[0]
        
        # Generate mean of latent modification
        latent_mean = self.mean_net(text_features)
        latent_mean = latent_mean.reshape(batch_size, 4, 64, 64)
        
        # Get standard deviation (expand to match batch size)
        std = torch.exp(self.log_std).reshape(1, 4, 64, 64).expand_as(latent_mean)
        
        # Sample modification from learned distribution
        if self.training:
            # During training, sample from the distribution
            eps = torch.randn_like(latent_mean)
            latent_modification = latent_mean + std * eps
        else:
            # During inference, use the mean
            latent_modification = latent_mean
        
        # Calculate log probability of the sampled modification
        log_prob = self._calculate_log_prob(latent_modification, latent_mean, std)
        # log_prob = -0.5 * torch.sum(latent_modification ** 2)
        
        # Scale the modification to be small
        scaled_modification = latent_modification * self.modification_scale
        
        return scaled_modification, log_prob
    
    def _calculate_log_prob(self, sample, mean, std):
        """Calculate log probability of sample under Gaussian distribution"""
        # Gaussian log probability: -0.5 * ((x - μ) / σ)² - log(σ) - 0.5*log(2π)
        log_2pi = torch.log(torch.tensor(2 * torch.pi, device=sample.device, dtype=sample.dtype))
        log_prob = -0.5 * torch.mean(((sample - mean) / std) ** 2) \
                   - torch.mean(torch.log(std)) \
                   - 0.5 * log_2pi
        
        return log_prob

class DiffusionPolicyNetwork(nn.Module):
    """
    Switchable Policy Network for Diffusion Models:
        - Supports both DIVERSITY_POLICY and LORA_UNET modes
        - Mode controlled by DEFAULT_TRAINING_MODE in constants.py
    Input:
        - Text prompt (converted to embeddings internally)
    Output:
        - Complete trajectory with log probabilities --> The Action we are trying to optimize
    """
    def __init__(self, sampler, num_inference_steps: int = 20):
        super(DiffusionPolicyNetwork, self).__init__()
        self.sampler = sampler
        self.unet = sampler.unet  # This is our "policy network"
        self.num_inference_steps = num_inference_steps
        
        # Import training mode from constants
        from ..utils.constants import DEFAULT_TRAINING_MODE
        self.training_mode = DEFAULT_TRAINING_MODE
        
        print(f"🎛️ Initializing policy in {self.training_mode} mode")
        
        if self.training_mode == "DIVERSITY_POLICY":
            self._setup_diversity_policy()
        elif self.training_mode == "LORA_UNET":
            self._setup_lora_unet()
        elif self.training_mode == "SCHEDULER_POLICY":
            self._setup_scheduler_policy()
        else:
            raise ValueError(f"Unknown training mode: {self.training_mode}")
    
    def _setup_diversity_policy(self):
        """Setup for diversity policy mode (current approach)"""
        print("🔧 Setting up Diversity Policy mode...")
        
        # Create small policy network instead of training UNet  
        # Stable Diffusion text embeddings are 768-dimensional
        self.diversity_policy = LatentDiversityPolicy(text_dim=768)
        
        # Move diversity policy to same device as sampler
        self.diversity_policy = self.diversity_policy.to(self.sampler.device)
        
        # FREEZE the UNet - we won't train it anymore
        for param in self.unet.parameters():
            param.requires_grad = False
        
        print("✅ UNet frozen - only training LatentDiversityPolicy")
    
    def _setup_scheduler_policy(self):
        """Setup for scheduler policy mode (controls diffusion parameters)"""
        print("🔧 Setting up Scheduler Policy mode...")
        
        # Create scheduler policy network
        self.scheduler_policy = SchedulerPolicy(text_dim=768, num_steps=self.num_inference_steps)
        
        # Move to same device as sampler
        self.scheduler_policy = self.scheduler_policy.to(self.sampler.device)
        
        # FREEZE the UNet - we won't train it
        for param in self.unet.parameters():
            param.requires_grad = False
        
        # No diversity policy in scheduler mode
        self.diversity_policy = None
        
        print("✅ UNet frozen - only training SchedulerPolicy")
        print(f"✅ Scheduler policy will control {self.num_inference_steps} denoising steps")
    
    def _setup_lora_unet(self):
        """Setup for LoRA UNet mode using PEFT library"""
        print("🔧 Setting up LoRA UNet mode...")
        
        try:
            from peft import LoraConfig, get_peft_model
            print("✅ PEFT library loaded successfully")
        except ImportError:
            print("❌ PEFT library not found. Install with: pip install peft")
            raise ImportError("PEFT library required for LoRA mode")
        
        # No diversity policy in LoRA mode - we train the UNet directly
        self.diversity_policy = None
        
        # Configure LoRA for UNet
        # Target attention layers in UNet for LoRA adaptation
        target_modules = [
            "to_q",  # Query projections in attention
            "to_k",  # Key projections in attention
            "to_v",  # Value projections in attention
            "to_out.0",  # Output projections in attention
        ]
        
        lora_config = LoraConfig(
            r=4,  # Rank of adaptation - balance between expressiveness and efficiency
            lora_alpha=8,  # LoRA scaling parameter
            target_modules=["to_q", "to_v"], # Target only query and value projections
            lora_dropout=0.1,  # Dropout for LoRA layers
            bias="none",  # Don't adapt bias terms
            # task_type removed - not needed for diffusion models
        )
        
        # Apply LoRA to UNet
        self.unet = get_peft_model(self.unet, lora_config)
        
        # Re-enable gradient checkpointing to handle complex log probability calculations
        # This trades compute for memory to support the full multivariate log probability
        try:
            if hasattr(self.unet, 'enable_gradient_checkpointing'):
                self.unet.enable_gradient_checkpointing()
                print("✅ Gradient checkpointing enabled for memory efficiency with complex log probability")
            elif hasattr(self.unet, 'gradient_checkpointing'):
                self.unet.gradient_checkpointing = True
                print("✅ Gradient checkpointing enabled via attribute")
        except Exception as e:
            print(f"⚠️ Could not enable gradient checkpointing: {e}")
            print("⚠️ Memory usage may be higher without checkpointing")
        
        # Print parameter counts
        total_params = sum(p.numel() for p in self.unet.parameters())
        trainable_params = sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
        
        print(f"✅ LoRA UNet configured:")
        print(f"   - Total parameters: {total_params:,}")
        print(f"   - Trainable parameters: {trainable_params:,}")
        print(f"   - Trainable percentage: {100 * trainable_params / total_params:.2f}%")
        print(f"   - LoRA rank: {lora_config.r}")
        print(f"   - LoRA alpha: {lora_config.lora_alpha}")
        
        # Store LoRA config for reference
        self.lora_config = lora_config
    
    def get_trainable_parameters(self):
        """Get parameters to train based on current mode"""
        if self.training_mode == "DIVERSITY_POLICY":
            return self.diversity_policy.parameters()
        elif self.training_mode == "LORA_UNET":
            # Return only LoRA parameters (PEFT automatically handles this)
            return filter(lambda p: p.requires_grad, self.unet.parameters())
        elif self.training_mode == "SCHEDULER_POLICY":
            return self.scheduler_policy.parameters()
        else:
            raise ValueError(f"Unknown training mode: {self.training_mode}")
    
    # def clear_gradients(self):
    #     """Clear gradients properly for LoRA training memory management"""
    #     if self.training_mode == "LORA_UNET":
    #         # Clear gradients from all LoRA parameters
    #         for param in self.get_trainable_parameters():
    #             if param.grad is not None:
    #                 param.grad = None
            
    #         # Also clear any optimizer state gradients
    #         if hasattr(self.unet, 'zero_grad'):
    #             self.unet.zero_grad()
                
    #         print("🧹 LoRA gradients cleared for memory management")
        
    def forward(self, prompt: str):
        """
        Generate trajectory for given prompt (mode-aware)
        Input:
            - Prompt
        Output:
            - complete trajectory (20 denoising actions)
        """
        if self.training_mode == "DIVERSITY_POLICY":
            return self._diversity_policy_forward(prompt)
        elif self.training_mode == "LORA_UNET":
            return self._lora_unet_forward(prompt)
        elif self.training_mode == "SCHEDULER_POLICY":
            return self._scheduler_policy_forward(prompt)
        else:
            raise ValueError(f"Unknown training mode: {self.training_mode}")
    
    def _diversity_policy_forward(self, prompt: str):
        """Forward pass for diversity policy mode"""
        return self.sampler.sample_with_policy_modification(
            prompt=prompt,
            policy_network=self.diversity_policy,
            num_inference_steps=self.num_inference_steps
        )
    
    def _lora_unet_forward(self, prompt: str):
        """Forward pass for LoRA UNet mode"""
        # Use standard sampling with LoRA-adapted UNet
        return self.sampler.sample_with_lora_unet(
            prompt=prompt,
            num_inference_steps=self.num_inference_steps
        )
    
    def _scheduler_policy_forward(self, prompt: str):
        """Forward pass for scheduler policy mode"""
        # Use sampling with custom scheduler parameters
        return self.sampler.sample_with_scheduler_policy(
            prompt=prompt,
            policy_network=self.scheduler_policy,
            num_inference_steps=self.num_inference_steps
        )
    
    def calculate_log_prob(self, trajectory) -> torch.Tensor:
        """
        Calculate log probability of trajectory (mode-aware)
        Arg:
            - A diffusion trajectory
        Return
            - the log probability this trajectory happens to generate images
        """
        if self.training_mode == "DIVERSITY_POLICY":
            # Use policy log probability from diversity policy
            return trajectory.policy_log_prob
        elif self.training_mode == "LORA_UNET":
            # Use denoising log probability from LoRA UNet
            return trajectory.denoising_log_prob
        elif self.training_mode == "SCHEDULER_POLICY":
            # Use scheduler policy log probability
            return trajectory.scheduler_log_prob
        else:
            raise ValueError(f"Unknown training mode: {self.training_mode}")
    def select_trajectory(self, prompt: str) -> Tuple:
        """
        Sample trajectory with log prob (equivalent to select_action)
        Mode-aware trajectory selection for both diversity policy and LoRA UNet
        Arg:
            - prompt
        Return
            - a diffusion trajectory
            - its log probability
        
        """
        trajectory = self.forward(prompt)
        log_prob = self.calculate_log_prob(trajectory)
        return trajectory, log_prob