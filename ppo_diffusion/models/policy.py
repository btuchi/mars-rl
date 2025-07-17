"""Policy network for diffusion models"""
from typing import TYPE_CHECKING, Tuple, Optional
import torch
import torch.nn as nn

# Add this new class to diffusion_ppo_agent.py
class SchedulerPolicy(nn.Module):
    """
    Policy that controls diffusion scheduler parameters for diversity
    Much more powerful than latent modifications - controls the entire denoising process
    """
    def __init__(self, text_dim=768, num_steps=20):
        super(SchedulerPolicy, self).__init__()
        self.num_steps = num_steps
        
        # Network to learn custom beta schedule (noise schedule)
        self.beta_network = nn.Sequential(
            nn.Linear(text_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_steps),
            nn.Tanh()  # [-1, 1] range for modifications
        )
        
        # Network to learn custom guidance scales per step
        self.guidance_network = nn.Sequential(
            nn.Linear(text_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_steps),
            nn.Tanh()  # [-1, 1] range for modifications
        )
        
        # Learnable modification scales that can adapt during training
        self.beta_modification_scale = nn.Parameter(torch.tensor(0.5))    # Learnable scale for beta
        self.guidance_modification_scale = nn.Parameter(torch.tensor(2.0))  # Learnable scale for guidance
        
        # Default values
        self.default_beta_start = 0.0001
        self.default_beta_end = 0.02
        self.default_guidance_scale = 7.5
        
        # Initialize weights conservatively
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights for better exploration"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)  # Increased from 0.1 to 0.5
                if m.bias is not None:
                    # Add larger random bias to break symmetry
                    nn.init.uniform_(m.bias, -0.1, 0.1)
    
    def forward(self, text_features):
        """
        Args:
            text_features: [batch_size, 768] text embeddings
        Returns:
            custom_betas: [batch_size, num_steps] custom beta schedule
            custom_guidance: [batch_size, num_steps] custom guidance scales
            log_prob: log probability of the modifications
        """
        batch_size = text_features.shape[0]
        
        # Generate modifications (in [-1, 1] range)
        beta_modifications = self.beta_network(text_features)
        guidance_modifications = self.guidance_network(text_features)
        
        # Add small noise during training to prevent collapse to zero
        if self.training:
            noise_scale = 0.01
            beta_modifications = beta_modifications + noise_scale * torch.randn_like(beta_modifications)
            guidance_modifications = guidance_modifications + noise_scale * torch.randn_like(guidance_modifications)
            # Clamp back to [-1, 1] range after noise addition
            beta_modifications = torch.clamp(beta_modifications, -1.0, 1.0)
            guidance_modifications = torch.clamp(guidance_modifications, -1.0, 1.0)
        
        # Create default schedules
        default_betas = torch.linspace(
            self.default_beta_start, 
            self.default_beta_end, 
            self.num_steps,
            device=text_features.device,
            dtype=text_features.dtype
        ).unsqueeze(0).expand(batch_size, -1)
        
        default_guidance = torch.full(
            (batch_size, self.num_steps),
            self.default_guidance_scale,
            device=text_features.device,
            dtype=text_features.dtype
        )
        
        # Apply conservative modifications
        custom_betas = default_betas + self.beta_modification_scale * beta_modifications
        custom_guidance = default_guidance + self.guidance_modification_scale * guidance_modifications
        
        # Clamp to safe ranges
        custom_betas = torch.clamp(custom_betas, 0.00005, 0.05)  # Safe beta range
        custom_guidance = torch.clamp(custom_guidance, 3.0, 15.0)  # Safe guidance range
        
        # Simplified log probability that encourages exploration
        # Use entropy-based calculation to prevent collapse to zero
        log_prob = torch.sum(torch.log(torch.abs(beta_modifications) + 1e-8)) + torch.sum(torch.log(torch.abs(guidance_modifications) + 1e-8))
        
        # Debug: Check if log_prob requires gradients (only occasionally)
        if torch.rand(1).item() < 0.05:  # 5% chance to print debug info
            print(f"🔍 Scheduler log_prob value: {log_prob.item():.6f}")
        
        return custom_betas, custom_guidance, log_prob

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