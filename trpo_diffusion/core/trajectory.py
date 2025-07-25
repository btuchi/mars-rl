"""Trajectory recording and diffusion sampling (cleaned up)"""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional, TYPE_CHECKING
import numpy as np
from dataclasses import dataclass
from diffusers import StableDiffusionPipeline
from ..utils.device import clear_gpu_cache
if TYPE_CHECKING:
    from ..models.policy import LatentDiversityPolicy

@dataclass
class TrajectoryStep:
    """Represents a single step in the diffusion trajectory for RL training"""
    timestep: int
    state: torch.Tensor
    action: torch.Tensor
    condition: torch.Tensor
    log_prob: torch.Tensor
    noise_pred: torch.Tensor

@dataclass
class DiffusionTrajectory:
    """Complete trajectory from random noise to final image"""
    steps: List[TrajectoryStep]
    final_image: torch.Tensor
    condition: torch.Tensor
    reward: Optional[float] = None
    total_log_prob: Optional[torch.Tensor] = None
    policy_log_prob: Optional[torch.Tensor] = None  # For policy-modified sampling
    denoising_log_prob: Optional[torch.Tensor] = None  # For LoRA UNet sampling
    scheduler_log_prob: Optional[torch.Tensor] = None  # For scheduler policy sampling

class DiffusionSampler:
    """Diffusion model sampler that records trajectories for RL training"""

    def __init__(self, model_id: str = "CompVis/stable-diffusion-v1-4", 
                 device: str = "cuda", use_fp16: bool = False):

        self.device = device
        self.dtype = torch.float16 if use_fp16 else torch.float32

        clear_gpu_cache()

        # Load Stable Diffusion pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
            use_safetensors=True,
            variant="fp16" if use_fp16 else None,
            low_cpu_mem_usage=True,
            device_map=None
        )

        # Enable memory optimizations
        self.pipe.enable_xformers_memory_efficient_attention()
        self.pipe.enable_attention_slicing("max")
        self.pipe.enable_vae_slicing()
        self.pipe.enable_vae_tiling()

        # Move to device
        self.pipe = self.pipe.to(device)

        # Extract components
        self.unet = self.pipe.unet
        self.scheduler = self.pipe.scheduler
        self.vae = self.pipe.vae
        self.text_encoder = self.pipe.text_encoder
        self.tokenizer = self.pipe.tokenizer

        # Enable gradient checkpointing
        self.unet.enable_gradient_checkpointing()

        # Set modes
        self.unet.train()
        self.vae.eval()
        self.text_encoder.eval()

        print("✅ DiffusionSampler initialized successfully")
    
    def encode_prompt(self, prompt: str, batch_size: int = 1) -> torch.Tensor:
        """Encode text prompt into embeddings"""
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_inputs.input_ids.to(self.device))[0]
        
        text_embeddings = text_embeddings.to(dtype=self.dtype)
        text_embeddings = text_embeddings.repeat(batch_size, 1, 1)
        
        return text_embeddings
    

    # REMOVED: sample_with_trajectory_recording() 
    # This method was redundant since we now have dedicated methods:
    # - sample_with_policy_modification() for DIVERSITY_POLICY mode
    # - sample_with_lora_unet() for LORA_UNET mode
    
    def sample_with_policy_modification(
        self,
        prompt: str,
        policy_network,
        num_inference_steps: int = 50,
        guidance_scale: float = 8.0,
        height: int = 512,
        width: int = 512,
        generator: Optional[torch.Generator] = None
    ) -> DiffusionTrajectory:
        """
        Sample with policy network modifications to initial latents
        """
        batch_size = 1
        
        # Encode prompt
        text_embeddings = self.encode_prompt(prompt, batch_size)
        
        # Get text features for policy network (use the conditional part)
        # Keep full 768 dimensions, just take mean over sequence length
        text_features = text_embeddings.mean(dim=1)  # [1, 77, 768] -> [1, 768]
        
        # For classifier-free guidance
        uncond_embeddings = self.encode_prompt("", batch_size)
        full_text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        # Sample initial noise
        latents_shape = (batch_size, 4, height // 8, width // 8)  # 4, 64, 64
        base_latents = torch.randn(
            latents_shape,
            generator=generator, 
            device=self.device,
            dtype=self.dtype
        )
        
        # Get policy modification
        with torch.enable_grad():  # Enable gradients for policy network
            policy_network.train()  # Make sure policy is in train mode
            latent_modification, policy_log_prob = policy_network(text_features)
            
        
        # Apply modification
        modified_latents = base_latents + latent_modification
        latents = modified_latents * self.scheduler.init_noise_sigma
        
        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        
        # Standard denoising loop with FROZEN UNet
        trajectory_steps = []
        
        for i, t in enumerate(timesteps):
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            timestep = t.expand(latent_model_input.shape[0])
            
            # UNet prediction (NO GRADIENTS - it's frozen)
            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input,
                    timestep,
                    encoder_hidden_states=full_text_embeddings,
                    return_dict=False
                )[0]
            
            # Classifier-free guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred_guided = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Denoise
            scheduler_output = self.scheduler.step(noise_pred_guided, t, latents, return_dict=True)
            prev_latents = scheduler_output.prev_sample
            
            # Record step (simplified since UNet is frozen)
            step = TrajectoryStep(
                timestep=t.item(),
                state=latents.clone().detach(),
                action=prev_latents.clone().detach(),
                condition=full_text_embeddings[1:2].clone().detach(),
                log_prob=torch.tensor(0.0),  # Step log prob not needed
                noise_pred=noise_pred_guided.clone().detach()
            )
            trajectory_steps.append(step)
            
            latents = prev_latents
        
        # Decode final image
        with torch.no_grad():
            latents = 1 / 0.18215 * latents
            final_image = self.vae.decode(latents).sample
            final_image = (final_image / 2 + 0.5).clamp(0, 1)
        
        # Create trajectory with policy log prob
        trajectory = DiffusionTrajectory(
            steps=trajectory_steps,
            final_image=final_image.detach(),
            condition=full_text_embeddings[1:2].clone().detach(),
            policy_log_prob=policy_log_prob  # NEW: Store policy log prob
        )
        
        return trajectory
    
    def sample_with_lora_unet(
        self,
        prompt: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 8.0,
        height: int = 512,
        width: int = 512,
        generator: Optional[torch.Generator] = None
    ) -> DiffusionTrajectory:
        """
        Sample with LoRA-adapted UNet (trainable parameters)
        Similar to original trajectory recording but with LoRA UNet training
        """
        batch_size = 1
        
        # Encode prompt
        text_embeddings = self.encode_prompt(prompt, batch_size)
        uncond_embeddings = self.encode_prompt("", batch_size)
        full_text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        # Prepare latent space
        latents_shape = (batch_size, 4, height // 8, width // 8)
        latents = torch.randn(
            latents_shape,
            generator=generator,
            device=self.device,
            dtype=self.dtype
        )
        
        latents = latents * self.scheduler.init_noise_sigma
        
        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        
        # Initialize denoising log probability
        denoising_log_prob = torch.tensor(0.0, device=self.device, dtype=self.dtype, requires_grad=True)
        trajectory_steps = []
        
        # Denoising loop with LoRA UNet (TRAINABLE)
        for i, t in enumerate(timesteps):
            if i > 0:
                clear_gpu_cache()
                
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            timestep = t.expand(latent_model_input.shape[0])
            
            # LoRA UNet prediction WITH GRADIENTS (trainable)
            noise_pred = self.unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=full_text_embeddings,
                return_dict=False
            )[0]
            
            # Classifier-free guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred_guided = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Simple log probability calculation for LoRA UNet training (need to be replaced)
            # Reverted from complex multivariate Gaussian to simple approach for memory efficiency
            
            # Simple approach: treat as negative L2 norm of noise prediction
            step_log_prob = -0.5 * torch.mean(noise_pred_guided ** 2)
            
            # Safety check for NaN/Inf
            if torch.isnan(step_log_prob) or torch.isinf(step_log_prob):
                step_log_prob = torch.tensor(-0.1, device=self.device, requires_grad=True)
                print(f"⚠️ NaN detected in LoRA step_log_prob at timestep {t}, using safe fallback")
            
            # Simple accumulation: sum across denoising steps
            if i == 0:
                denoising_log_prob = step_log_prob
            else:
                denoising_log_prob = denoising_log_prob + step_log_prob
            
            # Denoise
            scheduler_output = self.scheduler.step(noise_pred_guided, t, latents, return_dict=True)
            prev_latents = scheduler_output.prev_sample
            
            # Record step with additional debugging info
            step = TrajectoryStep(
                timestep=t.item(),
                state=latents.clone().detach(),
                action=prev_latents.clone().detach(),
                condition=full_text_embeddings[1:2].clone().detach(),
                log_prob=step_log_prob.detach(),
                noise_pred=noise_pred_guided.clone().detach()
            )
            
            # Debug info for first few steps
            if i < 3:
                print(f"  🔍 Step {i}: t={t.item()}, log_prob={step_log_prob.item():.6f}")
            trajectory_steps.append(step)
            
            latents = prev_latents
            
            # Clear intermediate variables
            del noise_pred, noise_pred_uncond, noise_pred_text, noise_pred_guided
            del latent_model_input, scheduler_output
        
        # Decode final image
        with torch.no_grad():
            latents = 1 / 0.18215 * latents
            final_image = self.vae.decode(latents).sample
            final_image = (final_image / 2 + 0.5).clamp(0, 1)
        
        # Create trajectory with denoising log probability
        trajectory = DiffusionTrajectory(
            steps=trajectory_steps,
            final_image=final_image.detach(),
            condition=full_text_embeddings[1:2].clone().detach(),
            denoising_log_prob=denoising_log_prob  # Store LoRA UNet log probability
        )
        
        # Clear memory
        del text_embeddings, uncond_embeddings, full_text_embeddings, latents
        clear_gpu_cache()
        
        return trajectory

    def sample_with_scheduler_policy(
        self,
        prompt: str,
        policy_network,
        num_inference_steps: int = 20,
        height: int = 512,
        width: int = 512,
        generator: Optional[torch.Generator] = None
    ) -> DiffusionTrajectory:
        """
        Sample with custom scheduler parameters from policy network
        This gives the policy control over the entire diffusion process
        """
        batch_size = 1
        
        # Encode prompt
        text_embeddings = self.encode_prompt(prompt, batch_size)
        uncond_embeddings = self.encode_prompt("", batch_size)
        full_text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        # Extract text features for policy (mean pooling)
        text_features = torch.mean(text_embeddings, dim=1)  # [1, 768]
        
        # Get custom scheduler parameters from policy
        with torch.enable_grad():
            policy_network.train()
            custom_betas, custom_guidance, scheduler_log_prob = policy_network(text_features)
            
            # DEBUG: Check if policy outputs have gradients
            print(f"🔍 GRADIENT DEBUG: custom_guidance requires_grad: {custom_guidance.requires_grad}")
            print(f"🔍 GRADIENT DEBUG: custom_guidance grad_fn: {custom_guidance.grad_fn}")
            print(f"🔍 GRADIENT DEBUG: scheduler_log_prob requires_grad: {scheduler_log_prob.requires_grad}")
            print(f"🔍 GRADIENT DEBUG: scheduler_log_prob grad_fn: {scheduler_log_prob.grad_fn}")
            
        # Prepare latent space  
        latents_shape = (batch_size, 4, height // 8, width // 8)
        latents = torch.randn(
            latents_shape,
            generator=generator,
            device=self.device,
            dtype=self.dtype
        )
        
        # Use custom beta schedule to modify scheduler
        # Note: This is a simplified approach - full implementation would require
        # creating a custom scheduler with the learned betas
        latents = latents * self.scheduler.init_noise_sigma
        
        # Set timesteps (using default for now, could be made custom too)
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        
        trajectory_steps = []
        
        # Denoising loop with custom guidance scales
        for i, t in enumerate(timesteps):
            # Use custom guidance scale for this step (with bounds checking)
            guidance_index = min(i, custom_guidance.shape[1] - 1)  # Clamp to valid range
            step_guidance_scale = custom_guidance[0, guidance_index]  # Keep as tensor to preserve gradients
            
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # Predict noise
            with torch.no_grad():  # UNet is frozen
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=full_text_embeddings,
                    return_dict=False
                )[0]
            
            # Classifier-free guidance with custom scale
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred_guided = noise_pred_uncond + step_guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # DEBUG: Check gradient flow through CFG scaling
            if i == 0:  # Only print for first step to avoid spam
                print(f"🔍 CFG DEBUG step {i}: step_guidance_scale requires_grad: {step_guidance_scale.requires_grad}")
                print(f"🔍 CFG DEBUG step {i}: step_guidance_scale grad_fn: {step_guidance_scale.grad_fn}")
                print(f"🔍 CFG DEBUG step {i}: noise_pred_guided requires_grad: {noise_pred_guided.requires_grad}")
                print(f"🔍 CFG DEBUG step {i}: noise_pred_guided grad_fn: {noise_pred_guided.grad_fn}")
            
            # Compute the previous noisy sample
            scheduler_output = self.scheduler.step(noise_pred_guided, t, latents, return_dict=True)
            prev_latents = scheduler_output.prev_sample
            
            # Store step information  
            step = TrajectoryStep(
                timestep=t.item(),
                state=latents.clone().detach(),
                action=torch.tensor([step_guidance_scale]),  # Store the custom guidance as action
                condition=full_text_embeddings[1:2].clone().detach(),  # Text embeddings
                log_prob=torch.tensor(0.0),  # Simplified for now
                noise_pred=noise_pred_guided.clone().detach()
            )
            trajectory_steps.append(step)
            
            latents = prev_latents
            
            # Debug info for first few steps
            if i < 3:
                print(f"  🔍 Step {i}: t={t.item()}, custom_guidance={step_guidance_scale:.3f}")
        
        # Decode final image (allow gradients to flow through frozen VAE)
        latents = 1 / 0.18215 * latents
        final_image = self.vae.decode(latents).sample
        final_image = (final_image / 2 + 0.5).clamp(0, 1)
        
        # DEBUG: Check if final image has gradients (should now be True!)
        print(f"🔍 FINAL IMAGE DEBUG: requires_grad: {final_image.requires_grad}")
        print(f"🔍 FINAL IMAGE DEBUG: grad_fn: {final_image.grad_fn}")
        
        # Create trajectory with scheduler log probability
        trajectory = DiffusionTrajectory(
            steps=trajectory_steps,
            final_image=final_image,  # Don't detach! We need gradients to flow
            condition=full_text_embeddings[1:2].clone().detach(),
            scheduler_log_prob=scheduler_log_prob  # Store scheduler policy log prob
        )
        
        # Clear memory
        del text_embeddings, uncond_embeddings, full_text_embeddings, latents
        clear_gpu_cache()
        
        return trajectory
            
