import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional
import numpy as np
from dataclasses import dataclass
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDPMScheduler
# Convert to PIL Image
from PIL import Image
import torchvision.transforms as transforms
# Extract features using CLIP (reuse your existing feature extraction code)
import clip

@dataclass
class TrajectoryStep:
    """
    Represents a single step in the diffusion trajectory for RL training.
    """
    timestep: int              # Current timestep t
    state: torch.Tensor        # Current noisy image x_t
    action: torch.Tensor       # Predicted denoised image x_{t-1}
    condition: torch.Tensor    # Text/class condition c
    log_prob: torch.Tensor     # Log probability of this action
    noise_pred: torch.Tensor   # Raw noise prediction from model

@dataclass
class DiffusionTrajectory:
    """
    Complete trajectory from random noise to final image.
    """
    steps: List[TrajectoryStep]
    final_image: torch.Tensor
    condition: torch.Tensor
    reward: Optional[float] = None

class DiffusionSampler:
    """
    Diffusion model sampler that records trajectories for RL training.
    """
    
    def __init__(self, model_id: str = "CompVis/stable-diffusion-v1-4", device: str = "cuda", use_fp16: bool = False):

        self.device = device
        self.dtype = torch.float16 if use_fp16 else torch.float32
        
        # Add memory optimization
        torch.cuda.empty_cache()

        # Load Stable Diffusion pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=self.dtype,           # Use float16 to save memory
            safety_checker=None,
            requires_safety_checker=False,
            use_safetensors=True,
            variant="fp16" if use_fp16 else None,
            low_cpu_mem_usage=True,              # This prevents meta device issues
            device_map=None                      # Disable automatic device mapping
        )

        # Enable memory efficient attention
        self.pipe.enable_xformers_memory_efficient_attention()
        self.pipe.enable_attention_slicing("max")       # Maximum slicing
        self.pipe.enable_vae_slicing()                 # VAE slicing
        self.pipe.enable_vae_tiling()                  # VAE tiling

        self.pipe = self.pipe.to(device)

        if not use_fp16:  # Only convert to float32 if not using fp16
            self.pipe.unet.to(torch.float32)
            self.pipe.vae.to(torch.float32)
            self.pipe.text_encoder.to(torch.float32)

        # Apply DataParallel for 2-GPU setup (do this AFTER device movement)
        if torch.cuda.device_count() > 1:
            print(f"Using DataParallel across {torch.cuda.device_count()} GPUs")
            self.pipe.unet = nn.DataParallel(self.pipe.unet)
            # Note: We skip CPU offload optimizations when using DataParallel
            # as they conflict with multi-GPU setup
        else:
            # Single GPU - can use CPU offload
            self.pipe.enable_model_cpu_offload()
            self.pipe.enable_sequential_cpu_offload()
        
        # Extract components we need
        self.unet = self.pipe.unet
        self.scheduler = self.pipe.scheduler
        self.vae = self.pipe.vae
        self.text_encoder = self.pipe.text_encoder
        self.tokenizer = self.pipe.tokenizer

        # Enable gradient checkpointing to save memory
        if hasattr(self.unet, 'module'):
            self.unet.module.enable_gradient_checkpointing()
        else:
            self.unet.enable_gradient_checkpointing()
        
        # Keep UNet in training mode but others in eval
        self.unet.train()
        self.vae.eval()
        self.text_encoder.eval()

        # Additional memory optimizations for 2-GPU setup
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
        torch.backends.cuda.matmul.allow_tf32 = True  # Use TF32 for faster computation
            
    def encode_prompt(self, prompt: str, batch_size: int = 1) -> torch.Tensor:
        """
        Encode text prompt into embeddings.
        
        Args:
            prompt: Text description
            batch_size: Number of images to generate
            
        Returns:
            text_embeddings: Encoded prompt embeddings
        """
        # Tokenize prompt
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        # Encode to embeddings
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_inputs.input_ids.to(self.device))[0]
        
        # Duplicate for batch
        text_embeddings = text_embeddings.to(dtype=self.dtype)
        text_embeddings = text_embeddings.repeat(batch_size, 1, 1)
        
        return text_embeddings
    
    def sample_with_trajectory_recording(
        self,
        prompt: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 8.0,
        height: int = 512,
        width: int = 512,
        generator: Optional[torch.Generator] = None
    ) -> DiffusionTrajectory:
        """
        Sample from diffusion model while recording the complete trajectory.
        
        Args:
            prompt: Text prompt for generation
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            height: Image height
            width: Image width
            generator: Random generator for reproducibility
            
        Returns:
            trajectory: Complete diffusion trajectory with all steps recorded
        """
        batch_size = 1
        
        # Encode prompt
        text_embeddings = self.encode_prompt(prompt, batch_size)
        
        # For classifier-free guidance, we need both conditional and unconditional embeddings
        uncond_embeddings = self.encode_prompt("", batch_size)
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        # Prepare latent space dimensions
        unet_config = self.unet.module.config if hasattr(self.unet, 'module') else self.unet.config
        latents_shape = (batch_size, unet_config.in_channels, height // 8, width // 8)
        
        # Sample initial noise
        latents = torch.randn(
            latents_shape, 
            generator=generator, 
            device=self.device,
            dtype=self.dtype  # Use the sampler's dtype
        )
        
        # Scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        
        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        
        # Initialize trajectory storage
        trajectory_steps = []
        
        # Denoising loop with trajectory recording
        for i, t in enumerate(timesteps):

            # Clear intermediate memory
            if i > 0:  # Don't clear on first iteration
                torch.cuda.empty_cache()
            
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # Prepare timestep
            timestep = t.expand(latent_model_input.shape[0])
            
            # Predict noise with gradient tracking (important for RL!)
            noise_pred = self.unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=text_embeddings,
                return_dict=False
            )[0]
            
            # Perform classifier-free guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            # Control how strongly the model adheres to the prompt via guisance scale
            noise_pred_guided = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Calculate log probability of this action
            # This is a simplified version - in practice, we might want more sophisticated probability calculation
            log_prob = -0.5 * torch.sum(noise_pred_guided ** 2)  # Simplified Gaussian log-prob


            with torch.no_grad():
                # Assuming Gaussian noise distribution
                noise_diff = noise_pred_guided - latents
                log_prob = -0.5 * torch.sum(noise_diff ** 2) / (latents.numel())
                log_prob = log_prob * 0.01  # Scale factor to prevent vanishing gradients

            # Make sure to remove no_grad when storing for training
            log_prob_with_grad = log_prob.detach().requires_grad_(True)
            
            # Denoise: Compute the previous noisy sample x_t -> x_{t-1}
            scheduler_output = self.scheduler.step(noise_pred_guided, t, latents, return_dict=True)
            prev_latents = scheduler_output.prev_sample
            
            # Record this step in the trajectory
            step = TrajectoryStep(
                timestep=t.item(),
                state=latents.clone().detach(),  
                action=prev_latents.clone().detach(),  
                condition=text_embeddings[1:2].clone().detach(),  
                log_prob=log_prob_with_grad.clone(),  
                noise_pred=noise_pred_guided.clone().detach()  
            )

            trajectory_steps.append(step)
            
            # Update latents for next iteration
            latents = prev_latents

            # Clear intermediate variables to save memory
            del noise_pred, noise_pred_uncond, noise_pred_text, noise_pred_guided
            del latent_model_input, scheduler_output
        
        # Decode final latents to image
        with torch.no_grad():
            latents = 1 / 0.18215 * latents
            final_image = self.vae.decode(latents).sample
            final_image = (final_image / 2 + 0.5).clamp(0, 1)
        
        # Create complete trajectory
        trajectory = DiffusionTrajectory(
            steps=trajectory_steps,
            final_image=final_image.detach(),
            condition=text_embeddings[1:2].clone().detach()  # Store conditional embedding
        )

        # Final memory cleanup
        del text_embeddings, uncond_embeddings, latents
        torch.cuda.empty_cache()
        
        return trajectory
    
    def sample_batch_with_trajectories(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[DiffusionTrajectory]:
        """
        Sample multiple images with trajectory recording.
        
        Args:
            prompts: List of text prompts
            **kwargs: Additional arguments for sampling
            
        Returns:
            trajectories: List of complete trajectories
        """
        trajectories = []
        
        for prompt in prompts:
            trajectory = self.sample_with_trajectory_recording(prompt, **kwargs)
            trajectories.append(trajectory)
            
        return trajectories

# Utility functions for working with trajectories
def extract_features_from_trajectory(trajectory: DiffusionTrajectory, feature_extractor) -> torch.Tensor:
    """
    Extract CLIP features from the final image of a trajectory.
    
    Args:
        trajectory: Diffusion trajectory
        feature_extractor: CLIP model for feature extraction
        
    Returns:
        features: Extracted features
    """
    # Convert final image to PIL format for CLIP
    final_image = trajectory.final_image.squeeze(0).cpu()
    final_image = torch.clamp(final_image, 0, 1)
    
    to_pil = transforms.ToPILImage()
    pil_image = to_pil(final_image)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    image_tensor = preprocess(pil_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        features = model.encode_image(image_tensor)
        features = features / features.norm(dim=-1, keepdim=True)
    
    return features.cpu().numpy().flatten()

def save_trajectory(trajectory: DiffusionTrajectory, filepath: str):
    """Save trajectory to disk for analysis."""
    torch.save(trajectory, filepath)

def load_trajectory(filepath: str) -> DiffusionTrajectory:
    """Load trajectory from disk."""
    return torch.load(filepath)

# Example usage and testing
def test_trajectory_recording():
    """
    Test the trajectory recording functionality.
    """
    print("Initializing diffusion sampler...")
    sampler = DiffusionSampler()
    
    # Test single trajectory
    print("Sampling with trajectory recording...")
    trajectory = sampler.sample_with_trajectory_recording(
        prompt="a photo of a mars crater",
        num_inference_steps=20  # Reduced for testing
    )
    
    print(f"Trajectory recorded with {len(trajectory.steps)} steps")
    print(f"Final image shape: {trajectory.final_image.shape}")
    
    # Verify gradient flow
    print("Checking gradient flow...")
    for i, step in enumerate(trajectory.steps[:3]):  # Check first 3 steps
        print(f"Step {i}: action requires_grad = {step.action.requires_grad}")
        print(f"Step {i}: log_prob requires_grad = {step.log_prob.requires_grad}")
    
    # Test feature extraction
    print("Testing feature extraction...")
    features = extract_features_from_trajectory(trajectory, None)
    print(f"Extracted features shape: {features.shape}")
    
    return trajectory

# Run test
if __name__ == "__main__":
    trajectory = test_trajectory_recording()