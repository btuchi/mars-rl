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
    total_log_prob: Optional[torch.Tensor] = None  # Total log probability for the trajectory
    

class DiffusionSampler:
    """
    Diffusion model sampler that records trajectories for RL training.
    """
    
    def __init__(self, model_id: str = "CompVis/stable-diffusion-v1-4", device: str = "cuda", use_fp16: bool = False):
        self.device = device
        self.dtype = torch.float16 if use_fp16 else torch.float32
        
        # Clear cache
        torch.cuda.empty_cache()

        # Load Stable Diffusion pipeline - SIMPLE approach
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

        # SIMPLE: Move everything to primary device
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
        """
        Encode text prompt into embeddings.
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
        
        # Convert to correct dtype and repeat for batch
        text_embeddings = text_embeddings.to(dtype=self.dtype)
        text_embeddings = text_embeddings.repeat(batch_size, 1, 1)
        
        return text_embeddings
    
    def sample_with_trajectory_recording(
        self,
        prompt: str,
        num_inference_steps: int = 20,
        guidance_scale: float = 8.0,
        height: int = 512,
        width: int = 512,
        generator: Optional[torch.Generator] = None
    ) -> DiffusionTrajectory:
        """
        Sample from diffusion model while recording the complete trajectory.
        """
        batch_size = 1
        
        # Encode prompt
        text_embeddings = self.encode_prompt(prompt, batch_size)
        
        # For classifier-free guidance
        uncond_embeddings = self.encode_prompt("", batch_size)
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        # Prepare latent space dimensions
        unet_config = self.unet.config
        latents_shape = (batch_size, unet_config.in_channels, height // 8, width // 8)
        
        # Sample initial noise
        latents = torch.randn(
            latents_shape,
            generator=generator, 
            device=self.device,
            dtype=self.dtype
        )
        
        # Scale by scheduler init noise sigma
        latents = latents * self.scheduler.init_noise_sigma
        
        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        # Initialize total log probability
        total_log_prob = torch.tensor(0.0, device=self.device, dtype=self.dtype, requires_grad=True)
        
        # Initialize trajectory storage
        trajectory_steps = []
        
        # Denoising loop with trajectory recording
        for i, t in enumerate(timesteps):
            # Clear cache periodically
            if i > 0:
                torch.cuda.empty_cache()
            
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # Prepare timestep
            timestep = t.expand(latent_model_input.shape[0])
            
            # Predict noise with gradient tracking
            noise_pred = self.unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=text_embeddings,
                return_dict=False
            )[0]
            
            # Perform classifier-free guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred_guided = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Clamp extreme values
            # noise_pred_clamped = torch.clamp(noise_pred_guided, -10.0, 10.0)
            noise_pred_clamped = noise_pred_guided

            # print(f"Step {i}:")
            # print(f"  noise_pred_guided min/max: {noise_pred_guided.min():.4f}/{noise_pred_guided.max():.4f}")
            # print(f"  noise_pred_clamped min/max: {noise_pred_clamped.min():.4f}/{noise_pred_clamped.max():.4f}")
            # print(f"  Contains NaN: {torch.isnan(noise_pred_clamped).any()}")

            # Simple but stable log probability
            mse = torch.mean((noise_pred_clamped) ** 2)
            step_log_prob = -mse * 0.01  # Small scale factor

            # Check for NaN and replace with safe value
            if torch.isnan(step_log_prob) or torch.isinf(step_log_prob):
                step_log_prob = torch.tensor(-0.01, device=noise_pred_guided.device, requires_grad=True)
                print("⚠️ NaN detected in step_log_prob, using safe fallback")
            
            # ONLY device fix we actually need:
            if step_log_prob.device != total_log_prob.device:
                step_log_prob = step_log_prob.to(total_log_prob.device)
            
            total_log_prob = total_log_prob + step_log_prob

            # Denoise
            scheduler_output = self.scheduler.step(noise_pred_guided, t, latents, return_dict=True)
            prev_latents = scheduler_output.prev_sample
            
            # Record step
            step = TrajectoryStep(
                timestep=t.item(),
                state=latents.clone().detach(),  
                action=prev_latents.clone().detach(),  
                condition=text_embeddings[1:2].clone().detach(),  
                log_prob=step_log_prob.detach(),  
                noise_pred=noise_pred_guided.clone().detach()  
            )

            trajectory_steps.append(step)
            
            # Update latents
            latents = prev_latents

            # Clear intermediate variables
            del noise_pred, noise_pred_uncond, noise_pred_text, noise_pred_guided
            del latent_model_input, scheduler_output, step_log_prob
        
        # Decode final latents to image
        with torch.no_grad():
            latents = 1 / 0.18215 * latents
            final_image = self.vae.decode(latents).sample
            final_image = (final_image / 2 + 0.5).clamp(0, 1)
        
        # Create trajectory
        trajectory = DiffusionTrajectory(
            steps=trajectory_steps,
            final_image=final_image.detach(),
            condition=text_embeddings[1:2].clone().detach(),
            total_log_prob=total_log_prob
        )
        
        # Final cleanup
        del text_embeddings, uncond_embeddings, latents
        torch.cuda.empty_cache()
        
        return trajectory
    
    def sample_batch_with_trajectories(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[DiffusionTrajectory]:
        """Sample multiple images with trajectory recording."""
        trajectories = []
        
        for prompt in prompts:
            trajectory = self.sample_with_trajectory_recording(prompt, **kwargs)
            trajectories.append(trajectory)
            
        return trajectories

# Utility functions
def extract_features_from_trajectory(trajectory: DiffusionTrajectory, model, preprocess) -> torch.Tensor:
    """Extract CLIP features from the final image of a trajectory."""
    # Convert final image to PIL format for CLIP
    final_image = trajectory.final_image.squeeze(0).cpu()
    final_image = torch.clamp(final_image, 0, 1)
    
    to_pil = transforms.ToPILImage()
    pil_image = to_pil(final_image)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model, preprocess = clip.load("ViT-B/32", device=device)
    
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

# import torch
# import torch.nn as nn
# from typing import List, Tuple, Dict, Optional
# import numpy as np
# from dataclasses import dataclass
# from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDPMScheduler
# # Convert to PIL Image
# from PIL import Image
# import torchvision.transforms as transforms
# # Extract features using CLIP (reuse your existing feature extraction code)
# import clip

# # def check_model_devices(model, model_name="Model"):
# #     """Debug function to check which devices model parameters are on"""
# #     devices = set()
# #     for name, param in model.named_parameters():
# #         devices.add(param.device)
# #         if len(devices) > 1:  # Found parameters on multiple devices
# #             print(f"❌ {model_name} has parameters on multiple devices!")
# #             for name, param in model.named_parameters():
# #                 print(f"  {name}: {param.device}")
# #             break
# #     else:
# #         device = list(devices)[0] if devices else "No parameters"
# #         print(f"✅ {model_name} parameters are all on: {device}")
# #     return devices

# @dataclass
# class TrajectoryStep:
#     """
#     Represents a single step in the diffusion trajectory for RL training.
#     """
#     timestep: int              # Current timestep t
#     state: torch.Tensor        # Current noisy image x_t
#     action: torch.Tensor       # Predicted denoised image x_{t-1}
#     condition: torch.Tensor    # Text/class condition c
#     log_prob: torch.Tensor     # Log probability of this action
#     noise_pred: torch.Tensor   # Raw noise prediction from model

# @dataclass
# class DiffusionTrajectory:
#     """
#     Complete trajectory from random noise to final image.
#     """
#     steps: List[TrajectoryStep]
#     final_image: torch.Tensor
#     condition: torch.Tensor
#     reward: Optional[float] = None
#     total_log_prob: Optional[torch.Tensor] = None  # Total log probability for the trajectory
    

# class DiffusionSampler:
#     """
#     Diffusion model sampler that records trajectories for RL training.
#     """
    
#     def __init__(self, model_id: str = "CompVis/stable-diffusion-v1-4", device: str = "cuda", use_fp16: bool = False):

#         self.device = device
#         self.dtype = torch.float16 if use_fp16 else torch.float32
        
#         # Add memory optimization
#         torch.cuda.empty_cache()

#         # Load Stable Diffusion pipeline
#         self.pipe = StableDiffusionPipeline.from_pretrained(
#             model_id,
#             torch_dtype=self.dtype,           # Use float16 to save memory
#             safety_checker=None,
#             requires_safety_checker=False,
#             use_safetensors=True,
#             variant="fp16" if use_fp16 else None,
#             low_cpu_mem_usage=True,              # This prevents meta device issues
#             device_map=None                      # Disable automatic device mapping
#         )

#         # Enable memory efficient attention
#         self.pipe.enable_xformers_memory_efficient_attention()
#         self.pipe.enable_attention_slicing("max")       # Maximum slicing
#         self.pipe.enable_vae_slicing()                 # VAE slicing
#         self.pipe.enable_vae_tiling()                  # VAE tiling

#         self.pipe = self.pipe.to(device)

#         if not use_fp16:  # Only convert to float32 if not using fp16
#             self.pipe.unet.to(torch.float32)
#             self.pipe.vae.to(torch.float32)
#             self.pipe.text_encoder.to(torch.float32)

        
#         # Extract components we need
#         self.unet = self.pipe.unet
#         self.scheduler = self.pipe.scheduler
#         self.vae = self.pipe.vae
#         self.text_encoder = self.pipe.text_encoder
#         self.tokenizer = self.pipe.tokenizer

#         torch.cuda.empty_cache()
#         with torch.cuda.device(1):
#             torch.cuda.empty_cache()

#         # SIMPLE DataParallel setup
#         if torch.cuda.device_count() > 1:
#             # print(f"Using simple DataParallel setup for {torch.cuda.device_count()} GPUs")
            
#             # # Make sure UNet is on cuda:0
#             # self.pipe.unet = self.pipe.unet.to("cuda:0")
            
#             # # Apply DataParallel
#             # self.pipe.unet = nn.DataParallel(self.pipe.unet)
            
#             # # Update references
#             # self.unet = self.pipe.unet
            
#             # print("Simple DataParallel setup complete")
#             print(f"DataParallel disabled - using single GPU mode")
            
#         else:
#             # Single GPU optimizations
#             # self.pipe.enable_model_cpu_offload()
#             # self.pipe.enable_sequential_cpu_offload()
#             print("Using single GPU mode on cuda:0")
#             # Keep UNet on cuda:0, no DataParallel
#             self.pipe.enable_model_cpu_offload()
#             self.pipe.enable_sequential_cpu_offload()

#         # Enable gradient checkpointing to save memory
#         if hasattr(self.unet, 'module'):
#             self.unet.module.enable_gradient_checkpointing()
#         else:
#             self.unet.enable_gradient_checkpointing()
        
#         # print("🔍 Checking device placement...")
#         # check_model_devices(self.unet, "UNet")
#         # check_model_devices(self.vae, "VAE") 
#         # check_model_devices(self.text_encoder, "TextEncoder")

#         # Force all components to cuda:0 to avoid DataParallel memory issues
#         print("🔧 Moving all components to cuda:0 to avoid DataParallel conflicts...")
#         self.vae = self.vae.to("cuda:0")
#         self.text_encoder = self.text_encoder.to("cuda:0")
#         self.pipe.vae = self.vae
#         self.pipe.text_encoder = self.text_encoder
#         print("✅ All components now on cuda:0")
        
#         # Keep UNet in training mode but others in eval
#         self.unet.train()
#         self.vae.eval()
#         self.text_encoder.eval()

#         # H100 optimizations
#         torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
#         torch.backends.cuda.matmul.allow_tf32 = True  # Use TF32 for faster computation
#         torch.backends.cudnn.allow_tf32 = True

#         print("✅ DiffusionSampler initialized successfully")

#         # Verify final device placement
#         # if hasattr(self.unet, 'module'):
#         #     final_device = next(self.unet.module.parameters()).device
#         # else:
#         #     final_device = next(self.unet.parameters()).device
#         # print(f"Final UNet device: {final_device}")

            
#     def encode_prompt(self, prompt: str, batch_size: int = 1) -> torch.Tensor:
#         """
#         Encode text prompt into embeddings.
        
#         Args:
#             prompt: Text description
#             batch_size: Number of images to generate
            
#         Returns:
#             text_embeddings: Encoded prompt embeddings
#         """

#         # Get the device where text_encoder is located
#         text_encoder_device = next(self.text_encoder.parameters()).device

#         # Tokenize prompt
#         text_inputs = self.tokenizer(
#             prompt,
#             padding="max_length",
#             max_length=self.tokenizer.model_max_length,
#             truncation=True,
#             return_tensors="pt",
#         )

#         # Move tokens to text_encoder's device
#         input_ids = text_inputs.input_ids.to(text_encoder_device)
        
#         # Encode to embeddings
#         with torch.no_grad():
#             text_embeddings = self.text_encoder(input_ids)[0]

#         # Convert to the correct dtype and move to UNet device (cuda:0)
#         text_embeddings = text_embeddings.to(dtype=self.dtype, device="cuda:0")
#         text_embeddings = text_embeddings.repeat(batch_size, 1, 1)
        
#         return text_embeddings
    
#     def sample_with_trajectory_recording(
#         self,
#         prompt: str,
#         num_inference_steps: int = 50,
#         guidance_scale: float = 8.0,
#         height: int = 512,
#         width: int = 512,
#         generator: Optional[torch.Generator] = None
#     ) -> DiffusionTrajectory:
#         """
#         Sample from diffusion model while recording the complete trajectory.
        
#         Args:
#             prompt: Text prompt for generation
#             num_inference_steps: Number of denoising steps
#             guidance_scale: Classifier-free guidance scale
#             height: Image height
#             width: Image width
#             generator: Random generator for reproducibility
            
#         Returns:
#             trajectory: Complete diffusion trajectory with all steps recorded
#         """
#         batch_size = 1
        
#         # Encode prompt
#         text_embeddings = self.encode_prompt(prompt, batch_size)
        
#         # For classifier-free guidance, we need both conditional and unconditional embeddings
#         uncond_embeddings = self.encode_prompt("", batch_size)
#         text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
#         # Prepare latent space dimensions
#         unet_config = self.unet.module.config if hasattr(self.unet, 'module') else self.unet.config
#         latents_shape = (batch_size, unet_config.in_channels, height // 8, width // 8)
        
#         # Sample initial noise
#         latents = torch.randn(
#             latents_shape,
#             generator=generator, 
#             device="cuda:0",
#             dtype=self.dtype  # Use the sampler's dtype
#         )
        
#         # Scale the initial noise by the standard deviation required by the scheduler
#         latents = latents * self.scheduler.init_noise_sigma
        
#         # Set timesteps
#         self.scheduler.set_timesteps(num_inference_steps, device="cuda:0")
#         timesteps = self.scheduler.timesteps

#         # Initialize TOTAL log probability (this is what we'll backprop through)
#         total_log_prob = torch.tensor(0.0, device=self.device, dtype=self.dtype, requires_grad=True)

        
#         # Initialize trajectory storage
#         trajectory_steps = []
        
#         # Denoising loop with trajectory recording
#         for i, t in enumerate(timesteps):

#             # Clear intermediate memory
#             if i > 0:  # Don't clear on first iteration
#                 torch.cuda.empty_cache()
            
#             # Expand latents for classifier-free guidance
#             latent_model_input = torch.cat([latents] * 2)
#             latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
#             # Prepare timestep
#             timestep = t.expand(latent_model_input.shape[0]).to("cuda:0")
            
#             # Predict noise with gradient tracking (important for RL!)
#             noise_pred = self.unet(
#                 latent_model_input,
#                 timestep,
#                 encoder_hidden_states=text_embeddings,
#                 return_dict=False
#             )[0]
            
#             # Perform classifier-free guidance
#             noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
#             # Control how strongly the model adheres to the prompt via guisance scale
#             noise_pred_guided = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
#             # Calculate log probability of this action
#             # This is a simplified version - in practice, we might want more sophisticated probability calculation
#             log_prob = -0.5 * torch.sum(noise_pred_guided ** 2)  # Simplified Gaussian log-prob


#             # with torch.no_grad():
#             # Assuming Gaussian noise distribution
#             noise_diff = noise_pred_guided - latents
#             log_prob = -0.5 * torch.sum(noise_diff ** 2) / (latents.numel())
#             step_log_prob = log_prob * 0.01  # Scale factor to prevent vanishing gradients
#             step_log_prob = step_log_prob.to(total_log_prob.device)

#             total_log_prob = total_log_prob + step_log_prob
#             log_prob_with_grad = log_prob

#             # Denoise: Compute the previous noisy sample x_t -> x_{t-1}
#             scheduler_output = self.scheduler.step(noise_pred_guided, t, latents, return_dict=True)
#             prev_latents = scheduler_output.prev_sample
            
#             # Record this step in the trajectory
#             step = TrajectoryStep(
#                 timestep=t.item(),
#                 state=latents.clone().detach(),  
#                 action=prev_latents.clone().detach(),  
#                 condition=text_embeddings[1:2].clone().detach(),  
#                 log_prob=step_log_prob.detach(),  
#                 noise_pred=noise_pred_guided.clone().detach()  
#             )

#             trajectory_steps.append(step)
            
#             # Update latents for next iteration
#             latents = prev_latents

#             # Clear intermediate variables to save memory
#             del noise_pred, noise_pred_uncond, noise_pred_text, noise_pred_guided
#             del latent_model_input, scheduler_output, step_log_prob  # Clear step log prob
        
#         vae_device = next(self.vae.parameters()).device

#         # Decode final latents to image
#         with torch.no_grad():
#             # Move latents to VAE device for decoding
#             latents_for_vae = (1 / 0.18215 * latents).to(vae_device)
#             final_image = self.vae.decode(latents_for_vae).sample
            
#             # Move result back to cuda:0 and normalize
#             final_image = final_image.to("cuda:0")
#             final_image = (final_image / 2 + 0.5).clamp(0, 1)
        
#         # Create complete trajectory
#         trajectory = DiffusionTrajectory(
#             steps=trajectory_steps,
#             final_image=final_image.detach(),
#             condition=text_embeddings[1:2].clone().detach(),  # Store conditional embedding
#             total_log_prob=total_log_prob
#         )
        

#         # Final memory cleanup
#         del text_embeddings, uncond_embeddings, latents
#         torch.cuda.empty_cache()
        
#         return trajectory
    
#     def sample_batch_with_trajectories(
#         self,
#         prompts: List[str],
#         **kwargs
#     ) -> List[DiffusionTrajectory]:
#         """
#         Sample multiple images with trajectory recording.
        
#         Args:
#             prompts: List of text prompts
#             **kwargs: Additional arguments for sampling
            
#         Returns:
#             trajectories: List of complete trajectories
#         """
#         trajectories = []
        
#         for prompt in prompts:
#             trajectory = self.sample_with_trajectory_recording(prompt, **kwargs)
#             trajectories.append(trajectory)
            
#         return trajectories

# # Utility functions for working with trajectories
# def extract_features_from_trajectory(trajectory: DiffusionTrajectory, feature_extractor) -> torch.Tensor:
#     """
#     Extract CLIP features from the final image of a trajectory.
    
#     Args:
#         trajectory: Diffusion trajectory
#         feature_extractor: CLIP model for feature extraction
        
#     Returns:
#         features: Extracted features
#     """
#     # Convert final image to PIL format for CLIP
#     final_image = trajectory.final_image.squeeze(0).cpu()
#     final_image = torch.clamp(final_image, 0, 1)
    
#     to_pil = transforms.ToPILImage()
#     pil_image = to_pil(final_image)
    
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model, preprocess = clip.load("ViT-B/32", device=device)
    
#     image_tensor = preprocess(pil_image).unsqueeze(0).to(device)
    
#     with torch.no_grad():
#         features = model.encode_image(image_tensor)
#         features = features / features.norm(dim=-1, keepdim=True)
    
#     return features.cpu().numpy().flatten()

# def save_trajectory(trajectory: DiffusionTrajectory, filepath: str):
#     """Save trajectory to disk for analysis."""
#     torch.save(trajectory, filepath)

# def load_trajectory(filepath: str) -> DiffusionTrajectory:
#     """Load trajectory from disk."""
#     return torch.load(filepath)

# # Example usage and testing
# def test_trajectory_recording():
#     """
#     Test the trajectory recording functionality.
#     """
#     print("Initializing diffusion sampler...")
#     sampler = DiffusionSampler()
    
#     # Test single trajectory
#     print("Sampling with trajectory recording...")
#     trajectory = sampler.sample_with_trajectory_recording(
#         prompt="a photo of a mars crater",
#         num_inference_steps=20  # Reduced for testing
#     )
    
#     print(f"Trajectory recorded with {len(trajectory.steps)} steps")
#     print(f"Final image shape: {trajectory.final_image.shape}")
    
#     # Verify gradient flow
#     print("Checking gradient flow...")
#     for i, step in enumerate(trajectory.steps[:3]):  # Check first 3 steps
#         print(f"Step {i}: action requires_grad = {step.action.requires_grad}")
#         print(f"Step {i}: log_prob requires_grad = {step.log_prob.requires_grad}")
    
#     # Test feature extraction
#     print("Testing feature extraction...")
#     features = extract_features_from_trajectory(trajectory, None)
#     print(f"Extracted features shape: {features.shape}")
    
#     return trajectory

# # Run test
# if __name__ == "__main__":
#     trajectory = test_trajectory_recording()