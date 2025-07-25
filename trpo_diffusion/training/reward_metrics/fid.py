"""
Fréchet Inception Distance (FID) metric using pytorch-fid library
for measuring quality and diversity of generated images.

FID measures the distance between feature distributions of generated and real images
using the Inception-v3 network. Lower FID indicates better quality and diversity.

Installation required: pip install pytorch-fid
"""

import torch
import numpy as np
from pathlib import Path
import tempfile
import os
from PIL import Image

try:
    from pytorch_fid import fid_score
    from pytorch_fid.inception import InceptionV3
    PYTORCH_FID_AVAILABLE = True
except ImportError:
    print("⚠️ pytorch-fid not found. Install with: pip install pytorch-fid")
    PYTORCH_FID_AVAILABLE = False


def save_images_to_temp_dir(images, prefix="temp_images"):
    """
    Save tensor images to temporary directory for FID calculation
    
    Args:
        images: torch.Tensor [batch_size, C, H, W] in range [0, 1]
        prefix: string prefix for temp directory
    
    Returns:
        temp_dir: Path to temporary directory containing images
    """
    temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
    
    for i, image_tensor in enumerate(images):
        # Convert from tensor to PIL Image
        if image_tensor.shape[0] == 1:  # Grayscale
            image_tensor = image_tensor.repeat(3, 1, 1)  # Convert to RGB
        elif image_tensor.shape[0] == 4:  # RGBA
            image_tensor = image_tensor[:3]  # Take RGB only
        
        # Convert to numpy and scale to [0, 255]
        image_np = (image_tensor.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        
        # Save as PNG
        image_pil = Image.fromarray(image_np)
        image_path = temp_dir / f"image_{i:04d}.png"
        image_pil.save(image_path)
    
    return temp_dir


def cleanup_temp_dir(temp_dir):
    """Remove temporary directory and all its contents"""
    import shutil
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


def calculate_fid_from_tensors(generated_images, reference_images, batch_size=50, device='cuda'):
    """
    Calculate FID between generated and reference images using pytorch-fid
    
    Args:
        generated_images: torch.Tensor [batch_size, C, H, W] in range [0, 1]
        reference_images: torch.Tensor [n_ref, C, H, W] in range [0, 1]
        batch_size: batch size for FID calculation
        device: device for computation
    
    Returns:
        fid_value: float, FID score (lower is better)
    """
    if not PYTORCH_FID_AVAILABLE:
        raise ImportError("pytorch-fid is required. Install with: pip install pytorch-fid")
    
    # Save images to temporary directories
    gen_temp_dir = None
    ref_temp_dir = None
    
    try:
        gen_temp_dir = save_images_to_temp_dir(generated_images, "gen_")
        ref_temp_dir = save_images_to_temp_dir(reference_images, "ref_")
        
        # Calculate FID using pytorch-fid
        fid_value = fid_score.calculate_fid_given_paths(
            [str(gen_temp_dir), str(ref_temp_dir)],
            batch_size=batch_size,
            device=device,
            dims=2048  # Inception-v3 feature dimension
        )
        
        return fid_value
        
    except Exception as e:
        print(f"❌ Error calculating FID: {e}")
        return float('inf')
        
    finally:
        # Clean up temporary directories
        if gen_temp_dir:
            cleanup_temp_dir(gen_temp_dir)
        if ref_temp_dir:
            cleanup_temp_dir(ref_temp_dir)


def calculate_fid_reward(generated_images, reference_images, reward_scale=1.0, device='cuda'):
    """
    Calculate FID-based reward for generated images
    
    Args:
        generated_images: torch.Tensor [batch_size, C, H, W] in range [0, 1]
        reference_images: torch.Tensor [n_ref, C, H, W] in range [0, 1]
        reward_scale: scaling factor for reward (negative FID)
        device: device for computation
    
    Returns:
        reward: float, higher is better (negative FID scaled by reward_scale)
        fid_score: float, the actual FID score (lower is better)
    """
    try:
        fid_value = calculate_fid_from_tensors(generated_images, reference_images, device=device)
        
        # Convert to reward (negative FID, scaled)
        reward = -fid_value * reward_scale
        
        return reward, fid_value
        
    except Exception as e:
        print(f"❌ Error in FID reward calculation: {e}")
        return 0.0, float('inf')


def calculate_fid_batch_rewards(generated_images, reference_images, reward_scale=1.0, device='cuda'):
    """
    Calculate FID rewards for a batch of generated images
    Since FID requires multiple images, we calculate one FID score for the entire batch
    
    Args:
        generated_images: torch.Tensor [batch_size, C, H, W]
        reference_images: torch.Tensor [n_ref, C, H, W]
        reward_scale: scaling factor for reward
        device: device for computation
    
    Returns:
        individual_rewards: numpy array of identical FID rewards (same for all images)
        avg_reward: float, FID reward
        fid_scores: numpy array of identical FID scores
    """
    # Calculate single FID score for the batch
    avg_reward, fid_value = calculate_fid_reward(
        generated_images, reference_images, reward_scale, device
    )
    
    batch_size = generated_images.shape[0]
    
    # All images get the same reward since FID is calculated on the batch
    individual_rewards = np.full(batch_size, avg_reward)
    fid_scores = np.full(batch_size, fid_value)
    
    return individual_rewards, avg_reward, fid_scores


# For integration with existing reward system
def calculate_fid_reward_simple(generated_images, reference_images, device='cuda'):
    """
    Simple FID reward calculation for integration with existing reward system
    
    Returns:
        reward: float, negative FID (higher is better)
    """
    reward, _ = calculate_fid_reward(generated_images, reference_images, reward_scale=1.0, device=device)
    return reward


if __name__ == "__main__":
    # Test FID calculation
    print("🧪 Testing FID calculation with pytorch-fid...")
    
    if not PYTORCH_FID_AVAILABLE:
        print("❌ pytorch-fid not available. Install with: pip install pytorch-fid")
        exit(1)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create dummy images (small batch for testing)
    print("Creating test images...")
    generated = torch.randn(4, 3, 256, 256).clamp(0, 1)
    reference = torch.randn(8, 3, 256, 256).clamp(0, 1)
    
    print("Calculating FID...")
    
    # Test FID calculation
    reward, fid_value = calculate_fid_reward(generated, reference, device=device)
    
    print(f"✅ FID calculation successful!")
    print(f"   FID Score: {fid_value:.4f}")
    print(f"   Reward: {reward:.4f}")
    
    # Test batch calculation
    individual_rewards, avg_reward, fid_scores = calculate_fid_batch_rewards(
        generated, reference, device=device
    )
    
    print(f"✅ Batch FID calculation successful!")
    print(f"   Batch FID score: {fid_scores[0]:.4f}")
    print(f"   Average reward: {avg_reward:.4f}")
    print(f"   All images get same reward: {np.all(individual_rewards == individual_rewards[0])}")