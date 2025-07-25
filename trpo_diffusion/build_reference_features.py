#!/usr/bin/env python3
"""
Build reference features from images in reference_images folder
Creates NPZ file with ResNet-18 features for diversity reward calculation

Usage:
    python build_reference_features.py

This script will:
1. Read all images from reference_images/crater/ 
2. Extract ResNet-18 features from each image
3. Save as NPZ file to reference_features/reference_crater_features_v2.npz
4. Verify the file can be loaded correctly

The resulting NPZ file will be used by the training script for diversity rewards.
"""

import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from typing import List

# Constants (hardcoded to avoid import conflicts)
DEFAULT_CATEGORY = "crater"

def load_resnet_model(device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Load ResNet-18 model for feature extraction"""
    print(f"Loading ResNet-18 model on device: {device}")
    model = models.resnet18(pretrained=True)
    # Remove the final classification layer to get features
    model = nn.Sequential(*list(model.children())[:-1])
    model = model.to(device)
    model.eval()
    
    # ResNet preprocessing (ImageNet normalization)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return model, preprocess, device

def load_and_preprocess_images(image_dir: Path, preprocess) -> tuple[List[torch.Tensor], List[str]]:
    """Load and preprocess all images from directory"""
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
    images = []
    filenames = []
    
    print(f"Loading images from: {image_dir}")
    
    # Get all image files
    image_files = []
    for ext in image_extensions:
        image_files.extend(image_dir.glob(f"*{ext}"))
        image_files.extend(image_dir.glob(f"*{ext.upper()}"))
    
    image_files = sorted(image_files)  # Sort for consistency
    
    print(f"Found {len(image_files)} image files")
    
    for image_path in image_files:
        try:
            # Load and preprocess image
            image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Preprocess for ResNet
            image_tensor = preprocess(image)
            images.append(image_tensor)
            filenames.append(image_path.name)
            
            print(f"  ✅ Loaded: {image_path.name}")
            
        except Exception as e:
            print(f"  ❌ Failed to load {image_path.name}: {e}")
    
    return images, filenames

def extract_features(model, images: List[torch.Tensor], device: str) -> np.ndarray:
    """Extract ResNet-18 features from preprocessed images"""
    print(f"Extracting features from {len(images)} images...")
    
    # Stack images into batch
    image_batch = torch.stack(images).to(device)
    
    # Extract features
    with torch.no_grad():
        features = model(image_batch)
        # ResNet output is [batch_size, 512, 1, 1], flatten to [batch_size, 512]
        features = features.flatten(start_dim=1)
        # Normalize features
        features = features / features.norm(dim=-1, keepdim=True)
    
    # Convert to numpy
    features_np = features.cpu().numpy()
    
    print(f"Extracted features shape: {features_np.shape}")
    print(f"Feature dimension: {features_np.shape[1]}")
    
    return features_np

def save_features_to_npz(features: np.ndarray, filenames: List[str], output_path: Path):
    """Save features to NPZ file with individual feature vectors as separate arrays"""
    
    # Create dictionary with each image's features as a separate array
    features_dict = {}
    
    for i, (filename, feature_vector) in enumerate(zip(filenames, features)):
        # Use clean filename as key (remove extension)
        clean_name = Path(filename).stem
        # Add index to avoid potential key conflicts
        key = f"image_{i:03d}_{clean_name}"
        features_dict[key] = feature_vector
    
    # Save to NPZ file
    np.savez_compressed(output_path, **features_dict)
    
    print(f"💾 Saved {len(features_dict)} feature vectors to: {output_path}")
    
    # Print some statistics
    print(f"📊 Feature statistics:")
    print(f"  - Number of images: {len(filenames)}")
    print(f"  - Feature dimension: {features.shape[1]}")
    print(f"  - Feature range: [{features.min():.3f}, {features.max():.3f}]")
    print(f"  - Feature mean: {features.mean():.3f}")
    print(f"  - Feature std: {features.std():.3f}")
    
    return features_dict

def verify_npz_file(npz_path: Path):
    """Verify the created NPZ file can be loaded correctly"""
    print(f"\n🔍 Verifying NPZ file: {npz_path}")
    
    try:
        # Load the NPZ file
        npz_data = np.load(npz_path)
        array_keys = list(npz_data.keys())
        
        print(f"  ✅ Successfully loaded NPZ file")
        print(f"  📋 Contains {len(array_keys)} feature vectors")
        
        # Check a few samples
        for i, key in enumerate(array_keys[:3]):
            feature_vector = npz_data[key]
            print(f"  🔍 {key}: shape={feature_vector.shape}, norm={np.linalg.norm(feature_vector):.3f}")
        
        if len(array_keys) > 3:
            print(f"  ... and {len(array_keys) - 3} more")
        
        # Test stacking (like the training code does)
        ref_features_list = []
        for key in array_keys:
            ref_features_list.append(npz_data[key])
        
        ref_features = np.stack(ref_features_list)
        print(f"  ✅ Successfully stacked into shape: {ref_features.shape}")
        
        npz_data.close()
        return True
        
    except Exception as e:
        print(f"  ❌ Error verifying NPZ file: {e}")
        return False

def main():
    """Main function to build reference features"""
    print("🚀 Building reference features for diversity reward calculation")
    print("=" * 60)
    
    # Get current directory and paths
    current_path = Path(__file__).parent
    category = DEFAULT_CATEGORY
    
    # Input and output paths
    images_dir = current_path / "reference_images" / category
    features_dir = current_path / "reference_features"
    features_dir.mkdir(exist_ok=True)
    
    output_path = features_dir / f"reference_{category}_features.npz"
    
    print(f"📂 Category: {category}")
    print(f"📂 Images directory: {images_dir}")
    print(f"📂 Output path: {output_path}")
    
    # Check if images directory exists
    if not images_dir.exists():
        print(f"❌ Images directory does not exist: {images_dir}")
        print(f"   Please create the directory and add reference images.")
        return
    
    # Load ResNet model
    model, preprocess, device = load_resnet_model()
    
    # Load and preprocess images
    images, filenames = load_and_preprocess_images(images_dir, preprocess)
    
    if len(images) == 0:
        print("❌ No images found! Please add reference images to the directory.")
        return
    
    # Extract features
    features = extract_features(model, images, device)
    
    # Save features to NPZ file
    features_dict = save_features_to_npz(features, filenames, output_path)
    
    # Verify the file
    verification_success = verify_npz_file(output_path)
    
    if verification_success:
        print(f"\n🎉 Successfully created reference features!")
        print(f"📁 Location: {output_path}")
        print(f"🔢 Ready for training with {len(features)} reference images")
    else:
        print(f"\n❌ Failed to verify NPZ file. Please check for issues.")

if __name__ == "__main__":
    main()