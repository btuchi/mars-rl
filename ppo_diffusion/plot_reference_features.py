#!/usr/bin/env python3
"""
Script to plot reference image feature distributions and t-SNE
This helps compare generated images against reference dataset
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from ppo_diffusion.core.features import FeatureExtractor
from ppo_diffusion.utils.constants import DEFAULT_CATEGORY
from PIL import Image
import torch

def load_reference_features():
    """Load reference features from the saved file"""
    # Try to find reference features file
    current_path = Path(__file__).parent
    reference_path = current_path / "reference_features" / f"reference_{DEFAULT_CATEGORY}_features.npz"
    
    print(f"📁 Loading reference features from: {reference_path}")
    if reference_path.suffix == '.npz':
        data = np.load(reference_path)
        print(f"🔍 Available keys in {reference_path.name}: {len(list(data.keys()))} images")
        
        # Collect all features from individual image keys
        all_features = []
        for key in sorted(data.keys()):  # Sort for consistent ordering
            feature_vector = data[key]
            all_features.append(feature_vector)
        
        # Stack into single array
        features = np.vstack(all_features)
        print(f"🔍 Collected features from {len(all_features)} images")
    else:
        features = np.load(reference_path)
    print(f"✅ Loaded {features.shape[0]} reference features with {features.shape[1]} dimensions")
    return features

# def load_reference_images():
#     """Load reference images if available"""
#     current_path = Path(__file__).parent
#     reference_images_path = current_path / "reference_features" / f"{DEFAULT_CATEGORY}_reference_images.npy"
    
#     if reference_images_path.exists():
#         print(f"📁 Loading reference images from: {reference_images_path}")
#         images = np.load(reference_images_path)
#         print(f"✅ Loaded {images.shape[0]} reference images")
#         return images
    
#     print("⚠️ No reference images found - will extract features from dataset")
#     return None

# def extract_fresh_features():
#     """Extract fresh features from reference dataset"""
#     try:
#         # Look for reference dataset
#         dataset_paths = [
#             f"ppo_diffusion/data/{DEFAULT_CATEGORY}",
#             f"ppo_diffusion/datasets/{DEFAULT_CATEGORY}",
#             f"data/{DEFAULT_CATEGORY}"
#         ]
        
#         dataset_path = None
#         for path in dataset_paths:
#             if os.path.exists(path):
#                 dataset_path = path
#                 break
        
#         if not dataset_path:
#             print("❌ No reference dataset found")
#             return None
        
#         print(f"📁 Extracting features from: {dataset_path}")
        
#         # Initialize feature extractor
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         feature_extractor = FeatureExtractor(device=device)
        
#         # Get all image files
#         image_files = []
#         for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
#             image_files.extend(Path(dataset_path).glob(ext))
        
#         print(f"📊 Found {len(image_files)} images")
        
#         # Extract features
#         features = []
#         for i, img_path in enumerate(image_files[:100]):  # Limit to 100 images
#             try:
#                 img = Image.open(img_path).convert('RGB')
#                 feature = feature_extractor.extract_image_features(img)
#                 features.append(feature)
                
#                 if (i + 1) % 10 == 0:
#                     print(f"  Processed {i+1}/{min(len(image_files), 100)} images")
                    
#             except Exception as e:
#                 print(f"⚠️ Error processing {img_path}: {e}")
#                 continue
        
#         if features:
#             features_array = np.vstack(features)
#             print(f"✅ Extracted {features_array.shape[0]} features with {features_array.shape[1]} dimensions")
#             return features_array
#         else:
#             print("❌ No features extracted")
#             return None
            
#     except Exception as e:
#         print(f"❌ Error extracting features: {e}")
#         return None

def plot_reference_feature_distribution(features, output_dir):
    """Plot per-image feature distribution for reference images (same as generated)"""
    print("📊 Creating reference per-image feature distribution plots...")
    
    # Create output directory
    dist_dir = Path(output_dir) / "reference_feature_plots" / "distribution"
    dist_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot distribution for each reference image (first 10 images)
    num_plots = features.shape[0]
    
    for img_idx in range(num_plots):
        feature_vector = features[img_idx]  # 512 features for this image
        
        # Create the plot (same format as generated images)
        plt.figure(figsize=(12, 8))
        
        # Main histogram
        plt.subplot(2, 2, 1)
        plt.hist(feature_vector, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        plt.title(f'Reference Image {img_idx+1} - Feature Distribution')
        plt.xlabel('Feature Value')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        
        # Box plot
        plt.subplot(2, 2, 2)
        plt.boxplot(feature_vector, vert=True)
        plt.title('Feature Distribution (Box Plot)')
        plt.ylabel('Feature Value')
        plt.grid(True, alpha=0.3)
        
        # Feature values over indices
        plt.subplot(2, 2, 3)
        plt.plot(feature_vector, alpha=0.7, color='darkgreen')
        plt.title('Feature Values by Index')
        plt.xlabel('Feature Index (0-511)')
        plt.ylabel('Feature Value')
        plt.grid(True, alpha=0.3)
        
        # Statistics text
        plt.subplot(2, 2, 4)
        plt.axis('off')
        stats_text = f"""
        Reference Image: {img_idx+1}
        Dataset: {DEFAULT_CATEGORY}
        
        Statistics:
        Mean: {np.mean(feature_vector):.4f}
        Std: {np.std(feature_vector):.4f}
        Min: {np.min(feature_vector):.4f}
        Max: {np.max(feature_vector):.4f}
        Range: {np.max(feature_vector) - np.min(feature_vector):.4f}
        Skew: {np.mean(((feature_vector - np.mean(feature_vector)) / np.std(feature_vector)) ** 3):.4f}
        """
        plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, 
                 verticalalignment='top', fontsize=10, fontfamily='monospace')
        
        plt.tight_layout()
        
        # Save plot
        filename = f"ref_img{img_idx+1:02d}_feature_dist.png"
        save_path = dist_dir / filename
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  📊 Reference image {img_idx+1} feature distribution saved: {filename}")
    
    print(f"✅ Created {num_plots} reference feature distribution plots")
    return str(dist_dir)

def plot_reference_tsne(features, output_dir):
    """Plot per-image t-SNE for reference images (512 features per image)"""
    print("📊 Creating reference per-image t-SNE plots...")
    
    try:
        from sklearn.manifold import TSNE
        
        # Create output directory
        tsne_dir = Path(output_dir) / "reference_feature_plots" / "tsne"
        tsne_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot t-SNE for each reference image (first 10 images)
        num_plots = features.shape[0]
        
        for img_idx in range(num_plots):
            print(f"🔄 Running t-SNE for reference image {img_idx+1}...")
            
            # Get 512 features for this image
            image_features = features[img_idx]  # Shape: (512,)
            
            # Prepare data for t-SNE: each feature becomes a "data point"
            feature_data = np.column_stack([
                image_features,  # Feature values
                np.arange(len(image_features))  # Feature indices
            ])
            
            # Run t-SNE on the 512 features
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(image_features)-1))
            tsne_features = tsne.fit_transform(feature_data)
            
            # Create the plot
            plt.figure(figsize=(10, 8))
            
            # Scatter plot - each point is one of the 512 features
            scatter = plt.scatter(tsne_features[:, 0], tsne_features[:, 1], 
                                 c=image_features, cmap='viridis', 
                                 s=30, alpha=0.7, edgecolors='black')
            
            plt.title(f't-SNE of 512 Features within Reference Image {img_idx+1}\nDataset: {DEFAULT_CATEGORY}')
            plt.xlabel('t-SNE Dimension 1')
            plt.ylabel('t-SNE Dimension 2')
            plt.grid(True, alpha=0.3)
            plt.colorbar(scatter, label='Feature Value')
            
            # Add statistics about feature clustering
            distances = []
            for i in range(len(tsne_features)):
                for j in range(i+1, len(tsne_features)):
                    dist = np.linalg.norm(tsne_features[i] - tsne_features[j])
                    distances.append(dist)
            
            if distances:
                plt.figtext(0.02, 0.02, 
                           f'Feature Clustering Stats:\nMean distance: {np.mean(distances):.3f}\nStd distance: {np.std(distances):.3f}\nTotal features: {len(image_features)}',
                           fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            
            plt.tight_layout()
            
            # Save plot
            filename = f"ref_img{img_idx+1:02d}_feature_tsne.png"
            save_path = tsne_dir / filename
            
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  📊 Reference image {img_idx+1} feature t-SNE saved: {filename}")
        
        print(f"✅ Created {num_plots} reference per-image t-SNE plots")
        return str(tsne_dir)
        
    except ImportError:
        print("⚠️ scikit-learn not installed - skipping t-SNE plot")
        return None
    except Exception as e:
        print(f"⚠️ Error creating t-SNE plot: {e}")
        return None

def main():
    """Main function to create reference feature plots"""
    print("🎯 Creating reference feature plots...")
    
    # Try to load existing features first
    features = load_reference_features()
    
    # If no features found, extract fresh ones
    # if features is None:
    #     features = extract_fresh_features()
    
    if features is None:
        print("❌ Could not load or extract reference features")
        return
    
    # Create output directory
    output_dir = Path("ppo_diffusion")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot feature distribution
    plot_reference_feature_distribution(features, output_dir)
    
    # Plot t-SNE
    plot_reference_tsne(features, output_dir)
    
    print("✅ Reference feature plots completed!")
    print(f"📁 Plots saved to: {output_dir}/reference_feature_plots/")

if __name__ == "__main__":
    main()