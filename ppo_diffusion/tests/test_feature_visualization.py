#!/usr/bin/env python3
"""
Test script for feature distribution visualization
Creates synthetic generated features and tests t-SNE plotting without GPU training
"""

import sys
import numpy as np
import pandas as pd
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def create_synthetic_training_data(category: str = "crater", num_episodes: int = 50, 
                                  images_per_episode: int = 4, feature_dim: int = 512):
    """
    Create synthetic training data that mimics real training logs
    
    Args:
        category: Category name (e.g., "crater")
        num_episodes: Number of training episodes to simulate
        images_per_episode: Number of images generated per episode
        feature_dim: Dimension of feature vectors (ResNet-18 = 512)
    """
    
    print(f"🎨 Creating synthetic training data for testing...")
    print(f"   Category: {category}")
    print(f"   Episodes: {num_episodes}")
    print(f"   Images per episode: {images_per_episode}")
    print(f"   Feature dimension: {feature_dim}")
    
    # Create timestamp for synthetic data
    timestamp = time.strftime("%Y%m%d%H%M%S") + "_synthetic"
    
    # Create logs directory
    current_path = Path(__file__).parent.parent
    logs_dir = current_path / "outputs" / "logs" / f"{category}_{timestamp}"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Created synthetic logs directory: {logs_dir}")
    
    # Load reference features to create realistic synthetic data
    ref_features_npz = current_path / "reference_features" / f"reference_{category}_features_v2.npz"
    if not ref_features_npz.exists():
        print(f"❌ Reference features not found: {ref_features_npz}")
        print("💡 Need reference features to create realistic synthetic data")
        return None
    
    # Load reference features
    ref_npz = np.load(ref_features_npz)
    ref_features_list = []
    for key in ref_npz.keys():
        ref_features_list.append(ref_npz[key])
    ref_features = np.stack(ref_features_list)
    ref_npz.close()
    
    print(f"📊 Loaded {len(ref_features)} reference features for realistic synthesis")
    
    # Create synthetic generated features that evolve over training
    generated_features_data = []
    
    prompts = [
        "a detailed crater on the moon surface",
        "lunar crater with rocky edges",
        "impact crater with shadows",
        "large crater formation",
        "crater with surrounding terrain"
    ]
    
    for episode in range(num_episodes):
        # Simulate learning progression: start random, gradually approach reference
        learning_progress = episode / num_episodes  # 0 to 1
        
        # Select random prompt
        prompt = prompts[episode % len(prompts)]
        
        for trajectory_idx in range(images_per_episode):
            # Create synthetic feature that evolves during training
            # Early episodes: more random
            # Later episodes: closer to reference features
            
            # Random base feature
            random_feature = np.random.randn(feature_dim)
            
            # Reference-like component (increases with learning)
            ref_idx = np.random.randint(0, len(ref_features))
            ref_component = ref_features[ref_idx]
            
            # Blend based on learning progress
            noise_level = 1.0 - learning_progress * 0.7  # Reduce noise over time
            ref_influence = learning_progress * 0.8  # Increase reference influence
            
            synthetic_feature = (
                noise_level * random_feature + 
                ref_influence * ref_component + 
                (1 - noise_level - ref_influence) * np.random.randn(feature_dim) * 0.1
            )
            
            # Add some realistic variation
            synthetic_feature += np.random.randn(feature_dim) * 0.05
            
            # Create feature entry
            feature_entry = {
                'episode': episode,
                'trajectory_idx': trajectory_idx,
                'prompt': prompt,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Add each feature dimension
            for j, feature_val in enumerate(synthetic_feature):
                feature_entry[f'feature_{j:03d}'] = float(feature_val)
            
            generated_features_data.append(feature_entry)
    
    # Save synthetic features to CSV
    features_csv = logs_dir / "generated_features.csv"
    df = pd.DataFrame(generated_features_data)
    df.to_csv(features_csv, index=False)
    
    print(f"💾 Saved {len(generated_features_data)} synthetic feature entries to CSV")
    
    # Create some minimal episode data for completeness
    episode_data = []
    for episode in range(num_episodes):
        episode_entry = {
            'episode': episode,
            'prompt': prompts[episode % len(prompts)],
            'avg_reward': 0.3 + 0.4 * (episode / num_episodes) + np.random.randn() * 0.1,
            'best_reward': 0.4 + 0.5 * (episode / num_episodes) + np.random.randn() * 0.1,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        episode_data.append(episode_entry)
    
    episode_csv = logs_dir / "episode_log.csv"
    episode_df = pd.DataFrame(episode_data)
    episode_df.to_csv(episode_csv, index=False)
    
    print(f"💾 Created episode log for completeness")
    
    return timestamp

def test_feature_visualization(timestamp: str, category: str = "crater"):
    """
    Test the feature distribution visualization with synthetic data
    
    Args:
        timestamp: Timestamp of the synthetic training data
        category: Category name
    """
    
    print(f"\n🎨 Testing feature distribution visualization...")
    print(f"   Timestamp: {timestamp}")
    print(f"   Category: {category}")
    
    try:
        # Import visualization function
        from utils.visualization import plot_feature_distributions
        
        # Test the visualization
        success = plot_feature_distributions(timestamp, category)
        
        if success:
            print(f"✅ Feature distribution plots created successfully!")
            print(f"📁 Plots saved to: outputs/plots/feature_distribution/{timestamp}/")
            print(f"   - feature_distribution_comparison_{category}_{timestamp}.png")
            print(f"   - feature_distribution_overlay_{category}_{timestamp}.png")
        else:
            print(f"❌ Feature distribution plotting failed")
            
        return success
        
    except Exception as e:
        print(f"❌ Error testing feature visualization: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_plot_test_simple(timestamp: str, category: str = "crater"):
    """
    Test the plot_test_simple.py functionality with synthetic data
    
    Args:
        timestamp: Timestamp of the synthetic training data
        category: Category name
    """
    
    print(f"\n📊 Testing plot_test_simple.py integration...")
    
    try:
        # Import visualization functions
        from utils.visualization import plot_from_csv, plot_feature_distributions
        
        print(f"📈 Testing training plot generation...")
        plot_from_csv(timestamp, category)
        
        print(f"🎨 Testing feature distribution plots...")
        success = plot_feature_distributions(timestamp, category)
        
        if success:
            print(f"✅ All plots generated successfully!")
            print(f"📁 Training plots: outputs/plots/training/")
            print(f"📁 Feature plots: outputs/plots/feature_distribution/{timestamp}/")
        else:
            print(f"✅ Training plots generated successfully!")
            print(f"⚠️ Feature distribution plots had issues")
            
        return success
        
    except Exception as e:
        print(f"❌ Error testing plot_test_simple integration: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    Main function to test the feature distribution visualization pipeline
    """
    
    print("🧪 Feature Distribution Visualization Test")
    print("=" * 60)
    print("This script creates synthetic training data and tests t-SNE visualization")
    print("without requiring GPU resources or actual training.")
    print("=" * 60)
    
    category = "crater"
    
    # Step 1: Create synthetic training data
    print("\n📝 Step 1: Creating synthetic training data...")
    timestamp = create_synthetic_training_data(
        category=category,
        num_episodes=30,  # Moderate number for testing
        images_per_episode=4,
        feature_dim=512
    )
    
    if not timestamp:
        print("❌ Failed to create synthetic data")
        return
    
    # Step 2: Test feature visualization directly
    print("\n📊 Step 2: Testing feature visualization...")
    viz_success = test_feature_visualization(timestamp, category)
    
    # Step 3: Test plot_test_simple integration
    print("\n🔧 Step 3: Testing plot_test_simple integration...")
    integration_success = test_plot_test_simple(timestamp, category)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    if viz_success and integration_success:
        print("✅ All tests passed!")
        print("🎉 Feature distribution visualization is working correctly!")
        print(f"📁 Check outputs/plots/feature_distribution/{timestamp}/ for results")
    elif viz_success:
        print("✅ Core visualization working")
        print("⚠️ Integration tests had issues")
    else:
        print("❌ Tests failed - check error messages above")
    
    # print(f"\n💡 Next steps:")
    # print(f"   1. Check the generated plots visually")
    # print(f"   2. Verify t-SNE shows progression from random to reference-like")
    # print(f"   3. When GPU available, run real training and compare results")

if __name__ == "__main__":
    main()