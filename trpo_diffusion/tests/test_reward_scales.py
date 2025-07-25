#!/usr/bin/env python3
"""
Test script to analyze the scales and ranges of MMD vs MI reward metrics
This helps determine appropriate weights for MMD_MI combination
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def test_reward_scales(category: str = "crater", num_samples: int = 20):
    """
    Test the scales of MMD and MI rewards to understand their ranges
    
    Args:
        category: Category to test (e.g., "crater")
        num_samples: Number of synthetic samples to generate for testing
    """
    
    print(f"🔬 Testing MMD vs MI reward scales for category: {category}")
    print("=" * 60)
    
    try:
        # Import reward metrics
        sys.path.insert(0, str(project_root))
        from training.reward_metrics import MMDRewardMetric, MIRewardMetric, FIDRewardMetric
        
        # Load reference data
        current_path = Path(__file__).parent.parent
        
        # Load reference features for MMD
        ref_features_npz = current_path / "reference_features" / f"reference_{category}_features_v2.npz"
        if not ref_features_npz.exists():
            print(f"❌ Reference features not found: {ref_features_npz}")
            return
            
        ref_npz = np.load(ref_features_npz)
        ref_features_list = []
        for key in ref_npz.keys():
            ref_features_list.append(ref_npz[key])
        ref_features = np.stack(ref_features_list)
        ref_npz.close()
        
        print(f"📊 Loaded {len(ref_features)} reference feature vectors (shape: {ref_features.shape})")
        
        # Load reference images for MI
        ref_images_npz = current_path / "reference_features" / f"reference_{category}_images.npz"
        ref_images = None
        if ref_images_npz.exists():
            images_npz = np.load(ref_images_npz)
            ref_images_list = []
            for key in images_npz.keys():
                ref_images_list.append(images_npz[key])
            ref_images = np.stack(ref_images_list)
            images_npz.close()
            print(f"📊 Loaded {len(ref_images)} reference images (shape: {ref_images.shape})")
        else:
            print(f"⚠️ Reference images not found: {ref_images_npz}")
            print(f"⚠️ Run: python create_reference_images_npz.py --category {category}")
            return
        
        # Initialize metrics
        mmd_metric = MMDRewardMetric()
        mi_metric = MIRewardMetric()
        
        # Test FID if pytorch-fid is available
        fid_metric = None
        try:
            fid_metric = FIDRewardMetric(reward_scale=0.1)
            print("🎯 FID metric initialized for testing")
        except Exception as e:
            print(f"⚠️ FID metric not available: {e}")
            print("💡 Install pytorch-fid to test FID scaling: pip install pytorch-fid")
        
        # Generate synthetic test data
        print(f"\n🧪 Generating {num_samples} synthetic samples for testing...")
        
        # Create various types of synthetic data to test different scales
        test_scenarios = {
            "Random Features": np.random.randn(num_samples, ref_features.shape[1]),
            "Scaled Reference": ref_features[:num_samples] + 0.1 * np.random.randn(num_samples, ref_features.shape[1]),
            "Noisy Features": np.random.randn(num_samples, ref_features.shape[1]) * 0.5,
            "Similar to Ref": ref_features[:num_samples] + 0.01 * np.random.randn(num_samples, ref_features.shape[1])
        }
        
        test_images_scenarios = {
            "Random Images": np.random.randint(0, 256, (num_samples, 512, 512, 3), dtype=np.uint8) / 255.0,
            "Noisy Reference": np.clip(ref_images[:num_samples] + np.random.randint(-50, 50, (num_samples, 512, 512, 3)), 0, 255).astype(np.uint8) / 255.0,
            "Similar Images": np.clip(ref_images[:num_samples] + np.random.randint(-10, 10, (num_samples, 512, 512, 3)), 0, 255).astype(np.uint8) / 255.0
        }
        
        # Convert reference images to [0,1] range for FID
        ref_images_normalized = ref_images.astype(np.float32) / 255.0
        
        print("\n" + "="*60)
        print("MMD REWARD SCALE ANALYSIS (Feature-based)")
        print("="*60)
        
        mmd_results = {}
        for scenario_name, test_features in test_scenarios.items():
            print(f"\n🔍 Testing MMD with: {scenario_name}")
            try:
                mmd_rewards = mmd_metric.calculate_rewards(test_features, ref_features)
                mmd_results[scenario_name] = mmd_rewards
                
                print(f"   Shape: {mmd_rewards.shape}")
                print(f"   Range: [{mmd_rewards.min():.6f}, {mmd_rewards.max():.6f}]")
                print(f"   Mean:  {mmd_rewards.mean():.6f} ± {mmd_rewards.std():.6f}")
                print(f"   Median: {np.median(mmd_rewards):.6f}")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                mmd_results[scenario_name] = None
        
        print("\n" + "="*60)
        print("MI REWARD SCALE ANALYSIS (Image-based)")
        print("="*60)
        
        mi_results = {}
        for scenario_name, test_images in test_images_scenarios.items():
            print(f"\n🔍 Testing MI with: {scenario_name}")
            try:
                mi_rewards = mi_metric.calculate_rewards(test_images, ref_images)
                mi_results[scenario_name] = mi_rewards
                
                print(f"   Shape: {mi_rewards.shape}")
                print(f"   Range: [{mi_rewards.min():.6f}, {mi_rewards.max():.6f}]")
                print(f"   Mean:  {mi_rewards.mean():.6f} ± {mi_rewards.std():.6f}")
                print(f"   Median: {np.median(mi_rewards):.6f}")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                mi_results[scenario_name] = None
        
        # Test FID if available
        if fid_metric is not None:
            print("\n" + "="*60)
            print("FID REWARD SCALE ANALYSIS (Image-based)")
            print("="*60)
            
            fid_results = {}
            device = 'cuda' if 'torch' in sys.modules and hasattr(sys.modules['torch'], 'cuda') and sys.modules['torch'].cuda.is_available() else 'cpu'
            
            # Test different FID reward scales
            test_scales = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
            
            print(f"\n🔍 Testing FID with different reward scales (using 'Random Images' scenario)")
            test_images = test_images_scenarios["Random Images"]
            
            try:
                for scale in test_scales:
                    print(f"\n   Testing scale {scale:.2f}:")
                    fid_metric_test = FIDRewardMetric(reward_scale=scale)
                    fid_rewards = fid_metric_test.calculate_rewards(test_images, ref_images_normalized, device=device)
                    fid_results[f"Scale_{scale}"] = fid_rewards
                    
                    print(f"     Range: [{fid_rewards.min():.6f}, {fid_rewards.max():.6f}]")
                    print(f"     Mean:  {fid_rewards.mean():.6f} ± {fid_rewards.std():.6f}")
                    
                print(f"\n💡 FID Scale Recommendations:")
                print(f"   - For rewards similar to MMD/MI: try scales 0.1-0.5")
                print(f"   - Lower scales (0.01-0.05): more conservative FID influence")
                print(f"   - Higher scales (0.5-1.0): stronger FID influence")
                    
            except Exception as e:
                print(f"   ❌ FID testing error: {e}")
                print(f"   💡 Make sure pytorch-fid is installed: pip install pytorch-fid")
        
        # Analyze and recommend weights
        print("\n" + "="*60)
        print("WEIGHT RECOMMENDATION ANALYSIS")
        print("="*60)
        
        # Get typical ranges
        mmd_ranges = []
        mi_ranges = []
        
        for scenario, rewards in mmd_results.items():
            if rewards is not None:
                mmd_ranges.extend([rewards.min(), rewards.max()])
        
        for scenario, rewards in mi_results.items():
            if rewards is not None:
                mi_ranges.extend([rewards.min(), rewards.max()])
        
        if mmd_ranges and mi_ranges:
            mmd_typical_range = max(mmd_ranges) - min(mmd_ranges)
            mi_typical_range = max(mi_ranges) - min(mi_ranges)
            
            print(f"\n📊 MMD typical range: {mmd_typical_range:.6f}")
            print(f"📊 MI typical range:  {mi_typical_range:.6f}")
            
            if mmd_typical_range > 0 and mi_typical_range > 0:
                # Suggest weights to balance the scales
                scale_ratio = mmd_typical_range / mi_typical_range
                
                print(f"\n💡 Scale ratio (MMD/MI): {scale_ratio:.3f}")
                
                if scale_ratio > 2:
                    print(f"💡 MMD has larger scale - consider higher MI weight")
                    suggested_mmd_weight = 0.3
                    suggested_mi_weight = 0.7
                elif scale_ratio < 0.5:
                    print(f"💡 MI has larger scale - consider higher MMD weight")
                    suggested_mmd_weight = 0.7
                    suggested_mi_weight = 0.3
                else:
                    print(f"💡 Scales are similar - balanced weights should work")
                    suggested_mmd_weight = 0.5
                    suggested_mi_weight = 0.5
                
                print(f"\n🎯 SUGGESTED WEIGHTS:")
                print(f"   MMD_WEIGHT = {suggested_mmd_weight}")
                print(f"   MI_WEIGHT = {suggested_mi_weight}")
                
                print(f"\n📝 Add these to constants.py:")
                print(f"   DEFAULT_REWARD_METRIC = \"MMD_MI\"")
                print(f"   MMD_WEIGHT = {suggested_mmd_weight}")
                print(f"   MI_WEIGHT = {suggested_mi_weight}")
        
        return mmd_results, mi_results
        
    except Exception as e:
        print(f"❌ Error in reward scale testing: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    """Run the reward scale testing"""
    print("🔬 Reward Scale Analysis Tool")
    print("Analyzes MMD and MI reward ranges to suggest optimal combination weights")
    
    # Test with crater category
    mmd_results, mi_results = test_reward_scales("crater", num_samples=10)
    
    if mmd_results and mi_results:
        print("\n✅ Scale analysis complete!")
        print("Use the suggested weights above for MMD_MI combination.")
    else:
        print("\n❌ Scale analysis failed. Check error messages above.")

if __name__ == "__main__":
    main()