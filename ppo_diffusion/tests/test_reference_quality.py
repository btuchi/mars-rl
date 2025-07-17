#!/usr/bin/env python3
"""
Simple test to check if reference images are causing reward flattening
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def test_reference_diversity(category: str = "crater"):
    """Test if reference images have enough diversity"""
    print(f"🔍 Testing reference image diversity for category: {category}")
    print("=" * 60)
    
    # Load reference images
    current_path = Path(__file__).parent
    ref_images_npz = project_root / "reference_features" / f"reference_{category}_images.npz"
    
    if not ref_images_npz.exists():
        print(f"❌ Reference images not found: {ref_images_npz}")
        return False
    
    # Load images
    images_npz = np.load(ref_images_npz)
    ref_images_list = []
    for key in images_npz.keys():
        ref_images_list.append(images_npz[key])
    ref_images = np.stack(ref_images_list)
    images_npz.close()
    
    print(f"📊 Loaded {len(ref_images)} reference images (shape: {ref_images.shape})")
    
    # Test 1: Visual diversity statistics
    print("\n🔍 Test 1: Visual Diversity Statistics")
    print("-" * 40)
    
    # Calculate pixel-level statistics
    mean_brightness = np.mean(ref_images, axis=(1,2,3))
    std_brightness = np.std(ref_images, axis=(1,2,3))
    
    print(f"Brightness variation across images:")
    print(f"  Range: [{mean_brightness.min():.2f}, {mean_brightness.max():.2f}]")
    print(f"  Std: {mean_brightness.std():.2f}")
    print(f"  Expected: Should be > 10 for good diversity")
    
    if mean_brightness.std() < 5:
        print("⚠️  WARNING: Low brightness diversity - images may be too similar")
    else:
        print("✅ Good brightness diversity")
    
    # Test 2: Pairwise differences
    print("\n🔍 Test 2: Pairwise Image Differences")
    print("-" * 40)
    
    # Calculate mean squared differences between all pairs
    n_images = len(ref_images)
    pairwise_diffs = []
    
    for i in range(min(n_images, 10)):  # Test first 10 to save time
        for j in range(i+1, min(n_images, 10)):
            diff = np.mean((ref_images[i] - ref_images[j]) ** 2)
            pairwise_diffs.append(diff)
    
    pairwise_diffs = np.array(pairwise_diffs)
    
    print(f"Pairwise MSE differences:")
    print(f"  Range: [{pairwise_diffs.min():.2f}, {pairwise_diffs.max():.2f}]")
    print(f"  Mean: {pairwise_diffs.mean():.2f}")
    print(f"  Expected: Should be > 100 for good diversity")
    
    if pairwise_diffs.mean() < 50:
        print("⚠️  WARNING: Low pairwise differences - images may be too similar")
    else:
        print("✅ Good pairwise diversity")
    
    # Test 3: Create synthetic diverse references for comparison
    print("\n🔍 Test 3: Compare with Synthetic Diverse References")
    print("-" * 40)
    
    try:
        from training.reward_metrics import MIRewardMetric
        mi_metric = MIRewardMetric()
        
        # Test with your actual references
        print("Testing MI rewards with your reference images...")
        
        # Create test generated images (varied)
        test_images_varied = []
        for i in range(4):
            # Create images with different characteristics
            if i == 0:
                img = np.random.randint(0, 50, (512, 512, 3), dtype=np.uint8)  # Dark
            elif i == 1:
                img = np.random.randint(200, 255, (512, 512, 3), dtype=np.uint8)  # Bright
            elif i == 2:
                img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)  # Random
            else:
                img = np.full((512, 512, 3), 128, dtype=np.uint8)  # Gray
            test_images_varied.append(img)
        
        test_images_varied = np.stack(test_images_varied)
        
        # Test rewards with your references
        rewards_your_refs = mi_metric.calculate_rewards(test_images_varied, ref_images)
        
        # Create synthetic diverse references
        synthetic_refs = []
        for i in range(len(ref_images_list)):
            if i % 4 == 0:
                img = np.random.randint(0, 80, (512, 512, 3), dtype=np.uint8)  # Dark
            elif i % 4 == 1:
                img = np.random.randint(180, 255, (512, 512, 3), dtype=np.uint8)  # Bright
            elif i % 4 == 2:
                img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)  # Random
            else:
                img = np.random.randint(60, 200, (512, 512, 3), dtype=np.uint8)  # Medium
            synthetic_refs.append(img)
        
        synthetic_refs = np.stack(synthetic_refs)
        
        # Test rewards with synthetic references
        rewards_synthetic_refs = mi_metric.calculate_rewards(test_images_varied, synthetic_refs)
        
        print(f"\nReward comparison:")
        print(f"Your references:     {rewards_your_refs}")
        print(f"Synthetic diverse:   {rewards_synthetic_refs}")
        print(f"Your reward range:   {rewards_your_refs.max() - rewards_your_refs.min():.6f}")
        print(f"Synthetic range:     {rewards_synthetic_refs.max() - rewards_synthetic_refs.min():.6f}")
        
        if rewards_synthetic_refs.max() - rewards_synthetic_refs.min() > 2 * (rewards_your_refs.max() - rewards_your_refs.min()):
            print("🚨 PROBLEM DETECTED: Your references may lack diversity!")
            print("💡 Solution: Generate more diverse reference images")
            return False
        else:
            print("✅ Your reference diversity seems reasonable")
            
    except Exception as e:
        print(f"❌ Error testing rewards: {e}")
    
    return True

def quick_fix_test(category: str = "crater"):
    """Quick test: temporarily use random references"""
    print(f"\n🧪 Quick Fix Test: Using Random References")
    print("=" * 60)
    
    # Create highly diverse random references
    print("Creating highly diverse random reference images...")
    random_refs = []
    for i in range(20):  # Create 20 diverse random images
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        random_refs.append(img)
    
    random_refs = np.stack(random_refs)
    
    # Save them temporarily
    current_path = Path(__file__).parent
    temp_file = current_path / "reference_features" / f"temp_random_{category}_images.npz"
    
    # Save as individual arrays (matching your format)
    save_dict = {}
    for i, img in enumerate(random_refs):
        save_dict[f"image_{i:03d}"] = img
    
    np.savez(temp_file, **save_dict)
    
    print(f"✅ Created temporary diverse references: {temp_file}")
    print(f"💡 To test:")
    print(f"   1. Backup your current reference file")
    print(f"   2. Rename temp file to: reference_{category}_images.npz")
    print(f"   3. Run 2-3 training episodes")
    print(f"   4. Check if rewards show more variation")
    print(f"   5. Restore original references")
    
    return temp_file

def main():
    """Main test function"""
    print("🔬 Reference Image Quality Test")
    print("Tests if your reference images have enough diversity for meaningful rewards")
    
    category = "crater"
    
    # Test current references
    is_diverse = test_reference_diversity(category)
    
    # Suggest quick fix
    if not is_diverse:
        quick_fix_test(category)
        print(f"\n🎯 RECOMMENDATION: Try the quick fix above to test if reference diversity is the issue")
    else:
        print(f"\n✅ Reference images seem fine - the issue may be elsewhere")

if __name__ == "__main__":
    main()