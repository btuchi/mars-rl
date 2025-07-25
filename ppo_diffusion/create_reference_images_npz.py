#!/usr/bin/env python3
"""
Create NPZ files from reference images for MI calculation
"""

import numpy as np
from PIL import Image
import os
from pathlib import Path
import argparse

def create_reference_images_npz(category: str = "crater", target_size: tuple = (512, 512)):
    """
    Create NPZ file from reference images in reference_images/{category}/ folder
    
    Args:
        category: Category name (e.g., "crater") 
        target_size: Target image size (width, height)
    """
    
    # Setup paths
    current_path = Path(__file__).parent
    images_folder = current_path / "reference_images" / category
    output_folder = current_path / "reference_features"
    output_folder.mkdir(exist_ok=True)
    
    print(f"🔍 Looking for images in: {images_folder}")
    
    # Check if images folder exists
    if not images_folder.exists():
        print(f"❌ Error: Images folder not found: {images_folder}")
        print(f"💡 Expected structure: reference_images/{category}/image1.jpg, image2.png, etc.")
        return False
    
    # Collect image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    image_files = []
    
    for file_path in images_folder.iterdir():
        if file_path.suffix.lower() in image_extensions:
            image_files.append(file_path)
    
    if not image_files:
        print(f"❌ Error: No image files found in {images_folder}")
        print(f"💡 Expected formats: {', '.join(image_extensions)}")
        return False
    
    print(f"📷 Found {len(image_files)} images")
    
    # Process images
    images_dict = {}
    processed_count = 0
    
    for i, img_path in enumerate(sorted(image_files)):
        try:
            print(f"📷 Processing {img_path.name} ({i+1}/{len(image_files)})")
            
            # Load and resize image
            with Image.open(img_path) as img:
                # Convert to RGB if needed (removes alpha channel, handles grayscale)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize to target size
                img = img.resize(target_size, Image.Resampling.LANCZOS)
                
                # Convert to numpy array (H, W, C) format
                img_array = np.array(img, dtype=np.uint8)
                
                # Store with descriptive key
                key = f"image_{i:03d}_{img_path.stem}"
                images_dict[key] = img_array
                
                processed_count += 1
                
                print(f"  ✅ Shape: {img_array.shape}, dtype: {img_array.dtype}")
                
        except Exception as e:
            print(f"  ⚠️ Error processing {img_path.name}: {e}")
            continue
    
    if processed_count == 0:
        print("❌ Error: No images were successfully processed")
        return False
    
    # Save NPZ file
    output_path = output_folder / f"reference_{category}_images.npz"
    np.savez_compressed(output_path, **images_dict)
    
    print(f"✅ Created {output_path}")
    print(f"📊 Summary:")
    print(f"   - Images processed: {processed_count}/{len(image_files)}")
    print(f"   - Image size: {target_size[0]}×{target_size[1]}×3")
    print(f"   - File size: {output_path.stat().st_size / (1024*1024):.1f} MB")
    
    # Verify the NPZ file
    print(f"🔍 Verifying NPZ file...")
    try:
        loaded_data = np.load(output_path)
        print(f"   - Keys: {list(loaded_data.keys())[:5]}{'...' if len(loaded_data.keys()) > 5 else ''}")
        print(f"   - First image shape: {loaded_data[list(loaded_data.keys())[0]].shape}")
        loaded_data.close()
        print(f"   ✅ NPZ file is valid")
    except Exception as e:
        print(f"   ❌ NPZ verification failed: {e}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Create NPZ file from reference images")
    parser.add_argument("--category", default="crater", help="Category name (default: crater)")
    parser.add_argument("--width", type=int, default=512, help="Target image width (default: 512)")
    parser.add_argument("--height", type=int, default=512, help="Target image height (default: 512)")
    
    args = parser.parse_args()
    
    print("🖼️ Reference Images to NPZ Converter")
    print("=" * 50)
    
    success = create_reference_images_npz(
        category=args.category,
        target_size=(args.width, args.height)
    )
    
    if success:
        print("\n🎉 Success! You can now use MI or MMD_MI reward metrics.")
        print(f"💡 Set DEFAULT_REWARD_METRIC = 'MI' or 'MMD_MI' in constants.py")
    else:
        print("\n❌ Failed to create NPZ file. Please check the error messages above.")

if __name__ == "__main__":
    main()