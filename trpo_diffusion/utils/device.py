"""Device management and optimizations"""

import torch
import os

def setup_h100_optimizations():
    """Setup H100-specific optimizations"""
    # Environment variables for H100s
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,roundup_power2_divisions:16'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async launches
    
    # Set memory fractions for each GPU
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_per_process_memory_fraction(0.85, device=i)
    
    # Enable optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    print("H100 optimizations enabled")

def get_device_info():
    """Get device information"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        print(f"Using device: {device} ({device_name}, {device_count} GPUs available)")
    else:
        print(f"Using device: {device}")
    return device

def clear_gpu_cache():
    """Clear GPU cache and collect garbage"""
    import gc
    torch.cuda.empty_cache()
    gc.collect()