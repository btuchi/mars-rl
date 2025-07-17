#!/usr/bin/env python3
"""Simple test script to call the existing visualization function"""

import sys
from pathlib import Path


# Add the project root directory to sys.path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def find_most_recent_logs(category: str = "crater"):
    """Find the most recent training logs for a given category"""
    current_path = Path(__file__).parent.parent
    logs_dir = current_path / "outputs" / "logs"
    
    if not logs_dir.exists():
        print(f"❌ Logs directory not found: {logs_dir}")
        return None
    
    # Find all log directories for this category
    pattern = f"{category}_*"
    log_dirs = [d for d in logs_dir.glob(pattern) if d.is_dir()]
    
    if not log_dirs:
        print(f"❌ No log directories found for category '{category}'")
        return None
    
    # Sort by timestamp (directory name contains timestamp)
    log_dirs.sort(key=lambda x: x.name.split('_')[-1], reverse=True)
    
    print(f"📋 Available training logs for '{category}':")
    for i, log_dir in enumerate(log_dirs[:5]):  # Show top 5
        timestamp = log_dir.name.split('_')[-1]
        files = [f.name for f in log_dir.glob("*.csv")]
        print(f"  {i+1}. {timestamp} - Files: {', '.join(files)}")
    
    # Check which ones have actual data
    for log_dir in log_dirs:
        episode_log = log_dir / "episode_log.csv"
        if episode_log.exists():
            timestamp = log_dir.name.split('_')[-1]
            print(f"📊 Most recent logs with data: {category}_{timestamp}")
            return timestamp
    
    print(f"❌ No log directories with episode data found for category '{category}'")
    return None

def main():
    """Main function to plot the most recent training logs"""
    category = "crater"
    
    print(f"🔍 Looking for most recent training logs for category: '{category}'")
    
    # Find the most recent one with data
    most_recent_timestamp = find_most_recent_logs(category)
    
    if not most_recent_timestamp:
        print("❌ No valid training logs found")
        return
    
    print(f"📈 Plotting training data for: {category}_{most_recent_timestamp}")
    
    try:
        # Import and call the existing visualization functions
        from ppo_diffusion.utils.visualization import plot_from_csv
        
        # Generate training progress plots
        print("📊 Generating training progress plots...")
        success = plot_from_csv(most_recent_timestamp, category)
        
        if success:
            print("✅ Training plots generated successfully!")
            print(f"📁 Training plots: outputs/plots/training/")

    except Exception as e:
        print(f"❌ Error generating plots: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()