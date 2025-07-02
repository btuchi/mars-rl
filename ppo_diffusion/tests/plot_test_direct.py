#!/usr/bin/env python3
"""Direct import test script that bypasses __init__.py"""

import sys
from pathlib import Path

# Add the project root directory to sys.path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def main():
    """Main function to plot the most recent training logs"""
    category = "crater"
    most_recent_timestamp = "20250701154607"  # The one with gradient data
    
    print(f"📈 Plotting training data for: {category}_{most_recent_timestamp}")
    
    try:
        # Direct import to bypass __init__.py
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from utils.visualization import plot_from_csv
        
        plot_from_csv(most_recent_timestamp, category)
        print("✅ Plots generated successfully!")
        
    except Exception as e:
        print(f"❌ Error generating plots: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()