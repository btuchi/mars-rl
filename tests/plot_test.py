from pathlib import Path
import sys
# Add the project root directory to sys.path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from ppo_diffusion.diffusion_log_utils import plot_from_csv

# Define the training timestamp and category
training_timestamp = "20250620211027"  # Replace with the actual timestamp from your logs
category = "crater"  # Replace with the category used during training

# Call the function to plot the data
plot_from_csv(training_timestamp, category)