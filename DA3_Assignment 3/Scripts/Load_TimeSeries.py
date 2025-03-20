import os
import pandas as pd

#%% User Defined Variables
# Define Base Directory
base_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))  # Navigate to project root

# Function to join paths
def jo(directory, filename):
    """Joins a directory and filename."""
    return os.path.join(directory, filename)



# Function to load a single dataset and return it as a DataFrame
def load_data(base_dir, filename="data.csv"):
    """Loads a single dataset from the time_series directory."""
    data_dir = jo(base_dir, "DA3_Ass3")  # Set the directory path
    file_path = jo(data_dir, filename)  # Full path to the dataset
    
    # Check if the file exists before loading
    if os.path.exists(file_path):
        print(f"✅ Loaded dataset from {file_path}")
        return pd.read_csv(file_path)  # Return as DataFrame
    else:
        print(f"⚠ Error: {file_path} not found.")
        return None  # Return None if the file is missing