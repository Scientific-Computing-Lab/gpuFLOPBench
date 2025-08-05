import json
import os

def load_kernels_data():
    """Loads the extracted CUDA kernels data from the JSON file."""
    # Path relative to this script file
    kernels_json_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "extracted_CUDA_kernels.json"))
    
    if not os.path.exists(kernels_json_path):
        print(f"Error: {kernels_json_path} not found.")
        return {}

    try:
        with open(kernels_json_path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {kernels_json_path}.")
        return {}

kernels_data = load_kernels_data()
target_names = list(kernels_data.keys()) if kernels_data else []
