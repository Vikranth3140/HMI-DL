import os
import json

def load_activitynet_data(dataset_dir):
    data = {}
    files = ["train.json", "val_1.json", "val_2.json"]
    for file in files:
        file_path = os.path.join(dataset_dir, file)
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                file_data = json.load(f)
                print(f"Loaded {len(file_data)} items from {file}")
                # Print a sample of the data to verify structure
                sample_key = list(file_data.keys())[0]
                print(f"Sample data for {sample_key}: {file_data[sample_key]}")
                data.update(file_data)
    return data

dataset_dir = "captions"
activitynet_data = load_activitynet_data(dataset_dir)
print(f"Loaded {len(activitynet_data)} examples from ActivityNet dataset.")