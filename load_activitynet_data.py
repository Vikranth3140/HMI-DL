import os
import json

def load_activitynet_data(dataset_dir):
    data = {}
    files = ["train.json", "val_1.json", "val_2.json"]
    for file in files:
        file_path = os.path.join(dataset_dir, file)
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data.update(json.load(f))
    return data

dataset_dir = "captions"
activitynet_data = load_activitynet_data(dataset_dir)
print(f"Loaded {len(activitynet_data)} examples from ActivityNet dataset.")