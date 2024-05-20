import os
import json

def load_msr_vtt_dataset(dataset_dir):
    data = []
    for filename in os.listdir(dataset_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(dataset_dir, filename)
            with open(file_path, 'r') as f:
                for line in f:
                    data.append(json.loads(line))
    return data

dataset_dir = "path_to_msr_vtt_dataset"
msr_vtt_data = load_msr_vtt_dataset(dataset_dir)
print(f"Loaded {len(msr_vtt_data)} examples from MSR-VTT dataset.")