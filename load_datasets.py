import os
import tarfile
import json

def extract_and_load_avsd_dataset(dataset_dir):
    data = []
    for filename in os.listdir(dataset_dir):
        if filename.endswith(".tar.gz"):
            file_path = os.path.join(dataset_dir, filename)
            print(f"Processing file: {file_path}")
            with tarfile.open(file_path, 'r:gz') as tar:
                for member in tar.getmembers():
                    print(f"Found member: {member.name}")
                    if member.isfile() and member.name.endswith('.json'):
                        f = tar.extractfile(member)
                        if f:
                            content = f.read().decode('utf-8')
                            data.append(json.loads(content))
    return data

def load_activitynet_data(dataset_dir):
    data = {}
    files = ["train.json", "val_1.json", "val_2.json"]
    for file in files:
        file_path = os.path.join(dataset_dir, file)
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data.update(json.load(f))
    return data

# Load AVSD dataset
avsd_dataset_dir = "OpenDataLab__AVSD/raw/.cache"
avsd_data = extract_and_load_avsd_dataset(avsd_dataset_dir)
print(f"Loaded {len(avsd_data)} training examples from AVSD dataset.")

# Load ActivityNet dataset
activitynet_dataset_dir = "path_to_extracted_activitynet_files"
activitynet_data = load_activitynet_data(activitynet_dataset_dir)
print(f"Loaded {len(activitynet_data)} examples from ActivityNet dataset.")