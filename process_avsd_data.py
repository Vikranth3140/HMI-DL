import os
import tarfile
import json

def extract_and_load_avsd_dataset(dataset_dir):
    data = []
    # Loop through all tar.gz files in the directory
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

# Provide the correct path to the dataset files
dataset_dir = "OpenDataLab___AVSD/raw/.cache"
avsd_data = extract_and_load_avsd_dataset(dataset_dir)

print(f"Loaded {len(avsd_data)} training examples from AVSD dataset.")