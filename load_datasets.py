import os
import tarfile
import json
from datasets import load_dataset

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

# Load AVSD dataset
dataset_dir = "OpenDataLab__AVSD/raw/.cache"
avsd_data = extract_and_load_avsd_dataset(dataset_dir)
print(f"Loaded {len(avsd_data)} training examples from AVSD dataset.")

# Load ActivityNet dataset
try:
    activitynet_dataset = load_dataset("huggingface/activitynet-captions")
    print("Loaded ActivityNet dataset.")
except Exception as e:
    print("Failed to load ActivityNet dataset. Error:", e)
    # Fallback to an alternative dataset
    try:
        msr_vtt_dataset = load_dataset("MSR-VTT")
        print("Loaded MSR-VTT dataset.")
    except Exception as e:
        print("Failed to load MSR-VTT dataset. Error:", e)