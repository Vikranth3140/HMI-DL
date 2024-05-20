from datasets import load_dataset

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