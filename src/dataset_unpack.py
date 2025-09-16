"""
#3 (Extra Step for Excel Export)
Simple Dataset Unpacker
Exports a sample of the unified dataset to Excel and CSV formats
"""

import pickle
import pandas as pd
import numpy as np

# Load pickle file
with open("/Users/HP/Desktop/University/Thesis/Code/multimodal-biomedical-monitoring/processed_unified_dataset/unified_dataset.pkl", "rb") as f:
    data = pickle.load(f)   # list of dicts
# print(f"Loaded {len(data)} samples")

# Take a representative sample from all datasets
n_samples_per_dataset = min(1000, len(data) // 3)  # 1000 samples per dataset
sample_data = []

# Get samples from each dataset
datasets = {}
for sample in data:
    dataset = sample['dataset']
    if dataset not in datasets:
        datasets[dataset] = []
    if len(datasets[dataset]) < n_samples_per_dataset:
        datasets[dataset].append(sample)

# Combine samples from all datasets
for dataset_samples in datasets.values():
    sample_data.extend(dataset_samples)

n_samples = len(sample_data)

print(f"Exporting {n_samples} samples to Excel...")

rows = []
for i, entry in enumerate(sample_data):
    if i % 100 == 0:
        print(f"Processing sample {i}/{n_samples}")
    
    window_data = entry["window_data"]  # numpy array
    labels = entry["labels"]

    # Create row dictionary with signal statistics instead of raw data
    # Label information - handle properly based on dataset
    activity_class = int(np.argmax(labels["activity"])) if np.sum(labels["activity"]) > 0 else -1
    stress_class = int(np.argmax(labels["stress"])) if np.sum(labels["stress"]) > 0 else -1
    arrhythmia_class = int(np.argmax(labels["arrhythmia"])) if np.sum(labels["arrhythmia"]) > 0 else -1
    
    # For display purposes, show "N/A" for datasets that don't have certain labels
    if entry["dataset"] == "PPG-DaLiA":
        stress_class = "N/A"  # PPG-DaLiA doesn't have stress labels
        arrhythmia_class = "N/A"  # PPG-DaLiA doesn't have arrhythmia labels
    elif entry["dataset"] == "MIT-BIH":
        activity_class = "N/A"  # MIT-BIH doesn't have activity labels
        stress_class = "N/A"  # MIT-BIH doesn't have stress labels
    elif entry["dataset"] == "WESAD":
        activity_class = "N/A"  # WESAD doesn't have activity labels
        arrhythmia_class = "N/A"  # WESAD doesn't have arrhythmia labels
    
    row = {
        "sample_id": f"{entry['dataset']}_{entry['subject_id']}_{entry['window_index']}",
        "subject_id": entry["subject_id"],
        "dataset": entry["dataset"],
        "window_index": entry["window_index"],
        "start_time": entry["start_time"],
        "activity_class": activity_class,
        "stress_class": stress_class,
        "arrhythmia_class": arrhythmia_class,
        # Label vectors as strings
        "activity_labels": str(labels["activity"].tolist()),
        "stress_labels": str(labels["stress"].tolist()),
        "arrhythmia_labels": str(labels["arrhythmia"].tolist()),
    }
    
    # Add signal statistics for each channel
    channel_names = ['ECG', 'PPG', 'Accel_X', 'Accel_Y', 'Accel_Z', 'EDA', 'Respiration', 'Temperature', 'EMG', 'EDA_Wrist', 'Temperature_Wrist']
    
    # Define which signals are available for each dataset
    dataset_signals = {
        'PPG-DaLiA': ['ECG', 'PPG', 'Accel_X', 'Accel_Y', 'Accel_Z', 'Respiration', 'Temperature', 'EMG', 'EDA_Wrist', 'Temperature_Wrist'],
        'MIT-BIH': ['ECG'],
        'WESAD': ['ECG', 'PPG', 'Accel_X', 'Accel_Y', 'Accel_Z', 'EDA', 'Respiration', 'Temperature', 'EMG', 'EDA_Wrist', 'Temperature_Wrist']
    }
    
    available_signals = dataset_signals.get(entry["dataset"], [])
    
    for j, ch_name in enumerate(channel_names):
        signal_data = window_data[j, :]
        
        # Check if signal is available for this dataset and has valid data
        if ch_name in available_signals and not np.all(np.isnan(signal_data)) and not np.all(signal_data == 0):
            row[f"{ch_name}_mean"] = float(np.mean(signal_data))
            row[f"{ch_name}_std"] = float(np.std(signal_data))
            row[f"{ch_name}_min"] = float(np.min(signal_data))
            row[f"{ch_name}_max"] = float(np.max(signal_data))
            row[f"{ch_name}_available"] = True
        else:
            row[f"{ch_name}_mean"] = "N/A"
            row[f"{ch_name}_std"] = "N/A"
            row[f"{ch_name}_min"] = "N/A"
            row[f"{ch_name}_max"] = "N/A"
            row[f"{ch_name}_available"] = False
    
    rows.append(row)

# Convert to DataFrame
df = pd.DataFrame(rows)

print(f"DataFrame shape: {df.shape}")
print("Columns:", df.columns.tolist())

# Save to Excel
print("Saving to Excel...")
df.to_excel("output.xlsx", index=False)
print("✅ Excel file saved as 'output.xlsx'")

# Also save as CSV for easier viewing
df.to_csv("output.csv", index=False)
print("✅ CSV file saved as 'output.csv'")

print(f"\nSummary:")
print(f"- Total samples in dataset: {len(data)}")
print(f"- Exported samples: {len(df)}")
print(f"- Columns: {len(df.columns)}")
print(f"- File size: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
