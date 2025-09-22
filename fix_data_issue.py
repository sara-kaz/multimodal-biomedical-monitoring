#!/usr/bin/env python3
"""
Fix data issue - identify and clean NaN/Inf values in the dataset
"""

import torch
import numpy as np
import pickle
from pathlib import Path

def analyze_and_fix_data():
    """Analyze and fix data issues"""
    
    print("🔍 Analyzing dataset for NaN/Inf issues...")
    
    # Load data
    data_path = "processed_unified_dataset/unified_dataset.pkl"
    with open(data_path, 'rb') as f:
        processed_data = pickle.load(f)
    
    print(f"📊 Loaded {len(processed_data)} samples")
    
    # Analyze data quality
    valid_samples = 0
    invalid_samples = 0
    nan_samples = 0
    inf_samples = 0
    
    for i, sample in enumerate(processed_data):
        if 'window_data' in sample and sample['window_data'] is not None:
            window_data = sample['window_data']
            
            if hasattr(window_data, 'shape') and len(window_data.shape) == 2:
                # Check for NaN and Inf values
                if np.isnan(window_data).any():
                    nan_samples += 1
                    print(f"⚠️  Sample {i} contains NaN values")
                    
                if np.isinf(window_data).any():
                    inf_samples += 1
                    print(f"⚠️  Sample {i} contains Inf values")
                    
                # Check data range
                if np.abs(window_data).max() > 1e6:
                    print(f"⚠️  Sample {i} has extreme values: {np.abs(window_data).max()}")
                
                valid_samples += 1
            else:
                invalid_samples += 1
        else:
            invalid_samples += 1
    
    print(f"\n📈 Data Quality Analysis:")
    print(f"✅ Valid samples: {valid_samples}")
    print(f"❌ Invalid samples: {invalid_samples}")
    print(f"⚠️  NaN samples: {nan_samples}")
    print(f"⚠️  Inf samples: {inf_samples}")
    
    # Clean the data
    print(f"\n🧹 Cleaning data...")
    cleaned_data = []
    
    for i, sample in enumerate(processed_data):
        if 'window_data' in sample and sample['window_data'] is not None:
            window_data = sample['window_data']
            
            if hasattr(window_data, 'shape') and len(window_data.shape) == 2:
                # Clean NaN and Inf values
                cleaned_window = window_data.copy()
                
                # Replace NaN with 0
                cleaned_window = np.nan_to_num(cleaned_window, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Clip extreme values
                cleaned_window = np.clip(cleaned_window, -1e6, 1e6)
                
                # Normalize to prevent extreme values
                if np.std(cleaned_window) > 0:
                    cleaned_window = (cleaned_window - np.mean(cleaned_window)) / np.std(cleaned_window)
                
                # Update sample
                sample['window_data'] = cleaned_window
                cleaned_data.append(sample)
    
    print(f"✅ Cleaned {len(cleaned_data)} samples")
    
    # Save cleaned data
    output_path = "processed_unified_dataset/cleaned_unified_dataset.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(cleaned_data, f)
    
    print(f"💾 Cleaned data saved to: {output_path}")
    
    # Test with a simple model
    print(f"\n🧪 Testing with simple model...")
    
    class TestModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(11 * 1000, 8)
        
        def forward(self, x):
            x = x.view(x.size(0), -1)
            return self.linear(x)
    
    model = TestModel()
    
    # Test with first few samples
    test_samples = cleaned_data[:4]
    test_data = []
    
    for sample in test_samples:
        window_data = sample['window_data']
        test_data.append(torch.FloatTensor(window_data))
    
    test_batch = torch.stack(test_data)
    print(f"Test batch shape: {test_batch.shape}")
    print(f"Test batch contains NaN: {torch.isnan(test_batch).any()}")
    print(f"Test batch contains Inf: {torch.isinf(test_batch).any()}")
    print(f"Test batch range: [{test_batch.min():.4f}, {test_batch.max():.4f}]")
    
    # Test forward pass
    with torch.no_grad():
        output = model(test_batch)
        print(f"Model output shape: {output.shape}")
        print(f"Model output contains NaN: {torch.isnan(output).any()}")
        print(f"Model output contains Inf: {torch.isinf(output).any()}")
        print(f"Model output range: [{output.min():.4f}, {output.max():.4f}]")
    
    return output_path

if __name__ == "__main__":
    cleaned_data_path = analyze_and_fix_data()
    print(f"\n✅ Data cleaning completed!")
    print(f"📁 Use cleaned data: {cleaned_data_path}")
