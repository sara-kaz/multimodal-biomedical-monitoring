# Multimodal Biomedical Dataset Integration Guide

This guide helps you combine and standardize PPG-DaLiA, MIT-BIH Arrhythmia, and WESAD datasets for your **Edge Intelligence for Multimodal Biomedical Monitoring** project.

## 📋 Prerequisites

### Required Python Packages
```bash
pip install requirements.txt
```

### Dataset Requirements
1. **PPG-DaLiA**: Download from [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/PPG-DaLiA)
2. **MIT-BIH Arrhythmia**: Download from [PhysioNet](https://physionet.org/content/mitdb/1.0.0/)
3. **WESAD**: Download from [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/WESAD)

## 🚀 Quick Start

### Step 1: Setup Directory Structure
```
your_project/
├── datasets/
│   ├── PPG_DaLiA/          # Extract PPG-DaLiA here
│   ├── mit-bih-arrhythmia/ # Extract MIT-BIH here  
│   └── WESAD/              # Extract WESAD here
├── processed_datasets/      # Will be created automatically
├── edge_deployment/         # Will be created automatically
└── your_scripts.py
```

### Step 2: Basic Usage
```python
from multimodal_dataset_processor import MultimodalDatasetProcessor
from data_loader_analyzer import DatasetAnalyzer, create_data_loaders

# Initialize processor
processor = MultimodalDatasetProcessor(output_dir="processed_datasets")

# Process all datasets
raw_data, windowed_data, raw_summary, windowed_summary = processor.process_all_datasets(
    ppg_dalia_path='datasets/PPG_DaLiA',
    mit_bih_path='datasets/mit-bih-arrhythmia',
    wesad_path='datasets/WESAD'
)

# Analyze processed data
analyzer = DatasetAnalyzer('processed_datasets/combined_windowed_data.pkl')
quality_metrics = analyzer.analyze_signal_quality()
label_distribution = analyzer.analyze_label_distribution()

# Create train/val/test splits
splits, subject_splits = analyzer.create_dataset_splits()

# Create PyTorch DataLoaders
train_loaders = create_data_loaders(splits['train'], batch_size=32)
```

## 📊 Dataset Characteristics After Processing

### Signal Types and Sampling Rates
- **ECG**: 250 Hz (from MIT-BIH and WESAD)
- **PPG**: 64 Hz (from PPG-DaLiA and WESAD)
- **Accelerometer**: 32 Hz (3-axis: X, Y, Z)
- **Gyroscope**: 32 Hz (from WESAD, if available)

### Label Types
- **Arrhythmia**: 5 classes (Normal, Supraventricular, Ventricular, Fusion, Other)
- **Stress**: 4 classes (Baseline, Stress, Amusement, Meditation)  
- **Activity**: Various activity types (dataset dependent)

### Processing Pipeline
1. **Filtering**: Bandpass filters optimized for each signal type
2. **Resampling**: Standardized sampling rates for edge deployment
3. **Normalization**: Percentile-based robust normalization
4. **Windowing**: 10-second windows with 50% overlap
5. **Alignment**: Time-synchronized multi-modal signals

## 🎯 For Your ESP32-S3 Project

### Memory-Optimized Features
- **Signal Compression**: int16 format reduces memory by 50%
- **Configurable Window Sizes**: Adjustable based on ESP32-S3 RAM
- **Normalization Stats**: Pre-computed for real-time inference
- **Compact Exports**: JSON format for embedded systems

### Edge Deployment Files
The processing pipeline generates several files optimized for edge deployment:

```
edge_deployment/
├── deployment_config.json          # Sampling rates, filter params
├── signal_normalization_stats.json # Real-time normalization
└── test_sample.json                # Sample data for testing
```

### ESP32-S3 Integration Example
```c
// Example normalization using exported stats
float normalize_ecg_sample(int16_t raw_value) {
    // Using exported p5/p95 percentiles for robust normalization
    float normalized = 2.0f * (raw_value - p5_ecg) / (p95_ecg - p5_ecg) - 1.0f;
    return fmaxf(-1.0f, fminf(1.0f, normalized));  // Clamp to [-1, 1]
}
```

## 🔧 Advanced Configuration

### Custom Signal Processing
```python
# Modify filtering parameters
processor.filter_params['ecg'] = {'highpass': 1.0, 'lowpass': 30}

# Adjust sampling rates for your hardware
processor.target_fs['ecg'] = 200  # Lower rate for resource constraints
```

### Custom Windowing
```python
# Create shorter windows for real-time processing
windowed_data = processor.create_windowed_samples(
    raw_data, 
    window_length=5,    # 5-second windows
    overlap=0.25        # 25% overlap
)
```

### Multi-Task Learning Setup
```python
# Create datasets for multiple tasks simultaneously
stress_dataset = MultimodalBiomedicalDataset(train_data, label_type='stress')
arrhythmia_dataset = MultimodalBiomedicalDataset(train_data, label_type='arrhythmia')
activity_dataset = MultimodalBiomedicalDataset(train_data, label_type='activity')
```

## 📈 Expected Results

### Dataset Statistics (Typical)
- **Total Samples**: ~50,000-100,000 windowed samples
- **PPG-DaLiA**: ~15 subjects, activity labels, PPG+ACC
- **MIT-BIH**: ~48 records, arrhythmia labels, ECG only
- **WESAD**: ~15 subjects, stress labels, ECG+PPG+ACC+others

### Signal Quality Metrics
- **ECG SNR**: 15-25 dB (good quality)
- **PPG SNR**: 10-20 dB (adequate for HR detection)
- **Accelerometer**: High dynamic range, motion artifacts handled

### Memory Footprint (10-sec windows)
- **ECG (250 Hz)**: 2,500 samples × 2 bytes = 5 KB
- **PPG (64 Hz)**: 640 samples × 2 bytes = 1.28 KB  
- **3-axis ACC (32 Hz)**: 320×3 samples × 2 bytes = 1.92 KB
- **Total per window**: ~8.2 KB (fits easily in ESP32-S3 RAM)

## 🐛 Troubleshooting

### Common Issues

1. **WFDB Import Error**
   ```bash
   pip install wfdb
   ```

2. **Memory Issues with Large Datasets**
   ```python
   # Process datasets one at a time
   ppg_data = processor.process_ppg_dalia(ppg_path)
   processor.save_processed_data(ppg_data, 'ppg_only.pkl')
   ```

3. **Missing Signal Types**
   ```python
   # Check available signals before creating datasets
   available_signals = set()
   for sample in processed_data:
       for sig_name, sig_data in sample['signals'].items():
           if sig_data is not None:
               available_signals.add(sig_name)
   print(f"Available signals: {available_signals}")
   ```

### Performance Optimization
- Use **multiprocessing** for large datasets
- **Batch process** signals to reduce memory usage
- **Pre-filter** samples by quality metrics


This integrated pipeline provides a solid foundation for your edge intelligence research. The standardized format ensures compatibility across datasets while maintaining the flexibility needed for advanced neural network architectures.
