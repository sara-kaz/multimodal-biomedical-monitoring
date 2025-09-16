# Data Export Summary

This document summarizes all the data export options available for the multimodal biomedical dataset.

## 📊 Available Export Files

### **1. Quick Export (Simple)**
- **Files**: `output.xlsx`, `output.csv`
- **Content**: 1,000 sample records with signal statistics
- **Columns**: 36 columns including sample metadata, label classes, and signal statistics
- **Size**: ~0.6 MB
- **Use Case**: Quick analysis, data exploration

### **2. Comprehensive Export (Advanced)**
- **Location**: `exports/` directory
- **Files**: 38 files total
- **Content**: Complete dataset analysis and samples

#### **Metadata Files**
- `dataset_metadata.json` - Complete dataset statistics in JSON format
- `dataset_metadata.csv` - Dataset summary in CSV format

#### **Sample Data Files**
- `sample_data.xlsx` - 1,000 sample records with full statistics
- `sample_data.csv` - Same data in CSV format

#### **Label Analysis Files**
- `label_analysis.xlsx` - All 60,510 samples with label information
- `label_analysis.csv` - Same data in CSV format
- `label_summary.xlsx` - Label distribution summary
- `label_summary.csv` - Same data in CSV format

#### **Signal Data Files**
- `signal_sample_000_*.csv` to `signal_sample_029_*.csv` - 30 individual signal samples
- Each file contains 10-second time series data for all 5 channels

## 🎯 Use Cases for Each Export

### **For Quick Analysis**
- Use `output.xlsx` or `output.csv`
- Contains 1,000 samples with signal statistics
- Easy to open in Excel or any data analysis tool

### **For Complete Dataset Analysis**
- Use `exports/label_analysis.xlsx` for all 60,510 samples
- Use `exports/dataset_metadata.json` for dataset statistics
- Use `exports/label_summary.xlsx` for class distributions

### **For Signal Processing**
- Use individual `signal_sample_*.csv` files
- Each contains raw 10-second time series data
- Perfect for signal processing and analysis

### **For Machine Learning**
- Use `exports/sample_data.xlsx` for training data
- Use `exports/label_analysis.xlsx` for complete dataset
- All files include proper label encodings

## 📋 Data Structure

### **Sample Data Columns**
- `sample_id` - Unique identifier
- `subject_id` - Subject identifier
- `dataset` - Source dataset (PPG-DaLiA, WESAD, MIT-BIH)
- `window_index` - Window number within subject
- `start_time` - Start time in seconds
- `*_class` - Class labels for each task
- `*_labels` - One-hot encoded labels
- `*_mean/std/min/max` - Signal statistics for each channel
- `*_available` - Channel availability flags

### **Signal Data Columns**
- `time` - Time axis (0-10 seconds)
- `ECG`, `PPG`, `Accel_X`, `Accel_Y`, `Accel_Z` - Signal channels
- `subject_id`, `dataset`, `window_index` - Metadata

## 🔧 Technical Details

### **File Formats**
- **Excel (.xlsx)**: Best for data analysis, includes formatting
- **CSV (.csv)**: Best for programming, universal compatibility
- **JSON (.json)**: Best for metadata, structured data

### **Data Types**
- **Signal Statistics**: Float values (mean, std, min, max)
- **Labels**: Integer class indices, one-hot vectors as strings
- **Metadata**: Strings and integers
- **Time Series**: Float arrays

### **Missing Data Handling**
- NaN values for unavailable channels
- -1 for missing class labels
- Proper handling of MIT-BIH (ECG-only) data

## 📈 Dataset Statistics

- **Total Samples**: 60,510 windows
- **Total Subjects**: 78 subjects
- **Datasets**: 3 (PPG-DaLiA, WESAD, MIT-BIH)
- **Channels**: 5 (ECG, PPG, Accel_X, Accel_Y, Accel_Z)
- **Window Length**: 10 seconds
- **Sampling Rate**: 100 Hz
- **Overlap**: 50%

## 🚀 Quick Start

### **For Excel Users**
1. Open `output.xlsx` for quick analysis
2. Use `exports/sample_data.xlsx` for more detailed analysis
3. Use `exports/label_analysis.xlsx` for complete dataset

### **For Python Users**
```python
import pandas as pd

# Load sample data
df = pd.read_csv('output.csv')

# Load complete dataset
df_complete = pd.read_csv('exports/label_analysis.csv')

# Load signal data
signal_df = pd.read_csv('exports/signal_sample_000_PPG-DaLiA_S5.csv')
```

### **For R Users**
```r
# Load sample data
df <- read.csv('output.csv')

# Load complete dataset
df_complete <- read.csv('exports/label_analysis.csv')
```

## 📝 Notes

- All files are UTF-8 encoded
- Excel files include proper formatting
- CSV files are comma-separated
- Signal data files are ready for time series analysis
- Label encodings follow the standard format defined in the dataset

---

*Generated on: September 15, 2024*  
*Dataset: Multimodal Biomedical Dataset (60,510 samples)*  
*Export Tools: Python pandas, openpyxl*
