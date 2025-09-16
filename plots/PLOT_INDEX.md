# Multimodal Biomedical Dataset - Plot Index

This directory contains all the plots generated for the thesis/paper on the multimodal biomedical dataset. All plots are saved in high-resolution PNG format (300 DPI) suitable for publication.

## 📊 Complete Plot Collection (26 plots)

### 1. **Dataset Overview & Summary**
- `dataset_overview.png` - Comprehensive overview showing dataset pipeline, statistics, and key metrics
- `dataset_summary.png` - Detailed dataset statistics, channel availability, and distribution analysis

### 2. **Signal Examples (9 plots)**
- `signal_examples_ppg-dalia_sample1.png` - PPG-DaLiA sample 1 (all 5 channels)
- `signal_examples_ppg-dalia_sample2.png` - PPG-DaLiA sample 2 (all 5 channels)
- `signal_examples_ppg-dalia_sample3.png` - PPG-DaLiA sample 3 (all 5 channels)
- `signal_examples_wesad_sample1.png` - WESAD sample 1 (all 5 channels)
- `signal_examples_wesad_sample2.png` - WESAD sample 2 (all 5 channels)
- `signal_examples_wesad_sample3.png` - WESAD sample 3 (all 5 channels)
- `signal_examples_mit-bih_sample1.png` - MIT-BIH sample 1 (ECG only)
- `signal_examples_mit-bih_sample2.png` - MIT-BIH sample 2 (ECG only)
- `signal_examples_mit-bih_sample3.png` - MIT-BIH sample 3 (ECG only)

### 3. **Label Distribution Analysis (3 plots)**
- `label_distribution_activity.png` - Activity label distribution (8 classes)
- `label_distribution_stress.png` - Stress label distribution (4 classes)
- `label_distribution_arrhythmia.png` - Arrhythmia label distribution (2 classes)

### 4. **Signal Quality Analysis (1 plot)**
- `signal_quality_analysis.png` - Comprehensive signal quality metrics including SNR, zero crossings, and RMS values

### 5. **Frequency Domain Analysis (3 plots)**
- `frequency_analysis_ppg-dalia.png` - Power spectral density analysis for PPG-DaLiA
- `frequency_analysis_wesad.png` - Power spectral density analysis for WESAD
- `frequency_analysis_mit-bih.png` - Power spectral density analysis for MIT-BIH

### 6. **Correlation Analysis (2 plots)**
- `correlation_analysis_ppg-dalia.png` - Channel correlation matrix for PPG-DaLiA
- `correlation_analysis_wesad.png` - Channel correlation matrix for WESAD

### 7. **Specialized Biomedical Analysis (5 plots)**
- `heart_rate_variability.png` - HRV analysis including RR intervals, Poincaré plot, and HRV metrics
- `activity_recognition_examples.png` - Examples of different activities for activity recognition
- `stress_detection_examples.png` - Examples of different stress states for stress detection
- `arrhythmia_examples.png` - Examples of normal vs abnormal heart rhythms
- `signal_statistics.png` - Comprehensive signal statistics including mean, std, skewness, kurtosis

### 8. **Legacy Plot (1 plot)**
- `sample_signals_0.png` - Original sample signals plot (legacy)

## 🎯 Plot Categories for Paper Sections

### **Introduction & Dataset Description**
- `dataset_overview.png` - Use as main figure for dataset description
- `dataset_summary.png` - Use for detailed statistics table

### **Methodology**
- `signal_examples_*.png` - Use to show signal examples from each dataset
- `frequency_analysis_*.png` - Use to demonstrate signal processing pipeline

### **Data Analysis**
- `label_distribution_*.png` - Use to show class distributions
- `signal_quality_analysis.png` - Use to demonstrate data quality
- `correlation_analysis_*.png` - Use to show inter-channel relationships

### **Applications**
- `activity_recognition_examples.png` - Use for activity recognition section
- `stress_detection_examples.png` - Use for stress detection section
- `arrhythmia_examples.png` - Use for arrhythmia detection section
- `heart_rate_variability.png` - Use for HRV analysis section

### **Results & Discussion**
- `signal_statistics.png` - Use for comprehensive statistical analysis
- `dataset_summary.png` - Use for dataset comparison

## 📋 Dataset Statistics Summary

- **Total Samples**: 60,510 windows
- **Total Subjects**: 78 subjects
- **Datasets**: 3 (PPG-DaLiA, WESAD, MIT-BIH)
- **Channels**: 5 (ECG, PPG, Accel_X, Accel_Y, Accel_Z)
- **Window Length**: 10 seconds
- **Sampling Rate**: 100 Hz
- **Overlap**: 50%

## 🔧 Technical Details

- **Resolution**: 300 DPI (publication quality)
- **Format**: PNG
- **Color Scheme**: Publication-friendly colors
- **Font Size**: Optimized for readability
- **Style**: Clean, professional appearance

## 📝 Usage Notes

1. All plots are ready for direct inclusion in academic papers
2. High-resolution format ensures quality in both digital and print formats
3. Color schemes are optimized for both color and grayscale printing
4. Font sizes are appropriate for standard paper formats
5. Plots include proper titles, labels, and legends

## 🎨 Customization

If you need to modify any plots, the source code is available in:
- `src/generate_paper_plots.py` - Main plotting script
- `src/generate_additional_plots.py` - Additional specialized plots
- `src/create_overview_plot.py` - Overview plot generator

---

*Generated on: September 15, 2024*  
*Dataset: Multimodal Biomedical Dataset (60,510 samples)*  
*Total Plots: 26*
