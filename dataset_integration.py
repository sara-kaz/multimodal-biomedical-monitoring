"""
Multimodal Biomedical Dataset Integration Pipeline
For Edge Intelligence Wearable Sensor-Fusion System

Combines and standardizes:
- PPG-DaLiA (PPG + Accelerometer + Activity labels)
- MIT-BIH Arrhythmia (ECG + Annotations)
- WESAD (ECG, PPG, EDA, EMG, Temp, Accelerometer, Gyroscope + Stress labels)
"""

import numpy as np
import pandas as pd
import scipy.io
from scipy import signal
from scipy.interpolate import interp1d
import h5py
import pickle
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class MultimodalDatasetProcessor:
    def __init__(self, output_dir="processed_datasets"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Standardized sampling rates for edge deployment
        self.target_fs = {
            'ecg': 250,      # Hz - Good balance for arrhythmia detection
            'ppg': 64,       # Hz - Sufficient for HR and SpO2
            'accel': 32,     # Hz - Adequate for activity recognition
            'gyro': 32,      # Hz - Adequate for orientation
            'other': 32      # Hz - For EDA, EMG, temp
        }
        
        # Standardized signal processing parameters
        self.filter_params = {
            'ecg': {'highpass': 0.5, 'lowpass': 40},
            'ppg': {'highpass': 0.5, 'lowpass': 8},
            'accel': {'highpass': 0.3, 'lowpass': 15},
            'gyro': {'highpass': 0.3, 'lowpass': 15}
        }
        
    def preprocess_signal(self, signal_data, signal_type, original_fs):
        """Standardized signal preprocessing pipeline"""
        # Remove DC component
        signal_data = signal_data - np.mean(signal_data)
        
        # Apply bandpass filtering
        if signal_type in self.filter_params:
            params = self.filter_params[signal_type]
            nyquist = original_fs / 2
            
            # High-pass filter
            if params['highpass'] < nyquist:
                sos_hp = signal.butter(4, params['highpass']/nyquist, 
                                     btype='high', output='sos')
                signal_data = signal.sosfilt(sos_hp, signal_data)
            
            # Low-pass filter
            if params['lowpass'] < nyquist:
                sos_lp = signal.butter(4, params['lowpass']/nyquist, 
                                     btype='low', output='sos')
                signal_data = signal.sosfilt(sos_lp, signal_data)
        
        # Resample to target frequency
        target_fs = self.target_fs.get(signal_type, self.target_fs['other'])
        if original_fs != target_fs:
            num_samples = int(len(signal_data) * target_fs / original_fs)
            signal_data = signal.resample(signal_data, num_samples)
        
        # Normalize to [-1, 1] range for neural network compatibility
        signal_data = self.normalize_signal(signal_data)
        
        return signal_data, target_fs
    
    def normalize_signal(self, data):
        """Robust normalization using percentile-based scaling"""
        p5, p95 = np.percentile(data, [5, 95])
        if p95 - p5 > 0:
            data = 2 * (data - p5) / (p95 - p5) - 1
            data = np.clip(data, -1, 1)
        return data
    
    def process_ppg_dalia(self, data_path):
        """Process PPG-DaLiA dataset"""
        print("Processing PPG-DaLiA dataset...")
        
        processed_data = []
        data_path = Path(data_path)
        
        # PPG-DaLiA contains .pkl files for each subject
        for pkl_file in data_path.glob("S*.pkl"):
            subject_id = pkl_file.stem
            print(f"  Processing {subject_id}")
            
            with open(pkl_file, 'rb') as f:
                subject_data = pickle.load(f, encoding='latin1')
            
            # Extract signals (original fs = 64 Hz for PPG, 32 Hz for accelerometer)
            ppg_signal = subject_data['signal']['wrist']['BVP']
            accel_x = subject_data['signal']['wrist']['ACC'][:, 0]
            accel_y = subject_data['signal']['wrist']['ACC'][:, 1] 
            accel_z = subject_data['signal']['wrist']['ACC'][:, 2]
            
            # Process signals
            ppg_processed, ppg_fs = self.preprocess_signal(ppg_signal, 'ppg', 64)
            accel_x_processed, accel_fs = self.preprocess_signal(accel_x, 'accel', 32)
            accel_y_processed, _ = self.preprocess_signal(accel_y, 'accel', 32)
            accel_z_processed, _ = self.preprocess_signal(accel_z, 'accel', 32)
            
            # Align time series (interpolate to common time base)
            min_length = min(len(ppg_processed), len(accel_x_processed))
            
            sample_data = {
                'subject_id': subject_id,
                'dataset': 'PPG-DaLiA',
                'signals': {
                    'ppg': ppg_processed[:min_length],
                    'accel_x': accel_x_processed[:min_length],
                    'accel_y': accel_y_processed[:min_length], 
                    'accel_z': accel_z_processed[:min_length],
                    'ecg': None  # Not available
                },
                'sampling_rates': {
                    'ppg': ppg_fs,
                    'accel': accel_fs,
                    'ecg': None
                },
                'labels': {
                    'activity': subject_data.get('activity', None),
                    'stress': None,  # Not available
                    'arrhythmia': None  # Not available
                },
                'duration': min_length / ppg_fs  # seconds
            }
            processed_data.append(sample_data)
            
        return processed_data
    
    def process_mit_bih(self, data_path):
        """Process MIT-BIH Arrhythmia dataset"""
        print("Processing MIT-BIH Arrhythmia dataset...")
        
        processed_data = []
        data_path = Path(data_path)
        
        # MIT-BIH uses .dat and .hea files
        for dat_file in data_path.glob("*.dat"):
            record_name = dat_file.stem
            if record_name.startswith('.'):
                continue
                
            print(f"  Processing {record_name}")
            
            try:
                # Read using wfdb library (install with: pip install wfdb)
                import wfdb
                record = wfdb.rdrecord(str(dat_file.parent / record_name))
                annotation = wfdb.rdann(str(dat_file.parent / record_name), 'atr')
                
                # Extract ECG signals (usually 2 leads, fs = 360 Hz)
                ecg_lead1 = record.p_signal[:, 0]
                ecg_lead2 = record.p_signal[:, 1] if record.p_signal.shape[1] > 1 else None
                
                # Process ECG signal
                ecg_processed, ecg_fs = self.preprocess_signal(ecg_lead1, 'ecg', record.fs)
                
                # Process annotations for arrhythmia labels
                arrhythmia_labels = self.process_mit_bih_annotations(
                    annotation, len(ecg_processed), record.fs, ecg_fs)
                
                sample_data = {
                    'subject_id': record_name,
                    'dataset': 'MIT-BIH',
                    'signals': {
                        'ecg': ecg_processed,
                        'ppg': None,  # Not available
                        'accel_x': None,
                        'accel_y': None,
                        'accel_z': None
                    },
                    'sampling_rates': {
                        'ecg': ecg_fs,
                        'ppg': None,
                        'accel': None
                    },
                    'labels': {
                        'arrhythmia': arrhythmia_labels,
                        'activity': None,
                        'stress': None
                    },
                    'duration': len(ecg_processed) / ecg_fs
                }
                processed_data.append(sample_data)
                
            except ImportError:
                print("  Warning: wfdb library not found. Install with: pip install wfdb")
                continue
            except Exception as e:
                print(f"  Error processing {record_name}: {e}")
                continue
                
        return processed_data
    
    def process_mit_bih_annotations(self, annotation, signal_length, orig_fs, new_fs):
        """Convert MIT-BIH annotations to sample-level labels"""
        # AAMI standard arrhythmia classes
        aami_classes = {
            'N': ['N', 'L', 'R', 'e', 'j'],  # Normal
            'S': ['A', 'a', 'J', 'S'],       # Supraventricular
            'V': ['V', 'E'],                  # Ventricular
            'F': ['F'],                       # Fusion
            'Q': ['/', 'f', 'Q']             # Unknown/Other
        }
        
        labels = np.zeros(signal_length, dtype=int)
        
        for i, (sample, symbol) in enumerate(zip(annotation.sample, annotation.symbol)):
            # Convert sample index to new sampling rate
            new_sample = int(sample * new_fs / orig_fs)
            if new_sample < signal_length:
                # Map symbol to AAMI class
                for class_idx, (class_name, symbols) in enumerate(aami_classes.items()):
                    if symbol in symbols:
                        labels[new_sample] = class_idx
                        break
                        
        return labels
    
    def process_wesad(self, data_path):
        """Process WESAD dataset"""
        print("Processing WESAD dataset...")
        
        processed_data = []
        data_path = Path(data_path)
        
        # WESAD contains .pkl files for each subject
        for pkl_file in data_path.glob("S*.pkl"):
            subject_id = pkl_file.stem
            print(f"  Processing {subject_id}")
            
            with open(pkl_file, 'rb') as f:
                subject_data = pickle.load(f, encoding='latin1')
            
            # Extract chest sensor data (higher quality for ECG)
            chest_data = subject_data['signal']['chest']
            wrist_data = subject_data['signal']['wrist']
            
            # Extract signals with their original sampling rates
            ecg_signal = chest_data['ECG'].flatten()  # 700 Hz
            ppg_signal = wrist_data['BVP'].flatten()  # 64 Hz
            accel_chest = chest_data['ACC']  # 700 Hz, 3D
            accel_wrist = wrist_data['ACC']  # 32 Hz, 3D
            
            # Process signals
            ecg_processed, ecg_fs = self.preprocess_signal(ecg_signal, 'ecg', 700)
            ppg_processed, ppg_fs = self.preprocess_signal(ppg_signal, 'ppg', 64)
            
            # Use wrist accelerometer (more relevant for activity)
            accel_x_processed, accel_fs = self.preprocess_signal(accel_wrist[:, 0], 'accel', 32)
            accel_y_processed, _ = self.preprocess_signal(accel_wrist[:, 1], 'accel', 32)
            accel_z_processed, _ = self.preprocess_signal(accel_wrist[:, 2], 'accel', 32)
            
            # Process stress labels (0=baseline, 1=stress, 2=amusement, 3=meditation)
            labels = subject_data['label'].flatten()
            stress_labels = self.process_wesad_labels(labels, len(ecg_processed), 700, ecg_fs)
            
            # Align all signals to ECG timebase
            min_length = len(ecg_processed)
            ppg_aligned = self.align_signals(ppg_processed, len(ecg_processed))
            accel_x_aligned = self.align_signals(accel_x_processed, len(ecg_processed))
            accel_y_aligned = self.align_signals(accel_y_processed, len(ecg_processed))
            accel_z_aligned = self.align_signals(accel_z_processed, len(ecg_processed))
            
            sample_data = {
                'subject_id': subject_id,
                'dataset': 'WESAD',
                'signals': {
                    'ecg': ecg_processed,
                    'ppg': ppg_aligned,
                    'accel_x': accel_x_aligned,
                    'accel_y': accel_y_aligned,
                    'accel_z': accel_z_aligned
                },
                'sampling_rates': {
                    'ecg': ecg_fs,
                    'ppg': ppg_fs,
                    'accel': accel_fs
                },
                'labels': {
                    'stress': stress_labels,
                    'activity': None,
                    'arrhythmia': None
                },
                'duration': len(ecg_processed) / ecg_fs
            }
            processed_data.append(sample_data)
            
        return processed_data
    
    def process_wesad_labels(self, labels, signal_length, orig_fs, new_fs):
        """Convert WESAD labels to new sampling rate"""
        # Resample labels to match signal
        if len(labels) != signal_length:
            # WESAD labels are at 700 Hz, resample to new rate
            label_indices = np.arange(len(labels)) * new_fs / orig_fs
            new_indices = np.arange(signal_length)
            
            # Use nearest neighbor interpolation for discrete labels
            f = interp1d(label_indices, labels, kind='nearest', 
                        bounds_error=False, fill_value=0)
            resampled_labels = f(new_indices).astype(int)
        else:
            resampled_labels = labels.astype(int)
            
        return resampled_labels
    
    def align_signals(self, source_signal, target_length):
        """Align signal length using interpolation"""
        if len(source_signal) == target_length:
            return source_signal
            
        x_old = np.linspace(0, 1, len(source_signal))
        x_new = np.linspace(0, 1, target_length)
        f = interp1d(x_old, source_signal, kind='linear', 
                    bounds_error=False, fill_value='extrapolate')
        return f(x_new)
    
    def create_windowed_samples(self, processed_data, window_length=10, overlap=0.5):
        """Create fixed-length windows for training"""
        print("Creating windowed samples...")
        
        windowed_samples = []
        
        for sample in processed_data:
            # Determine primary sampling rate
            if sample['signals']['ecg'] is not None:
                fs = sample['sampling_rates']['ecg']
                signal_length = len(sample['signals']['ecg'])
            elif sample['signals']['ppg'] is not None:
                fs = sample['sampling_rates']['ppg']
                signal_length = len(sample['signals']['ppg'])
            else:
                continue
            
            window_samples = int(window_length * fs)
            step_size = int(window_samples * (1 - overlap))
            
            for start_idx in range(0, signal_length - window_samples + 1, step_size):
                end_idx = start_idx + window_samples
                
                windowed_sample = {
                    'subject_id': sample['subject_id'],
                    'dataset': sample['dataset'],
                    'window_start': start_idx / fs,
                    'window_length': window_length,
                    'signals': {},
                    'sampling_rates': sample['sampling_rates'].copy(),
                    'labels': {}
                }
                
                # Extract windowed signals
                for signal_name, signal_data in sample['signals'].items():
                    if signal_data is not None:
                        windowed_sample['signals'][signal_name] = signal_data[start_idx:end_idx]
                    else:
                        windowed_sample['signals'][signal_name] = None
                
                # Extract windowed labels (use majority vote for classification)
                for label_name, label_data in sample['labels'].items():
                    if label_data is not None and len(label_data) > start_idx:
                        window_labels = label_data[start_idx:end_idx]
                        # Use majority vote for window label
                        unique, counts = np.unique(window_labels, return_counts=True)
                        windowed_sample['labels'][label_name] = unique[np.argmax(counts)]
                    else:
                        windowed_sample['labels'][label_name] = None
                
                windowed_samples.append(windowed_sample)
        
        return windowed_samples
    
    def save_processed_data(self, processed_data, filename):
        """Save processed data with metadata"""
        save_path = self.output_dir / filename
        
        # Create summary statistics
        summary = {
            'total_samples': len(processed_data),
            'datasets': {},
            'signal_types': set(),
            'label_types': set(),
            'sampling_rates': self.target_fs,
            'processing_params': self.filter_params
        }
        
        for sample in processed_data:
            dataset = sample['dataset']
            if dataset not in summary['datasets']:
                summary['datasets'][dataset] = {
                    'count': 0,
                    'subjects': set(),
                    'total_duration': 0
                }
            
            summary['datasets'][dataset]['count'] += 1
            summary['datasets'][dataset]['subjects'].add(sample['subject_id'])
            summary['datasets'][dataset]['total_duration'] += sample.get('duration', 0)
            
            # Track available signal and label types
            for signal_name, signal_data in sample['signals'].items():
                if signal_data is not None:
                    summary['signal_types'].add(signal_name)
            
            for label_name, label_data in sample['labels'].items():
                if label_data is not None:
                    summary['label_types'].add(label_name)
        
        # Convert sets to lists for JSON serialization
        for dataset_info in summary['datasets'].values():
            dataset_info['subjects'] = list(dataset_info['subjects'])
        summary['signal_types'] = list(summary['signal_types'])
        summary['label_types'] = list(summary['label_types'])
        
        # Save data
        data_to_save = {
            'processed_data': processed_data,
            'summary': summary
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(data_to_save, f)
        
        print(f"Processed data saved to: {save_path}")
        print(f"Summary: {summary['total_samples']} samples from {len(summary['datasets'])} datasets")
        
        return summary
    
    def process_all_datasets(self, ppg_dalia_path=None, mit_bih_path=None, wesad_path=None):
        """Main processing pipeline"""
        all_processed_data = []
        
        # Process each dataset
        if ppg_dalia_path:
            ppg_data = self.process_ppg_dalia(ppg_dalia_path)
            all_processed_data.extend(ppg_data)
        
        if mit_bih_path:
            mit_data = self.process_mit_bih(mit_bih_path)
            all_processed_data.extend(mit_data)
        
        if wesad_path:
            wesad_data = self.process_wesad(wesad_path)
            all_processed_data.extend(wesad_data)
        
        # Save raw processed data
        summary = self.save_processed_data(all_processed_data, 'combined_raw_data.pkl')
        
        # Create windowed samples for training
        windowed_data = self.create_windowed_samples(all_processed_data)
        windowed_summary = self.save_processed_data(windowed_data, 'combined_windowed_data.pkl')
        
        return all_processed_data, windowed_data, summary, windowed_summary

# Usage example
if __name__ == "__main__":
    processor = MultimodalDatasetProcessor()
    
    # Update these paths to your dataset locations
    dataset_paths = {
        'ppg_dalia_path': '/path/to/PPG_DaLiA',
        'mit_bih_path': '/path/to/mit-bih-arrhythmia-database',
        'wesad_path': '/path/to/WESAD'
    }
    
    # Process all datasets
    raw_data, windowed_data, raw_summary, windowed_summary = processor.process_all_datasets(
        **dataset_paths
    )
    
    print("\nProcessing completed!")
    print(f"Raw data: {len(raw_data)} samples")
    print(f"Windowed data: {len(windowed_data)} samples")
    print(f"Available signals: {raw_summary['signal_types']}")
    print(f"Available labels: {raw_summary['label_types']}")
