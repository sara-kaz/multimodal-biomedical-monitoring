"""
#1
Multimodal Biomedical Dataset Integration Pipeline
Following Exact Specifications for Edge Intelligence Thesis Project

Unified Format:
- Sampling Rate: 100 Hz (all signals)
- Window Size: 10 seconds = 1000 samples
- Shape: [channels × 1000 samples]
- Labels: One-hot encoded vectors
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

class UnifiedBiomedicalDataProcessor:
    def __init__(self, output_dir="processed_unified_dataset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # UNIFIED SPECIFICATIONS
        self.TARGET_FS = 100  # Hz - unified sampling rate
        self.WINDOW_LENGTH = 10  # seconds
        self.WINDOW_SAMPLES = 1000  # samples (10s × 100Hz)
        self.OVERLAP = 0.5  # 50% overlap
        
        # Unified channel mapping
        self.CHANNEL_MAPPING = {
            'ecg': 0,
            'ppg': 1, 
            'accel_x': 2,
            'accel_y': 3,
            'accel_z': 4,
            'eda': 5,
            'respiration': 6,
            'temperature': 7,
            'emg': 8,
            'eda_wrist': 9,
            'temperature_wrist': 10
        }
        self.N_CHANNELS = 11  # 5 original + 6 additional WESAD signals
        
        # Label encodings (one-hot)
        self.LABEL_ENCODINGS = {
            'activity': {
                'sitting': 0, 'walking': 1, 'cycling': 2, 'driving': 3, 
                'working': 4, 'stairs': 5, 'table_soccer': 6, 'lunch': 7
            },
            'stress': {
                'baseline': 0, 'stress': 1, 'amusement': 2, 'meditation': 3
            },
            'arrhythmia': {
                'normal': 0, 'abnormal': 1  # Binary classification
            }
        }
        
    def create_empty_window(self):
        """Create empty window with NaN for missing channels"""
        return np.full((self.N_CHANNELS, self.WINDOW_SAMPLES), np.nan)
    
    def resample_signal(self, signal_data, original_fs):
        """Resample signal to unified 100 Hz"""
        if original_fs == self.TARGET_FS:
            return signal_data
        
        # Calculate new length
        new_length = int(len(signal_data) * self.TARGET_FS / original_fs)
        
        # Use scipy.signal.resample for high-quality resampling
        resampled = signal.resample(signal_data, new_length)
        return resampled
    
    def normalize_signal(self, signal_data):
        """Z-score normalization: (x - mean) / std"""
        mean_val = np.mean(signal_data)
        std_val = np.std(signal_data)
        
        if std_val > 1e-8:  # Avoid division by zero
            normalized = (signal_data - mean_val) / std_val
        else:
            normalized = signal_data - mean_val
            
        return normalized
    
    def create_windows(self, signals_dict, labels_dict=None):
        """
        Create 10-second windows with 50% overlap
        
        Args:
            signals_dict: {'ecg': array, 'ppg': array, 'accel_x': array, ...}
            labels_dict: {'activity': value, 'stress': value, 'arrhythmia': value}
        
        Returns:
            List of windowed samples
        """
        # Find the longest signal to determine number of windows
        max_length = 0
        for signal_name, signal_data in signals_dict.items():
            if signal_data is not None:
                max_length = max(max_length, len(signal_data))
        
        if max_length < self.WINDOW_SAMPLES:
            return []  # Signal too short
        
        # Calculate window parameters
        step_size = int(self.WINDOW_SAMPLES * (1 - self.OVERLAP))  # 500 samples
        n_windows = (max_length - self.WINDOW_SAMPLES) // step_size + 1
        
        windowed_samples = []
        
        for window_idx in range(n_windows):
            start_idx = window_idx * step_size
            end_idx = start_idx + self.WINDOW_SAMPLES
            
            # Create empty window
            window_data = self.create_empty_window()
            
            # Fill available channels
            for signal_name, signal_data in signals_dict.items():
                if signal_data is not None and signal_name in self.CHANNEL_MAPPING:
                    channel_idx = self.CHANNEL_MAPPING[signal_name]
                    if end_idx <= len(signal_data):
                        window_data[channel_idx, :] = signal_data[start_idx:end_idx]
                    else:
                        # Handle edge case where signal is shorter
                        available_samples = len(signal_data) - start_idx
                        if available_samples > 0:
                            window_data[channel_idx, :available_samples] = signal_data[start_idx:]
            
            # Create one-hot labels
            one_hot_labels = self.create_one_hot_labels(labels_dict)
            
            sample = {
                'window_data': window_data,  # Shape: [5, 1000]
                'labels': one_hot_labels,
                'window_index': window_idx,
                'start_time': start_idx / self.TARGET_FS
            }
            
            windowed_samples.append(sample)
        
        return windowed_samples
    
    def create_one_hot_labels(self, labels_dict):
        """Create one-hot encoded labels"""
        one_hot = {}
        
        if labels_dict is None:
            labels_dict = {}
        
        for label_type, encoding_map in self.LABEL_ENCODINGS.items():
            n_classes = len(encoding_map)
            one_hot_vector = np.zeros(n_classes)
            
            if label_type in labels_dict and labels_dict[label_type] is not None:
                label_value = labels_dict[label_type]
                if label_value in encoding_map:
                    class_idx = encoding_map[label_value]
                    one_hot_vector[class_idx] = 1
            
            one_hot[label_type] = one_hot_vector
        
        return one_hot
    
    def process_ppg_dalia(self, data_path):
        """Process PPG-DaLiA Dataset"""
        print("📊 Processing PPG-DaLiA Dataset...")
        
        all_windows = []
        data_path = Path(data_path)
        
        print(f"Looking for files in: {data_path}")
        pkl_files = list(data_path.glob("S*/S*.pkl"))
        print(f"Found {len(pkl_files)} PKL files")
        
        for pkl_file in pkl_files:
            subject_id = pkl_file.stem
            print(f"  Processing {subject_id}")
            
            with open(pkl_file, 'rb') as f:
                subject_data = pickle.load(f, encoding='latin1')
            
            # Extract signals
            # Note: PPG-DaLiA has chest ECG, wrist PPG/ACC
            signals = {}
            
            # ECG from chest sensor (64 Hz)
            if 'chest' in subject_data['signal']:
                ecg_raw = subject_data['signal']['chest']['ECG'].flatten()
                ecg_resampled = self.resample_signal(ecg_raw, 64)
                signals['ecg'] = self.normalize_signal(ecg_resampled)
                
                # Additional PPG-DaLiA signals from chest (64 Hz)
                if 'EDA' in subject_data['signal']['chest']:
                    eda_raw = subject_data['signal']['chest']['EDA'].flatten()
                    eda_resampled = self.resample_signal(eda_raw, 64)
                    signals['eda'] = self.normalize_signal(eda_resampled)
                
                if 'Resp' in subject_data['signal']['chest']:
                    resp_raw = subject_data['signal']['chest']['Resp'].flatten()
                    resp_resampled = self.resample_signal(resp_raw, 64)
                    signals['respiration'] = self.normalize_signal(resp_resampled)
                
                if 'Temp' in subject_data['signal']['chest']:
                    temp_raw = subject_data['signal']['chest']['Temp'].flatten()
                    temp_resampled = self.resample_signal(temp_raw, 64)
                    signals['temperature'] = self.normalize_signal(temp_resampled)
                
                if 'EMG' in subject_data['signal']['chest']:
                    emg_raw = subject_data['signal']['chest']['EMG'].flatten()
                    emg_resampled = self.resample_signal(emg_raw, 64)
                    signals['emg'] = self.normalize_signal(emg_resampled)
            
            # PPG from wrist (64 Hz) 
            ppg_raw = subject_data['signal']['wrist']['BVP'].flatten()
            ppg_resampled = self.resample_signal(ppg_raw, 64)
            signals['ppg'] = self.normalize_signal(ppg_resampled)
            
            # Accelerometer from wrist (32 Hz, 3-axis)
            acc_raw = subject_data['signal']['wrist']['ACC']  # [N, 3]
            acc_x_resampled = self.resample_signal(acc_raw[:, 0], 32)
            acc_y_resampled = self.resample_signal(acc_raw[:, 1], 32)
            acc_z_resampled = self.resample_signal(acc_raw[:, 2], 32)
            
            signals['accel_x'] = self.normalize_signal(acc_x_resampled)
            signals['accel_y'] = self.normalize_signal(acc_y_resampled)
            signals['accel_z'] = self.normalize_signal(acc_z_resampled)
            
            # Additional PPG-DaLiA signals from wrist (32 Hz)
            if 'EDA' in subject_data['signal']['wrist']:
                eda_wrist_raw = subject_data['signal']['wrist']['EDA'].flatten()
                eda_wrist_resampled = self.resample_signal(eda_wrist_raw, 32)
                signals['eda_wrist'] = self.normalize_signal(eda_wrist_resampled)
            
            if 'TEMP' in subject_data['signal']['wrist']:
                temp_wrist_raw = subject_data['signal']['wrist']['TEMP'].flatten()
                temp_wrist_resampled = self.resample_signal(temp_wrist_raw, 32)
                signals['temperature_wrist'] = self.normalize_signal(temp_wrist_resampled)
            
            # Extract activity labels
            activity_labels = subject_data.get('activity', None)
            
            # Map activity labels to our standard format
            activity_mapping = {
                0: 'sitting', 1: 'walking', 2: 'cycling', 3: 'driving',
                4: 'working', 5: 'stairs', 6: 'table_soccer', 7: 'lunch'
            }
            
            if activity_labels is not None:
                # Get majority activity for each potential window
                step_size = int(self.WINDOW_SAMPLES * (1 - self.OVERLAP))
                signal_length = len(signals['ppg'])
                
                for start_idx in range(0, signal_length - self.WINDOW_SAMPLES + 1, step_size):
                    end_idx = start_idx + self.WINDOW_SAMPLES
                    
                    # Get activity labels for this window
                    window_activity_indices = np.arange(start_idx, end_idx)
                    window_activities = []
                    
                    for idx in window_activity_indices:
                        if idx < len(activity_labels):
                            activity_code = int(activity_labels[idx])  # Convert to int
                            if activity_code in activity_mapping:
                                window_activities.append(activity_mapping[activity_code])
                    
                    # Use majority vote
                    if window_activities:
                        majority_activity = max(set(window_activities), key=window_activities.count)
                    else:
                        majority_activity = 'sitting'  # default
                    
                    labels = {'activity': majority_activity}
                    
                    # Create window for this segment
                    window_signals = {}
                    for sig_name, sig_data in signals.items():
                        if end_idx <= len(sig_data):
                            window_signals[sig_name] = sig_data[start_idx:end_idx]
                    
                    if len(window_signals) > 0:
                        windows = self.create_windows(window_signals, labels)
                        for window in windows:
                            window['subject_id'] = subject_id
                            window['dataset'] = 'PPG-DaLiA'
                        all_windows.extend(windows)
            else:
                # No activity labels available, create windows without activity labels
                labels = {'activity': None}
                windows = self.create_windows(signals, labels)
                for window in windows:
                    window['subject_id'] = subject_id
                    window['dataset'] = 'PPG-DaLiA'
                all_windows.extend(windows)
        
        print(f"  ✅ PPG-DaLiA: {len(all_windows)} windows created")
        return all_windows
    
    def process_mit_bih(self, data_path):
        """Process MIT-BIH Arrhythmia Dataset"""
        print("Processing MIT-BIH Arrhythmia Dataset...")
        
        all_windows = []
        data_path = Path(data_path)
        
        for dat_file in data_path.glob("*.dat"):
            record_name = dat_file.stem
            if record_name.startswith('.'):
                continue
            
            print(f"  Processing {record_name}")
            
            try:
                import wfdb
                
                # Read record and annotations
                record = wfdb.rdrecord(str(dat_file.parent / record_name))
                annotation = wfdb.rdann(str(dat_file.parent / record_name), 'atr')
                
                # Extract ECG (Lead II preferred, or first available lead)
                ecg_raw = record.p_signal[:, 0]  # First lead
                
                # Resample from 360 Hz to 100 Hz
                ecg_resampled = self.resample_signal(ecg_raw, record.fs)
                ecg_normalized = self.normalize_signal(ecg_resampled)
                
                # Create binary arrhythmia labels (normal vs abnormal)
                arrhythmia_labels = self.create_mit_bih_labels(
                    annotation, len(ecg_normalized), record.fs
                )
                
                # Create windows
                step_size = int(self.WINDOW_SAMPLES * (1 - self.OVERLAP))
                
                for start_idx in range(0, len(ecg_normalized) - self.WINDOW_SAMPLES + 1, step_size):
                    end_idx = start_idx + self.WINDOW_SAMPLES
                    
                    # Get majority arrhythmia label for this window
                    window_labels = arrhythmia_labels[start_idx:end_idx]
                    majority_label = 'abnormal' if np.mean(window_labels) > 0.5 else 'normal'
                    
                    # Create signals dict (only ECG available)
                    signals = {
                        'ecg': ecg_normalized[start_idx:end_idx],
                        'ppg': None,
                        'accel_x': None,
                        'accel_y': None,
                        'accel_z': None
                    }
                    
                    labels = {'arrhythmia': majority_label}
                    
                    windows = self.create_windows(signals, labels)
                    for window in windows:
                        window['subject_id'] = record_name
                        window['dataset'] = 'MIT-BIH'
                    all_windows.extend(windows)
                    
            except ImportError:
                print("  ⚠️  wfdb library not found. Install with: pip install wfdb")
                continue
            except Exception as e:
                print(f"  ❌ Error processing {record_name}: {e}")
                continue
        
        print(f"  ✅ MIT-BIH: {len(all_windows)} windows created")
        return all_windows
    
    def create_mit_bih_labels(self, annotation, signal_length, original_fs):
        """Create binary arrhythmia labels (normal vs abnormal)"""
        
        # AAMI arrhythmia classification
        normal_beats = ['N', 'L', 'R', 'e', 'j']  # Normal beats
        abnormal_beats = ['A', 'a', 'J', 'S', 'V', 'E', 'F', '/', 'f', 'Q']  # Abnormal
        
        # Create sample-level labels (resampled to 100 Hz)
        labels = np.zeros(signal_length)
        
        for sample, symbol in zip(annotation.sample, annotation.symbol):
            # Convert sample index to 100 Hz
            new_sample = int(sample * self.TARGET_FS / original_fs)
            if new_sample < signal_length:
                if symbol in abnormal_beats:
                    # Mark as abnormal (with context window)
                    start_ctx = max(0, new_sample - 50)  # 0.5s before
                    end_ctx = min(signal_length, new_sample + 50)  # 0.5s after
                    labels[start_ctx:end_ctx] = 1
        
        return labels
    
    def process_wesad(self, data_path):
        """Process WESAD Dataset"""
        print("Processing WESAD Dataset...")
        
        all_windows = []
        data_path = Path(data_path)
        
        print(f"Looking for files in: {data_path}")
        pkl_files = list(data_path.glob("S*/S*.pkl"))
        print(f"Found {len(pkl_files)} PKL files")
        
        for pkl_file in pkl_files:
            subject_id = pkl_file.stem
            print(f"  Processing {subject_id}")
            
            with open(pkl_file, 'rb') as f:
                subject_data = pickle.load(f, encoding='latin1')
            
            # Extract signals
            chest_data = subject_data['signal']['chest']
            wrist_data = subject_data['signal']['wrist']
            
            signals = {}
            
            # ECG from chest (700 Hz)
            ecg_raw = chest_data['ECG'].flatten()
            ecg_resampled = self.resample_signal(ecg_raw, 700)
            signals['ecg'] = self.normalize_signal(ecg_resampled)
            
            # PPG from wrist (64 Hz)
            ppg_raw = wrist_data['BVP'].flatten() 
            ppg_resampled = self.resample_signal(ppg_raw, 64)
            signals['ppg'] = self.normalize_signal(ppg_resampled)
            
            # Accelerometer from wrist (32 Hz, 3-axis)
            acc_raw = wrist_data['ACC']  # [N, 3]
            acc_x_resampled = self.resample_signal(acc_raw[:, 0], 32)
            acc_y_resampled = self.resample_signal(acc_raw[:, 1], 32) 
            acc_z_resampled = self.resample_signal(acc_raw[:, 2], 32)
            
            signals['accel_x'] = self.normalize_signal(acc_x_resampled)
            signals['accel_y'] = self.normalize_signal(acc_y_resampled)
            signals['accel_z'] = self.normalize_signal(acc_z_resampled)
            
            # Additional WESAD signals from chest (700 Hz)
            if 'EDA' in chest_data:
                eda_raw = chest_data['EDA'].flatten()
                eda_resampled = self.resample_signal(eda_raw, 700)
                signals['eda'] = self.normalize_signal(eda_resampled)
            
            if 'Resp' in chest_data:
                resp_raw = chest_data['Resp'].flatten()
                resp_resampled = self.resample_signal(resp_raw, 700)
                signals['respiration'] = self.normalize_signal(resp_resampled)
            
            if 'Temp' in chest_data:
                temp_raw = chest_data['Temp'].flatten()
                temp_resampled = self.resample_signal(temp_raw, 700)
                signals['temperature'] = self.normalize_signal(temp_resampled)
            
            if 'EMG' in chest_data:
                emg_raw = chest_data['EMG'].flatten()
                emg_resampled = self.resample_signal(emg_raw, 700)
                signals['emg'] = self.normalize_signal(emg_resampled)
            
            # Additional WESAD signals from wrist (32 Hz)
            if 'EDA' in wrist_data:
                eda_wrist_raw = wrist_data['EDA'].flatten()
                eda_wrist_resampled = self.resample_signal(eda_wrist_raw, 32)
                signals['eda_wrist'] = self.normalize_signal(eda_wrist_resampled)
            
            if 'TEMP' in wrist_data:
                temp_wrist_raw = wrist_data['TEMP'].flatten()
                temp_wrist_resampled = self.resample_signal(temp_wrist_raw, 32)
                signals['temperature_wrist'] = self.normalize_signal(temp_wrist_resampled)
            
            # Extract stress labels (700 Hz)
            stress_labels_raw = subject_data['label'].flatten()
            
            # Map WESAD stress labels: 0=baseline, 1=stress, 2=amusement, 3=meditation
            stress_mapping = {0: 'baseline', 1: 'stress', 2: 'amusement', 3: 'meditation'}
            
            # Resample stress labels to 100 Hz
            stress_labels_resampled = self.resample_signal(stress_labels_raw.astype(float), 700)
            stress_labels_resampled = np.round(stress_labels_resampled).astype(int)
            
            # Create windows
            step_size = int(self.WINDOW_SAMPLES * (1 - self.OVERLAP))
            signal_length = len(signals['ecg'])
            
            for start_idx in range(0, signal_length - self.WINDOW_SAMPLES + 1, step_size):
                end_idx = start_idx + self.WINDOW_SAMPLES
                
                # Get majority stress label for this window
                if start_idx < len(stress_labels_resampled):
                    window_stress_labels = stress_labels_resampled[start_idx:end_idx]
                    majority_stress_code = int(np.median(window_stress_labels))  # Use median
                    
                    if majority_stress_code in stress_mapping:
                        majority_stress = stress_mapping[majority_stress_code]
                    else:
                        majority_stress = 'baseline'  # default
                else:
                    majority_stress = 'baseline'
                
                # Create signals dict for this window
                window_signals = {}
                for sig_name, sig_data in signals.items():
                    if end_idx <= len(sig_data):
                        window_signals[sig_name] = sig_data[start_idx:end_idx]
                
                labels = {'stress': majority_stress}
                
                if len(window_signals) > 0:
                    windows = self.create_windows(window_signals, labels)
                    for window in windows:
                        window['subject_id'] = subject_id
                        window['dataset'] = 'WESAD'
                    all_windows.extend(windows)
        
        print(f"  ✅ WESAD: {len(all_windows)} windows created")
        return all_windows
    
    def combine_all_datasets(self, ppg_dalia_path=None, mit_bih_path=None, wesad_path=None):
        """
        Main processing pipeline - combines all datasets into unified format
        
        Returns:
            unified_dataset: List of samples with format:
                {
                    'window_data': np.array([5, 1000]),  # [channels × samples]
                    'labels': {
                        'activity': one_hot_vector,
                        'stress': one_hot_vector, 
                        'arrhythmia': one_hot_vector
                    },
                    'subject_id': str,
                    'dataset': str,
                    'window_index': int,
                    'start_time': float
                }
        """
        
        print("Starting Unified Dataset Creation...")
        print(f"- Target Format: {self.N_CHANNELS} channels × {self.WINDOW_SAMPLES} samples")
        print(f"- Sampling Rate: {self.TARGET_FS} Hz")
        print(f"- Window Length: {self.WINDOW_LENGTH} seconds")
        print(f"- Overlap: {self.OVERLAP * 100}%")
        
        all_windows = []
        
        # Process each dataset
        if ppg_dalia_path:
            ppg_windows = self.process_ppg_dalia(ppg_dalia_path)
            all_windows.extend(ppg_windows)
        
        if mit_bih_path:
            mit_windows = self.process_mit_bih(mit_bih_path)
            all_windows.extend(mit_windows)
            
        if wesad_path:
            wesad_windows = self.process_wesad(wesad_path)
            all_windows.extend(wesad_windows)
        
        print(f"\n✅ Total Windows Created: {len(all_windows)}")
        
        # Save unified dataset
        self.save_unified_dataset(all_windows)
        
        # Create summary statistics
        summary = self.create_summary_statistics(all_windows)
        
        return all_windows, summary
    
    def save_unified_dataset(self, unified_dataset):
        """Save unified dataset"""
        
        # Save as pickle
        save_path = self.output_dir / 'unified_dataset.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(unified_dataset, f)
        
        print(f"✅ Unified dataset saved: {save_path}")
        
        # Save metadata
        metadata = {
            'format': {
                'channels': self.N_CHANNELS,
                'samples_per_window': self.WINDOW_SAMPLES,
                'sampling_rate_hz': self.TARGET_FS,
                'window_length_sec': self.WINDOW_LENGTH,
                'overlap': self.OVERLAP
            },
            'channel_mapping': self.CHANNEL_MAPPING,
            'label_encodings': self.LABEL_ENCODINGS,
            'total_windows': len(unified_dataset)
        }
        
        metadata_path = self.output_dir / 'dataset_metadata.pkl'
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
            
        print(f"📋 Metadata saved: {metadata_path}")
    
    def create_summary_statistics(self, unified_dataset):
        """Create summary statistics"""
        
        summary = {
            'total_windows': len(unified_dataset),
            'datasets': {},
            'subjects': set(),
            'label_distributions': {}
        }
        
        # Analyze by dataset
        for window in unified_dataset:
            dataset = window['dataset']
            subject = window['subject_id']
            
            if dataset not in summary['datasets']:
                summary['datasets'][dataset] = {'count': 0, 'subjects': set()}
            
            summary['datasets'][dataset]['count'] += 1
            summary['datasets'][dataset]['subjects'].add(subject)
            summary['subjects'].add(f"{dataset}_{subject}")
        
        # Convert sets to lists
        for dataset_info in summary['datasets'].values():
            dataset_info['subjects'] = list(dataset_info['subjects'])
        summary['subjects'] = list(summary['subjects'])
        
        # Analyze label distributions
        for label_type in self.LABEL_ENCODINGS.keys():
            summary['label_distributions'][label_type] = {}
            
            for window in unified_dataset:
                one_hot = window['labels'][label_type]
                if np.sum(one_hot) > 0:  # Has valid label
                    class_idx = np.argmax(one_hot)
                    # Find class name
                    for class_name, idx in self.LABEL_ENCODINGS[label_type].items():
                        if idx == class_idx:
                            if class_name not in summary['label_distributions'][label_type]:
                                summary['label_distributions'][label_type][class_name] = 0
                            summary['label_distributions'][label_type][class_name] += 1
                            break
        
        # Print summary
        print(f"\n--- DATASET SUMMARY ---")
        print(f"Total Windows: {summary['total_windows']}")
        print(f"Total Subjects: {len(summary['subjects'])}")
        
        for dataset, info in summary['datasets'].items():
            print(f"{dataset}: {info['count']} windows, {len(info['subjects'])} subjects")
        
        for label_type, distribution in summary['label_distributions'].items():
            if distribution:
                print(f"{label_type.title()} Labels: {distribution}")
        
        return summary

# Usage Example
if __name__ == "__main__":
    
    # Initialize processor
    processor = UnifiedBiomedicalDataProcessor()
    
    # Process all datasets - UPDATE THESE PATHS
    dataset_paths = {
        'ppg_dalia_path': '/Users/HP/Desktop/University/Thesis/Code/data/ppg+dalia/PPG_FieldStudy',
        'mit_bih_path': '/Users/HP/Desktop/University/Thesis/Code/data/mit-bih-arrhythmia-database-1.0.0', 
        'wesad_path': '/Users/HP/Desktop/University/Thesis/Code/data/WESAD'
    }
    
    # Create unified dataset
    unified_dataset, summary = processor.combine_all_datasets(**dataset_paths)
    
    print("\n🎉 UNIFIED DATASET CREATED!")
    print(f"Format: [{processor.N_CHANNELS} channels × {processor.WINDOW_SAMPLES} samples]")
    print(f"Sampling Rate: {processor.TARGET_FS} Hz")
    print(f"Total Windows: {len(unified_dataset)}")
    
    # Example: Access first window
    if unified_dataset:
        first_window = unified_dataset[0]
        print(f"\nExample Window:")
        print(f"Shape: {first_window['window_data'].shape}")
        print(f"Dataset: {first_window['dataset']}")
        print(f"Subject: {first_window['subject_id']}")
        print(f"Activity Label: {first_window['labels']['activity']}")
        print(f"Stress Label: {first_window['labels']['stress']}")  
        print(f"Arrhythmia Label: {first_window['labels']['arrhythmia']}")