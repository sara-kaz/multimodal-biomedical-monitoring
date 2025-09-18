"""
Data Utilities for Multi-task Training
Includes data loading, preprocessing, and splitting utilities
"""

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pickle
from pathlib import Path


class MultimodalBiomedicalDataset(Dataset):
    """
    PyTorch Dataset for multimodal biomedical data with multi-task support
    """
    
    def __init__(self, 
                 processed_data: List[Dict],
                 signal_types: List[str] = None,
                 task_configs: Dict[str, Dict] = None,
                 normalize_per_sample: bool = True,
                 augment: bool = False):
        """
        Initialize dataset
        
        Args:
            processed_data: List of processed data samples
            signal_types: List of signal types to include
            task_configs: Task configuration dictionary
            normalize_per_sample: Whether to normalize each sample individually
            augment: Whether to apply data augmentation
        """
        self.data = processed_data
        self.normalize_per_sample = normalize_per_sample
        self.augment = augment
        
        # Default signal types
        if signal_types is None:
            signal_types = ['ecg', 'ppg', 'accel_x', 'accel_y', 'accel_z', 
                          'eda', 'respiration', 'temperature', 'emg', 
                          'eda_wrist', 'temperature_wrist']
        self.signal_types = signal_types
        
        # Default task configurations
        if task_configs is None:
            task_configs = {
                'activity': {'num_classes': 8, 'weight': 1.0},
                'stress': {'num_classes': 4, 'weight': 1.0},
                'arrhythmia': {'num_classes': 2, 'weight': 1.0}
            }
        self.task_configs = task_configs
        
        # Filter valid samples
        self.valid_samples = self._filter_valid_samples()
        
        print(f"Valid samples: {len(self.valid_samples)}/{len(processed_data)}")
        
        # Create label encoders for each task
        self.label_encoders = self._create_label_encoders()
    
    def _filter_valid_samples(self) -> List[Dict]:
        """Filter samples that have required signals and labels"""
        valid_samples = []
        
        for sample in self.data:
            # Check if sample has required signals
            has_signals = True
            for signal_type in self.signal_types:
                if signal_type not in sample.get('signals', {}):
                    has_signals = False
                    break
                if sample['signals'][signal_type] is None:
                    has_signals = False
                    break
            
            # Check if sample has at least one valid label
            has_labels = False
            for task_name in self.task_configs.keys():
                if (task_name in sample.get('labels', {}) and 
                    sample['labels'][task_name] is not None):
                    has_labels = True
                    break
            
            if has_signals and has_labels:
                valid_samples.append(sample)
        
        return valid_samples
    
    def _create_label_encoders(self) -> Dict[str, Dict]:
        """Create label encoders for each task"""
        encoders = {}
        
        for task_name in self.task_configs.keys():
            # Collect all unique labels for this task
            labels = []
            for sample in self.valid_samples:
                if (task_name in sample.get('labels', {}) and 
                    sample['labels'][task_name] is not None):
                    labels.append(sample['labels'][task_name])
            
            if labels:
                unique_labels = sorted(list(set(labels)))
                label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
                idx_to_label = {idx: label for label, idx in label_to_idx.items()}
                
                encoders[task_name] = {
                    'label_to_idx': label_to_idx,
                    'idx_to_label': idx_to_label,
                    'num_classes': len(unique_labels)
                }
            else:
                encoders[task_name] = {
                    'label_to_idx': {},
                    'idx_to_label': {},
                    'num_classes': 0
                }
        
        return encoders
    
    def __len__(self) -> int:
        return len(self.valid_samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.valid_samples[idx]
        
        # Extract signals
        signals = []
        for signal_type in self.signal_types:
            signal_data = sample['signals'][signal_type]
            
            if self.normalize_per_sample:
                # Additional per-sample normalization
                signal_data = (signal_data - np.mean(signal_data)) / (np.std(signal_data) + 1e-8)
            
            signals.append(signal_data)
        
        # Stack signals into multi-channel tensor
        signals = np.stack(signals, axis=0)  # Shape: [channels, sequence_length]
        
        # Apply data augmentation if enabled
        if self.augment:
            signals = self._apply_augmentation(signals)
        
        # Convert to tensor
        signals = torch.FloatTensor(signals)
        
        # Create targets for each task
        targets = {}
        for task_name in self.task_configs.keys():
            if (task_name in sample.get('labels', {}) and 
                sample['labels'][task_name] is not None):
                label = sample['labels'][task_name]
                if task_name in self.label_encoders and label in self.label_encoders[task_name]['label_to_idx']:
                    target_idx = self.label_encoders[task_name]['label_to_idx'][label]
                    targets[task_name] = torch.LongTensor([target_idx])
                else:
                    # Use -1 for missing labels
                    targets[task_name] = torch.LongTensor([-1])
            else:
                targets[task_name] = torch.LongTensor([-1])
        
        return {
            'signals': signals,
            **targets,
            'subject_id': sample.get('subject_id', 'unknown'),
            'dataset': sample.get('dataset', 'unknown'),
            'window_index': sample.get('window_index', 0),
            'start_time': sample.get('start_time', 0.0)
        }
    
    def _apply_augmentation(self, signals: np.ndarray) -> np.ndarray:
        """Apply data augmentation to signals"""
        # Add noise
        noise_std = 0.01
        noise = np.random.normal(0, noise_std, signals.shape)
        signals = signals + noise
        
        # Time shift (circular shift)
        if np.random.random() > 0.5:
            shift = np.random.randint(-50, 50)  # Shift up to 0.5 seconds
            signals = np.roll(signals, shift, axis=1)
        
        # Scale variation
        if np.random.random() > 0.5:
            scale_factor = np.random.uniform(0.9, 1.1)
            signals = signals * scale_factor
        
        return signals
    
    def get_class_distribution(self) -> Dict[str, Dict[str, int]]:
        """Get class distribution for each task"""
        distributions = {}
        
        for task_name in self.task_configs.keys():
            class_counts = {}
            for sample in self.valid_samples:
                if (task_name in sample.get('labels', {}) and 
                    sample['labels'][task_name] is not None):
                    label = sample['labels'][task_name]
                    class_counts[label] = class_counts.get(label, 0) + 1
            
            distributions[task_name] = class_counts
        
        return distributions


def create_subject_splits(processed_data: List[Dict], 
                         test_size: float = 0.2, 
                         val_size: float = 0.1,
                         random_state: int = 42) -> Tuple[Dict[str, List[Dict]], Dict[str, List[str]]]:
    """
    Create train/validation/test splits with subject-wise splitting
    
    Args:
        processed_data: List of processed data samples
        test_size: Fraction of subjects for test set
        val_size: Fraction of subjects for validation set
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (sample_splits, subject_splits)
    """
    # Group samples by subject
    subject_groups = {}
    for sample in processed_data:
        subject_id = sample.get('subject_id', 'unknown')
        if subject_id not in subject_groups:
            subject_groups[subject_id] = []
        subject_groups[subject_id].append(sample)
    
    subjects = list(subject_groups.keys())
    
    # Split subjects (not individual samples)
    train_subjects, test_subjects = train_test_split(
        subjects, test_size=test_size, random_state=random_state
    )
    
    train_subjects, val_subjects = train_test_split(
        train_subjects, test_size=val_size/(1-test_size), random_state=random_state
    )
    
    # Create sample splits
    train_samples = []
    val_samples = []
    test_samples = []
    
    for subject_id, samples in subject_groups.items():
        if subject_id in train_subjects:
            train_samples.extend(samples)
        elif subject_id in val_subjects:
            val_samples.extend(samples)
        else:
            test_samples.extend(samples)
    
    print(f"Train: {len(train_samples)} samples from {len(train_subjects)} subjects")
    print(f"Val: {len(val_samples)} samples from {len(val_subjects)} subjects")
    print(f"Test: {len(test_samples)} samples from {len(test_subjects)} subjects")
    
    return {
        'train': train_samples,
        'val': val_samples,
        'test': test_samples
    }, {
        'train_subjects': train_subjects,
        'val_subjects': val_subjects,
        'test_subjects': test_subjects
    }


def create_data_loaders(processed_data: List[Dict],
                       task_configs: Dict[str, Dict],
                       batch_size: int = 32,
                       num_workers: int = 4,
                       augment_train: bool = True) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for training, validation, and test sets
    
    Args:
        processed_data: List of processed data samples
        task_configs: Task configuration dictionary
        batch_size: Batch size for DataLoaders
        num_workers: Number of worker processes
        augment_train: Whether to apply augmentation to training set
    
    Returns:
        Dictionary of DataLoaders
    """
    # Create subject-wise splits
    sample_splits, subject_splits = create_subject_splits(processed_data)
    
    data_loaders = {}
    
    for split_name, samples in sample_splits.items():
        if len(samples) == 0:
            continue
        
        # Create dataset
        augment = augment_train and split_name == 'train'
        dataset = MultimodalBiomedicalDataset(
            samples, 
            task_configs=task_configs,
            augment=augment
        )
        
        if len(dataset) == 0:
            print(f"Warning: No valid samples in {split_name} set")
            continue
        
        # Create DataLoader
        shuffle = split_name == 'train'
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=split_name == 'train'
        )
        
        data_loaders[split_name] = data_loader
        
        print(f"{split_name.capitalize()} DataLoader: {len(dataset)} samples, {len(data_loader)} batches")
    
    return data_loaders


def load_processed_data(data_path: str) -> List[Dict]:
    """
    Load processed data from pickle file
    
    Args:
        data_path: Path to processed data file
    
    Returns:
        List of processed data samples
    """
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # Handle different data formats
    if isinstance(data, dict) and 'processed_data' in data:
        return data['processed_data']
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Unsupported data format in {data_path}")


def create_balanced_sampler(dataset: MultimodalBiomedicalDataset, 
                           task_name: str) -> torch.utils.data.WeightedRandomSampler:
    """
    Create balanced sampler for imbalanced datasets
    
    Args:
        dataset: Dataset to create sampler for
        task_name: Task name to balance
    
    Returns:
        WeightedRandomSampler for balanced sampling
    """
    # Get class distribution
    class_counts = {}
    for sample in dataset.valid_samples:
        if (task_name in sample.get('labels', {}) and 
            sample['labels'][task_name] is not None):
            label = sample['labels'][task_name]
            class_counts[label] = class_counts.get(label, 0) + 1
    
    # Calculate weights
    total_samples = sum(class_counts.values())
    class_weights = {label: total_samples / count for label, count in class_counts.items()}
    
    # Assign weights to samples
    sample_weights = []
    for sample in dataset.valid_samples:
        if (task_name in sample.get('labels', {}) and 
            sample['labels'][task_name] is not None):
            label = sample['labels'][task_name]
            weight = class_weights[label]
        else:
            weight = 1.0  # Default weight for samples without label
        sample_weights.append(weight)
    
    return torch.utils.data.WeightedRandomSampler(
        sample_weights, 
        num_samples=len(sample_weights),
        replacement=True
    )


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for multi-task data
    
    Args:
        batch: List of samples from dataset
    
    Returns:
        Batched data dictionary
    """
    # Stack signals
    signals = torch.stack([sample['signals'] for sample in batch])
    
    # Stack targets for each task
    targets = {}
    for key in batch[0].keys():
        if key not in ['signals', 'subject_id', 'dataset', 'window_index', 'start_time']:
            # This is a task target
            task_targets = []
            for sample in batch:
                if sample[key].item() != -1:  # Valid label
                    task_targets.append(sample[key])
                else:
                    # Use 0 as default for missing labels
                    task_targets.append(torch.tensor(0))
            targets[key] = torch.stack(task_targets)
    
    # Collect metadata
    subject_ids = [sample['subject_id'] for sample in batch]
    datasets = [sample['dataset'] for sample in batch]
    window_indices = [sample['window_index'] for sample in batch]
    start_times = [sample['start_time'] for sample in batch]
    
    return {
        'signals': signals,
        **targets,
        'subject_ids': subject_ids,
        'datasets': datasets,
        'window_indices': window_indices,
        'start_times': start_times
    }


# Example usage and testing
if __name__ == "__main__":
    # Test data loading
    data_path = "processed_unified_dataset/unified_dataset.pkl"
    
    try:
        processed_data = load_processed_data(data_path)
        print(f"Loaded {len(processed_data)} samples")
        
        # Test dataset creation
        task_configs = {
            'activity': {'num_classes': 8, 'weight': 1.0},
            'stress': {'num_classes': 4, 'weight': 1.0},
            'arrhythmia': {'num_classes': 2, 'weight': 1.0}
        }
        
        dataset = MultimodalBiomedicalDataset(processed_data, task_configs=task_configs)
        print(f"Created dataset with {len(dataset)} valid samples")
        
        # Test data loader creation
        data_loaders = create_data_loaders(processed_data, task_configs, batch_size=8)
        
        # Test data loader
        if 'train' in data_loaders:
            train_loader = data_loaders['train']
            for batch in train_loader:
                print(f"Batch signals shape: {batch['signals'].shape}")
                for task_name in task_configs.keys():
                    if task_name in batch:
                        print(f"Batch {task_name} shape: {batch[task_name].shape}")
                break
        
        print("✅ Data utilities tested successfully")
        
    except FileNotFoundError:
        print(f"Data file not found: {data_path}")
        print("Please run dataset processing first")
    except Exception as e:
        print(f"Error testing data utilities: {e}")
