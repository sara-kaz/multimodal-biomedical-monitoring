#!/usr/bin/env python3
"""
Comprehensive Training Script for Multimodal Biomedical Monitoring
Trains CNN/Transformer models on combined PPG-DaLiA, MIT-BIH, and WESAD datasets
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import pickle
from pathlib import Path
import time
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Import the CNN/Transformer-Lite model
from src.models.cnn_transformer_lite import CNNTransformerLite


class MultimodalBiomedicalDataset(Dataset):
    """Dataset for multimodal biomedical signals with proper label handling"""
    
    def __init__(self, data, task_configs):
        self.data = data
        self.task_configs = task_configs
        
        # Filter valid samples
        self.valid_samples = []
        for sample in data:
            if 'window_data' in sample and sample['window_data'] is not None:
                window_data = sample['window_data']
                if hasattr(window_data, 'shape') and len(window_data.shape) == 2:
                    self.valid_samples.append(sample)
        
        print(f"Valid samples: {len(self.valid_samples)}")
        
        # Analyze label distribution for each task
        for task_name in task_configs.keys():
            labels = []
            for sample in self.valid_samples:
                if task_name in sample.get('labels', {}):
                    label = sample['labels'][task_name]
                    if isinstance(label, (list, np.ndarray)) and len(label) > 0:
                        class_idx = np.argmax(label)
                        labels.append(class_idx)
            
            if labels:
                unique_labels = set(labels)
                print(f"Task {task_name}: {len(unique_labels)} unique classes")
                print(f"  Distribution: {np.bincount(labels)}")
    
    def __len__(self):
        return len(self.valid_samples)
    
    def __getitem__(self, idx):
        sample = self.valid_samples[idx]
        
        # Get window data
        window_data = sample['window_data']  # [11, 1000]
        
        # Convert to tensor and ensure proper shape
        signals = torch.FloatTensor(window_data)
        if signals.shape != (11, 1000):
            signals = signals.view(11, 1000)
        
        # Get labels - convert one-hot to class indices
        targets = {}
        for task_name in self.task_configs.keys():
            if task_name in sample.get('labels', {}):
                label = sample['labels'][task_name]
                if isinstance(label, (list, np.ndarray)) and len(label) > 0:
                    class_idx = np.argmax(label)
                    targets[task_name] = torch.LongTensor([class_idx])
                else:
                    targets[task_name] = torch.LongTensor([0])
            else:
                targets[task_name] = torch.LongTensor([0])
        
        return {
            'signals': signals,
            **targets
        }


def train_multimodal_model(data_path, epochs=10, batch_size=32, output_dir="training_results"):
    """Comprehensive training function for multimodal biomedical monitoring"""
    
    print("Starting Multimodal Biomedical Training")
    print(f"- Data path: {data_path}")
    print(f"- Epochs: {epochs}")
    print(f"- Batch size: {batch_size}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\nLoading cleaned data...")
    cleaned_data_path = "processed_unified_dataset/cleaned_unified_dataset.pkl"
    with open(cleaned_data_path, 'rb') as f:
        processed_data = pickle.load(f)
    print(f"✅ Loaded {len(processed_data)} samples")
    
    # Create dataset
    print("\nCreating dataset...")
    task_configs = {
        'activity': {'num_classes': 8, 'weight': 1.0},
        'stress': {'num_classes': 4, 'weight': 1.0},
        'arrhythmia': {'num_classes': 2, 'weight': 1.0}
    }
    
    dataset = MultimodalBiomedicalDataset(processed_data, task_configs)
    
    # Create train/val/test split (60/20/20)
    train_size = int(0.6 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\n✅ Train samples: {len(train_dataset)}")
    print(f"✅ Val samples: {len(val_dataset)}")
    print(f"✅ Test samples: {len(test_dataset)}")
    
    # Create model
    print("\nCreating CNN/Transformer-Lite model...")
    model = CNNTransformerLite(
        n_channels=11,
        n_samples=1000,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        task_configs=task_configs
    )
    
    # Use CPU for stability (MPS has bus error issues on some M2 Macs)
    device = 'cpu'
    
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"- Model created with {total_params:,} parameters")
    print(f"- Using device: {device}")
    
    # Create loss function and optimizer with lower learning rate
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    print(f"\n\x1b[32mStarting training for {epochs} epochs...\x1b[0m")
    
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'epoch_times': []
    }
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        print(f"\n\x1b[32m--- Epoch {epoch+1}/{epochs} ---\x1b[0m")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                # Move batch to device
                signals = batch['signals'].to(device)
                targets = {task: batch[task].to(device) for task in task_configs.keys()}
                
                # Forward pass
                optimizer.zero_grad()
                predictions = model(signals)
                
                # Calculate loss for each task with NaN checks
                total_loss = 0.0
                valid_tasks = 0
                for task_name in task_configs.keys():
                    if task_name in predictions and task_name in targets:
                        pred = predictions[task_name]
                        target = targets[task_name].squeeze()
                        
                        # Check for NaN in predictions
                        if torch.isnan(pred).any() or torch.isinf(pred).any():
                            print(f"⚠️  NaN/Inf in {task_name} predictions, skipping batch")
                            continue
                            
                        loss = criterion(pred, target)
                        if not torch.isnan(loss) and not torch.isinf(loss) and loss.item() > 0:
                            total_loss += loss
                            valid_tasks += 1
                
                # Skip if no valid losses
                if valid_tasks == 0:
                    continue
                
                # Backward pass with better gradient clipping
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                
                # Calculate accuracy
                train_loss += total_loss.item()
                train_batches += 1
                
                # Count correct predictions for each task
                for task_name in task_configs.keys():
                    if task_name in predictions and task_name in targets:
                        pred_classes = torch.argmax(predictions[task_name], dim=1)
                        correct = (pred_classes == targets[task_name].squeeze()).sum().item()
                        train_correct += correct
                        train_total += targets[task_name].size(0)
                
                # Progress update every 200 batches
                if batch_idx % 200 == 0:
                    print(f"  Batch {batch_idx}/{len(train_loader)} - Loss: {total_loss.item():.4f}")
                    
            except Exception as e:
                print(f"⚠️  Error in batch {batch_idx}: {e}")
                if device == 'mps':
                    print("🔄 Switching to CPU due to MPS error...")
                    device = 'cpu'
                    model.to(device)
                continue
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                try:
                    signals = batch['signals'].to(device)
                    targets = {task: batch[task].to(device) for task in task_configs.keys()}
                    
                    predictions = model(signals)
                    
                    # Calculate loss for each task
                    batch_loss = 0.0
                    valid_tasks = 0
                    for task_name in task_configs.keys():
                        if task_name in predictions and task_name in targets:
                            loss = criterion(predictions[task_name], targets[task_name].squeeze())
                            if not torch.isnan(loss) and not torch.isinf(loss):
                                batch_loss += loss
                                valid_tasks += 1
                    
                    if valid_tasks > 0:
                        val_loss += batch_loss.item()
                        
                        # Count correct predictions
                        for task_name in task_configs.keys():
                            if task_name in predictions and task_name in targets:
                                pred_classes = torch.argmax(predictions[task_name], dim=1)
                                correct = (pred_classes == targets[task_name].squeeze()).sum().item()
                                val_correct += correct
                                val_total += targets[task_name].size(0)
                except Exception as e:
                    print(f"⚠️  Error in validation batch: {e}")
                    continue
        
        
        # Calculate metrics
        avg_train_loss = train_loss / max(train_batches, 1)
        avg_val_loss = val_loss / max(len(val_loader), 1)
        train_accuracy = train_correct / max(train_total, 1)
        val_accuracy = val_correct / max(val_total, 1)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Store history
        training_history['train_loss'].append(avg_train_loss)
        training_history['val_loss'].append(avg_val_loss)
        training_history['train_accuracy'].append(train_accuracy)
        training_history['val_accuracy'].append(val_accuracy)
        training_history['epoch_times'].append(epoch_time)
        
        # Print results
        print(f"- Epoch time: {epoch_time:.1f}s")
        print(f"- \033[1mTrain     \033[0m: Loss = {avg_train_loss:.4f} | Accuracy = {train_accuracy:.2%}")
        print(f"- \033[1mValidation\033[0m: Loss = {avg_val_loss:.4f} | Accuracy = {val_accuracy:.2%}")
        print(f"⚡ \033[1mSpeed\033[0m: {len(train_loader)/epoch_time:.1f} batches/sec")
        
        # Update learning rate
        scheduler.step()
    
    # Test evaluation
    print(f"\nEvaluating on Test Set...")
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    test_batches = 0
    
    with torch.no_grad():
        for batch in test_loader:
            signals = batch['signals'].to(device)
            targets = {task: batch[task].to(device) for task in task_configs.keys()}
            
            predictions = model(signals)
            
            # Calculate loss for each task
            batch_loss = 0.0
            valid_tasks = 0
            for task_name in task_configs.keys():
                if task_name in predictions and task_name in targets:
                    loss = criterion(predictions[task_name], targets[task_name].squeeze())
                    if not torch.isnan(loss) and not torch.isinf(loss):
                        batch_loss += loss
                        valid_tasks += 1
            
            if valid_tasks > 0:
                test_loss += batch_loss.item()
                test_batches += 1
                
                # Count correct predictions
                for task_name in task_configs.keys():
                    if task_name in predictions and task_name in targets:
                        pred_classes = torch.argmax(predictions[task_name], dim=1)
                        correct = (pred_classes == targets[task_name].squeeze()).sum().item()
                        test_correct += correct
                        test_total += targets[task_name].size(0)
    
    # Calculate final metrics
    final_train_acc = training_history['train_accuracy'][-1]
    final_val_acc = training_history['val_accuracy'][-1]
    final_test_acc = test_correct / max(test_total, 1)
    
    print(f"\n✅ Training completed!")
    print(f"- Final Train Accuracy: {final_train_acc:.2%}")
    print(f"- Final Val Accuracy: {final_val_acc:.2%}")
    print(f"\033[92m- Final Test Accuracy: {final_test_acc:.2%}\033[0m")
    
    # Save results
    results = {
        'training_history': training_history,
        'final_metrics': {
            'train_accuracy': final_train_acc,
            'val_accuracy': final_val_acc,
            'test_accuracy': final_test_acc,
            'total_parameters': total_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)
        }
    }
    
    # Save to file
    with open(output_dir / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save model
    torch.save(model.state_dict(), output_dir / 'model.pth')
    
    print(f"Results saved to: {output_dir}") 
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Multimodal biomedical training')
    parser.add_argument('--data_path', type=str, 
                       default='processed_unified_dataset/unified_dataset.pkl',
                       help='Path to processed dataset')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--output_dir', type=str, default='training_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    results = train_multimodal_model(
        data_path=args.data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )
