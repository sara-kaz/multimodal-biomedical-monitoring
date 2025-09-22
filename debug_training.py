#!/usr/bin/env python3
"""
Debug training script to identify 0% accuracy issue
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pickle
from pathlib import Path
import sys

# Import the CNN/Transformer-Lite model
from src.models.cnn_transformer_lite import CNNTransformerLite

class DebugDataset(Dataset):
    """Debug dataset to understand data issues"""
    
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
        
        # Debug label analysis
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
                print(f"  Sample labels: {labels[:10]}")
    
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

def debug_training():
    """Debug the training process"""
    
    print("🔍 Debug Training Analysis")
    
    # Load data
    data_path = "processed_unified_dataset/unified_dataset.pkl"
    print(f"📊 Loading data from: {data_path}")
    
    with open(data_path, 'rb') as f:
        processed_data = pickle.load(f)
    print(f"✅ Loaded {len(processed_data)} samples")
    
    # Create dataset
    task_configs = {
        'activity': {'num_classes': 8, 'weight': 1.0},
        'stress': {'num_classes': 4, 'weight': 1.0},
        'arrhythmia': {'num_classes': 2, 'weight': 1.0}
    }
    
    dataset = DebugDataset(processed_data, task_configs)
    
    # Create data loader
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Create model
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
    
    # Set device
    device = 'cpu'  # Use CPU for debugging
    model.to(device)
    
    # Create optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\n🧪 Testing with small batch...")
    
    # Test with one batch
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= 1:  # Only test first batch
            break
            
        print(f"\n📦 Batch {batch_idx}:")
        print(f"  Signals shape: {batch['signals'].shape}")
        
        # Move to device
        signals = batch['signals'].to(device)
        targets = {task: batch[task].to(device) for task in task_configs.keys()}
        
        print(f"  Targets shapes:")
        for task, target in targets.items():
            print(f"    {task}: {target.shape} = {target.squeeze().tolist()}")
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            predictions = model(signals)
        
        print(f"  Predictions shapes:")
        for task, pred in predictions.items():
            print(f"    {task}: {pred.shape}")
            print(f"    {task} logits: {pred[0].detach().cpu().numpy()}")
            print(f"    {task} argmax: {torch.argmax(pred, dim=1).cpu().numpy()}")
        
        # Test loss calculation
        print(f"\n🔍 Loss Analysis:")
        for task_name in task_configs.keys():
            if task_name in predictions and task_name in targets:
                pred = predictions[task_name]
                target = targets[task_name].squeeze()
                
                print(f"  {task_name}:")
                print(f"    Prediction shape: {pred.shape}")
                print(f"    Target shape: {target.shape}")
                print(f"    Target values: {target.cpu().numpy()}")
                
                try:
                    loss = criterion(pred, target)
                    print(f"    Loss: {loss.item():.4f}")
                    
                    # Check accuracy
                    pred_classes = torch.argmax(pred, dim=1)
                    correct = (pred_classes == target).sum().item()
                    total = target.size(0)
                    accuracy = correct / total
                    print(f"    Accuracy: {correct}/{total} = {accuracy:.4f}")
                    
                except Exception as e:
                    print(f"    Error calculating loss: {e}")
        
        # Test training step
        print(f"\n🚀 Testing Training Step:")
        model.train()
        optimizer.zero_grad()
        
        predictions = model(signals)
        
        total_loss = 0.0
        valid_tasks = 0
        for task_name in task_configs.keys():
            if task_name in predictions and task_name in targets:
                loss = criterion(predictions[task_name], targets[task_name].squeeze())
                if not torch.isnan(loss) and not torch.isinf(loss):
                    total_loss += loss
                    valid_tasks += 1
                    print(f"  {task_name} loss: {loss.item():.4f}")
        
        print(f"  Total loss: {total_loss.item():.4f}")
        print(f"  Valid tasks: {valid_tasks}")
        
        if valid_tasks > 0:
            total_loss.backward()
            print(f"  Gradients computed successfully")
            optimizer.step()
            print(f"  Optimizer step completed")
        else:
            print(f"  ⚠️  No valid tasks for training!")

if __name__ == "__main__":
    debug_training()
