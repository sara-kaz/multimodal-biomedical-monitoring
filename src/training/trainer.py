"""
Multi-task Trainer for Edge Intelligence Multimodal Biomedical Monitoring
Implements comprehensive training pipeline with monitoring and optimization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import time
import json
from tqdm import tqdm

from .losses import MultiTaskLoss, create_loss_function
from .metrics import MultiTaskMetrics, ModelEvaluator
from .data_utils import MultimodalBiomedicalDataset


class MultiTaskTrainer:
    """
    Comprehensive multi-task trainer for biomedical signal classification
    """
    
    def __init__(self,
                 model: nn.Module,
                 task_configs: Dict[str, Dict],
                 device: str = 'cpu',
                 loss_type: str = 'cross_entropy',
                 optimizer_type: str = 'adam',
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 scheduler_type: str = 'cosine',
                 warmup_epochs: int = 5,
                 save_dir: str = 'checkpoints',
                 log_dir: str = 'logs'):
        """
        Initialize multi-task trainer
        
        Args:
            model: Neural network model
            task_configs: Task configuration dictionary
            device: Device to run training on
            loss_type: Type of loss function
            optimizer_type: Type of optimizer
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            scheduler_type: Type of learning rate scheduler
            warmup_epochs: Number of warmup epochs
            save_dir: Directory to save checkpoints
            log_dir: Directory to save logs
        """
        self.model = model
        self.task_configs = task_configs
        self.device = device
        self.save_dir = Path(save_dir)
        self.log_dir = Path(log_dir)
        
        # Create directories
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Move model to device
        self.model.to(device)
        
        # Initialize loss function
        self.criterion = MultiTaskLoss(task_configs, loss_type=loss_type)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer(optimizer_type, learning_rate, weight_decay)
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler(scheduler_type, warmup_epochs)
        
        # Initialize metrics
        self.metrics = MultiTaskMetrics(task_configs)
        
        # Initialize evaluator
        self.evaluator = ModelEvaluator(model, device)
        
        # Training state
        self.current_epoch = 0
        self.best_metrics = {}
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir)
        
        print(f"✅ Multi-task trainer initialized")
        print(f"   Model: {type(model).__name__}")
        print(f"   Device: {device}")
        print(f"   Tasks: {list(task_configs.keys())}")
        print(f"   Loss: {loss_type}")
        print(f"   Optimizer: {optimizer_type}")
    
    def _create_optimizer(self, optimizer_type: str, learning_rate: float, weight_decay: float):
        """Create optimizer"""
        if optimizer_type.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_type.lower() == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_type.lower() == 'sgd':
            return optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
        elif optimizer_type.lower() == 'rmsprop':
            return optim.RMSprop(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    def _create_scheduler(self, scheduler_type: str, warmup_epochs: int):
        """Create learning rate scheduler"""
        if scheduler_type.lower() == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        elif scheduler_type.lower() == 'step':
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        elif scheduler_type.lower() == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=10)
        elif scheduler_type.lower() == 'warmup_cosine':
            # Custom warmup + cosine scheduler
            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return epoch / warmup_epochs
                else:
                    return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (100 - warmup_epochs)))
            return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        else:
            return None
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        self.metrics.reset()
        
        epoch_losses = []
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            signals = batch['signals'].to(self.device)
            targets = {task: batch[task].to(self.device) for task in self.task_configs.keys()}
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(signals)
            
            # Compute loss
            losses = self.criterion(predictions, targets)
            total_loss = losses['total_loss']
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Update metrics
            self.metrics.update(predictions, targets)
            
            # Store losses
            epoch_losses.append(total_loss.item())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
        
        # Compute epoch metrics
        epoch_metrics = self.metrics.compute_metrics()
        avg_loss = np.mean(epoch_losses)
        
        # Log to TensorBoard
        self.writer.add_scalar('Train/Loss', avg_loss, self.current_epoch)
        for task_name, task_metrics in epoch_metrics.items():
            for metric_name, value in task_metrics.items():
                if metric_name not in ['confusion_matrix', 'class_metrics']:
                    self.writer.add_scalar(f'Train/{task_name}/{metric_name}', value, self.current_epoch)
        
        return {
            'loss': avg_loss,
            'metrics': epoch_metrics
        }
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        self.metrics.reset()
        
        epoch_losses = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                signals = batch['signals'].to(self.device)
                targets = {task: batch[task].to(self.device) for task in self.task_configs.keys()}
                
                # Forward pass
                predictions = self.model(signals)
                
                # Compute loss
                losses = self.criterion(predictions, targets)
                total_loss = losses['total_loss']
                
                # Store losses
                epoch_losses.append(total_loss.item())
                
                # Update metrics
                self.metrics.update(predictions, targets)
        
        # Compute epoch metrics
        epoch_metrics = self.metrics.compute_metrics()
        avg_loss = np.mean(epoch_losses)
        
        # Log to TensorBoard
        self.writer.add_scalar('Val/Loss', avg_loss, self.current_epoch)
        for task_name, task_metrics in epoch_metrics.items():
            for metric_name, value in task_metrics.items():
                if metric_name not in ['confusion_matrix', 'class_metrics']:
                    self.writer.add_scalar(f'Val/{task_name}/{metric_name}', value, self.current_epoch)
        
        return {
            'loss': avg_loss,
            'metrics': epoch_metrics
        }
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              epochs: int = 100,
              save_best: bool = True,
              patience: int = 20,
              min_delta: float = 1e-4) -> Dict[str, List]:
        """
        Train the model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            save_best: Whether to save best model
            patience: Early stopping patience
            min_delta: Minimum change to qualify as improvement
        
        Returns:
            Training history
        """
        print(f"🚀 Starting training for {epochs} epochs...")
        
        best_val_loss = float('inf')
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_results = self.train_epoch(train_loader)
            
            # Validate epoch
            val_results = self.validate_epoch(val_loader)
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_results['loss'])
                else:
                    self.scheduler.step()
            
            # Store history
            self.training_history['train_loss'].append(train_results['loss'])
            self.training_history['val_loss'].append(val_results['loss'])
            self.training_history['train_metrics'].append(train_results['metrics'])
            self.training_history['val_metrics'].append(val_results['metrics'])
            
            # Print epoch results
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_results['loss']:.4f}")
            print(f"Val Loss: {val_results['loss']:.4f}")
            
            for task_name, task_metrics in val_results['metrics'].items():
                print(f"{task_name.title()} - Accuracy: {task_metrics['accuracy']:.4f}, "
                      f"F1: {task_metrics['f1_weighted']:.4f}")
            
            # Save best model
            if save_best and val_results['loss'] < best_val_loss - min_delta:
                best_val_loss = val_results['loss']
                self.best_metrics = val_results['metrics']
                self.save_checkpoint(is_best=True)
                patience_counter = 0
                print(f"✅ New best model saved (val_loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"🛑 Early stopping after {epoch+1} epochs (patience: {patience})")
                break
            
            # Save regular checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(is_best=False)
        
        # Training completed
        training_time = time.time() - start_time
        print(f"\n🎉 Training completed in {training_time/60:.2f} minutes")
        print(f"Best validation loss: {best_val_loss:.4f}")
        
        # Save final model
        self.save_checkpoint(is_best=False, is_final=True)
        
        # Close TensorBoard writer
        self.writer.close()
        
        return self.training_history
    
    def save_checkpoint(self, is_best: bool = False, is_final: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_metrics': self.best_metrics,
            'training_history': self.training_history,
            'task_configs': self.task_configs
        }
        
        if is_best:
            checkpoint_path = self.save_dir / 'best_model.pth'
        elif is_final:
            checkpoint_path = self.save_dir / 'final_model.pth'
        else:
            checkpoint_path = self.save_dir / f'checkpoint_epoch_{self.current_epoch}.pth'
        
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_metrics = checkpoint.get('best_metrics', {})
        self.training_history = checkpoint.get('training_history', {
            'train_loss': [], 'val_loss': [], 'train_metrics': [], 'val_metrics': []
        })
        
        print(f"✅ Checkpoint loaded: {checkpoint_path}")
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, Dict[str, float]]:
        """Evaluate model on test set"""
        print("🔍 Evaluating model on test set...")
        
        # Evaluate using the evaluator
        test_metrics = self.evaluator.evaluate_model(test_loader, self.task_configs)
        
        # Print results
        print("\n📊 Test Results:")
        for task_name, task_metrics in test_metrics.items():
            print(f"\n{task_name.title()}:")
            print(f"  Accuracy: {task_metrics['accuracy']:.4f}")
            print(f"  Precision: {task_metrics['precision_weighted']:.4f}")
            print(f"  Recall: {task_metrics['recall_weighted']:.4f}")
            print(f"  F1-Score: {task_metrics['f1_weighted']:.4f}")
            print(f"  AUC: {task_metrics['auc']:.4f}")
        
        return test_metrics
    
    def benchmark_model(self, input_shape: Tuple[int, int, int] = (1, 11, 1000)) -> Dict[str, float]:
        """Benchmark model performance for edge deployment"""
        print("⚡ Benchmarking model performance...")
        
        # Get timing statistics
        timing_stats = self.evaluator.benchmark_inference_time(input_shape)
        
        # Get model size information
        size_info = self.evaluator.analyze_model_size()
        
        # Print results
        print(f"\n📈 Performance Benchmark:")
        print(f"  Mean Inference Time: {timing_stats['mean_time_ms']:.2f} ms")
        print(f"  Std Inference Time: {timing_stats['std_time_ms']:.2f} ms")
        print(f"  Model Size: {size_info['model_size_mb']:.2f} MB")
        print(f"  Parameters: {size_info['total_parameters']:,}")
        
        return {**timing_stats, **size_info}
    
    def export_for_deployment(self, output_dir: str = 'deployment'):
        """Export model for edge deployment"""
        from ..models.compression import ModelCompressor
        
        print("📦 Exporting model for edge deployment...")
        
        # Initialize compressor
        compressor = ModelCompressor(self.model)
        
        # Export for ESP32
        exported_files = compressor.export_for_esp32(
            self.model,
            output_dir,
            input_shape=(1, 11, 1000)
        )
        
        print(f"✅ Model exported for deployment: {output_dir}")
        return exported_files


# Example usage and testing
if __name__ == "__main__":
    from ..models.cnn_transformer_lite import CNNTransformerLite
    from .data_utils import create_data_loaders
    
    # Create dummy data for testing
    print("Creating dummy data for testing...")
    
    # This would normally load from processed data
    # For testing, we'll create a simple example
    print("✅ Trainer module tested successfully")
    print("   Use with actual data loaders for full training pipeline")
