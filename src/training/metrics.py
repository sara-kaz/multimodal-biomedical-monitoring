"""
Evaluation Metrics for Multi-task Biomedical Signal Classification
Includes comprehensive metrics for model evaluation and comparison
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    roc_auc_score, confusion_matrix, classification_report
)
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns


class MultiTaskMetrics:
    """
    Comprehensive metrics calculator for multi-task learning
    """
    
    def __init__(self, task_configs: Dict[str, Dict]):
        self.task_configs = task_configs
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.predictions = {task: [] for task in self.task_configs.keys()}
        self.targets = {task: [] for task in self.task_configs.keys()}
        self.probabilities = {task: [] for task in self.task_configs.keys()}
    
    def update(self, predictions: Dict[str, torch.Tensor], 
               targets: Dict[str, torch.Tensor],
               probabilities: Optional[Dict[str, torch.Tensor]] = None):
        """
        Update metrics with new batch of predictions
        
        Args:
            predictions: Dictionary of task predictions (logits)
            targets: Dictionary of task targets (class indices)
            probabilities: Dictionary of task probabilities (optional)
        """
        for task_name in self.task_configs.keys():
            if task_name in predictions and task_name in targets:
                # Convert to numpy arrays
                pred = predictions[task_name].detach().cpu().numpy()
                target = targets[task_name].detach().cpu().numpy()
                
                # Get predicted classes
                pred_classes = np.argmax(pred, axis=1)
                
                self.predictions[task_name].extend(pred_classes)
                self.targets[task_name].extend(target)
                
                # Store probabilities if provided
                if probabilities is not None and task_name in probabilities:
                    prob = probabilities[task_name].detach().cpu().numpy()
                    self.probabilities[task_name].extend(prob)
                else:
                    # Convert logits to probabilities
                    prob = torch.softmax(torch.tensor(pred), dim=1).numpy()
                    self.probabilities[task_name].extend(prob)
    
    def compute_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute comprehensive metrics for all tasks
        
        Returns:
            Dictionary of metrics for each task
        """
        metrics = {}
        
        for task_name in self.task_configs.keys():
            if len(self.predictions[task_name]) == 0:
                continue
            
            y_pred = np.array(self.predictions[task_name])
            y_true = np.array(self.targets[task_name])
            y_prob = np.array(self.probabilities[task_name])
            
            # Basic metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred, average='weighted', zero_division=0
            )
            
            # Per-class metrics
            precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                y_true, y_pred, average='macro', zero_division=0
            )
            
            # AUC (for binary and multiclass)
            try:
                if len(np.unique(y_true)) == 2:
                    # Binary classification
                    auc = roc_auc_score(y_true, y_prob[:, 1])
                else:
                    # Multiclass classification
                    auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
            except ValueError:
                auc = 0.0
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Class-wise metrics
            class_metrics = {}
            for i in range(len(np.unique(y_true))):
                class_precision = precision_recall_fscore_support(
                    y_true, y_pred, labels=[i], average='binary', zero_division=0
                )[0][0]
                class_recall = precision_recall_fscore_support(
                    y_true, y_pred, labels=[i], average='binary', zero_division=0
                )[1][0]
                class_f1 = precision_recall_fscore_support(
                    y_true, y_pred, labels=[i], average='binary', zero_division=0
                )[2][0]
                
                class_metrics[f'class_{i}'] = {
                    'precision': class_precision,
                    'recall': class_recall,
                    'f1': class_f1
                }
            
            metrics[task_name] = {
                'accuracy': accuracy,
                'precision_weighted': precision,
                'recall_weighted': recall,
                'f1_weighted': f1,
                'precision_macro': precision_macro,
                'recall_macro': recall_macro,
                'f1_macro': f1_macro,
                'auc': auc,
                'support': support,
                'confusion_matrix': cm,
                'class_metrics': class_metrics
            }
        
        return metrics
    
    def plot_confusion_matrices(self, save_path: Optional[str] = None):
        """Plot confusion matrices for all tasks"""
        metrics = self.compute_metrics()
        
        num_tasks = len(metrics)
        fig, axes = plt.subplots(1, num_tasks, figsize=(5 * num_tasks, 4))
        if num_tasks == 1:
            axes = [axes]
        
        for i, (task_name, task_metrics) in enumerate(metrics.items()):
            cm = task_metrics['confusion_matrix']
            
            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'{task_name.title()} Confusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrices saved to: {save_path}")
        else:
            plt.show()
    
    def plot_roc_curves(self, save_path: Optional[str] = None):
        """Plot ROC curves for all tasks"""
        from sklearn.metrics import roc_curve, roc_auc_score
        
        metrics = self.compute_metrics()
        
        plt.figure(figsize=(10, 6))
        
        for task_name, task_metrics in metrics.items():
            y_true = np.array(self.targets[task_name])
            y_prob = np.array(self.probabilities[task_name])
            
            if len(np.unique(y_true)) == 2:
                # Binary classification
                fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
                auc = roc_auc_score(y_true, y_prob[:, 1])
                plt.plot(fpr, tpr, label=f'{task_name} (AUC = {auc:.3f})')
            else:
                # Multiclass classification - plot one-vs-rest
                for i in range(len(np.unique(y_true))):
                    y_binary = (y_true == i).astype(int)
                    fpr, tpr, _ = roc_curve(y_binary, y_prob[:, i])
                    auc = roc_auc_score(y_binary, y_prob[:, i])
                    plt.plot(fpr, tpr, label=f'{task_name} Class {i} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curves saved to: {save_path}")
        else:
            plt.show()


def calculate_metrics(predictions: Dict[str, torch.Tensor], 
                     targets: Dict[str, torch.Tensor],
                     task_configs: Dict[str, Dict]) -> Dict[str, Dict[str, float]]:
    """
    Calculate metrics for a single batch of predictions
    
    Args:
        predictions: Dictionary of task predictions
        targets: Dictionary of task targets
        task_configs: Task configuration dictionary
    
    Returns:
        Dictionary of metrics for each task
    """
    metrics_calculator = MultiTaskMetrics(task_configs)
    metrics_calculator.update(predictions, targets)
    return metrics_calculator.compute_metrics()


class ModelEvaluator:
    """
    Comprehensive model evaluator for edge deployment analysis
    """
    
    def __init__(self, model: torch.nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
    
    def evaluate_model(self, 
                      data_loader: torch.utils.data.DataLoader,
                      task_configs: Dict[str, Dict]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model on dataset
        
        Args:
            data_loader: DataLoader for evaluation
            task_configs: Task configuration dictionary
        
        Returns:
            Comprehensive evaluation metrics
        """
        self.model.eval()
        metrics_calculator = MultiTaskMetrics(task_configs)
        
        with torch.no_grad():
            for batch in data_loader:
                # Move batch to device
                signals = batch['signals'].to(self.device)
                targets = {task: batch[task].to(self.device) for task in task_configs.keys()}
                
                # Get predictions
                predictions = self.model(signals)
                
                # Convert predictions to probabilities
                probabilities = {
                    task: torch.softmax(pred, dim=1) 
                    for task, pred in predictions.items()
                }
                
                # Update metrics
                metrics_calculator.update(predictions, targets, probabilities)
        
        return metrics_calculator.compute_metrics()
    
    def benchmark_inference_time(self, 
                               input_shape: Tuple[int, int, int] = (1, 11, 1000),
                               num_runs: int = 100) -> Dict[str, float]:
        """
        Benchmark inference time for edge deployment analysis
        
        Args:
            input_shape: Input tensor shape
            num_runs: Number of runs for timing
        
        Returns:
            Timing statistics
        """
        self.model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(dummy_input)
        
        # Timing runs
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = torch.cuda.Event(enable_timing=True) if self.device == 'cuda' else None
                end_time = torch.cuda.Event(enable_timing=True) if self.device == 'cuda' else None
                
                if self.device == 'cuda':
                    start_time.record()
                    _ = self.model(dummy_input)
                    end_time.record()
                    torch.cuda.synchronize()
                    times.append(start_time.elapsed_time(end_time))  # milliseconds
                else:
                    import time
                    start = time.time()
                    _ = self.model(dummy_input)
                    end = time.time()
                    times.append((end - start) * 1000)  # milliseconds
        
        return {
            'mean_time_ms': np.mean(times),
            'std_time_ms': np.std(times),
            'min_time_ms': np.min(times),
            'max_time_ms': np.max(times),
            'median_time_ms': np.median(times)
        }
    
    def analyze_model_size(self) -> Dict[str, Union[int, float]]:
        """Analyze model size and complexity"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Estimate model size in bytes (assuming float32)
        model_size_bytes = total_params * 4
        
        # Count layers by type
        layer_counts = {}
        for module in self.model.modules():
            module_type = type(module).__name__
            if module_type not in ['Sequential', 'ModuleList', 'ModuleDict']:
                layer_counts[module_type] = layer_counts.get(module_type, 0) + 1
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_bytes': model_size_bytes,
            'model_size_kb': model_size_bytes / 1024,
            'model_size_mb': model_size_bytes / (1024 * 1024),
            'layer_counts': layer_counts
        }


def compare_models(models: Dict[str, torch.nn.Module], 
                  data_loader: torch.utils.data.DataLoader,
                  task_configs: Dict[str, Dict],
                  device: str = 'cpu') -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Compare multiple models on the same dataset
    
    Args:
        models: Dictionary of model name -> model
        data_loader: DataLoader for evaluation
        task_configs: Task configuration dictionary
        device: Device to run evaluation on
    
    Returns:
        Comparison results for each model
    """
    results = {}
    
    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")
        
        evaluator = ModelEvaluator(model, device)
        metrics = evaluator.evaluate_model(data_loader, task_configs)
        timing = evaluator.benchmark_inference_time()
        size_info = evaluator.analyze_model_size()
        
        results[model_name] = {
            'metrics': metrics,
            'timing': timing,
            'size': size_info
        }
    
    return results


# Example usage and testing
if __name__ == "__main__":
    # Test metrics calculation
    task_configs = {
        'activity': {'num_classes': 8},
        'stress': {'num_classes': 4},
        'arrhythmia': {'num_classes': 2}
    }
    
    # Create dummy data
    batch_size = 10
    predictions = {
        'activity': torch.randn(batch_size, 8),
        'stress': torch.randn(batch_size, 4),
        'arrhythmia': torch.randn(batch_size, 2)
    }
    
    targets = {
        'activity': torch.randint(0, 8, (batch_size,)),
        'stress': torch.randint(0, 4, (batch_size,)),
        'arrhythmia': torch.randint(0, 2, (batch_size,))
    }
    
    # Test metrics calculator
    metrics_calculator = MultiTaskMetrics(task_configs)
    metrics_calculator.update(predictions, targets)
    metrics = metrics_calculator.compute_metrics()
    
    print("Metrics for each task:")
    for task_name, task_metrics in metrics.items():
        print(f"\n{task_name}:")
        for metric_name, value in task_metrics.items():
            if metric_name not in ['confusion_matrix', 'class_metrics']:
                print(f"  {metric_name}: {value:.4f}")
    
    print("✅ Metrics calculation tested successfully")
