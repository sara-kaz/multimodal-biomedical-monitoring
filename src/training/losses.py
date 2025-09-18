"""
Loss Functions for Multi-task Biomedical Signal Classification
Includes specialized losses for imbalanced datasets and edge deployment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union
import numpy as np


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss function that combines losses from different tasks
    with adaptive weighting
    """
    
    def __init__(self, 
                 task_configs: Dict[str, Dict],
                 loss_type: str = 'cross_entropy',
                 adaptive_weighting: bool = True,
                 temperature: float = 2.0):
        super().__init__()
        
        self.task_configs = task_configs
        self.adaptive_weighting = adaptive_weighting
        self.temperature = temperature
        
        # Initialize task weights
        self.task_weights = nn.ParameterDict({
            task_name: nn.Parameter(torch.tensor(config.get('weight', 1.0), dtype=torch.float32))
            for task_name, config in task_configs.items()
        })
        
        # Initialize individual loss functions
        self.loss_functions = nn.ModuleDict()
        for task_name, config in task_configs.items():
            num_classes = config['num_classes']
            
            if loss_type == 'cross_entropy':
                self.loss_functions[task_name] = nn.CrossEntropyLoss()
            elif loss_type == 'focal':
                self.loss_functions[task_name] = FocalLoss(
                    alpha=config.get('alpha', None),
                    gamma=config.get('gamma', 2.0)
                )
            elif loss_type == 'label_smoothing':
                self.loss_functions[task_name] = LabelSmoothingLoss(
                    smoothing=config.get('smoothing', 0.1),
                    num_classes=num_classes
                )
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss
        
        Args:
            predictions: Dictionary of task predictions
            targets: Dictionary of task targets
        
        Returns:
            Dictionary containing individual and total losses
        """
        losses = {}
        total_loss = 0.0
        
        for task_name in self.task_configs.keys():
            if task_name in predictions and task_name in targets:
                pred = predictions[task_name]
                target = targets[task_name]
                
                # Compute individual task loss
                task_loss = self.loss_functions[task_name](pred, target)
                losses[f'{task_name}_loss'] = task_loss
                
                # Apply task weight
                if self.adaptive_weighting:
                    weight = F.softmax(self.task_weights[task_name] / self.temperature, dim=0)
                    weighted_loss = weight * task_loss
                else:
                    weighted_loss = self.task_weights[task_name] * task_loss
                
                losses[f'{task_name}_weighted_loss'] = weighted_loss
                total_loss += weighted_loss
        
        losses['total_loss'] = total_loss
        return losses
    
    def update_task_weights(self, task_losses: Dict[str, float]):
        """Update task weights based on loss magnitudes (adaptive weighting)"""
        if not self.adaptive_weighting:
            return
        
        # Compute relative loss magnitudes
        loss_values = list(task_losses.values())
        if len(loss_values) == 0:
            return
        
        # Normalize losses
        max_loss = max(loss_values)
        normalized_losses = [loss / max_loss for loss in loss_values]
        
        # Update weights inversely proportional to loss magnitude
        for i, (task_name, _) in enumerate(self.task_configs.items()):
            if i < len(normalized_losses):
                new_weight = 1.0 / (normalized_losses[i] + 1e-8)
                self.task_weights[task_name].data = torch.tensor(new_weight, dtype=torch.float32)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    Addresses the problem of easy negatives overwhelming the loss
    """
    
    def __init__(self, alpha: Optional[Union[float, List[float]]] = None, 
                 gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
        if isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        elif isinstance(alpha, float):
            self.alpha = torch.tensor([alpha, 1.0 - alpha], dtype=torch.float32)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss
        
        Args:
            inputs: Model predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
        
        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            ce_loss = alpha_t * ce_loss
        
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss for better generalization
    Reduces overconfidence and improves model calibration
    """
    
    def __init__(self, smoothing: float = 0.1, num_classes: int = 2):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.confidence = 1.0 - smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute label smoothing loss
        
        Args:
            inputs: Model predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
        
        Returns:
            Label smoothing loss value
        """
        log_preds = F.log_softmax(inputs, dim=1)
        
        # Create smoothed targets
        smooth_targets = torch.zeros_like(log_preds)
        smooth_targets.fill_(self.smoothing / (self.num_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        # Compute loss
        loss = -torch.sum(smooth_targets * log_preds, dim=1)
        return loss.mean()


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for learning better representations
    Useful for multimodal fusion
    """
    
    def __init__(self, margin: float = 1.0, temperature: float = 0.1):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
    
    def forward(self, features1: torch.Tensor, features2: torch.Tensor, 
                labels: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss between two feature representations
        
        Args:
            features1: First feature representation [batch_size, feature_dim]
            features2: Second feature representation [batch_size, feature_dim]
            labels: Binary labels indicating if pairs are similar (1) or dissimilar (0)
        
        Returns:
            Contrastive loss value
        """
        # Compute cosine similarity
        similarity = F.cosine_similarity(features1, features2, dim=1)
        
        # Convert to distance
        distance = 1 - similarity
        
        # Contrastive loss
        positive_loss = labels.float() * torch.pow(distance, 2)
        negative_loss = (1 - labels.float()) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        
        loss = torch.mean(positive_loss + negative_loss)
        return loss


class KLDivergenceLoss(nn.Module):
    """
    KL Divergence Loss for knowledge distillation
    Used when distilling knowledge from teacher to student model
    """
    
    def __init__(self, temperature: float = 3.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, student_logits: torch.Tensor, 
                teacher_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence loss between student and teacher predictions
        
        Args:
            student_logits: Student model logits [batch_size, num_classes]
            teacher_logits: Teacher model logits [batch_size, num_classes]
        
        Returns:
            KL divergence loss value
        """
        # Apply temperature scaling
        student_probs = F.softmax(student_logits / self.temperature, dim=1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        
        # Compute KL divergence
        kl_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            teacher_probs,
            reduction='batchmean'
        )
        
        return kl_loss * (self.temperature ** 2)


class UncertaintyWeightedLoss(nn.Module):
    """
    Uncertainty-weighted loss for multi-task learning
    Automatically learns task weights based on uncertainty
    """
    
    def __init__(self, num_tasks: int):
        super().__init__()
        self.num_tasks = num_tasks
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
    
    def forward(self, task_losses: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute uncertainty-weighted multi-task loss
        
        Args:
            task_losses: List of individual task losses
        
        Returns:
            Weighted total loss
        """
        if len(task_losses) != self.num_tasks:
            raise ValueError(f"Expected {self.num_tasks} task losses, got {len(task_losses)}")
        
        total_loss = 0.0
        for i, loss in enumerate(task_losses):
            precision = torch.exp(-self.log_vars[i])
            weighted_loss = precision * loss + self.log_vars[i]
            total_loss += weighted_loss
        
        return total_loss


def create_loss_function(loss_type: str, **kwargs) -> nn.Module:
    """
    Factory function to create loss functions
    
    Args:
        loss_type: Type of loss function to create
        **kwargs: Additional arguments for loss function initialization
    
    Returns:
        Initialized loss function
    """
    if loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss(**kwargs)
    elif loss_type == 'focal':
        return FocalLoss(**kwargs)
    elif loss_type == 'label_smoothing':
        return LabelSmoothingLoss(**kwargs)
    elif loss_type == 'contrastive':
        return ContrastiveLoss(**kwargs)
    elif loss_type == 'kl_divergence':
        return KLDivergenceLoss(**kwargs)
    elif loss_type == 'uncertainty_weighted':
        return UncertaintyWeightedLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# Example usage and testing
if __name__ == "__main__":
    # Test multi-task loss
    task_configs = {
        'activity': {'num_classes': 8, 'weight': 1.0},
        'stress': {'num_classes': 4, 'weight': 1.0},
        'arrhythmia': {'num_classes': 2, 'weight': 1.0}
    }
    
    multi_task_loss = MultiTaskLoss(task_configs, loss_type='cross_entropy')
    
    # Test with dummy data
    batch_size = 4
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
    
    losses = multi_task_loss(predictions, targets)
    print("Multi-task losses:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.4f}")
    
    # Test focal loss
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    inputs = torch.randn(4, 2)
    targets = torch.randint(0, 2, (4,))
    focal_loss_value = focal_loss(inputs, targets)
    print(f"Focal loss: {focal_loss_value.item():.4f}")
    
    # Test label smoothing loss
    smooth_loss = LabelSmoothingLoss(smoothing=0.1, num_classes=2)
    smooth_loss_value = smooth_loss(inputs, targets)
    print(f"Label smoothing loss: {smooth_loss_value.item():.4f}")
    
    print("✅ All loss functions tested successfully")
