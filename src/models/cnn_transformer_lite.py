"""
CNN/Transformer-Lite Architecture for Multimodal Biomedical Signal Processing
Optimized for ESP32-S3 Edge Deployment

This module implements a lightweight neural network architecture that combines:
1. 1D CNN layers for temporal feature extraction
2. Transformer-Lite encoder for multimodal fusion
3. Multi-task classification heads for simultaneous prediction
4. Optimized for memory and computational constraints of ESP32-S3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple


class PositionalEncoding(nn.Module):
    """Lightweight positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        self.d_model = d_model
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Use only sine encoding to reduce parameters
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:, :x.size(1)]


class TransformerLiteEncoder(nn.Module):
    """Lightweight Transformer encoder optimized for edge deployment"""
    
    def __init__(self, d_model: int, nhead: int, num_layers: int, 
                 dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        
        # Feed-forward network (simplified)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Stack multiple layers
        self.layers = nn.ModuleList([
            self._create_layer() for _ in range(num_layers)
        ])
    
    def _create_layer(self):
        """Create a single transformer layer"""
        return nn.ModuleDict({
            'self_attn': nn.MultiheadAttention(
                self.d_model, self.nhead, dropout=0.1, batch_first=True
            ),
            'ffn': nn.Sequential(
                nn.Linear(self.d_model, self.d_model * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.d_model * 2, self.d_model),
                nn.Dropout(0.1)
            ),
            'norm1': nn.LayerNorm(self.d_model),
            'norm2': nn.LayerNorm(self.d_model)
        })
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            mask: Optional attention mask
        Returns:
            Encoded tensor of same shape
        """
        for layer in self.layers:
            # Self-attention with residual connection
            attn_out, _ = layer['self_attn'](x, x, x, attn_mask=mask)
            x = layer['norm1'](x + self.dropout(attn_out))
            
            # Feed-forward with residual connection
            ffn_out = layer['ffn'](x)
            x = layer['norm2'](x + self.dropout(ffn_out))
        
        return x


class CNNTransformerLite(nn.Module):
    """
    CNN/Transformer-Lite architecture for multimodal biomedical signal processing
    
    Architecture:
    1. 1D CNN feature extractor for each signal type
    2. Transformer-Lite encoder for multimodal fusion
    3. Multi-task classification heads
    4. Optimized for ESP32-S3 deployment constraints
    """
    
    def __init__(self, 
                 n_channels: int = 11,  # ECG, PPG, 3xACC, EDA, Resp, Temp, EMG, EDA_wrist, Temp_wrist
                 n_samples: int = 1000,  # 10 seconds at 100Hz
                 d_model: int = 64,  # Reduced for edge deployment
                 nhead: int = 4,  # Reduced attention heads
                 num_layers: int = 2,  # Minimal transformer layers
                 dim_feedforward: int = 128,  # Reduced FFN dimension
                 dropout: float = 0.1,
                 task_configs: Optional[Dict] = None):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.d_model = d_model
        
        # Default task configurations
        if task_configs is None:
            task_configs = {
                'activity': {'num_classes': 8, 'weight': 1.0},
                'stress': {'num_classes': 4, 'weight': 1.0},
                'arrhythmia': {'num_classes': 2, 'weight': 1.0}
            }
        self.task_configs = task_configs
        
        # 1D CNN Feature Extractor
        self.cnn_layers = self._build_cnn_layers()
        
        # Channel projection to transformer dimension
        self.channel_projection = nn.Linear(n_channels * 32, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, n_samples)
        
        # Transformer-Lite encoder
        self.transformer = TransformerLiteEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Multi-task classification heads
        self.task_heads = nn.ModuleDict()
        for task_name, config in task_configs.items():
            self.task_heads[task_name] = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, config['num_classes'])
            )
        
        # Task weights for multi-task learning
        self.task_weights = nn.ParameterDict({
            task_name: nn.Parameter(torch.tensor(config['weight'], dtype=torch.float32))
            for task_name, config in task_configs.items()
        })
        
        # Initialize weights
        self._init_weights()
    
    def _build_cnn_layers(self) -> nn.Module:
        """Build 1D CNN layers for temporal feature extraction"""
        return nn.Sequential(
            # First conv block
            nn.Conv1d(self.n_channels, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 1000 -> 500
            
            # Second conv block
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 500 -> 250
            
            # Third conv block
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # No pooling to preserve temporal information
        )
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape [batch_size, n_channels, n_samples]
        
        Returns:
            Dictionary of task predictions
        """
        batch_size = x.size(0)
        
        # 1D CNN feature extraction
        cnn_features = self.cnn_layers(x)  # [batch_size, 32, 250]
        
        # Reshape for transformer input
        # [batch_size, 32, 250] -> [batch_size, 250, 32]
        cnn_features = cnn_features.transpose(1, 2)
        
        # Project to transformer dimension
        # [batch_size, 250, 32] -> [batch_size, 250, d_model]
        projected = self.channel_projection(
            cnn_features.reshape(batch_size, 250, -1)
        )
        
        # Add positional encoding
        pos_encoded = self.pos_encoding(projected)
        
        # Transformer encoding
        transformer_out = self.transformer(pos_encoded)
        
        # Global average pooling
        # [batch_size, 250, d_model] -> [batch_size, d_model]
        pooled = self.global_pool(transformer_out.transpose(1, 2)).squeeze(-1)
        
        # Multi-task predictions
        predictions = {}
        for task_name, head in self.task_heads.items():
            predictions[task_name] = head(pooled)
        
        return predictions
    
    def get_model_size(self) -> Dict[str, int]:
        """Get model size information for deployment analysis"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Estimate model size in bytes (assuming float32)
        model_size_bytes = total_params * 4
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_bytes': model_size_bytes,
            'model_size_kb': model_size_bytes / 1024,
            'model_size_mb': model_size_bytes / (1024 * 1024)
        }
    
    def get_flops_estimate(self, input_shape: Tuple[int, int, int]) -> int:
        """Estimate FLOPs for given input shape"""
        # This is a simplified estimation
        # In practice, you'd use tools like ptflops or thop for accurate measurement
        
        batch_size, n_channels, n_samples = input_shape
        
        # CNN layers FLOPs (approximate)
        cnn_flops = 0
        # Conv1d layers
        cnn_flops += n_channels * 16 * 7 * n_samples  # First conv
        cnn_flops += 16 * 32 * 5 * (n_samples // 2)  # Second conv
        cnn_flops += 32 * 32 * 3 * (n_samples // 4)  # Third conv
        
        # Transformer FLOPs (approximate)
        seq_len = n_samples // 4  # After CNN pooling
        transformer_flops = 0
        # Self-attention: O(n^2 * d)
        transformer_flops += seq_len * seq_len * self.d_model
        # FFN: O(n * d^2)
        transformer_flops += seq_len * self.d_model * self.d_model * 2
        
        total_flops = batch_size * (cnn_flops + transformer_flops)
        return int(total_flops)


class SingleModalityBaseline(nn.Module):
    """Single modality baseline models for comparison"""
    
    def __init__(self, modality: str, n_samples: int = 1000, num_classes: int = 2):
        super().__init__()
        self.modality = modality
        self.n_samples = n_samples
        
        # Simple CNN for single modality
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, n_samples]
        x = x.unsqueeze(1)  # Add channel dimension
        features = self.feature_extractor(x)
        features = features.squeeze(-1)  # Remove last dimension
        return self.classifier(features)


def create_model(model_type: str = 'cnn_transformer_lite', **kwargs) -> nn.Module:
    """
    Factory function to create different model architectures
    
    Args:
        model_type: Type of model to create
        **kwargs: Additional arguments for model initialization
    
    Returns:
        Initialized model
    """
    if model_type == 'cnn_transformer_lite':
        return CNNTransformerLite(**kwargs)
    elif model_type == 'single_modality_ecg':
        return SingleModalityBaseline(modality='ecg', **kwargs)
    elif model_type == 'single_modality_ppg':
        return SingleModalityBaseline(modality='ppg', **kwargs)
    elif model_type == 'single_modality_accel':
        return SingleModalityBaseline(modality='accel', **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Example usage and testing
if __name__ == "__main__":
    # Test the model
    model = CNNTransformerLite(
        n_channels=11,
        n_samples=1000,
        d_model=64,
        nhead=4,
        num_layers=2
    )
    
    # Create dummy input
    batch_size = 4
    dummy_input = torch.randn(batch_size, 11, 1000)
    
    # Forward pass
    with torch.no_grad():
        predictions = model(dummy_input)
    
    print("Model Architecture:")
    print(f"Input shape: {dummy_input.shape}")
    for task, pred in predictions.items():
        print(f"{task}: {pred.shape}")
    
    # Model size analysis
    size_info = model.get_model_size()
    print(f"\nModel Size Analysis:")
    for key, value in size_info.items():
        print(f"{key}: {value}")
    
    # FLOPs estimation
    flops = model.get_flops_estimate((batch_size, 11, 1000))
    print(f"Estimated FLOPs: {flops:,}")
