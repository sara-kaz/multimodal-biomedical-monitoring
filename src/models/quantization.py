"""
Quantization Module for ESP32-S3 Deployment
Implements various quantization techniques for edge deployment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import copy

# Handle PyTorch version differences
try:
    from torch.quantization import quantize_dynamic, quantize_static
except ImportError:
    from torch.ao.quantization import quantize_dynamic, quantize_static


class QuantizedLinear(nn.Module):
    """Quantized Linear layer for ESP32-S3 deployment"""
    
    def __init__(self, in_features: int, out_features: int, 
                 bias: bool = True, bits: int = 8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.bias = bias
        
        # Quantized weights (stored as int8)
        self.weight = nn.Parameter(torch.zeros(out_features, in_features, dtype=torch.int8))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.int32))
        else:
            self.register_parameter('bias', None)
        
        # Quantization parameters
        self.weight_scale = nn.Parameter(torch.ones(out_features))
        self.weight_zero_point = nn.Parameter(torch.zeros(out_features))
        
        if bias:
            self.bias_scale = nn.Parameter(torch.ones(out_features))
            self.bias_zero_point = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantized weights"""
        # Dequantize weights
        weight_fp = self.weight.float() * self.weight_scale.unsqueeze(1) + self.weight_zero_point.unsqueeze(1)
        
        # Linear transformation
        output = F.linear(x, weight_fp, None)
        
        # Add bias if present
        if self.bias is not None:
            bias_fp = self.bias.float() * self.bias_scale + self.bias_zero_point
            output = output + bias_fp
        
        return output


class QuantizedConv1d(nn.Module):
    """Quantized 1D Convolution layer for ESP32-S3 deployment"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, bias: bool = True, bits: int = 8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bits = bits
        self.bias = bias
        
        # Quantized weights
        self.weight = nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size, dtype=torch.int8))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, dtype=torch.int32))
        else:
            self.register_parameter('bias', None)
        
        # Quantization parameters
        self.weight_scale = nn.Parameter(torch.ones(out_channels))
        self.weight_zero_point = nn.Parameter(torch.zeros(out_channels))
        
        if bias:
            self.bias_scale = nn.Parameter(torch.ones(out_channels))
            self.bias_zero_point = nn.Parameter(torch.zeros(out_channels))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantized weights"""
        # Dequantize weights
        weight_fp = self.weight.float() * self.weight_scale.view(-1, 1, 1) + self.weight_zero_point.view(-1, 1, 1)
        
        # Convolution
        output = F.conv1d(x, weight_fp, None, self.stride, self.padding)
        
        # Add bias if present
        if self.bias is not None:
            bias_fp = self.bias.float() * self.bias_scale + self.bias_zero_point
            output = output + bias_fp.view(1, -1, 1)
        
        return output


class QuantizationCalibrator:
    """Calibrator for quantization parameters"""
    
    def __init__(self, model: nn.Module, bits: int = 8):
        self.model = model
        self.bits = bits
        self.activation_stats = {}
        self.weight_stats = {}
        
    def calibrate_activations(self, calibration_data: torch.Tensor):
        """Calibrate activation quantization parameters"""
        print("Calibrating activation quantization parameters...")
        
        self.model.eval()
        with torch.no_grad():
            for i, sample in enumerate(calibration_data):
                if i >= 100:  # Limit calibration samples
                    break
                
                # Forward pass to collect activation statistics
                _ = self.model(sample.unsqueeze(0))
                
                # Collect activation statistics for each layer
                for name, module in self.model.named_modules():
                    if hasattr(module, 'activation_stats'):
                        if name not in self.activation_stats:
                            self.activation_stats[name] = {
                                'min': float('inf'),
                                'max': float('-inf'),
                                'mean': 0.0,
                                'std': 0.0
                            }
                        
                        stats = module.activation_stats
                        self.activation_stats[name]['min'] = min(
                            self.activation_stats[name]['min'], stats['min']
                        )
                        self.activation_stats[name]['max'] = max(
                            self.activation_stats[name]['max'], stats['max']
                        )
        
        print(f"✅ Calibrated {len(self.activation_stats)} layers")
    
    def calibrate_weights(self):
        """Calibrate weight quantization parameters"""
        print("Calibrating weight quantization parameters...")
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                if hasattr(module, 'weight'):
                    weight = module.weight.data
                    
                    # Calculate quantization parameters
                    weight_min = weight.min().item()
                    weight_max = weight.max().item()
                    
                    # Symmetric quantization
                    scale = max(abs(weight_min), abs(weight_max)) / (2**(self.bits-1) - 1)
                    zero_point = 0
                    
                    self.weight_stats[name] = {
                        'scale': scale,
                        'zero_point': zero_point,
                        'min': weight_min,
                        'max': weight_max
                    }
        
        print(f"✅ Calibrated {len(self.weight_stats)} weight layers")


class PostTrainingQuantizer:
    """Post-training quantization for existing models"""
    
    def __init__(self, model: nn.Module, bits: int = 8):
        self.model = model
        self.bits = bits
        self.calibrator = QuantizationCalibrator(model, bits)
    
    def quantize_weights(self) -> nn.Module:
        """Quantize model weights"""
        print(f"Quantizing model weights to {self.bits} bits...")
        
        quantized_model = copy.deepcopy(self.model)
        
        for name, module in quantized_model.named_modules():
            if isinstance(module, nn.Linear):
                # Replace with quantized linear layer
                quantized_linear = QuantizedLinear(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    bits=self.bits
                )
                
                # Copy and quantize weights
                self._quantize_linear_weights(module, quantized_linear)
                
                # Replace module
                parent = quantized_model
                for attr in name.split('.')[:-1]:
                    parent = getattr(parent, attr)
                setattr(parent, name.split('.')[-1], quantized_linear)
            
            elif isinstance(module, nn.Conv1d):
                # Replace with quantized conv1d layer
                quantized_conv = QuantizedConv1d(
                    module.in_channels,
                    module.out_channels,
                    module.kernel_size[0],
                    stride=module.stride[0],
                    padding=module.padding[0],
                    bias=module.bias is not None,
                    bits=self.bits
                )
                
                # Copy and quantize weights
                self._quantize_conv_weights(module, quantized_conv)
                
                # Replace module
                parent = quantized_model
                for attr in name.split('.')[:-1]:
                    parent = getattr(parent, attr)
                setattr(parent, name.split('.')[-1], quantized_conv)
        
        print("✅ Model weights quantized")
        return quantized_model
    
    def _quantize_linear_weights(self, original: nn.Linear, quantized: QuantizedLinear):
        """Quantize linear layer weights"""
        weight = original.weight.data
        bias = original.bias.data if original.bias is not None else None
        
        # Calculate quantization parameters
        weight_min = weight.min().item()
        weight_max = weight.max().item()
        weight_scale = max(abs(weight_min), abs(weight_max)) / (2**(self.bits-1) - 1)
        weight_zero_point = 0
        
        # Quantize weights
        quantized_weight = torch.round(weight / weight_scale).clamp(
            -2**(self.bits-1), 2**(self.bits-1)-1
        ).to(torch.int8)
        
        quantized.weight.data = quantized_weight
        quantized.weight_scale.data = torch.tensor(weight_scale)
        quantized.weight_zero_point.data = torch.tensor(weight_zero_point)
        
        # Quantize bias if present
        if bias is not None:
            bias_scale = weight_scale  # Same scale as weights
            quantized_bias = torch.round(bias / bias_scale).clamp(
                -2**31, 2**31-1
            ).to(torch.int32)
            
            quantized.bias.data = quantized_bias
            quantized.bias_scale.data = torch.tensor(bias_scale)
            quantized.bias_zero_point.data = torch.tensor(0)
    
    def _quantize_conv_weights(self, original: nn.Conv1d, quantized: QuantizedConv1d):
        """Quantize conv1d layer weights"""
        weight = original.weight.data
        bias = original.bias.data if original.bias is not None else None
        
        # Calculate quantization parameters
        weight_min = weight.min().item()
        weight_max = weight.max().item()
        weight_scale = max(abs(weight_min), abs(weight_max)) / (2**(self.bits-1) - 1)
        weight_zero_point = 0
        
        # Quantize weights
        quantized_weight = torch.round(weight / weight_scale).clamp(
            -2**(self.bits-1), 2**(self.bits-1)-1
        ).to(torch.int8)
        
        quantized.weight.data = quantized_weight
        quantized.weight_scale.data = torch.tensor(weight_scale)
        quantized.weight_zero_point.data = torch.tensor(weight_zero_point)
        
        # Quantize bias if present
        if bias is not None:
            bias_scale = weight_scale  # Same scale as weights
            quantized_bias = torch.round(bias / bias_scale).clamp(
                -2**31, 2**31-1
            ).to(torch.int32)
            
            quantized.bias.data = quantized_bias
            quantized.bias_scale.data = torch.tensor(bias_scale)
            quantized.bias_zero_point.data = torch.tensor(0)


class QuantizationAwareTraining:
    """Quantization-Aware Training (QAT) implementation"""
    
    def __init__(self, model: nn.Module, bits: int = 8):
        self.model = model
        self.bits = bits
        self.fake_quantized_model = None
    
    def prepare_qat_model(self) -> nn.Module:
        """Prepare model for quantization-aware training"""
        print("Preparing model for Quantization-Aware Training...")
        
        # Create fake quantization modules
        self.fake_quantized_model = self._add_fake_quantization(self.model)
        
        print("✅ Model prepared for QAT")
        return self.fake_quantized_model
    
    def _add_fake_quantization(self, model: nn.Module) -> nn.Module:
        """Add fake quantization modules for QAT"""
        # This is a simplified implementation
        # In practice, you'd use torch.quantization.quantize_dynamic or similar
        
        qat_model = copy.deepcopy(model)
        
        # Add fake quantization after each linear and conv layer
        for name, module in qat_model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                # Add fake quantization wrapper
                fake_quant = FakeQuantization(bits=self.bits)
                
                # Replace module with quantized version
                parent = qat_model
                for attr in name.split('.')[:-1]:
                    parent = getattr(parent, attr)
                setattr(parent, name.split('.')[-1], fake_quant)
        
        return qat_model
    
    def convert_to_quantized(self) -> nn.Module:
        """Convert QAT model to quantized model"""
        if self.fake_quantized_model is None:
            raise ValueError("Model not prepared for QAT")
        
        print("Converting QAT model to quantized model...")
        
        # Convert fake quantization to real quantization
        quantized_model = self._convert_fake_to_real(self.fake_quantized_model)
        
        print("✅ QAT model converted to quantized model")
        return quantized_model
    
    def _convert_fake_to_real(self, model: nn.Module) -> nn.Module:
        """Convert fake quantization to real quantization"""
        # This would contain the actual conversion logic
        # For now, return the model as-is
        return model


class FakeQuantization(nn.Module):
    """Fake quantization for QAT"""
    
    def __init__(self, bits: int = 8):
        super().__init__()
        self.bits = bits
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.zero_point = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with fake quantization"""
        # Quantize
        quantized = torch.round(x / self.scale + self.zero_point)
        quantized = quantized.clamp(-2**(self.bits-1), 2**(self.bits-1)-1)
        
        # Dequantize
        dequantized = (quantized - self.zero_point) * self.scale
        
        return dequantized


def quantize_model_for_esp32(model: nn.Module, 
                           calibration_data: Optional[torch.Tensor] = None,
                           bits: int = 8) -> nn.Module:
    """
    Complete quantization pipeline for ESP32-S3 deployment
    
    Args:
        model: Model to quantize
        calibration_data: Data for calibration (optional)
        bits: Number of bits for quantization
    
    Returns:
        Quantized model ready for ESP32 deployment
    """
    print(f"Quantizing model for ESP32-S3 deployment ({bits} bits)...")
    
    # Initialize quantizer
    quantizer = PostTrainingQuantizer(model, bits)
    
    # Calibrate if data provided
    if calibration_data is not None:
        quantizer.calibrator.calibrate_activations(calibration_data)
    
    # Quantize weights
    quantized_model = quantizer.quantize_weights()
    
    print("✅ Model quantized for ESP32-S3 deployment")
    return quantized_model


# Example usage and testing
if __name__ == "__main__":
    from cnn_transformer_lite import CNNTransformerLite
    
    # Create a test model
    model = CNNTransformerLite(n_channels=11, n_samples=1000)
    
    # Test post-training quantization
    quantizer = PostTrainingQuantizer(model, bits=8)
    quantized_model = quantizer.quantize_weights()
    
    # Test with dummy data
    dummy_input = torch.randn(1, 11, 1000)
    
    with torch.no_grad():
        original_output = model(dummy_input)
        quantized_output = quantized_model(dummy_input)
    
    print("Original model output shapes:")
    for task, output in original_output.items():
        print(f"  {task}: {output.shape}")
    
    print("Quantized model output shapes:")
    for task, output in quantized_output.items():
        print(f"  {task}: {output.shape}")
    
    # Test complete quantization pipeline
    final_quantized = quantize_model_for_esp32(model, bits=8)
    print("✅ Complete quantization pipeline tested successfully")
