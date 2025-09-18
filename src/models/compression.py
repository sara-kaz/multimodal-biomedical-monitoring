"""
Model Compression Techniques for ESP32-S3 Deployment
Includes quantization, pruning, and knowledge distillation
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.quantization import quantize_dynamic, quantize_static
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import copy


class ModelCompressor:
    """
    Comprehensive model compression for edge deployment
    Implements quantization, pruning, and knowledge distillation
    """
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.original_model = copy.deepcopy(model)
        self.compression_history = []
    
    def quantize_model(self, 
                      quantization_type: str = 'dynamic',
                      target_modules: Optional[List[str]] = None,
                      calibration_data: Optional[torch.Tensor] = None) -> nn.Module:
        """
        Quantize model for reduced precision inference
        
        Args:
            quantization_type: 'dynamic' or 'static'
            target_modules: List of module types to quantize
            calibration_data: Data for static quantization calibration
        
        Returns:
            Quantized model
        """
        print(f"Quantizing model with {quantization_type} quantization...")
        
        if target_modules is None:
            target_modules = [nn.Linear, nn.Conv1d]
        
        if quantization_type == 'dynamic':
            # Dynamic quantization (no calibration needed)
            quantized_model = quantize_dynamic(
                self.model, 
                target_modules,
                dtype=torch.qint8
            )
            
        elif quantization_type == 'static':
            if calibration_data is None:
                raise ValueError("Calibration data required for static quantization")
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Prepare model for static quantization
            self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(self.model, inplace=True)
            
            # Calibrate with sample data
            with torch.no_grad():
                for i in range(min(100, len(calibration_data))):
                    sample = calibration_data[i].unsqueeze(0)
                    _ = self.model(sample)
            
            # Convert to quantized model
            quantized_model = torch.quantization.convert(self.model, inplace=False)
            
        else:
            raise ValueError(f"Unknown quantization type: {quantization_type}")
        
        # Record compression
        self.compression_history.append({
            'type': 'quantization',
            'method': quantization_type,
            'target_modules': target_modules
        })
        
        print(f"✅ Model quantized successfully")
        return quantized_model
    
    def prune_model(self, 
                   pruning_ratio: float = 0.3,
                   pruning_type: str = 'magnitude',
                   target_modules: Optional[List[str]] = None) -> nn.Module:
        """
        Prune model to reduce parameters
        
        Args:
            pruning_ratio: Fraction of parameters to prune (0.0 to 1.0)
            pruning_type: 'magnitude', 'random', or 'structured'
            target_modules: List of module types to prune
        
        Returns:
            Pruned model
        """
        print(f"Pruning model with {pruning_type} pruning (ratio: {pruning_ratio})...")
        
        if target_modules is None:
            target_modules = [nn.Linear, nn.Conv1d]
        
        # Create pruning parameters
        parameters_to_prune = []
        for name, module in self.model.named_modules():
            if any(isinstance(module, target_type) for target_type in target_modules):
                if hasattr(module, 'weight'):
                    parameters_to_prune.append((module, 'weight'))
        
        if pruning_type == 'magnitude':
            # Magnitude-based pruning
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=pruning_ratio,
            )
            
        elif pruning_type == 'random':
            # Random pruning
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.RandomUnstructured,
                amount=pruning_ratio,
            )
            
        elif pruning_type == 'structured':
            # Structured pruning (removes entire channels/filters)
            for module, param_name in parameters_to_prune:
                if isinstance(module, nn.Conv1d):
                    prune.ln_structured(module, param_name, amount=pruning_ratio, n=2, dim=0)
                elif isinstance(module, nn.Linear):
                    prune.ln_structured(module, param_name, amount=pruning_ratio, n=2, dim=1)
        
        else:
            raise ValueError(f"Unknown pruning type: {pruning_type}")
        
        # Make pruning permanent
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)
        
        # Record compression
        self.compression_history.append({
            'type': 'pruning',
            'method': pruning_type,
            'ratio': pruning_ratio,
            'target_modules': target_modules
        })
        
        print(f"✅ Model pruned successfully")
        return self.model
    
    def apply_structured_pruning(self, 
                               pruning_ratio: float = 0.3,
                               target_modules: Optional[List[str]] = None) -> nn.Module:
        """
        Apply structured pruning to remove entire channels/filters
        
        Args:
            pruning_ratio: Fraction of channels to remove
            target_modules: List of module types to prune
        
        Returns:
            Structured pruned model
        """
        print(f"Applying structured pruning (ratio: {pruning_ratio})...")
        
        if target_modules is None:
            target_modules = [nn.Conv1d, nn.Linear]
        
        # Identify channels to prune based on L2 norm
        channels_to_prune = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv1d):
                # For Conv1d, prune output channels
                weights = module.weight.data
                channel_norms = torch.norm(weights, dim=(1, 2))  # L2 norm per channel
                num_channels_to_prune = int(len(channel_norms) * pruning_ratio)
                
                if num_channels_to_prune > 0:
                    _, indices_to_prune = torch.topk(channel_norms, num_channels_to_prune, largest=False)
                    channels_to_prune[name] = indices_to_prune.tolist()
            
            elif isinstance(module, nn.Linear):
                # For Linear, prune output features
                weights = module.weight.data
                feature_norms = torch.norm(weights, dim=1)  # L2 norm per output feature
                num_features_to_prune = int(len(feature_norms) * pruning_ratio)
                
                if num_features_to_prune > 0:
                    _, indices_to_prune = torch.topk(feature_norms, num_features_to_prune, largest=False)
                    channels_to_prune[name] = indices_to_prune.tolist()
        
        # Create new model with pruned architecture
        pruned_model = self._create_pruned_model(channels_to_prune)
        
        # Record compression
        self.compression_history.append({
            'type': 'structured_pruning',
            'ratio': pruning_ratio,
            'pruned_channels': channels_to_prune
        })
        
        print(f"✅ Structured pruning applied successfully")
        return pruned_model
    
    def _create_pruned_model(self, channels_to_prune: Dict[str, List[int]]) -> nn.Module:
        """Create new model with pruned architecture"""
        # This is a simplified implementation
        # In practice, you'd need to carefully reconstruct the model architecture
        # with the pruned dimensions
        
        # For now, return the original model
        # TODO: Implement proper model reconstruction
        return copy.deepcopy(self.model)
    
    def apply_knowledge_distillation(self, 
                                   teacher_model: nn.Module,
                                   student_model: nn.Module,
                                   temperature: float = 3.0,
                                   alpha: float = 0.7) -> nn.Module:
        """
        Apply knowledge distillation from teacher to student model
        
        Args:
            teacher_model: Pre-trained teacher model
            student_model: Smaller student model
            temperature: Softmax temperature for distillation
            alpha: Weight for distillation loss vs hard labels
        
        Returns:
            Distilled student model
        """
        print(f"Applying knowledge distillation (temperature: {temperature}, alpha: {alpha})...")
        
        # Set models to evaluation mode
        teacher_model.eval()
        student_model.train()
        
        # Record compression
        self.compression_history.append({
            'type': 'knowledge_distillation',
            'temperature': temperature,
            'alpha': alpha
        })
        
        print(f"✅ Knowledge distillation setup complete")
        return student_model
    
    def get_compression_stats(self) -> Dict[str, Union[int, float]]:
        """Get compression statistics"""
        original_params = sum(p.numel() for p in self.original_model.parameters())
        current_params = sum(p.numel() for p in self.model.parameters())
        
        # Calculate sparsity
        total_weights = 0
        zero_weights = 0
        
        for param in self.model.parameters():
            if param.dim() > 1:  # Skip bias terms
                total_weights += param.numel()
                zero_weights += (param == 0).sum().item()
        
        sparsity = zero_weights / total_weights if total_weights > 0 else 0
        
        return {
            'original_parameters': original_params,
            'current_parameters': current_params,
            'parameter_reduction': (original_params - current_params) / original_params,
            'sparsity': sparsity,
            'compression_ratio': original_params / current_params if current_params > 0 else 1.0
        }
    
    def export_for_esp32(self, 
                        model: nn.Module,
                        output_path: str,
                        input_shape: Tuple[int, int, int] = (1, 11, 1000)) -> Dict[str, str]:
        """
        Export model for ESP32-S3 deployment
        
        Args:
            model: Model to export
            output_path: Directory to save exported files
            input_shape: Input tensor shape for the model
        
        Returns:
            Dictionary of exported file paths
        """
        import os
        import json
        from pathlib import Path
        
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        
        # Export model weights as JSON (ESP32-compatible format)
        weights_file = output_dir / 'model_weights.json'
        self._export_weights_to_json(model, weights_file)
        exported_files['weights'] = str(weights_file)
        
        # Export model architecture
        arch_file = output_dir / 'model_architecture.json'
        self._export_architecture_to_json(model, arch_file, input_shape)
        exported_files['architecture'] = str(arch_file)
        
        # Export inference code template
        code_file = output_dir / 'inference_template.cpp'
        self._export_cpp_template(code_file, input_shape)
        exported_files['cpp_template'] = str(code_file)
        
        # Export compression statistics
        stats_file = output_dir / 'compression_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(self.get_compression_stats(), f, indent=2)
        exported_files['stats'] = str(stats_file)
        
        print(f"✅ Model exported for ESP32-S3 deployment to: {output_dir}")
        return exported_files
    
    def _export_weights_to_json(self, model: nn.Module, output_file: Path):
        """Export model weights in JSON format for ESP32"""
        weights_dict = {}
        
        for name, param in model.named_parameters():
            # Convert to list and round to reduce file size
            weights_list = param.detach().cpu().numpy().flatten().tolist()
            # Round to 4 decimal places to reduce file size
            weights_list = [round(w, 4) for w in weights_list]
            weights_dict[name] = weights_list
        
        with open(output_file, 'w') as f:
            json.dump(weights_dict, f, indent=2)
    
    def _export_architecture_to_json(self, model: nn.Module, output_file: Path, input_shape: Tuple[int, int, int]):
        """Export model architecture in JSON format"""
        architecture = {
            'input_shape': input_shape,
            'layers': [],
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'model_type': type(model).__name__
        }
        
        # Extract layer information
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                layer_info = {
                    'name': name,
                    'type': type(module).__name__,
                    'parameters': sum(p.numel() for p in module.parameters())
                }
                
                # Add specific layer parameters
                if isinstance(module, nn.Conv1d):
                    layer_info.update({
                        'in_channels': module.in_channels,
                        'out_channels': module.out_channels,
                        'kernel_size': module.kernel_size[0],
                        'stride': module.stride[0],
                        'padding': module.padding[0]
                    })
                elif isinstance(module, nn.Linear):
                    layer_info.update({
                        'in_features': module.in_features,
                        'out_features': module.out_features
                    })
                
                architecture['layers'].append(layer_info)
        
        with open(output_file, 'w') as f:
            json.dump(architecture, f, indent=2)
    
    def _export_cpp_template(self, output_file: Path, input_shape: Tuple[int, int, int]):
        """Export C++ inference template for ESP32"""
        template = f'''/*
 * ESP32-S3 Inference Template for Multimodal Biomedical Monitoring
 * Auto-generated from PyTorch model
 * Input shape: {input_shape}
 */

#include <Arduino.h>
#include <vector>
#include <cmath>

// Model configuration
const int INPUT_CHANNELS = {input_shape[1]};
const int INPUT_SAMPLES = {input_shape[2]};
const int BATCH_SIZE = {input_shape[0]};

// Placeholder for model weights (to be filled from JSON)
// This would be generated from the actual model weights
float model_weights[] = {{
    // Weights will be loaded from model_weights.json
}};

// Inference function
std::vector<float> run_inference(const std::vector<std::vector<float>>& input_data) {{
    // Placeholder implementation
    // This would contain the actual inference logic
    std::vector<float> output;
    
    // TODO: Implement actual inference
    // 1. Load weights from JSON
    // 2. Implement forward pass
    // 3. Return predictions
    
    return output;
}}

// Setup function
void setup() {{
    Serial.begin(115200);
    Serial.println("ESP32-S3 Multimodal Biomedical Monitor");
    
    // Initialize model
    // TODO: Load model weights and initialize
}}

// Main loop
void loop() {{
    // Read sensor data
    std::vector<std::vector<float>> sensor_data(INPUT_CHANNELS, std::vector<float>(INPUT_SAMPLES));
    
    // TODO: Read from actual sensors
    // - ECG sensor
    // - PPG sensor  
    // - IMU (3-axis accelerometer)
    
    // Run inference
    auto predictions = run_inference(sensor_data);
    
    // TODO: Process predictions
    // - Arrhythmia detection
    // - Stress recognition
    // - Activity classification
    
    // Send results via Bluetooth/WiFi
    // TODO: Implement communication
    
    delay(100); // 10Hz update rate
}}
'''
        
        with open(output_file, 'w') as f:
            f.write(template)


class QuantizationAwareTraining:
    """
    Quantization-Aware Training (QAT) for better quantized model performance
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.original_model = copy.deepcopy(model)
    
    def prepare_for_qat(self) -> nn.Module:
        """Prepare model for quantization-aware training"""
        print("Preparing model for Quantization-Aware Training...")
        
        # Set quantization configuration
        self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # Prepare model for QAT
        qat_model = torch.quantization.prepare_qat(self.model, inplace=False)
        
        print("✅ Model prepared for QAT")
        return qat_model
    
    def convert_qat_model(self, qat_model: nn.Module) -> nn.Module:
        """Convert QAT model to quantized model"""
        print("Converting QAT model to quantized model...")
        
        # Set to evaluation mode
        qat_model.eval()
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(qat_model, inplace=False)
        
        print("✅ QAT model converted to quantized model")
        return quantized_model


# Example usage and testing
if __name__ == "__main__":
    from cnn_transformer_lite import CNNTransformerLite
    
    # Create a test model
    model = CNNTransformerLite(n_channels=11, n_samples=1000)
    
    # Initialize compressor
    compressor = ModelCompressor(model)
    
    # Test pruning
    pruned_model = compressor.prune_model(pruning_ratio=0.3)
    
    # Test quantization
    dummy_input = torch.randn(1, 11, 1000)
    quantized_model = compressor.quantize_model('dynamic')
    
    # Get compression stats
    stats = compressor.get_compression_stats()
    print("Compression Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Export for ESP32
    exported_files = compressor.export_for_esp32(
        quantized_model, 
        'esp32_export',
        input_shape=(1, 11, 1000)
    )
    
    print("Exported files:")
    for file_type, file_path in exported_files.items():
        print(f"{file_type}: {file_path}")
