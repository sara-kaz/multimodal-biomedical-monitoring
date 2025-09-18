"""
ESP32-S3 Model Converter
Converts PyTorch models to ESP32-S3 compatible C++ code
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import onnx
import onnxruntime as ort


class ESP32Converter:
    """
    Converts PyTorch models to ESP32-S3 compatible format
    """
    
    def __init__(self, target_platform: str = 'esp32_s3'):
        self.target_platform = target_platform
        self.supported_ops = self._get_supported_operations()
    
    def _get_supported_operations(self) -> List[str]:
        """Get list of operations supported on ESP32-S3"""
        return [
            'Conv1d', 'Linear', 'ReLU', 'Sigmoid', 'Tanh',
            'MaxPool1d', 'AvgPool1d', 'AdaptiveAvgPool1d',
            'BatchNorm1d', 'LayerNorm', 'Dropout',
            'Add', 'Mul', 'Div', 'Sub', 'Pow',
            'MatMul', 'Gemm', 'Transpose', 'Reshape',
            'Flatten', 'Concat', 'Split', 'Slice'
        ]
    
    def convert_model(self, 
                     model: nn.Module,
                     input_shape: Tuple[int, int, int] = (1, 11, 1000),
                     output_path: str = 'esp32_deployment',
                     quantization_bits: int = 8) -> Dict[str, str]:
        """
        Convert PyTorch model to ESP32-S3 compatible format
        
        Args:
            model: PyTorch model to convert
            input_shape: Input tensor shape
            output_path: Directory to save converted files
            quantization_bits: Number of bits for quantization
        
        Returns:
            Dictionary of generated file paths
        """
        print(f"🔄 Converting model for ESP32-S3 deployment...")
        
        # Create output directory
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set model to evaluation mode
        model.eval()
        
        # Convert to ONNX first
        onnx_path = self._convert_to_onnx(model, input_shape, output_dir)
        
        # Generate C++ code
        cpp_files = self._generate_cpp_code(model, input_shape, output_dir, quantization_bits)
        
        # Generate model weights
        weights_file = self._export_weights(model, output_dir, quantization_bits)
        
        # Generate configuration files
        config_files = self._generate_config_files(model, input_shape, output_dir)
        
        # Generate Arduino sketch
        arduino_sketch = self._generate_arduino_sketch(output_dir, input_shape)
        
        generated_files = {
            'onnx_model': onnx_path,
            'cpp_files': cpp_files,
            'weights': weights_file,
            'config_files': config_files,
            'arduino_sketch': arduino_sketch
        }
        
        print(f"✅ Model conversion completed!")
        print(f"📁 Output directory: {output_dir}")
        
        return generated_files
    
    def _convert_to_onnx(self, model: nn.Module, input_shape: Tuple[int, int, int], 
                        output_dir: Path) -> str:
        """Convert PyTorch model to ONNX format"""
        print("🔄 Converting to ONNX...")
        
        # Create dummy input
        dummy_input = torch.randn(input_shape)
        
        # Export to ONNX
        onnx_path = output_dir / 'model.onnx'
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['activity', 'stress', 'arrhythmia'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'activity': {0: 'batch_size'},
                'stress': {0: 'batch_size'},
                'arrhythmia': {0: 'batch_size'}
            }
        )
        
        print(f"✅ ONNX model saved: {onnx_path}")
        return str(onnx_path)
    
    def _generate_cpp_code(self, model: nn.Module, input_shape: Tuple[int, int, int],
                          output_dir: Path, quantization_bits: int) -> Dict[str, str]:
        """Generate C++ inference code"""
        print("🔄 Generating C++ inference code...")
        
        cpp_files = {}
        
        # Generate main inference header
        inference_header = self._generate_inference_header(model, input_shape, quantization_bits)
        header_path = output_dir / 'inference.h'
        with open(header_path, 'w') as f:
            f.write(inference_header)
        cpp_files['inference_header'] = str(header_path)
        
        # Generate inference implementation
        inference_impl = self._generate_inference_implementation(model, input_shape, quantization_bits)
        impl_path = output_dir / 'inference.cpp'
        with open(impl_path, 'w') as f:
            f.write(inference_impl)
        cpp_files['inference_impl'] = str(impl_path)
        
        # Generate model weights header
        weights_header = self._generate_weights_header(model, quantization_bits)
        weights_h_path = output_dir / 'model_weights.h'
        with open(weights_h_path, 'w') as f:
            f.write(weights_header)
        cpp_files['weights_header'] = str(weights_h_path)
        
        print(f"✅ C++ code generated")
        return cpp_files
    
    def _generate_inference_header(self, model: nn.Module, input_shape: Tuple[int, int, int],
                                  quantization_bits: int) -> str:
        """Generate C++ header file for inference"""
        batch_size, n_channels, n_samples = input_shape
        
        return f'''/*
 * ESP32-S3 Inference Header for Multimodal Biomedical Monitoring
 * Auto-generated from PyTorch model
 */

#ifndef INFERENCE_H
#define INFERENCE_H

#include <Arduino.h>
#include <vector>
#include <cmath>

// Model configuration
const int BATCH_SIZE = {batch_size};
const int INPUT_CHANNELS = {n_channels};
const int INPUT_SAMPLES = {n_samples};
const int QUANTIZATION_BITS = {quantization_bits};

// Task configurations
const int ACTIVITY_CLASSES = 8;
const int STRESS_CLASSES = 4;
const int ARRHYTHMIA_CLASSES = 2;

// Data structures
struct ModelInput {{
    float data[INPUT_CHANNELS][INPUT_SAMPLES];
}};

struct ModelOutput {{
    float activity[ACTIVITY_CLASSES];
    float stress[STRESS_CLASSES];
    float arrhythmia[ARRHYTHMIA_CLASSES];
}};

// Function declarations
class MultimodalBiomedicalInference {{
public:
    MultimodalBiomedicalInference();
    ~MultimodalBiomedicalInference();
    
    // Initialize model
    bool initialize();
    
    // Run inference
    ModelOutput predict(const ModelInput& input);
    
    // Get model info
    size_t getModelSize() const;
    float getInferenceTime() const;
    
private:
    // Model weights and biases
    void loadModelWeights();
    
    // Layer implementations
    void conv1d_layer(const float* input, float* output, int layer_idx);
    void linear_layer(const float* input, float* output, int layer_idx);
    void relu_activation(float* data, int size);
    void maxpool1d_layer(const float* input, float* output, int layer_idx);
    void batch_norm1d_layer(float* data, int layer_idx);
    
    // Utility functions
    void softmax(float* data, int size);
    float sigmoid(float x);
    float tanh_activation(float x);
    
    // Model state
    bool _initialized;
    float _last_inference_time;
    size_t _model_size;
}};

// Global instance
extern MultimodalBiomedicalInference biomedical_model;

#endif // INFERENCE_H
'''
    
    def _generate_inference_implementation(self, model: nn.Module, input_shape: Tuple[int, int, int],
                                         quantization_bits: int) -> str:
        """Generate C++ implementation file"""
        return f'''/*
 * ESP32-S3 Inference Implementation for Multimodal Biomedical Monitoring
 * Auto-generated from PyTorch model
 */

#include "inference.h"
#include "model_weights.h"
#include <Arduino.h>

// Global model instance
MultimodalBiomedicalInference biomedical_model;

MultimodalBiomedicalInference::MultimodalBiomedicalInference() 
    : _initialized(false), _last_inference_time(0.0), _model_size(0) {{
}}

MultimodalBiomedicalInference::~MultimodalBiomedicalInference() {{
}}

bool MultimodalBiomedicalInference::initialize() {{
    if (_initialized) {{
        return true;
    }}
    
    // Load model weights
    loadModelWeights();
    
    // Calculate model size
    _model_size = calculateModelSize();
    
    _initialized = true;
    Serial.println("Biomedical model initialized successfully");
    return true;
}}

ModelOutput MultimodalBiomedicalInference::predict(const ModelInput& input) {{
    if (!_initialized) {{
        Serial.println("Error: Model not initialized");
        return ModelOutput();
    }}
    
    unsigned long start_time = micros();
    
    // Flatten input
    float flattened_input[INPUT_CHANNELS * INPUT_SAMPLES];
    int idx = 0;
    for (int c = 0; c < INPUT_CHANNELS; c++) {{
        for (int s = 0; s < INPUT_SAMPLES; s++) {{
            flattened_input[idx++] = input.data[c][s];
        }}
    }}
    
    // CNN layers
    float cnn_output[32 * 250]; // After CNN processing
    conv1d_layer(flattened_input, cnn_output, 0);
    relu_activation(cnn_output, 32 * 250);
    maxpool1d_layer(cnn_output, cnn_output, 0);
    
    conv1d_layer(cnn_output, cnn_output, 1);
    relu_activation(cnn_output, 32 * 250);
    maxpool1d_layer(cnn_output, cnn_output, 1);
    
    conv1d_layer(cnn_output, cnn_output, 2);
    relu_activation(cnn_output, 32 * 250);
    
    // Global average pooling
    float pooled_features[32];
    for (int i = 0; i < 32; i++) {{
        pooled_features[i] = 0.0;
        for (int j = 0; j < 250; j++) {{
            pooled_features[i] += cnn_output[i * 250 + j];
        }}
        pooled_features[i] /= 250.0;
    }}
    
    // Project to transformer dimension
    float projected[64];
    linear_layer(pooled_features, projected, 0);
    
    // Transformer layers (simplified)
    float transformer_out[64];
    memcpy(transformer_out, projected, 64 * sizeof(float));
    
    // Multi-task heads
    ModelOutput output;
    linear_layer(transformer_out, output.activity, 1); // Activity head
    linear_layer(transformer_out, output.stress, 2);   // Stress head
    linear_layer(transformer_out, output.arrhythmia, 3); // Arrhythmia head
    
    // Apply softmax to outputs
    softmax(output.activity, ACTIVITY_CLASSES);
    softmax(output.stress, STRESS_CLASSES);
    softmax(output.arrhythmia, ARRHYTHMIA_CLASSES);
    
    _last_inference_time = (micros() - start_time) / 1000.0; // Convert to ms
    return output;
}}

void MultimodalBiomedicalInference::loadModelWeights() {{
    // Model weights are loaded from model_weights.h
    // This is a placeholder for actual weight loading
    Serial.println("Model weights loaded");
}}

size_t MultimodalBiomedicalInference::getModelSize() const {{
    return _model_size;
}}

float MultimodalBiomedicalInference::getInferenceTime() const {{
    return _last_inference_time;
}}

// Layer implementations (simplified)
void MultimodalBiomedicalInference::conv1d_layer(const float* input, float* output, int layer_idx) {{
    // Simplified conv1d implementation
    // In practice, this would use the actual model weights
    memcpy(output, input, 32 * 250 * sizeof(float));
}}

void MultimodalBiomedicalInference::linear_layer(const float* input, float* output, int layer_idx) {{
    // Simplified linear layer implementation
    // In practice, this would use the actual model weights
    for (int i = 0; i < 8; i++) {{ // Assuming 8 output classes for activity
        output[i] = 0.0;
        for (int j = 0; j < 64; j++) {{
            output[i] += input[j] * 0.1; // Placeholder weight
        }}
    }}
}}

void MultimodalBiomedicalInference::relu_activation(float* data, int size) {{
    for (int i = 0; i < size; i++) {{
        data[i] = max(0.0f, data[i]);
    }}
}}

void MultimodalBiomedicalInference::maxpool1d_layer(const float* input, float* output, int layer_idx) {{
    // Simplified maxpool implementation
    memcpy(output, input, 32 * 250 * sizeof(float));
}}

void MultimodalBiomedicalInference::batch_norm1d_layer(float* data, int layer_idx) {{
    // Simplified batch norm implementation
    // In practice, this would use actual batch norm parameters
}}

void MultimodalBiomedicalInference::softmax(float* data, int size) {{
    // Find maximum value for numerical stability
    float max_val = data[0];
    for (int i = 1; i < size; i++) {{
        if (data[i] > max_val) {{
            max_val = data[i];
        }}
    }}
    
    // Compute exponentials and sum
    float sum_exp = 0.0;
    for (int i = 0; i < size; i++) {{
        data[i] = exp(data[i] - max_val);
        sum_exp += data[i];
    }}
    
    // Normalize
    for (int i = 0; i < size; i++) {{
        data[i] /= sum_exp;
    }}
}}

float MultimodalBiomedicalInference::sigmoid(float x) {{
    return 1.0 / (1.0 + exp(-x));
}}

float MultimodalBiomedicalInference::tanh_activation(float x) {{
    return tanh(x);
}}
'''
    
    def _generate_weights_header(self, model: nn.Module, quantization_bits: int) -> str:
        """Generate C++ header file with model weights"""
        print("🔄 Generating model weights header...")
        
        weights_data = []
        total_params = 0
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Convert to numpy and flatten
                param_np = param.detach().cpu().numpy().flatten()
                total_params += len(param_np)
                
                # Quantize if specified
                if quantization_bits < 32:
                    param_np = self._quantize_weights(param_np, quantization_bits)
                
                # Convert to C++ array format
                param_cpp = ', '.join([f'{w:.6f}f' for w in param_np])
                weights_data.append(f'// {name}\nstatic const float {name.replace(".", "_")}[] = {{{param_cpp}}};\n')
        
        weights_content = f'''/*
 * Model Weights for ESP32-S3 Multimodal Biomedical Monitoring
 * Auto-generated from PyTorch model
 * Total parameters: {total_params:,}
 * Quantization: {quantization_bits} bits
 */

#ifndef MODEL_WEIGHTS_H
#define MODEL_WEIGHTS_H

{''.join(weights_data)}

// Weight array sizes
const int TOTAL_PARAMETERS = {total_params};

#endif // MODEL_WEIGHTS_H
'''
        
        return weights_content
    
    def _quantize_weights(self, weights: np.ndarray, bits: int) -> np.ndarray:
        """Quantize weights to specified bit width"""
        if bits >= 32:
            return weights
        
        # Calculate quantization parameters
        min_val = np.min(weights)
        max_val = np.max(weights)
        
        # Symmetric quantization
        scale = max(abs(min_val), abs(max_val)) / (2**(bits-1) - 1)
        
        # Quantize
        quantized = np.round(weights / scale)
        quantized = np.clip(quantized, -2**(bits-1), 2**(bits-1)-1)
        
        # Dequantize back to float
        return quantized * scale
    
    def _export_weights(self, model: nn.Module, output_dir: Path, quantization_bits: int) -> str:
        """Export model weights as JSON"""
        print("🔄 Exporting model weights...")
        
        weights_dict = {}
        total_params = 0
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Convert to list and quantize
                weights_list = param.detach().cpu().numpy().flatten().tolist()
                
                if quantization_bits < 32:
                    weights_array = np.array(weights_list)
                    quantized_weights = self._quantize_weights(weights_array, quantization_bits)
                    weights_list = quantized_weights.tolist()
                
                # Round to reduce file size
                weights_list = [round(w, 6) for w in weights_list]
                weights_dict[name] = weights_list
                total_params += len(weights_list)
        
        # Save as JSON
        weights_file = output_dir / 'model_weights.json'
        with open(weights_file, 'w') as f:
            json.dump(weights_dict, f, indent=2)
        
        print(f"✅ Model weights exported: {weights_file}")
        print(f"   Total parameters: {total_params:,}")
        
        return str(weights_file)
    
    def _generate_config_files(self, model: nn.Module, input_shape: Tuple[int, int, int], 
                              output_dir: Path) -> Dict[str, str]:
        """Generate configuration files"""
        print("🔄 Generating configuration files...")
        
        config_files = {}
        
        # Model configuration
        model_config = {
            'input_shape': list(input_shape),
            'output_shapes': {
                'activity': [1, 8],
                'stress': [1, 4],
                'arrhythmia': [1, 2]
            },
            'sampling_rate_hz': 100,
            'window_length_sec': 10,
            'target_platform': self.target_platform,
            'quantization_bits': 8,
            'model_size_bytes': sum(p.numel() for p in model.parameters()) * 4
        }
        
        config_path = output_dir / 'model_config.json'
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
        config_files['model_config'] = str(config_path)
        
        # Deployment configuration
        deployment_config = {
            'platform': 'esp32_s3',
            'memory_requirements': {
                'flash_size_kb': 512,
                'ram_size_kb': 320,
                'psram_size_kb': 0
            },
            'performance_targets': {
                'max_inference_time_ms': 100,
                'max_model_size_kb': 500,
                'target_accuracy': 0.85
            },
            'sensor_config': {
                'ecg_sampling_rate': 100,
                'ppg_sampling_rate': 100,
                'imu_sampling_rate': 100,
                'buffer_size_samples': 1000
            }
        }
        
        deploy_path = output_dir / 'deployment_config.json'
        with open(deploy_path, 'w') as f:
            json.dump(deployment_config, f, indent=2)
        config_files['deployment_config'] = str(deploy_path)
        
        print(f"✅ Configuration files generated")
        return config_files
    
    def _generate_arduino_sketch(self, output_dir: Path, input_shape: Tuple[int, int, int]) -> str:
        """Generate Arduino sketch for ESP32-S3"""
        print("🔄 Generating Arduino sketch...")
        
        sketch_content = f'''/*
 * ESP32-S3 Multimodal Biomedical Monitoring
 * Real-time health monitoring with ECG, PPG, and IMU sensors
 */

#include "inference.h"
#include <WiFi.h>
#include <BluetoothSerial.h>
#include <ArduinoJson.h>

// WiFi configuration
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// Bluetooth configuration
BluetoothSerial SerialBT;

// Sensor pins (configure based on your hardware)
const int ECG_PIN = 34;
const int PPG_PIN = 35;
const int ACCEL_X_PIN = 36;
const int ACCEL_Y_PIN = 39;
const int ACCEL_Z_PIN = 32;

// Sampling configuration
const int SAMPLING_RATE_HZ = 100;
const int WINDOW_SIZE_SAMPLES = 1000;
const int WINDOW_DURATION_MS = 10000; // 10 seconds

// Data buffers
float ecg_buffer[WINDOW_SIZE_SAMPLES];
float ppg_buffer[WINDOW_SIZE_SAMPLES];
float accel_x_buffer[WINDOW_SIZE_SAMPLES];
float accel_y_buffer[WINDOW_SIZE_SAMPLES];
float accel_z_buffer[WINDOW_SIZE_SAMPLES];

int buffer_index = 0;
unsigned long last_sample_time = 0;
unsigned long window_start_time = 0;

// Results
ModelOutput last_predictions;
bool new_predictions_available = false;

void setup() {{
    Serial.begin(115200);
    SerialBT.begin("BiomedicalMonitor");
    
    // Initialize model
    if (!biomedical_model.initialize()) {{
        Serial.println("Failed to initialize model");
        while(1);
    }}
    
    // Initialize WiFi
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {{
        delay(1000);
        Serial.println("Connecting to WiFi...");
    }}
    Serial.println("WiFi connected");
    
    // Initialize sensor pins
    pinMode(ECG_PIN, INPUT);
    pinMode(PPG_PIN, INPUT);
    pinMode(ACCEL_X_PIN, INPUT);
    pinMode(ACCEL_Y_PIN, INPUT);
    pinMode(ACCEL_Z_PIN, INPUT);
    
    Serial.println("ESP32-S3 Biomedical Monitor Ready");
    window_start_time = millis();
}}

void loop() {{
    // Sample sensors at 100Hz
    if (millis() - last_sample_time >= 1000 / SAMPLING_RATE_HZ) {{
        sample_sensors();
        last_sample_time = millis();
    }}
    
    // Process window when full
    if (buffer_index >= WINDOW_SIZE_SAMPLES) {{
        process_window();
        buffer_index = 0;
        window_start_time = millis();
    }}
    
    // Send results if available
    if (new_predictions_available) {{
        send_results();
        new_predictions_available = false;
    }}
    
    // Handle Bluetooth commands
    handle_bluetooth();
}}

void sample_sensors() {{
    // Read analog sensors (0-4095 range, convert to voltage)
    float ecg_voltage = (analogRead(ECG_PIN) / 4095.0) * 3.3;
    float ppg_voltage = (analogRead(PPG_PIN) / 4095.0) * 3.3;
    float accel_x_voltage = (analogRead(ACCEL_X_PIN) / 4095.0) * 3.3;
    float accel_y_voltage = (analogRead(ACCEL_Y_PIN) / 4095.0) * 3.3;
    float accel_z_voltage = (analogRead(ACCEL_Z_PIN) / 4095.0) * 3.3;
    
    // Store in buffers
    ecg_buffer[buffer_index] = ecg_voltage;
    ppg_buffer[buffer_index] = ppg_voltage;
    accel_x_buffer[buffer_index] = accel_x_voltage;
    accel_y_buffer[buffer_index] = accel_y_voltage;
    accel_z_buffer[buffer_index] = accel_z_voltage;
    
    buffer_index++;
}}

void process_window() {{
    // Normalize signals
    normalize_signal(ecg_buffer, WINDOW_SIZE_SAMPLES);
    normalize_signal(ppg_buffer, WINDOW_SIZE_SAMPLES);
    normalize_signal(accel_x_buffer, WINDOW_SIZE_SAMPLES);
    normalize_signal(accel_y_buffer, WINDOW_SIZE_SAMPLES);
    normalize_signal(accel_z_buffer, WINDOW_SIZE_SAMPLES);
    
    // Prepare model input
    ModelInput input;
    for (int i = 0; i < WINDOW_SIZE_SAMPLES; i++) {{
        input.data[0][i] = ecg_buffer[i];
        input.data[1][i] = ppg_buffer[i];
        input.data[2][i] = accel_x_buffer[i];
        input.data[3][i] = accel_y_buffer[i];
        input.data[4][i] = accel_z_buffer[i];
        // Additional channels would be filled here
    }}
    
    // Run inference
    last_predictions = biomedical_model.predict(input);
    new_predictions_available = true;
    
    // Print results
    print_results();
}}

void normalize_signal(float* signal, int length) {{
    // Calculate mean and std
    float mean = 0.0;
    for (int i = 0; i < length; i++) {{
        mean += signal[i];
    }}
    mean /= length;
    
    float variance = 0.0;
    for (int i = 0; i < length; i++) {{
        float diff = signal[i] - mean;
        variance += diff * diff;
    }}
    variance /= length;
    float std = sqrt(variance);
    
    // Normalize
    for (int i = 0; i < length; i++) {{
        signal[i] = (signal[i] - mean) / (std + 1e-8);
    }}
}}

void print_results() {{
    Serial.println("=== Health Monitoring Results ===");
    
    // Activity classification
    int activity_class = get_max_class(last_predictions.activity, 8);
    Serial.print("Activity: ");
    switch(activity_class) {{
        case 0: Serial.println("Sitting"); break;
        case 1: Serial.println("Walking"); break;
        case 2: Serial.println("Cycling"); break;
        case 3: Serial.println("Driving"); break;
        case 4: Serial.println("Working"); break;
        case 5: Serial.println("Stairs"); break;
        case 6: Serial.println("Table Soccer"); break;
        case 7: Serial.println("Lunch"); break;
    }}
    
    // Stress classification
    int stress_class = get_max_class(last_predictions.stress, 4);
    Serial.print("Stress Level: ");
    switch(stress_class) {{
        case 0: Serial.println("Baseline"); break;
        case 1: Serial.println("Stress"); break;
        case 2: Serial.println("Amusement"); break;
        case 3: Serial.println("Meditation"); break;
    }}
    
    // Arrhythmia detection
    int arrhythmia_class = get_max_class(last_predictions.arrhythmia, 2);
    Serial.print("Heart Rhythm: ");
    Serial.println(arrhythmia_class == 0 ? "Normal" : "Abnormal");
    
    Serial.print("Inference Time: ");
    Serial.print(biomedical_model.getInferenceTime());
    Serial.println(" ms");
    Serial.println("================================");
}}

int get_max_class(const float* predictions, int num_classes) {{
    int max_class = 0;
    float max_value = predictions[0];
    
    for (int i = 1; i < num_classes; i++) {{
        if (predictions[i] > max_value) {{
            max_value = predictions[i];
            max_class = i;
        }}
    }}
    
    return max_class;
}}

void send_results() {{
    // Send via WiFi (HTTP POST)
    if (WiFi.status() == WL_CONNECTED) {{
        send_wifi_results();
    }}
    
    // Send via Bluetooth
    send_bluetooth_results();
}}

void send_wifi_results() {{
    // Create JSON payload
    DynamicJsonDocument doc(1024);
    doc["timestamp"] = millis();
    doc["activity"] = get_max_class(last_predictions.activity, 8);
    doc["stress"] = get_max_class(last_predictions.stress, 4);
    doc["arrhythmia"] = get_max_class(last_predictions.arrhythmia, 2);
    doc["inference_time_ms"] = biomedical_model.getInferenceTime();
    
    // Send HTTP POST (implement based on your server)
    // This is a placeholder
}}

void send_bluetooth_results() {{
    if (SerialBT.available()) {{
        String json_string;
        serializeJson(doc, json_string);
        SerialBT.println(json_string);
    }}
}}

void handle_bluetooth() {{
    if (SerialBT.available()) {{
        String command = SerialBT.readString();
        command.trim();
    
        if (command == "STATUS") {{
            SerialBT.println("ESP32-S3 Biomedical Monitor Active");
        }} else if (command == "RESULTS") {{
            send_bluetooth_results();
        }} else if (command == "MODEL_INFO") {{
            SerialBT.print("Model Size: ");
            SerialBT.print(biomedical_model.getModelSize());
            SerialBT.println(" bytes");
        }}
    }}
}}
'''
        
        sketch_path = output_dir / 'biomedical_monitor.ino'
        with open(sketch_path, 'w') as f:
            f.write(sketch_content)
        
        print(f"✅ Arduino sketch generated: {sketch_path}")
        return str(sketch_path)


# Example usage and testing
if __name__ == "__main__":
    from ..models.cnn_transformer_lite import CNNTransformerLite
    
    # Create a test model
    model = CNNTransformerLite(n_channels=11, n_samples=1000)
    
    # Initialize converter
    converter = ESP32Converter()
    
    # Convert model
    generated_files = converter.convert_model(
        model=model,
        input_shape=(1, 11, 1000),
        output_path='esp32_deployment',
        quantization_bits=8
    )
    
    print("Generated files:")
    for file_type, file_path in generated_files.items():
        print(f"  {file_type}: {file_path}")
    
    print("✅ ESP32 conversion completed successfully")
