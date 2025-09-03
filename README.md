# Edge Intelligence for Multimodal Biomedical Monitoring

**A Wearable Sensor-Fusion System (ECG/PPG + IMU) on ESP32-S3 Using Compressed CNN/Transformer-Lite**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)
![ESP32](https://img.shields.io/badge/ESP32--S3-Deployment-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🎯 **Project Overview**

This thesis project develops an **edge intelligence system** for real-time multimodal biomedical monitoring using wearable sensors. The system combines ECG, PPG, and IMU sensors with compressed neural networks deployed on ESP32-S3 microcontrollers for:

- **🫀 Arrhythmia Detection** (Normal vs. Abnormal)
- **🧠 Stress Recognition** (Baseline, Stress, Amusement, Meditation)  
- **🏃 Activity Classification** (Walking, Cycling, Sitting, etc.)

### **Key Innovation**
- **Multi-task Learning**: Simultaneous classification across three biomedical tasks
- **Sensor Fusion**: ECG + PPG + 3-axis accelerometer integration
- **Edge Deployment**: Compressed CNN/Transformer-Lite running on ESP32-S3
- **Real-time Processing**: 100Hz sampling with 10-second sliding windows

---

## 📊 **Datasets**

The project integrates and standardizes three major biomedical datasets:

| Dataset | Signals | Labels | Subjects | Sample Rate |
|---------|---------|---------|----------|-------------|
| **PPG-DaLiA** | PPG, ACC, (ECG) | 8 Activities | 15 | 64Hz → 100Hz |
| **MIT-BIH Arrhythmia** | ECG (2-lead) | 5 Arrhythmia Classes | 48 Records | 360Hz → 100Hz |
| **WESAD** | ECG, PPG, ACC, EDA, EMG | 4 Stress States | 15 | 700Hz → 100Hz |

### **Unified Format**
- **Sampling Rate**: 100Hz (standardized)
- **Window Size**: 10 seconds (1000 samples)
- **Shape**: `[5 channels × 1000 samples]`
- **Channels**: ECG, PPG, Accel_X, Accel_Y, Accel_Z
- **Labels**: One-hot encoded vectors for multi-task learning

---

## 🏗️ **System Architecture**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Wearable      │    │   ESP32-S3       │    │   Mobile App    │
│   Sensors       │    │   Edge AI        │    │   Dashboard     │
│                 │    │                  │    │                 │
│ ├─ ECG          │───▶│ ├─ Preprocessing  │───▶│ ├─ Heart Health │
│ ├─ PPG          │    │ ├─ CNN/Trans-Lite│    │ ├─ Stress Level │
│ └─ IMU (3-axis) │    │ └─ Multi-task ML │    │ └─ Activity     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### **Neural Network Pipeline**
1. **Signal Preprocessing**: Bandpass filtering, normalization, resampling
2. **Feature Extraction**: 1D CNN layers for temporal patterns
3. **Multi-modal Fusion**: Transformer-Lite attention mechanism
4. **Multi-task Heads**: Separate classifiers for each biomedical task
5. **Model Compression**: Quantization, pruning for ESP32-S3 deployment

---

## 🚀 **Quick Start**

### **1. Installation**
```bash
# Clone repository
git clone https://github.com/yourusername/edge-biomedical-monitoring.git
cd edge-biomedical-monitoring

# Install dependencies
pip install -r requirements.txt

# Download datasets (see Dataset Setup section)
```

### **2. Dataset Processing**
```bash
# Create unified dataset from all three sources
python process_datasets.py \
    --ppg_dalia_path datasets/PPG_DaLiA \
    --mit_bih_path datasets/mit-bih-arrhythmia \
    --wesad_path datasets/WESAD \
    --output_dir processed_unified_dataset
```

### **3. Model Training**
```bash
# Train multi-task CNN/Transformer-Lite
python train_model.py \
    --data_path processed_unified_dataset/unified_dataset.pkl \
    --model_type cnn_transformer_lite \
    --batch_size 32 \
    --epochs 100 \
    --tasks activity stress arrhythmia
```

### **4. Model Compression**
```bash
# Optimize for ESP32-S3 deployment
python compress_model.py \
    --model_path models/best_model.pth \
    --output_format onnx \
    --quantization int8 \
    --target_platform esp32_s3
```

### **5. ESP32-S3 Deployment**
```bash
# Generate embedded C code and flash to device
python deploy_to_esp32.py \
    --model_path models/compressed_model.onnx \
    --esp32_port /dev/ttyUSB0
```

---

## 📁 **Project Structure**

```
edge-biomedical-monitoring/
├── 📄 README.md                          # This file
├── 📄 requirements.txt                   # Python dependencies
├── 📁 datasets/                          # Raw datasets
│   ├── 📁 PPG_DaLiA/                    # PPG-DaLiA dataset
│   ├── 📁 mit-bih-arrhythmia/           # MIT-BIH Arrhythmia dataset
│   └── 📁 WESAD/                        # WESAD dataset
├── 📁 src/                              # Source code
│   ├── 📄 dataset_processor.py          # Unified dataset processing
│   ├── 📄 unified_dataset.py            # PyTorch dataset class
│   ├── 📄 models/                       # Neural network architectures
│   │   ├── 📄 cnn_transformer_lite.py  # Main model architecture
│   │   ├── 📄 compression.py           # Model optimization
│   │   └── 📄 quantization.py          # INT8 quantization
│   ├── 📄 training/                     # Training utilities
│   │   ├── 📄 trainer.py               # Multi-task training loop
│   │   ├── 📄 losses.py                # Loss functions
│   │   └── 📄 metrics.py               # Evaluation metrics
│   └── 📄 deployment/                   # ESP32-S3 deployment
│       ├── 📄 esp32_converter.py       # Model-to-C conversion
│       ├── 📄 esp32_firmware/           # ESP32-S3 C++ code
│       └── 📄 mobile_app/               # Flutter mobile app
├── 📁 processed_unified_dataset/        # Processed data (generated)
├── 📁 models/                           # Trained models (generated)
├── 📁 results/                          # Experiment results
├── 📁 notebooks/                        # Jupyter notebooks
│   ├── 📄 01_dataset_exploration.ipynb # Data analysis
│   ├── 📄 02_model_development.ipynb   # Model prototyping
│   └── 📄 03_performance_analysis.ipynb# Results visualization
└── 📁 docs/                            # Documentation
```

---

## 🔧 **Key Components**

### **Dataset Processing Pipeline**
- **Unified Format**: All datasets converted to `[5 channels × 1000 samples]` @ 100Hz
- **Signal Processing**: Bandpass filtering, z-score normalization, resampling
- **Windowing**: 10-second windows with 50% overlap for training
- **Label Encoding**: One-hot vectors for multi-task learning
- **Subject-wise Splits**: Prevents data leakage in train/val/test

### **Neural Network Architecture**
```python
class CNNTransformerLite(nn.Module):
    def __init__(self):
        # 1D CNN Feature Extractor
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(5, 32, kernel_size=7, padding=3),  # 5 input channels
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # ... more layers
        )
        
        # Lightweight Transformer Encoder
        self.transformer = nn.TransformerEncoder(...)
        
        # Multi-task Classification Heads
        self.activity_head = nn.Linear(hidden_dim, 8)    # 8 activity classes
        self.stress_head = nn.Linear(hidden_dim, 4)      # 4 stress classes  
        self.arrhythmia_head = nn.Linear(hidden_dim, 2)  # 2 arrhythmia classes
```

### **ESP32-S3 Optimization**
- **Model Compression**: 90%+ size reduction through pruning and quantization
- **Memory Management**: Optimized for 512KB RAM constraint
- **Real-time Inference**: <100ms latency for 10-second windows
- **Power Efficiency**: Optimized for battery-powered wearable devices

---

## 📊 **Expected Results**

### **Dataset Statistics**
| Metric | Value |
|--------|-------|
| Total Windows | ~50,000-100,000 |
| Subjects | 78 (across all datasets) |
| Training Data | ~70% (subject-wise split) |
| Validation Data | ~15% |
| Test Data | ~15% |

### **Model Performance Targets**
| Task | Accuracy | F1-Score | Inference Time |
|------|----------|----------|----------------|
| **Arrhythmia Detection** | >95% | >0.93 | <50ms |
| **Stress Recognition** | >85% | >0.82 | <50ms |
| **Activity Classification** | >90% | >0.88 | <50ms |

### **Edge Deployment Specs**
| Specification | Value |
|---------------|-------|
| **Model Size** | <500KB (compressed) |
| **RAM Usage** | <300KB |
| **Power Consumption** | <50mW (inference) |
| **Sampling Rate** | 100Hz continuous |
| **Battery Life** | >24 hours |

---

## 🔬 **Research Contributions**

### **1. Unified Multimodal Dataset**
- First standardized integration of PPG-DaLiA, MIT-BIH, and WESAD
- Common 100Hz sampling rate with synchronized multi-modal signals
- Subject-independent evaluation protocol

### **2. Multi-task Edge AI Architecture**
- Novel CNN/Transformer-Lite design for simultaneous biomedical tasks
- Attention-based sensor fusion for ECG, PPG, and IMU data
- Optimized for resource-constrained ESP32-S3 deployment

### **3. Real-world Validation**
- Cross-dataset generalization analysis
- Subject-independent performance evaluation  
- Real-time wearable device implementation

### **4. Open-source Contribution**
- Complete preprocessing pipeline for biomedical datasets
- ESP32-S3 deployment framework for neural networks
- Mobile app for real-time health monitoring

---

## 📚 **Usage Examples**

### **Dataset Analysis**
```python
from src.unified_dataset import UnifiedBiomedicalDataset
from src.dataset_processor import UnifiedBiomedicalDataProcessor

# Load processed dataset
dataset = UnifiedBiomedicalDataset('processed_unified_dataset/unified_dataset.pkl')
print(f"Total samples: {len(dataset)}")

# Visualize sample
dataset.plot_sample(idx=0)

# Create data loaders
splits = dataset.create_subject_splits()
data_loaders = create_data_loaders(dataset, splits, batch_size=32)
```

### **Model Training**
```python
from src.models.cnn_transformer_lite import CNNTransformerLite
from src.training.trainer import MultiTaskTrainer

# Initialize model and trainer
model = CNNTransformerLite(n_channels=5, n_samples=1000)
trainer = MultiTaskTrainer(model, data_loaders)

# Train with multi-task learning
trainer.train(epochs=100, tasks=['activity', 'stress', 'arrhythmia'])
```

### **Model Deployment**
```python
from src.deployment.esp32_converter import ESP32Converter

# Convert to ESP32-S3 compatible format
converter = ESP32Converter()
converter.convert_model('models/best_model.pth', 
                       output_path='esp32_firmware/model.c',
                       quantization='int8')
```

---

## 🛠️ **Development Setup**

### **Prerequisites**
- Python 3.8+
- ESP32-S3 development board
- PlatformIO or Arduino IDE
- CUDA GPU (recommended for training)

### **Development Installation**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install in development mode
pip install -e .
pip install -r requirements.txt

# Setup pre-commit hooks
pre-commit install
```

### **Testing**
```bash
# Run unit tests
pytest tests/

# Run integration tests
pytest tests/integration/

# Performance benchmarks
pytest tests/benchmarks/
```

---

## 📈 **Monitoring & Logging**

### **TensorBoard Integration**
```bash
# Start TensorBoard
tensorboard --logdir=runs/

# View at http://localhost:6006
```

### **Weights & Biases (Optional)**
```python
import wandb

# Initialize experiment tracking
wandb.init(project="edge-biomedical-monitoring")
wandb.config.update(hyperparameters)
```


### **Development Guidelines**
- Follow PEP 8 style guide
- Add unit tests for new features
- Update documentation for API changes
- Ensure ESP32-S3 compatibility for deployment code

---

## 📧 **Contact & Support**

- **Author**: Sara Khaled Aly
- **Email**: sara.ali.949@my.csun.edu
- **Institution**: California State University, Northridge
- **Thesis Advisor**: Dr. Shahnam Mirzaei

### **Citation**
```bibtex
@mastersthesis{your2024edge,
  title={Edge Intelligence for Multimodal Biomedical Monitoring: A Wearable Sensor-Fusion System on ESP32-S3},
  author={Your Name},
  school={Your University},
  year={2024},
  type={Master's Thesis}
}
```

---

## 🙏 **Acknowledgments**

- **PPG-DaLiA Dataset**: University of California, Irvine
- **MIT-BIH Arrhythmia Database**: PhysioNet, Harvard-MIT
- **WESAD Dataset**: University of Siegen
- **ESP32-S3**: Espressif Systems
- **PyTorch Community**: Facebook AI Research

---

**⚡ Ready to revolutionize wearable health monitoring with edge AI!** 🚀
