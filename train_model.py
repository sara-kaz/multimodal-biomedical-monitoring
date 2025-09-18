#!/usr/bin/env python3
"""
Main Training Script for Edge Intelligence Multimodal Biomedical Monitoring
Trains CNN/Transformer-Lite models for simultaneous biomedical signal classification
"""

import argparse
import torch
import numpy as np
import random
from pathlib import Path
import json
import yaml
from typing import Dict, Any

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent / 'src'))

from src.models.cnn_transformer_lite import CNNTransformerLite, create_model
from src.training.trainer import MultiTaskTrainer
from src.training.data_utils import create_data_loaders, load_processed_data
from src.training.metrics import ModelEvaluator


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            return yaml.safe_load(f)
        else:
            return json.load(f)


def create_model_from_config(config: Dict[str, Any]) -> torch.nn.Module:
    """Create model from configuration"""
    model_config = config['model']
    
    if model_config['type'] == 'cnn_transformer_lite':
        return CNNTransformerLite(
            n_channels=model_config.get('n_channels', 11),
            n_samples=model_config.get('n_samples', 1000),
            d_model=model_config.get('d_model', 64),
            nhead=model_config.get('nhead', 4),
            num_layers=model_config.get('num_layers', 2),
            dim_feedforward=model_config.get('dim_feedforward', 128),
            dropout=model_config.get('dropout', 0.1),
            task_configs=config['tasks']
        )
    else:
        return create_model(model_config['type'], **model_config.get('params', {}))


def main():
    parser = argparse.ArgumentParser(description='Train multimodal biomedical monitoring model')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to processed dataset')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Path to configuration file')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use for training (cpu, cuda, auto)')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='cnn_transformer_lite',
                       help='Type of model to train')
    parser.add_argument('--d_model', type=int, default=64,
                       help='Model dimension for transformer')
    parser.add_argument('--nhead', type=int, default=4,
                       help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of transformer layers')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory for models and logs')
    parser.add_argument('--experiment_name', type=str, default='multimodal_biomedical',
                       help='Name of the experiment')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--eval_only', action='store_true',
                       help='Only evaluate the model')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"🚀 Starting training with device: {device}")
    print(f"📁 Data path: {args.data_path}")
    print(f"⚙️  Config: {args.config}")
    
    # Load configuration
    if Path(args.config).exists():
        config = load_config(args.config)
        print(f"✅ Loaded configuration from {args.config}")
    else:
        # Create default configuration
        config = {
            'model': {
                'type': args.model_type,
                'n_channels': 11,
                'n_samples': 1000,
                'd_model': args.d_model,
                'nhead': args.nhead,
                'num_layers': args.num_layers,
                'dim_feedforward': 128,
                'dropout': 0.1
            },
            'tasks': {
                'activity': {'num_classes': 8, 'weight': 1.0},
                'stress': {'num_classes': 4, 'weight': 1.0},
                'arrhythmia': {'num_classes': 2, 'weight': 1.0}
            },
            'training': {
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'learning_rate': args.learning_rate,
                'loss_type': 'cross_entropy',
                'optimizer_type': 'adamw',
                'scheduler_type': 'cosine',
                'weight_decay': 1e-4,
                'warmup_epochs': 5
            }
        }
        print("⚠️  Using default configuration")
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Load processed data
    print("📊 Loading processed data...")
    try:
        processed_data = load_processed_data(args.data_path)
        print(f"✅ Loaded {len(processed_data)} samples")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Create data loaders
    print("🔄 Creating data loaders...")
    data_loaders = create_data_loaders(
        processed_data,
        config['tasks'],
        batch_size=config['training']['batch_size'],
        num_workers=4,
        augment_train=True
    )
    
    if not data_loaders:
        print("❌ No valid data loaders created")
        return
    
    print(f"✅ Created data loaders: {list(data_loaders.keys())}")
    
    # Create model
    print("🏗️  Creating model...")
    model = create_model_from_config(config)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    print("🎯 Initializing trainer...")
    trainer = MultiTaskTrainer(
        model=model,
        task_configs=config['tasks'],
        device=device,
        loss_type=config['training']['loss_type'],
        optimizer_type=config['training']['optimizer_type'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        scheduler_type=config['training']['scheduler_type'],
        warmup_epochs=config['training']['warmup_epochs'],
        save_dir=str(output_dir / 'checkpoints'),
        log_dir=str(output_dir / 'logs')
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"📂 Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    if not args.eval_only:
        # Train model
        print("🚀 Starting training...")
        training_history = trainer.train(
            train_loader=data_loaders['train'],
            val_loader=data_loaders['val'],
            epochs=config['training']['epochs'],
            save_best=True,
            patience=20
        )
        
        # Save training history
        with open(output_dir / 'training_history.json', 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            history_serializable = {}
            for key, value in training_history.items():
                if isinstance(value, list) and len(value) > 0:
                    if isinstance(value[0], dict):
                        # Handle metrics dictionaries
                        history_serializable[key] = value
                    else:
                        # Handle simple lists
                        history_serializable[key] = [float(v) if isinstance(v, (int, float, np.number)) else v for v in value]
                else:
                    history_serializable[key] = value
            json.dump(history_serializable, f, indent=2)
        
        print("✅ Training completed!")
    
    # Evaluate model
    if 'test' in data_loaders:
        print("🔍 Evaluating model...")
        test_metrics = trainer.evaluate(data_loaders['test'])
        
        # Save test results
        with open(output_dir / 'test_results.json', 'w') as f:
            json.dump(test_metrics, f, indent=2)
        
        print("✅ Evaluation completed!")
    
    # Benchmark model
    print("⚡ Benchmarking model performance...")
    benchmark_results = trainer.benchmark_model()
    
    # Save benchmark results
    with open(output_dir / 'benchmark_results.json', 'w') as f:
        json.dump(benchmark_results, f, indent=2)
    
    print("✅ Benchmarking completed!")
    
    # Export for deployment
    print("📦 Exporting model for deployment...")
    deployment_files = trainer.export_for_deployment(str(output_dir / 'deployment'))
    
    print("✅ Deployment export completed!")
    
    # Print summary
    print(f"\n🎉 Training pipeline completed!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Checkpoints: {output_dir / 'checkpoints'}")
    print(f"📈 Logs: {output_dir / 'logs'}")
    print(f"🚀 Deployment files: {output_dir / 'deployment'}")
    
    if 'test' in data_loaders:
        print(f"\n📊 Test Results Summary:")
        for task_name, task_metrics in test_metrics.items():
            print(f"  {task_name.title()}:")
            print(f"    Accuracy: {task_metrics['accuracy']:.4f}")
            print(f"    F1-Score: {task_metrics['f1_weighted']:.4f}")
            print(f"    AUC: {task_metrics['auc']:.4f}")
    
    print(f"\n⚡ Performance Summary:")
    print(f"  Model Size: {benchmark_results['model_size_mb']:.2f} MB")
    print(f"  Parameters: {benchmark_results['total_parameters']:,}")
    print(f"  Inference Time: {benchmark_results['mean_time_ms']:.2f} ms")


if __name__ == "__main__":
    main()
