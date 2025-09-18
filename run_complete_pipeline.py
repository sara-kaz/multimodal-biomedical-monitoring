#!/usr/bin/env python3
"""
Complete Pipeline Script for Edge Intelligence Multimodal Biomedical Monitoring
Demonstrates the entire workflow from data processing to ESP32-S3 deployment
"""

import argparse
import torch
import numpy as np
import json
import time
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.dataset_integration import UnifiedBiomedicalDataProcessor
from src.models.cnn_transformer_lite import CNNTransformerLite
from src.training.trainer import MultiTaskTrainer
from src.training.data_utils import create_data_loaders
from src.training.metrics import ModelEvaluator
from src.models.compression import ModelCompressor
from src.deployment.esp32_converter import ESP32Converter


def run_data_processing(data_paths: dict, output_dir: Path):
    """Step 1: Process datasets into unified format"""
    print("=" * 60)
    print("STEP 1: DATASET PROCESSING")
    print("=" * 60)
    
    processor = UnifiedBiomedicalDataProcessor(output_dir=str(output_dir / 'processed_data'))
    
    # Process all datasets
    unified_dataset, summary = processor.combine_all_datasets(
        ppg_dalia_path=data_paths.get('ppg_dalia'),
        mit_bih_path=data_paths.get('mit_bih'),
        wesad_path=data_paths.get('wesad')
    )
    
    print(f"✅ Processed {len(unified_dataset)} samples")
    print(f"📊 Summary: {summary}")
    
    return unified_dataset, summary


def run_model_training(unified_dataset, output_dir: Path, config: dict):
    """Step 2: Train multimodal model"""
    print("=" * 60)
    print("STEP 2: MODEL TRAINING")
    print("=" * 60)
    
    # Create data loaders
    data_loaders = create_data_loaders(
        unified_dataset,
        config['tasks'],
        batch_size=config['training']['batch_size'],
        num_workers=4
    )
    
    # Create model
    model = CNNTransformerLite(
        n_channels=config['model']['n_channels'],
        n_samples=config['model']['n_samples'],
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_layers=config['model']['num_layers'],
        task_configs=config['tasks']
    )
    
    print(f"🏗️  Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Initialize trainer
    trainer = MultiTaskTrainer(
        model=model,
        task_configs=config['tasks'],
        device='cuda' if torch.cuda.is_available() else 'cpu',
        loss_type=config['training']['loss_type'],
        optimizer_type=config['training']['optimizer_type'],
        learning_rate=config['training']['learning_rate'],
        save_dir=str(output_dir / 'checkpoints'),
        log_dir=str(output_dir / 'logs')
    )
    
    # Train model
    print("🚀 Starting training...")
    start_time = time.time()
    
    history = trainer.train(
        train_loader=data_loaders['train'],
        val_loader=data_loaders['val'],
        epochs=config['training']['epochs'],
        save_best=True,
        patience=20
    )
    
    training_time = time.time() - start_time
    print(f"✅ Training completed in {training_time/60:.2f} minutes")
    
    return model, trainer, data_loaders


def run_model_evaluation(model, data_loaders, output_dir: Path):
    """Step 3: Evaluate model performance"""
    print("=" * 60)
    print("STEP 3: MODEL EVALUATION")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model, device)
    
    # Evaluate on test set
    if 'test' in data_loaders:
        print("🔍 Evaluating on test set...")
        test_metrics = evaluator.evaluate_model(data_loaders['test'], {
            'activity': {'num_classes': 8},
            'stress': {'num_classes': 4},
            'arrhythmia': {'num_classes': 2}
        })
        
        # Print results
        print("\n📊 Test Results:")
        for task_name, task_metrics in test_metrics.items():
            print(f"  {task_name.title()}:")
            print(f"    Accuracy: {task_metrics['accuracy']:.4f}")
            print(f"    F1-Score: {task_metrics['f1_weighted']:.4f}")
            print(f"    AUC: {task_metrics['auc']:.4f}")
    
    # Benchmark performance
    print("\n⚡ Benchmarking performance...")
    timing_stats = evaluator.benchmark_inference_time()
    size_info = evaluator.analyze_model_size()
    
    print(f"  Inference Time: {timing_stats['mean_time_ms']:.2f} ms")
    print(f"  Model Size: {size_info['model_size_mb']:.2f} MB")
    print(f"  Parameters: {size_info['total_parameters']:,}")
    
    # Save results
    results = {
        'test_metrics': test_metrics if 'test' in data_loaders else {},
        'timing_stats': timing_stats,
        'size_info': size_info
    }
    
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return results


def run_model_compression(model, output_dir: Path, config: dict):
    """Step 4: Compress model for ESP32-S3 deployment"""
    print("=" * 60)
    print("STEP 4: MODEL COMPRESSION")
    print("=" * 60)
    
    # Initialize compressor
    compressor = ModelCompressor(model)
    
    # Apply pruning if enabled
    if config['compression']['pruning']['enabled']:
        print("✂️  Applying pruning...")
        pruned_model = compressor.prune_model(
            pruning_ratio=config['compression']['pruning']['ratio'],
            pruning_type=config['compression']['pruning']['type']
        )
        print(f"✅ Pruning applied")
    else:
        pruned_model = model
    
    # Apply quantization if enabled
    if config['compression']['quantization']['enabled']:
        print("🔢 Applying quantization...")
        quantized_model = compressor.quantize_model(
            quantization_type=config['compression']['quantization']['type'],
            target_modules=[torch.nn.Linear, torch.nn.Conv1d]
        )
        print(f"✅ Quantization applied")
    else:
        quantized_model = pruned_model
    
    # Get compression statistics
    stats = compressor.get_compression_stats()
    print(f"\n📊 Compression Statistics:")
    print(f"  Original Parameters: {stats['original_parameters']:,}")
    print(f"  Compressed Parameters: {stats['current_parameters']:,}")
    print(f"  Compression Ratio: {stats['compression_ratio']:.2f}x")
    print(f"  Sparsity: {stats['sparsity']:.2%}")
    
    # Save compressed model
    torch.save(quantized_model.state_dict(), output_dir / 'compressed_model.pth')
    
    return quantized_model, stats


def run_esp32_deployment(compressed_model, output_dir: Path, config: dict):
    """Step 5: Deploy model to ESP32-S3"""
    print("=" * 60)
    print("STEP 5: ESP32-S3 DEPLOYMENT")
    print("=" * 60)
    
    # Initialize converter
    converter = ESP32Converter(target_platform='esp32_s3')
    
    # Convert model
    print("🔄 Converting model for ESP32-S3...")
    deployment_files = converter.convert_model(
        model=compressed_model,
        input_shape=(1, 11, 1000),
        output_path=str(output_dir / 'esp32_deployment'),
        quantization_bits=config['deployment']['quantization_bits']
    )
    
    print(f"✅ ESP32-S3 deployment files generated:")
    for file_type, file_path in deployment_files.items():
        print(f"  {file_type}: {file_path}")
    
    return deployment_files


def run_performance_analysis(results: dict, output_dir: Path):
    """Step 6: Analyze performance and generate report"""
    print("=" * 60)
    print("STEP 6: PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    # Create performance report
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model_performance': results.get('evaluation', {}),
        'compression_stats': results.get('compression', {}),
        'deployment_info': results.get('deployment', {}),
        'recommendations': []
    }
    
    # Analyze performance
    if 'evaluation' in results:
        eval_results = results['evaluation']
        
        # Check if targets are met
        if 'timing_stats' in eval_results:
            inference_time = eval_results['timing_stats']['mean_time_ms']
            if inference_time > 100:
                report['recommendations'].append(
                    f"Inference time ({inference_time:.1f}ms) exceeds ESP32-S3 target (100ms)"
                )
            else:
                report['recommendations'].append(
                    f"Inference time ({inference_time:.1f}ms) meets ESP32-S3 target"
                )
        
        if 'size_info' in eval_results:
            model_size = eval_results['size_info']['model_size_mb']
            if model_size > 0.5:
                report['recommendations'].append(
                    f"Model size ({model_size:.2f}MB) exceeds ESP32-S3 target (0.5MB)"
                )
            else:
                report['recommendations'].append(
                    f"Model size ({model_size:.2f}MB) meets ESP32-S3 target"
                )
    
    # Save report
    with open(output_dir / 'performance_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print("📊 Performance Analysis Complete")
    print(f"📄 Report saved to: {output_dir / 'performance_report.json'}")
    
    return report


def main():
    parser = argparse.ArgumentParser(description='Run complete multimodal biomedical monitoring pipeline')
    
    # Data paths
    parser.add_argument('--ppg_dalia_path', type=str, 
                       default='data/ppg+dalia/PPG_FieldStudy',
                       help='Path to PPG-DaLiA dataset')
    parser.add_argument('--mit_bih_path', type=str,
                       default='data/mit-bih-arrhythmia-database-1.0.0',
                       help='Path to MIT-BIH dataset')
    parser.add_argument('--wesad_path', type=str,
                       default='data/WESAD',
                       help='Path to WESAD dataset')
    
    # Configuration
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='complete_pipeline_output',
                       help='Output directory for all results')
    
    # Pipeline steps
    parser.add_argument('--skip_data_processing', action='store_true',
                       help='Skip data processing step')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training step')
    parser.add_argument('--skip_evaluation', action='store_true',
                       help='Skip evaluation step')
    parser.add_argument('--skip_compression', action='store_true',
                       help='Skip compression step')
    parser.add_argument('--skip_deployment', action='store_true',
                       help='Skip deployment step')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    if Path(args.config).exists():
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default configuration
        config = {
            'model': {
                'n_channels': 11, 'n_samples': 1000, 'd_model': 64,
                'nhead': 4, 'num_layers': 2
            },
            'tasks': {
                'activity': {'num_classes': 8, 'weight': 1.0},
                'stress': {'num_classes': 4, 'weight': 1.0},
                'arrhythmia': {'num_classes': 2, 'weight': 1.0}
            },
            'training': {
                'epochs': 50, 'batch_size': 32, 'learning_rate': 1e-3,
                'loss_type': 'cross_entropy', 'optimizer_type': 'adamw'
            },
            'compression': {
                'pruning': {'enabled': True, 'ratio': 0.3, 'type': 'magnitude'},
                'quantization': {'enabled': True, 'type': 'dynamic', 'bits': 8}
            },
            'deployment': {'quantization_bits': 8}
        }
    
    print("🚀 Starting Complete Multimodal Biomedical Monitoring Pipeline")
    print(f"📁 Output directory: {output_dir}")
    print(f"⚙️  Configuration: {args.config}")
    
    # Initialize results
    results = {}
    
    # Step 1: Data Processing
    if not args.skip_data_processing:
        data_paths = {
            'ppg_dalia': args.ppg_dalia_path,
            'mit_bih': args.mit_bih_path,
            'wesad': args.wesad_path
        }
        unified_dataset, summary = run_data_processing(data_paths, output_dir)
        results['dataset'] = {'samples': len(unified_dataset), 'summary': summary}
    else:
        print("⏭️  Skipping data processing")
        # Load existing dataset
        dataset_path = output_dir / 'processed_data' / 'unified_dataset.pkl'
        if dataset_path.exists():
            import pickle
            with open(dataset_path, 'rb') as f:
                unified_dataset = pickle.load(f)
        else:
            print("❌ No existing dataset found. Please run data processing first.")
            return
    
    # Step 2: Model Training
    if not args.skip_training:
        model, trainer, data_loaders = run_model_training(unified_dataset, output_dir, config)
        results['training'] = {'model': model, 'trainer': trainer}
    else:
        print("⏭️  Skipping training")
        # Load existing model
        model_path = output_dir / 'checkpoints' / 'best_model.pth'
        if model_path.exists():
            model = CNNTransformerLite(
                n_channels=config['model']['n_channels'],
                n_samples=config['model']['n_samples'],
                d_model=config['model']['d_model'],
                nhead=config['model']['nhead'],
                num_layers=config['model']['num_layers'],
                task_configs=config['tasks']
            )
            checkpoint = torch.load(model_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Create data loaders
            data_loaders = create_data_loaders(unified_dataset, config['tasks'])
        else:
            print("❌ No existing model found. Please run training first.")
            return
    
    # Step 3: Model Evaluation
    if not args.skip_evaluation:
        evaluation_results = run_model_evaluation(model, data_loaders, output_dir)
        results['evaluation'] = evaluation_results
    else:
        print("⏭️  Skipping evaluation")
    
    # Step 4: Model Compression
    if not args.skip_compression:
        compressed_model, compression_stats = run_model_compression(model, output_dir, config)
        results['compression'] = compression_stats
    else:
        print("⏭️  Skipping compression")
        compressed_model = model
    
    # Step 5: ESP32-S3 Deployment
    if not args.skip_deployment:
        deployment_files = run_esp32_deployment(compressed_model, output_dir, config)
        results['deployment'] = deployment_files
    else:
        print("⏭️  Skipping deployment")
    
    # Step 6: Performance Analysis
    performance_report = run_performance_analysis(results, output_dir)
    results['performance_report'] = performance_report
    
    # Final summary
    print("=" * 60)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    print(f"📁 All results saved to: {output_dir}")
    print(f"📊 Performance report: {output_dir / 'performance_report.json'}")
    
    if 'evaluation' in results:
        eval_results = results['evaluation']
        if 'test_metrics' in eval_results:
            print(f"\n📈 Model Performance:")
            for task_name, task_metrics in eval_results['test_metrics'].items():
                print(f"  {task_name.title()}: {task_metrics['accuracy']:.4f} accuracy")
    
    if 'compression' in results:
        comp_stats = results['compression']
        print(f"\n🗜️  Compression: {comp_stats['compression_ratio']:.2f}x reduction")
    
    if 'deployment' in results:
        print(f"\n🚀 ESP32-S3 files generated in: {output_dir / 'esp32_deployment'}")
    
    print(f"\n💡 Recommendations:")
    for recommendation in performance_report.get('recommendations', []):
        print(f"  - {recommendation}")
    
    print(f"\n🎯 Ready for your thesis defense!")


if __name__ == "__main__":
    main()
