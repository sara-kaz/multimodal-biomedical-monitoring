#!/usr/bin/env python3
"""
Model Evaluation Script for Edge Intelligence Multimodal Biomedical Monitoring
Comprehensive evaluation against single-modality baselines and performance analysis
"""

import argparse
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent / 'src'))

from src.models.cnn_transformer_lite import CNNTransformerLite, create_model, SingleModalityBaseline
from src.training.data_utils import create_data_loaders, load_processed_data
from src.training.metrics import ModelEvaluator, compare_models
from src.models.compression import ModelCompressor


def evaluate_single_modality_baselines(data_loaders: Dict, device: str = 'cpu') -> Dict[str, Dict]:
    """
    Evaluate single-modality baseline models for comparison
    
    Args:
        data_loaders: Dictionary of data loaders
        device: Device to run evaluation on
    
    Returns:
        Dictionary of baseline results
    """
    print("🔍 Evaluating single-modality baselines...")
    
    baseline_results = {}
    
    # Create single-modality models
    baseline_models = {
        'ecg_only': SingleModalityBaseline(modality='ecg', num_classes=2),  # Arrhythmia
        'ppg_only': SingleModalityBaseline(modality='ppg', num_classes=4),  # Stress
        'accel_only': SingleModalityBaseline(modality='accel', num_classes=8)  # Activity
    }
    
    # Evaluate each baseline
    for model_name, model in baseline_models.items():
        print(f"  Evaluating {model_name}...")
        
        model.to(device)
        evaluator = ModelEvaluator(model, device)
        
        # Determine which data loader to use based on model type
        if 'ecg' in model_name:
            task_name = 'arrhythmia'
        elif 'ppg' in model_name:
            task_name = 'stress'
        else:
            task_name = 'activity'
        
        if task_name in data_loaders:
            # Create single-task data loader
            single_task_loader = create_single_task_loader(data_loaders[task_name], task_name)
            
            # Evaluate model
            metrics = evaluator.evaluate_model(single_task_loader, {task_name: {'num_classes': model.num_classes}})
            timing = evaluator.benchmark_inference_time()
            size_info = evaluator.analyze_model_size()
            
            baseline_results[model_name] = {
                'metrics': metrics,
                'timing': timing,
                'size': size_info
            }
    
    return baseline_results


def create_single_task_loader(data_loader, task_name: str):
    """Create single-task data loader from multi-task loader"""
    # This is a simplified implementation
    # In practice, you'd need to properly extract single-task data
    return data_loader


def evaluate_multimodal_model(model: torch.nn.Module, data_loaders: Dict, device: str = 'cpu') -> Dict:
    """
    Evaluate multimodal model
    
    Args:
        model: Multimodal model to evaluate
        data_loaders: Dictionary of data loaders
        device: Device to run evaluation on
    
    Returns:
        Dictionary of evaluation results
    """
    print("🔍 Evaluating multimodal model...")
    
    model.to(device)
    evaluator = ModelEvaluator(model, device)
    
    # Evaluate on test set
    test_metrics = {}
    if 'test' in data_loaders:
        test_metrics = evaluator.evaluate_model(data_loaders['test'], {
            'activity': {'num_classes': 8},
            'stress': {'num_classes': 4},
            'arrhythmia': {'num_classes': 2}
        })
    
    # Benchmark performance
    timing_stats = evaluator.benchmark_inference_time()
    size_info = evaluator.analyze_model_size()
    
    return {
        'metrics': test_metrics,
        'timing': timing_stats,
        'size': size_info
    }


def analyze_performance_tradeoffs(multimodal_results: Dict, baseline_results: Dict) -> Dict:
    """
    Analyze performance tradeoffs between multimodal and single-modality models
    
    Args:
        multimodal_results: Results from multimodal model
        baseline_results: Results from single-modality baselines
    
    Returns:
        Analysis results
    """
    print("📊 Analyzing performance tradeoffs...")
    
    analysis = {
        'accuracy_comparison': {},
        'efficiency_comparison': {},
        'size_comparison': {},
        'recommendations': []
    }
    
    # Compare accuracy
    if 'metrics' in multimodal_results:
        for task_name, task_metrics in multimodal_results['metrics'].items():
            if task_name in ['activity', 'stress', 'arrhythmia']:
                multimodal_acc = task_metrics['accuracy']
                
                # Find corresponding baseline
                baseline_key = None
                if task_name == 'arrhythmia':
                    baseline_key = 'ecg_only'
                elif task_name == 'stress':
                    baseline_key = 'ppg_only'
                elif task_name == 'activity':
                    baseline_key = 'accel_only'
                
                if baseline_key and baseline_key in baseline_results:
                    baseline_acc = baseline_results[baseline_key]['metrics'][task_name]['accuracy']
                    improvement = multimodal_acc - baseline_acc
                    
                    analysis['accuracy_comparison'][task_name] = {
                        'multimodal': multimodal_acc,
                        'single_modality': baseline_acc,
                        'improvement': improvement,
                        'improvement_pct': (improvement / baseline_acc) * 100
                    }
    
    # Compare efficiency
    multimodal_time = multimodal_results.get('timing', {}).get('mean_time_ms', 0)
    multimodal_size = multimodal_results.get('size', {}).get('model_size_mb', 0)
    
    for baseline_name, baseline_result in baseline_results.items():
        baseline_time = baseline_result.get('timing', {}).get('mean_time_ms', 0)
        baseline_size = baseline_result.get('size', {}).get('model_size_mb', 0)
        
        analysis['efficiency_comparison'][baseline_name] = {
            'multimodal_time_ms': multimodal_time,
            'baseline_time_ms': baseline_time,
            'time_overhead': multimodal_time - baseline_time,
            'multimodal_size_mb': multimodal_size,
            'baseline_size_mb': baseline_size,
            'size_overhead': multimodal_size - baseline_size
        }
    
    # Generate recommendations
    if analysis['accuracy_comparison']:
        avg_improvement = np.mean([
            comp['improvement_pct'] for comp in analysis['accuracy_comparison'].values()
        ])
        
        if avg_improvement > 5:
            analysis['recommendations'].append(
                f"Multimodal fusion provides {avg_improvement:.1f}% average accuracy improvement"
            )
        else:
            analysis['recommendations'].append(
                "Multimodal fusion provides minimal accuracy improvement"
            )
    
    if multimodal_time > 100:
        analysis['recommendations'].append(
            f"Inference time ({multimodal_time:.1f}ms) exceeds ESP32-S3 target (100ms)"
        )
    
    if multimodal_size > 0.5:
        analysis['recommendations'].append(
            f"Model size ({multimodal_size:.2f}MB) exceeds ESP32-S3 target (0.5MB)"
        )
    
    return analysis


def create_performance_plots(results: Dict, output_dir: Path):
    """Create performance comparison plots"""
    print("📈 Creating performance plots...")
    
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Accuracy comparison plot
    if 'accuracy_comparison' in results:
        tasks = list(results['accuracy_comparison'].keys())
        multimodal_accs = [results['accuracy_comparison'][task]['multimodal'] for task in tasks]
        baseline_accs = [results['accuracy_comparison'][task]['single_modality'] for task in tasks]
        
        plt.figure(figsize=(10, 6))
        x = np.arange(len(tasks))
        width = 0.35
        
        plt.bar(x - width/2, baseline_accs, width, label='Single Modality', alpha=0.8)
        plt.bar(x + width/2, multimodal_accs, width, label='Multimodal', alpha=0.8)
        
        plt.xlabel('Task')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Comparison: Single Modality vs Multimodal')
        plt.xticks(x, tasks)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Efficiency comparison plot
    if 'efficiency_comparison' in results:
        baselines = list(results['efficiency_comparison'].keys())
        multimodal_times = [results['efficiency_comparison'][baseline]['multimodal_time_ms'] for baseline in baselines]
        baseline_times = [results['efficiency_comparison'][baseline]['baseline_time_ms'] for baseline in baselines]
        
        plt.figure(figsize=(10, 6))
        x = np.arange(len(baselines))
        width = 0.35
        
        plt.bar(x - width/2, baseline_times, width, label='Single Modality', alpha=0.8)
        plt.bar(x + width/2, multimodal_times, width, label='Multimodal', alpha=0.8)
        
        plt.xlabel('Baseline Model')
        plt.ylabel('Inference Time (ms)')
        plt.title('Inference Time Comparison')
        plt.xticks(x, baselines, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'inference_time_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✅ Plots saved to: {plots_dir}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate multimodal biomedical monitoring model')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to processed dataset')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    
    # Evaluation arguments
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use for evaluation')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--compare_baselines', action='store_true',
                       help='Compare against single-modality baselines')
    
    # Other arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--create_plots', action='store_true',
                       help='Create performance comparison plots')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"🔍 Starting model evaluation with device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
        {
            'activity': {'num_classes': 8, 'weight': 1.0},
            'stress': {'num_classes': 4, 'weight': 1.0},
            'arrhythmia': {'num_classes': 2, 'weight': 1.0}
        },
        batch_size=args.batch_size,
        num_workers=4
    )
    
    # Load trained model
    print("🏗️  Loading trained model...")
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        model = CNNTransformerLite(
            n_channels=11,
            n_samples=1000,
            d_model=64,
            nhead=4,
            num_layers=2
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Evaluate multimodal model
    print("🔍 Evaluating multimodal model...")
    multimodal_results = evaluate_multimodal_model(model, data_loaders, device)
    
    # Evaluate single-modality baselines if requested
    baseline_results = {}
    if args.compare_baselines:
        baseline_results = evaluate_single_modality_baselines(data_loaders, device)
    
    # Analyze performance tradeoffs
    print("📊 Analyzing performance tradeoffs...")
    analysis_results = analyze_performance_tradeoffs(multimodal_results, baseline_results)
    
    # Create plots if requested
    if args.create_plots:
        create_performance_plots(analysis_results, output_dir)
    
    # Save results
    print("💾 Saving evaluation results...")
    
    # Save multimodal results
    with open(output_dir / 'multimodal_results.json', 'w') as f:
        json.dump(multimodal_results, f, indent=2, default=str)
    
    # Save baseline results
    if baseline_results:
        with open(output_dir / 'baseline_results.json', 'w') as f:
            json.dump(baseline_results, f, indent=2, default=str)
    
    # Save analysis results
    with open(output_dir / 'analysis_results.json', 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    
    # Print summary
    print(f"\n🎉 Evaluation completed!")
    print(f"📁 Results saved to: {output_dir}")
    
    if 'metrics' in multimodal_results:
        print(f"\n📊 Multimodal Model Results:")
        for task_name, task_metrics in multimodal_results['metrics'].items():
            print(f"  {task_name.title()}:")
            print(f"    Accuracy: {task_metrics['accuracy']:.4f}")
            print(f"    F1-Score: {task_metrics['f1_weighted']:.4f}")
            print(f"    AUC: {task_metrics['auc']:.4f}")
    
    if 'timing' in multimodal_results:
        print(f"\n⚡ Performance Summary:")
        print(f"  Inference Time: {multimodal_results['timing']['mean_time_ms']:.2f} ms")
        print(f"  Model Size: {multimodal_results['size']['model_size_mb']:.2f} MB")
        print(f"  Parameters: {multimodal_results['size']['total_parameters']:,}")
    
    if analysis_results.get('recommendations'):
        print(f"\n💡 Recommendations:")
        for recommendation in analysis_results['recommendations']:
            print(f"  - {recommendation}")


if __name__ == "__main__":
    main()
