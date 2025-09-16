"""
Comprehensive Plot Generation for Multimodal Biomedical Dataset Paper
Generates all necessary plots for thesis/paper inclusion
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
from scipy import signal
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class PaperPlotGenerator:
    def __init__(self, data_file_path):
        self.data_file_path = Path(data_file_path)
        self.load_data()
        self.plots_dir = Path('plots')
        self.plots_dir.mkdir(exist_ok=True)
        
        # Set up plotting parameters
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
        plt.rcParams['legend.fontsize'] = 9
        
    def load_data(self):
        """Load the unified dataset"""
        with open(self.data_file_path, 'rb') as f:
            self.data = pickle.load(f)
        
        print(f"Loaded {len(self.data)} samples")
        
        # Load metadata
        metadata_path = self.data_file_path.parent / 'dataset_metadata.pkl'
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            self.metadata = {}
    
    def plot_signal_examples(self, n_samples_per_dataset=3, duration_sec=10):
        """Plot signal examples from each dataset and channel"""
        print("Generating signal example plots...")
        
        # Get samples from each dataset
        datasets = {}
        for sample in self.data:
            dataset = sample['dataset']
            if dataset not in datasets:
                datasets[dataset] = []
            if len(datasets[dataset]) < n_samples_per_dataset:
                datasets[dataset].append(sample)
        
        channel_names = ['ECG', 'PPG', 'Accel_X', 'Accel_Y', 'Accel_Z']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for dataset_name, samples in datasets.items():
            for sample_idx, sample in enumerate(samples):
                fig, axes = plt.subplots(5, 1, figsize=(12, 10))
                fig.suptitle(f'{dataset_name} - Sample {sample_idx+1} ({sample["subject_id"]})', 
                           fontsize=14, fontweight='bold')
                
                window_data = sample['window_data']
                time_axis = np.linspace(0, duration_sec, window_data.shape[1])
                
                for ch_idx, (ch_name, color) in enumerate(zip(channel_names, colors)):
                    signal_data = window_data[ch_idx, :]
                    
                    if not np.all(np.isnan(signal_data)):
                        axes[ch_idx].plot(time_axis, signal_data, color=color, linewidth=1.2)
                        axes[ch_idx].set_title(f'{ch_name} (fs=100Hz)', fontweight='bold')
                        axes[ch_idx].set_ylabel('Normalized Amplitude')
                        axes[ch_idx].grid(True, alpha=0.3)
                        axes[ch_idx].set_xlim(0, duration_sec)
                    else:
                        axes[ch_idx].text(0.5, 0.5, 'Data Not Available', 
                                        ha='center', va='center', transform=axes[ch_idx].transAxes,
                                        fontsize=12, color='red')
                        axes[ch_idx].set_title(f'{ch_name} (Not Available)', fontweight='bold')
                        axes[ch_idx].set_ylabel('Normalized Amplitude')
                        axes[ch_idx].grid(True, alpha=0.3)
                
                axes[-1].set_xlabel('Time (seconds)')
                plt.tight_layout()
                
                filename = f'signal_examples_{dataset_name.lower()}_sample{sample_idx+1}.png'
                plt.savefig(self.plots_dir / filename, dpi=300, bbox_inches='tight')
                plt.close()
        
        print(f"✅ Signal example plots saved to {self.plots_dir}")
    
    def plot_label_distributions(self):
        """Plot label distributions for all tasks"""
        print("Generating label distribution plots...")
        
        # Collect label data
        label_data = {'activity': {}, 'stress': {}, 'arrhythmia': {}}
        
        for sample in self.data:
            dataset = sample['dataset']
            labels = sample['labels']
            
            for task in label_data.keys():
                if task in labels:
                    one_hot = labels[task]
                    if np.sum(one_hot) > 0:
                        class_idx = np.argmax(one_hot)
                        if dataset not in label_data[task]:
                            label_data[task][dataset] = {}
                        if class_idx not in label_data[task][dataset]:
                            label_data[task][dataset][class_idx] = 0
                        label_data[task][dataset][class_idx] += 1
        
        # Create plots for each task
        for task, task_data in label_data.items():
            if not task_data:
                continue
                
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle(f'{task.title()} Label Distribution', fontsize=16, fontweight='bold')
            
            # Overall distribution
            overall_counts = {}
            for dataset_counts in task_data.values():
                for class_idx, count in dataset_counts.items():
                    if class_idx not in overall_counts:
                        overall_counts[class_idx] = 0
                    overall_counts[class_idx] += count
            
            # Get class names from metadata
            if task in self.metadata.get('label_encodings', {}):
                class_names = list(self.metadata['label_encodings'][task].keys())
            else:
                class_names = [f'Class {i}' for i in range(len(overall_counts))]
            
            classes = [class_names[i] for i in sorted(overall_counts.keys())]
            counts = [overall_counts[i] for i in sorted(overall_counts.keys())]
            
            bars1 = ax1.bar(classes, counts, color=sns.color_palette("husl", len(classes)))
            ax1.set_title('Overall Distribution')
            ax1.set_ylabel('Number of Samples')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, count in zip(bars1, counts):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                        f'{count}', ha='center', va='bottom', fontweight='bold')
            
            # Per-dataset distribution
            dataset_names = list(task_data.keys())
            x = np.arange(len(classes))
            width = 0.8 / len(dataset_names)
            
            for i, dataset in enumerate(dataset_names):
                dataset_counts = [task_data[dataset].get(j, 0) for j in sorted(overall_counts.keys())]
                ax2.bar(x + i * width, dataset_counts, width, label=dataset, alpha=0.8)
            
            ax2.set_title('Distribution by Dataset')
            ax2.set_ylabel('Number of Samples')
            ax2.set_xlabel('Classes')
            ax2.set_xticks(x + width * (len(dataset_names) - 1) / 2)
            ax2.set_xticklabels(classes, rotation=45)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / f'label_distribution_{task}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✅ Label distribution plots saved to {self.plots_dir}")
    
    def plot_signal_quality_analysis(self):
        """Plot signal quality metrics"""
        print("Generating signal quality analysis plots...")
        
        # Collect quality metrics
        quality_data = []
        
        for sample in self.data:
            dataset = sample['dataset']
            window_data = sample['window_data']
            channel_names = ['ECG', 'PPG', 'Accel_X', 'Accel_Y', 'Accel_Z']
            
            for ch_idx, ch_name in enumerate(channel_names):
                signal_data = window_data[ch_idx, :]
                
                if not np.all(np.isnan(signal_data)):
                    # Calculate metrics
                    signal_power = np.mean(signal_data ** 2)
                    noise_estimate = np.var(np.diff(signal_data))
                    snr_db = 10 * np.log10(signal_power / (noise_estimate + 1e-10))
                    
                    zero_crossings = len(np.where(np.diff(np.sign(signal_data)))[0])
                    rms_value = np.sqrt(np.mean(signal_data ** 2))
                    
                    quality_data.append({
                        'dataset': dataset,
                        'channel': ch_name,
                        'snr_db': snr_db,
                        'zero_crossings': zero_crossings,
                        'rms_value': rms_value,
                        'signal_length': len(signal_data)
                    })
        
        df_quality = pd.DataFrame(quality_data)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Signal Quality Analysis', fontsize=16, fontweight='bold')
        
        # SNR by dataset and channel
        snr_pivot = df_quality.pivot_table(values='snr_db', index='channel', 
                                          columns='dataset', aggfunc='mean')
        sns.heatmap(snr_pivot, annot=True, fmt='.1f', cmap='viridis', ax=axes[0,0])
        axes[0,0].set_title('Signal-to-Noise Ratio (dB)')
        axes[0,0].set_xlabel('Dataset')
        axes[0,0].set_ylabel('Channel')
        
        # Zero crossings by dataset and channel
        zc_pivot = df_quality.pivot_table(values='zero_crossings', index='channel', 
                                         columns='dataset', aggfunc='mean')
        sns.heatmap(zc_pivot, annot=True, fmt='.1f', cmap='plasma', ax=axes[0,1])
        axes[0,1].set_title('Zero Crossings (per 10s window)')
        axes[0,1].set_xlabel('Dataset')
        axes[0,1].set_ylabel('Channel')
        
        # SNR distribution by channel
        sns.boxplot(data=df_quality, x='channel', y='snr_db', hue='dataset', ax=axes[1,0])
        axes[1,0].set_title('SNR Distribution by Channel')
        axes[1,0].set_xlabel('Channel')
        axes[1,0].set_ylabel('SNR (dB)')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # RMS value distribution by channel
        sns.boxplot(data=df_quality, x='channel', y='rms_value', hue='dataset', ax=axes[1,1])
        axes[1,1].set_title('RMS Value Distribution by Channel')
        axes[1,1].set_xlabel('Channel')
        axes[1,1].set_ylabel('RMS Value')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'signal_quality_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Signal quality analysis plot saved to {self.plots_dir}")
    
    def plot_frequency_analysis(self, n_samples=5):
        """Plot frequency domain analysis for each channel type"""
        print("Generating frequency domain analysis plots...")
        
        # Collect samples from each dataset
        datasets = {}
        for sample in self.data:
            dataset = sample['dataset']
            if dataset not in datasets:
                datasets[dataset] = []
            if len(datasets[dataset]) < n_samples:
                datasets[dataset].append(sample)
        
        channel_names = ['ECG', 'PPG', 'Accel_X', 'Accel_Y', 'Accel_Z']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        fs = 100  # Sampling frequency
        
        for dataset_name, samples in datasets.items():
            fig, axes = plt.subplots(5, 1, figsize=(12, 15))
            fig.suptitle(f'{dataset_name} - Frequency Domain Analysis', 
                        fontsize=16, fontweight='bold')
            
            for ch_idx, (ch_name, color) in enumerate(zip(channel_names, colors)):
                # Calculate average power spectral density
                psd_avg = None
                valid_samples = 0
                
                for sample in samples:
                    signal_data = sample['window_data'][ch_idx, :]
                    if not np.all(np.isnan(signal_data)):
                        # Compute PSD
                        freqs, psd = signal.welch(signal_data, fs=fs, nperseg=256)
                        
                        if psd_avg is None:
                            psd_avg = psd
                        else:
                            psd_avg += psd
                        valid_samples += 1
                
                if valid_samples > 0:
                    psd_avg /= valid_samples
                    axes[ch_idx].semilogy(freqs, psd_avg, color=color, linewidth=1.5)
                    axes[ch_idx].set_title(f'{ch_name} Power Spectral Density')
                    axes[ch_idx].set_ylabel('Power/Frequency (dB/Hz)')
                    axes[ch_idx].grid(True, alpha=0.3)
                    axes[ch_idx].set_xlim(0, fs/2)
                else:
                    axes[ch_idx].text(0.5, 0.5, 'Data Not Available', 
                                    ha='center', va='center', transform=axes[ch_idx].transAxes,
                                    fontsize=12, color='red')
                    axes[ch_idx].set_title(f'{ch_name} (Not Available)')
                    axes[ch_idx].set_ylabel('Power/Frequency (dB/Hz)')
                    axes[ch_idx].grid(True, alpha=0.3)
            
            axes[-1].set_xlabel('Frequency (Hz)')
            plt.tight_layout()
            plt.savefig(self.plots_dir / f'frequency_analysis_{dataset_name.lower()}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✅ Frequency analysis plots saved to {self.plots_dir}")
    
    def plot_dataset_summary(self):
        """Plot dataset summary statistics"""
        print("Generating dataset summary plots...")
        
        # Collect dataset statistics
        dataset_stats = {}
        for sample in self.data:
            dataset = sample['dataset']
            if dataset not in dataset_stats:
                dataset_stats[dataset] = {
                    'total_samples': 0,
                    'subjects': set(),
                    'channels_available': {'ECG': 0, 'PPG': 0, 'Accel_X': 0, 'Accel_Y': 0, 'Accel_Z': 0}
                }
            
            dataset_stats[dataset]['total_samples'] += 1
            dataset_stats[dataset]['subjects'].add(sample['subject_id'])
            
            # Check channel availability
            window_data = sample['window_data']
            channel_names = ['ECG', 'PPG', 'Accel_X', 'Accel_Y', 'Accel_Z']
            for ch_idx, ch_name in enumerate(channel_names):
                if not np.all(np.isnan(window_data[ch_idx, :])):
                    dataset_stats[dataset]['channels_available'][ch_name] += 1
        
        # Convert sets to counts
        for dataset in dataset_stats:
            dataset_stats[dataset]['subject_count'] = len(dataset_stats[dataset]['subjects'])
            del dataset_stats[dataset]['subjects']
        
        # Create summary plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Dataset Summary Statistics', fontsize=16, fontweight='bold')
        
        # Sample count by dataset
        datasets = list(dataset_stats.keys())
        sample_counts = [dataset_stats[d]['total_samples'] for d in datasets]
        subject_counts = [dataset_stats[d]['subject_count'] for d in datasets]
        
        x = np.arange(len(datasets))
        width = 0.35
        
        bars1 = axes[0,0].bar(x - width/2, sample_counts, width, label='Samples', alpha=0.8)
        bars2 = axes[0,0].bar(x + width/2, subject_counts, width, label='Subjects', alpha=0.8)
        
        axes[0,0].set_title('Sample and Subject Counts by Dataset')
        axes[0,0].set_ylabel('Count')
        axes[0,0].set_xlabel('Dataset')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels(datasets)
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                axes[0,0].text(bar.get_x() + bar.get_width()/2., height + max(sample_counts)*0.01,
                              f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        # Channel availability heatmap
        channel_names = ['ECG', 'PPG', 'Accel_X', 'Accel_Y', 'Accel_Z']
        availability_matrix = []
        for dataset in datasets:
            row = []
            for channel in channel_names:
                total = dataset_stats[dataset]['total_samples']
                available = dataset_stats[dataset]['channels_available'][channel]
                row.append(available / total * 100)  # Percentage
            availability_matrix.append(row)
        
        im = axes[0,1].imshow(availability_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        axes[0,1].set_title('Channel Availability by Dataset (%)')
        axes[0,1].set_xlabel('Channel')
        axes[0,1].set_ylabel('Dataset')
        axes[0,1].set_xticks(range(len(channel_names)))
        axes[0,1].set_xticklabels(channel_names)
        axes[0,1].set_yticks(range(len(datasets)))
        axes[0,1].set_yticklabels(datasets)
        
        # Add text annotations
        for i in range(len(datasets)):
            for j in range(len(channel_names)):
                text = axes[0,1].text(j, i, f'{availability_matrix[i][j]:.1f}%',
                                    ha="center", va="center", color="black", fontweight='bold')
        
        # Sample distribution pie chart
        axes[1,0].pie(sample_counts, labels=datasets, autopct='%1.1f%%', startangle=90)
        axes[1,0].set_title('Sample Distribution Across Datasets')
        
        # Subject distribution pie chart
        axes[1,1].pie(subject_counts, labels=datasets, autopct='%1.1f%%', startangle=90)
        axes[1,1].set_title('Subject Distribution Across Datasets')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'dataset_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Dataset summary plot saved to {self.plots_dir}")
    
    def plot_correlation_analysis(self):
        """Plot correlation analysis between channels"""
        print("Generating correlation analysis plots...")
        
        # Collect data for correlation analysis
        datasets = {}
        for sample in self.data:
            dataset = sample['dataset']
            if dataset not in datasets:
                datasets[dataset] = []
            if len(datasets[dataset]) < 100:  # Limit samples for performance
                datasets[dataset].append(sample)
        
        channel_names = ['ECG', 'PPG', 'Accel_X', 'Accel_Y', 'Accel_Z']
        
        for dataset_name, samples in datasets.items():
            # Create correlation matrix
            correlation_data = []
            valid_samples = 0
            
            for sample in samples:
                window_data = sample['window_data']
                row = []
                valid_row = True
                
                for ch_idx, ch_name in enumerate(channel_names):
                    signal_data = window_data[ch_idx, :]
                    if not np.all(np.isnan(signal_data)):
                        row.append(signal_data)
                    else:
                        valid_row = False
                        break
                
                if valid_row and len(row) == len(channel_names):
                    correlation_data.append(np.concatenate(row))
                    valid_samples += 1
            
            if valid_samples > 0:
                correlation_data = np.array(correlation_data)
                
                # Calculate correlation matrix
                corr_matrix = np.corrcoef(correlation_data.T)
                
                # Reshape to channel-wise correlation
                n_channels = len(channel_names)
                samples_per_channel = correlation_data.shape[1] // n_channels
                
                channel_corr = np.zeros((n_channels, n_channels))
                for i in range(n_channels):
                    for j in range(n_channels):
                        start_i = i * samples_per_channel
                        end_i = (i + 1) * samples_per_channel
                        start_j = j * samples_per_channel
                        end_j = (j + 1) * samples_per_channel
                        
                        channel_corr[i, j] = np.mean(corr_matrix[start_i:end_i, start_j:end_j])
                
                # Plot correlation heatmap
                fig, ax = plt.subplots(1, 1, figsize=(10, 8))
                im = ax.imshow(channel_corr, cmap='RdBu_r', vmin=-1, vmax=1)
                
                ax.set_title(f'{dataset_name} - Channel Correlation Matrix', 
                           fontsize=14, fontweight='bold')
                ax.set_xlabel('Channel')
                ax.set_ylabel('Channel')
                ax.set_xticks(range(len(channel_names)))
                ax.set_xticklabels(channel_names)
                ax.set_yticks(range(len(channel_names)))
                ax.set_yticklabels(channel_names)
                
                # Add correlation values
                for i in range(len(channel_names)):
                    for j in range(len(channel_names)):
                        text = ax.text(j, i, f'{channel_corr[i, j]:.2f}',
                                     ha="center", va="center", color="black", fontweight='bold')
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Correlation Coefficient')
                
                plt.tight_layout()
                plt.savefig(self.plots_dir / f'correlation_analysis_{dataset_name.lower()}.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
        
        print(f"✅ Correlation analysis plots saved to {self.plots_dir}")
    
    def generate_all_plots(self):
        """Generate all plots for the paper"""
        print("🎨 Generating all plots for paper inclusion...")
        print("=" * 60)
        
        # Create plots directory
        self.plots_dir.mkdir(exist_ok=True)
        
        # Generate all plot types
        self.plot_signal_examples(n_samples_per_dataset=3, duration_sec=10)
        self.plot_label_distributions()
        self.plot_signal_quality_analysis()
        self.plot_frequency_analysis(n_samples=5)
        self.plot_dataset_summary()
        self.plot_correlation_analysis()
        
        print("=" * 60)
        print("✅ All plots generated successfully!")
        print(f"📁 Plots saved in: {self.plots_dir.absolute()}")
        
        # List all generated files
        plot_files = list(self.plots_dir.glob("*.png"))
        print(f"📊 Generated {len(plot_files)} plot files:")
        for file in sorted(plot_files):
            print(f"   - {file.name}")

def main():
    """Main function to generate all plots"""
    
    # Find the dataset file
    dataset_paths = [
        'processed_unified_dataset/unified_dataset.pkl',
        '../processed_unified_dataset/unified_dataset.pkl',
        '/Users/HP/Desktop/University/Thesis/Code/multimodal-biomedical-monitoring/processed_unified_dataset/unified_dataset.pkl'
    ]
    
    dataset_path = None
    for path in dataset_paths:
        if Path(path).exists():
            dataset_path = path
            break
    
    if dataset_path is None:
        print("❌ Dataset file not found!")
        return
    
    print(f"📂 Using dataset: {dataset_path}")
    
    # Generate all plots
    plot_generator = PaperPlotGenerator(dataset_path)
    plot_generator.generate_all_plots()

if __name__ == "__main__":
    main()
