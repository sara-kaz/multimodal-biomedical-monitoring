"""
Additional Specialized Plots for Biomedical Paper
Generates additional plots commonly needed in biomedical research papers
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
from scipy import signal
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class AdditionalPlotGenerator:
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
    
    def plot_heart_rate_variability(self, n_samples=10):
        """Plot heart rate variability analysis"""
        print("Generating heart rate variability plots...")
        
        # Find samples with ECG data
        ecg_samples = []
        for sample in self.data:
            if not np.all(np.isnan(sample['window_data'][0, :])):  # ECG is channel 0
                ecg_samples.append(sample)
                if len(ecg_samples) >= n_samples:
                    break
        
        if not ecg_samples:
            print("No ECG samples found for HRV analysis")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Heart Rate Variability Analysis', fontsize=16, fontweight='bold')
        
        # Collect HRV metrics
        rri_intervals = []
        hrv_metrics = {'rmssd': [], 'sdnn': [], 'pnn50': []}
        
        for i, sample in enumerate(ecg_samples[:5]):  # Analyze first 5 samples
            ecg_signal = sample['window_data'][0, :]
            
            # Simple R-peak detection (for demonstration)
            # In practice, you'd use more sophisticated methods
            peaks, _ = signal.find_peaks(ecg_signal, height=np.mean(ecg_signal) + 2*np.std(ecg_signal), 
                                       distance=50)
            
            if len(peaks) > 3:
                # Calculate RR intervals
                rr_intervals = np.diff(peaks) / 100.0  # Convert to seconds (100 Hz sampling)
                rri_intervals.extend(rr_intervals)
                
                # Calculate HRV metrics
                rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
                sdnn = np.std(rr_intervals)
                pnn50 = np.sum(np.abs(np.diff(rr_intervals)) > 0.05) / len(rr_intervals) * 100
                
                hrv_metrics['rmssd'].append(rmssd)
                hrv_metrics['sdnn'].append(sdnn)
                hrv_metrics['pnn50'].append(pnn50)
        
        # Plot RR interval time series
        if rri_intervals:
            time_axis = np.cumsum(rri_intervals)
            axes[0,0].plot(time_axis, rri_intervals, 'b-', linewidth=1.5)
            axes[0,0].set_title('RR Interval Time Series')
            axes[0,0].set_xlabel('Time (s)')
            axes[0,0].set_ylabel('RR Interval (s)')
            axes[0,0].grid(True, alpha=0.3)
        
        # Plot RR interval histogram
        if rri_intervals:
            axes[0,1].hist(rri_intervals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0,1].set_title('RR Interval Distribution')
            axes[0,1].set_xlabel('RR Interval (s)')
            axes[0,1].set_ylabel('Frequency')
            axes[0,1].grid(True, alpha=0.3)
        
        # Plot HRV metrics box plot
        if hrv_metrics['rmssd']:
            metrics_data = [hrv_metrics['rmssd'], hrv_metrics['sdnn'], hrv_metrics['pnn50']]
            metrics_labels = ['RMSSD (s)', 'SDNN (s)', 'pNN50 (%)']
            
            bp = axes[1,0].boxplot(metrics_data, labels=metrics_labels, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
            axes[1,0].set_title('HRV Metrics Distribution')
            axes[1,0].set_ylabel('Value')
            axes[1,0].grid(True, alpha=0.3)
        
        # Plot Poincaré plot
        if len(rri_intervals) > 1:
            axes[1,1].scatter(rri_intervals[:-1], rri_intervals[1:], alpha=0.6, s=20)
            axes[1,1].set_title('Poincaré Plot')
            axes[1,1].set_xlabel('RR(n) (s)')
            axes[1,1].set_ylabel('RR(n+1) (s)')
            axes[1,1].grid(True, alpha=0.3)
            
            # Add diagonal lines
            min_val = min(min(rri_intervals[:-1]), min(rri_intervals[1:]))
            max_val = max(max(rri_intervals[:-1]), max(rri_intervals[1:]))
            axes[1,1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'heart_rate_variability.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Heart rate variability plot saved to {self.plots_dir}")
    
    def plot_activity_recognition_examples(self):
        """Plot examples of different activities for activity recognition"""
        print("Generating activity recognition examples...")
        
        # Find samples with activity labels
        activity_samples = {}
        for sample in self.data:
            if 'activity' in sample['labels']:
                one_hot = sample['labels']['activity']
                if np.sum(one_hot) > 0:
                    class_idx = np.argmax(one_hot)
                    if class_idx not in activity_samples:
                        activity_samples[class_idx] = []
                    if len(activity_samples[class_idx]) < 3:  # 3 samples per activity
                        activity_samples[class_idx].append(sample)
        
        # Activity class names
        activity_names = ['sitting', 'walking', 'cycling', 'driving', 'working', 'stairs', 'table_soccer', 'lunch']
        
        # Create subplot for each activity
        n_activities = len(activity_samples)
        fig, axes = plt.subplots(n_activities, 1, figsize=(15, 3*n_activities))
        if n_activities == 1:
            axes = [axes]
        
        fig.suptitle('Activity Recognition Examples', fontsize=16, fontweight='bold')
        
        for class_idx, samples in activity_samples.items():
            if class_idx < len(activity_names):
                activity_name = activity_names[class_idx]
                
                # Plot accelerometer data (most relevant for activity recognition)
                sample = samples[0]  # Use first sample
                window_data = sample['window_data']
                time_axis = np.linspace(0, 10, window_data.shape[1])
                
                # Plot all three accelerometer channels
                axes[class_idx].plot(time_axis, window_data[2, :], 'r-', label='Accel_X', linewidth=1.2)
                axes[class_idx].plot(time_axis, window_data[3, :], 'g-', label='Accel_Y', linewidth=1.2)
                axes[class_idx].plot(time_axis, window_data[4, :], 'b-', label='Accel_Z', linewidth=1.2)
                
                axes[class_idx].set_title(f'{activity_name.title()} Activity', fontweight='bold')
                axes[class_idx].set_ylabel('Acceleration (normalized)')
                axes[class_idx].legend()
                axes[class_idx].grid(True, alpha=0.3)
                axes[class_idx].set_xlim(0, 10)
        
        axes[-1].set_xlabel('Time (seconds)')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'activity_recognition_examples.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Activity recognition examples plot saved to {self.plots_dir}")
    
    def plot_stress_detection_examples(self):
        """Plot examples of different stress states"""
        print("Generating stress detection examples...")
        
        # Find samples with stress labels
        stress_samples = {}
        for sample in self.data:
            if 'stress' in sample['labels']:
                one_hot = sample['labels']['stress']
                if np.sum(one_hot) > 0:
                    class_idx = np.argmax(one_hot)
                    if class_idx not in stress_samples:
                        stress_samples[class_idx] = []
                    if len(stress_samples[class_idx]) < 3:  # 3 samples per stress state
                        stress_samples[class_idx].append(sample)
        
        # Stress class names
        stress_names = ['baseline', 'stress', 'amusement', 'meditation']
        
        # Create subplot for each stress state
        n_states = len(stress_samples)
        fig, axes = plt.subplots(n_states, 1, figsize=(15, 3*n_states))
        if n_states == 1:
            axes = [axes]
        
        fig.suptitle('Stress Detection Examples', fontsize=16, fontweight='bold')
        
        for class_idx, samples in stress_samples.items():
            if class_idx < len(stress_names):
                stress_name = stress_names[class_idx]
                
                # Plot ECG and PPG (most relevant for stress detection)
                sample = samples[0]  # Use first sample
                window_data = sample['window_data']
                time_axis = np.linspace(0, 10, window_data.shape[1])
                
                # Plot ECG and PPG
                axes[class_idx].plot(time_axis, window_data[0, :], 'r-', label='ECG', linewidth=1.2)
                if not np.all(np.isnan(window_data[1, :])):
                    axes[class_idx].plot(time_axis, window_data[1, :], 'b-', label='PPG', linewidth=1.2)
                
                axes[class_idx].set_title(f'{stress_name.title()} State', fontweight='bold')
                axes[class_idx].set_ylabel('Amplitude (normalized)')
                axes[class_idx].legend()
                axes[class_idx].grid(True, alpha=0.3)
                axes[class_idx].set_xlim(0, 10)
        
        axes[-1].set_xlabel('Time (seconds)')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'stress_detection_examples.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Stress detection examples plot saved to {self.plots_dir}")
    
    def plot_arhythmia_examples(self):
        """Plot examples of normal vs abnormal heart rhythms"""
        print("Generating arrhythmia examples...")
        
        # Find samples with arrhythmia labels
        normal_samples = []
        abnormal_samples = []
        
        for sample in self.data:
            if 'arrhythmia' in sample['labels']:
                one_hot = sample['labels']['arrhythmia']
                if np.sum(one_hot) > 0:
                    class_idx = np.argmax(one_hot)
                    if class_idx == 0:  # normal
                        if len(normal_samples) < 5:
                            normal_samples.append(sample)
                    else:  # abnormal
                        if len(abnormal_samples) < 5:
                            abnormal_samples.append(sample)
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle('Arrhythmia Detection Examples', fontsize=16, fontweight='bold')
        
        # Plot normal rhythms
        for i, sample in enumerate(normal_samples[:3]):
            ecg_signal = sample['window_data'][0, :]
            time_axis = np.linspace(0, 10, len(ecg_signal))
            axes[0].plot(time_axis, ecg_signal + i*2, label=f'Normal {i+1}', linewidth=1.2)
        
        axes[0].set_title('Normal Heart Rhythms', fontweight='bold')
        axes[0].set_ylabel('ECG Amplitude (normalized)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(0, 10)
        
        # Plot abnormal rhythms
        for i, sample in enumerate(abnormal_samples[:3]):
            ecg_signal = sample['window_data'][0, :]
            time_axis = np.linspace(0, 10, len(ecg_signal))
            axes[1].plot(time_axis, ecg_signal + i*2, label=f'Abnormal {i+1}', linewidth=1.2)
        
        axes[1].set_title('Abnormal Heart Rhythms', fontweight='bold')
        axes[1].set_ylabel('ECG Amplitude (normalized)')
        axes[1].set_xlabel('Time (seconds)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(0, 10)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'arrhythmia_examples.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Arrhythmia examples plot saved to {self.plots_dir}")
    
    def plot_signal_statistics(self):
        """Plot comprehensive signal statistics"""
        print("Generating signal statistics plots...")
        
        # Collect statistics for each channel and dataset
        stats_data = []
        
        for sample in self.data:
            dataset = sample['dataset']
            window_data = sample['window_data']
            channel_names = ['ECG', 'PPG', 'Accel_X', 'Accel_Y', 'Accel_Z']
            
            for ch_idx, ch_name in enumerate(channel_names):
                signal_data = window_data[ch_idx, :]
                
                if not np.all(np.isnan(signal_data)):
                    stats_data.append({
                        'dataset': dataset,
                        'channel': ch_name,
                        'mean': np.mean(signal_data),
                        'std': np.std(signal_data),
                        'skewness': self._calculate_skewness(signal_data),
                        'kurtosis': self._calculate_kurtosis(signal_data),
                        'range': np.max(signal_data) - np.min(signal_data)
                    })
        
        df_stats = pd.DataFrame(stats_data)
        
        # Create comprehensive statistics plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comprehensive Signal Statistics', fontsize=16, fontweight='bold')
        
        # Mean values by channel and dataset
        sns.boxplot(data=df_stats, x='channel', y='mean', hue='dataset', ax=axes[0,0])
        axes[0,0].set_title('Mean Values by Channel')
        axes[0,0].set_xlabel('Channel')
        axes[0,0].set_ylabel('Mean Value')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Standard deviation by channel and dataset
        sns.boxplot(data=df_stats, x='channel', y='std', hue='dataset', ax=axes[0,1])
        axes[0,1].set_title('Standard Deviation by Channel')
        axes[0,1].set_xlabel('Channel')
        axes[0,1].set_ylabel('Standard Deviation')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Skewness by channel and dataset
        sns.boxplot(data=df_stats, x='channel', y='skewness', hue='dataset', ax=axes[0,2])
        axes[0,2].set_title('Skewness by Channel')
        axes[0,2].set_xlabel('Channel')
        axes[0,2].set_ylabel('Skewness')
        axes[0,2].tick_params(axis='x', rotation=45)
        
        # Kurtosis by channel and dataset
        sns.boxplot(data=df_stats, x='channel', y='kurtosis', hue='dataset', ax=axes[1,0])
        axes[1,0].set_title('Kurtosis by Channel')
        axes[1,0].set_xlabel('Channel')
        axes[1,0].set_ylabel('Kurtosis')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Range by channel and dataset
        sns.boxplot(data=df_stats, x='channel', y='range', hue='dataset', ax=axes[1,1])
        axes[1,1].set_title('Range by Channel')
        axes[1,1].set_xlabel('Channel')
        axes[1,1].set_ylabel('Range')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        # Signal-to-noise ratio by channel and dataset
        df_stats['snr'] = df_stats['mean'] / (df_stats['std'] + 1e-8)
        sns.boxplot(data=df_stats, x='channel', y='snr', hue='dataset', ax=axes[1,2])
        axes[1,2].set_title('Signal-to-Noise Ratio by Channel')
        axes[1,2].set_xlabel('Channel')
        axes[1,2].set_ylabel('SNR')
        axes[1,2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'signal_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Signal statistics plot saved to {self.plots_dir}")
    
    def _calculate_skewness(self, data):
        """Calculate skewness of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data):
        """Calculate kurtosis of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def generate_additional_plots(self):
        """Generate all additional plots"""
        print("🎨 Generating additional specialized plots...")
        print("=" * 60)
        
        self.plot_heart_rate_variability(n_samples=10)
        self.plot_activity_recognition_examples()
        self.plot_stress_detection_examples()
        self.plot_arhythmia_examples()
        self.plot_signal_statistics()
        
        print("=" * 60)
        print("✅ All additional plots generated successfully!")
        print(f"📁 Plots saved in: {self.plots_dir.absolute()}")

def main():
    """Main function to generate additional plots"""
    
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
    
    # Generate additional plots
    plot_generator = AdditionalPlotGenerator(dataset_path)
    plot_generator.generate_additional_plots()

if __name__ == "__main__":
    main()
