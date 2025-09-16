"""
Create a comprehensive overview plot for the paper
Shows the complete dataset pipeline and key results
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import pickle
from pathlib import Path

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8')

def create_overview_plot():
    """Create a comprehensive overview plot for the paper"""
    
    # Load dataset for statistics
    dataset_path = 'processed_unified_dataset/unified_dataset.pkl'
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('Multimodal Biomedical Dataset: Comprehensive Overview', 
                fontsize=20, fontweight='bold', y=0.95)
    
    # Create a grid layout
    gs = fig.add_gridspec(4, 6, hspace=0.3, wspace=0.3)
    
    # 1. Dataset Overview (top left)
    ax1 = fig.add_subplot(gs[0, :2])
    
    # Calculate dataset statistics
    datasets = {}
    for sample in data:
        dataset = sample['dataset']
        if dataset not in datasets:
            datasets[dataset] = {'samples': 0, 'subjects': set()}
        datasets[dataset]['samples'] += 1
        datasets[dataset]['subjects'].add(sample['subject_id'])
    
    dataset_names = list(datasets.keys())
    sample_counts = [datasets[d]['samples'] for d in dataset_names]
    subject_counts = [len(datasets[d]['subjects']) for d in dataset_names]
    
    x = np.arange(len(dataset_names))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, sample_counts, width, label='Samples', alpha=0.8, color='skyblue')
    bars2 = ax1.bar(x + width/2, subject_counts, width, label='Subjects', alpha=0.8, color='lightcoral')
    
    ax1.set_title('Dataset Overview', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Count')
    ax1.set_xlabel('Dataset')
    ax1.set_xticks(x)
    ax1.set_xticklabels(dataset_names)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(sample_counts)*0.01,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Channel Availability (top right)
    ax2 = fig.add_subplot(gs[0, 2:4])
    
    # Calculate channel availability
    channel_names = ['ECG', 'PPG', 'Accel_X', 'Accel_Y', 'Accel_Z']
    availability_data = []
    
    for dataset in dataset_names:
        row = []
        for ch_idx, ch_name in enumerate(channel_names):
            available = 0
            total = 0
            for sample in data:
                if sample['dataset'] == dataset:
                    total += 1
                    if not np.all(np.isnan(sample['window_data'][ch_idx, :])):
                        available += 1
            row.append(available / total * 100 if total > 0 else 0)
        availability_data.append(row)
    
    im = ax2.imshow(availability_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    ax2.set_title('Channel Availability (%)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Channel')
    ax2.set_ylabel('Dataset')
    ax2.set_xticks(range(len(channel_names)))
    ax2.set_xticklabels(channel_names)
    ax2.set_yticks(range(len(dataset_names)))
    ax2.set_yticklabels(dataset_names)
    
    # Add text annotations
    for i in range(len(dataset_names)):
        for j in range(len(channel_names)):
            text = ax2.text(j, i, f'{availability_data[i][j]:.0f}%',
                          ha="center", va="center", color="black", fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Availability (%)')
    
    # 3. Label Distribution (top right)
    ax3 = fig.add_subplot(gs[0, 4:])
    
    # Calculate label distributions
    label_counts = {'activity': 0, 'stress': 0, 'arrhythmia': 0}
    for sample in data:
        for task in label_counts.keys():
            if task in sample['labels']:
                one_hot = sample['labels'][task]
                if np.sum(one_hot) > 0:
                    label_counts[task] += 1
    
    tasks = list(label_counts.keys())
    counts = list(label_counts.values())
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    
    bars = ax3.bar(tasks, counts, color=colors, alpha=0.8)
    ax3.set_title('Label Distribution', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Number of Samples')
    ax3.set_xlabel('Task')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, count in zip(bars, counts):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(counts)*0.01,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Signal Examples (middle row)
    ax4 = fig.add_subplot(gs[1, :3])
    
    # Plot sample signals from different datasets
    sample_datasets = ['PPG-DaLiA', 'WESAD', 'MIT-BIH']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, dataset_name in enumerate(sample_datasets):
        # Find first sample from this dataset
        sample = None
        for s in data:
            if s['dataset'] == dataset_name:
                sample = s
                break
        
        if sample:
            window_data = sample['window_data']
            time_axis = np.linspace(0, 10, window_data.shape[1])
            
            # Plot available channels
            for ch_idx, (ch_name, color) in enumerate(zip(channel_names, colors)):
                signal_data = window_data[ch_idx, :]
                if not np.all(np.isnan(signal_data)):
                    ax4.plot(time_axis + i*12, signal_data + ch_idx*2, 
                            color=color, linewidth=1, label=f'{ch_name}' if i == 0 else "")
    
    ax4.set_title('Signal Examples (10s windows)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Normalized Amplitude')
    ax4.set_xlabel('Time (seconds)')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    # 5. Signal Quality Metrics (middle right)
    ax5 = fig.add_subplot(gs[1, 3:])
    
    # Calculate signal quality metrics
    quality_data = {'dataset': [], 'channel': [], 'snr': []}
    
    for sample in data:
        dataset = sample['dataset']
        window_data = sample['window_data']
        
        for ch_idx, ch_name in enumerate(channel_names):
            signal_data = window_data[ch_idx, :]
            if not np.all(np.isnan(signal_data)):
                signal_power = np.mean(signal_data ** 2)
                noise_estimate = np.var(np.diff(signal_data))
                snr_db = 10 * np.log10(signal_power / (noise_estimate + 1e-10))
                
                quality_data['dataset'].append(dataset)
                quality_data['channel'].append(ch_name)
                quality_data['snr'].append(snr_db)
    
    # Create box plot for SNR by channel
    import pandas as pd
    df_quality = pd.DataFrame(quality_data)
    
    snr_by_channel = []
    channel_labels = []
    for ch_name in channel_names:
        ch_data = df_quality[df_quality['channel'] == ch_name]['snr']
        if len(ch_data) > 0:
            snr_by_channel.append(ch_data)
            channel_labels.append(ch_name)
    
    bp = ax5.boxplot(snr_by_channel, labels=channel_labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    ax5.set_title('Signal-to-Noise Ratio by Channel', fontsize=14, fontweight='bold')
    ax5.set_ylabel('SNR (dB)')
    ax5.set_xlabel('Channel')
    ax5.grid(True, alpha=0.3)
    ax5.tick_params(axis='x', rotation=45)
    
    # 6. Dataset Pipeline (bottom row)
    ax6 = fig.add_subplot(gs[2:, :])
    ax6.set_xlim(0, 10)
    ax6.set_ylim(0, 4)
    ax6.axis('off')
    
    # Draw pipeline diagram
    pipeline_steps = [
        ('Raw Datasets', 1, 3.5, 'lightblue'),
        ('PPG-DaLiA\n(15 subjects)', 1, 2.5, 'lightgreen'),
        ('WESAD\n(15 subjects)', 1, 1.5, 'lightcoral'),
        ('MIT-BIH\n(48 subjects)', 1, 0.5, 'lightyellow'),
    ]
    
    for step, x, y, color in pipeline_steps:
        rect = FancyBboxPatch((x-0.4, y-0.3), 0.8, 0.6, 
                             boxstyle="round,pad=0.1", 
                             facecolor=color, edgecolor='black', linewidth=2)
        ax6.add_patch(rect)
        ax6.text(x, y, step, ha='center', va='center', fontweight='bold', fontsize=10)
    
    # Arrows
    arrow_props = dict(arrowstyle='->', lw=2, color='black')
    ax6.annotate('', xy=(2.5, 2), xytext=(1.5, 2), arrowprops=arrow_props)
    
    # Processing steps
    processing_steps = [
        ('Resampling\nto 100 Hz', 3, 3.5, 'lightgray'),
        ('Normalization\n(Z-score)', 3, 2.5, 'lightgray'),
        ('Windowing\n(10s, 50% overlap)', 3, 1.5, 'lightgray'),
        ('Channel Mapping\n(5 channels)', 3, 0.5, 'lightgray'),
    ]
    
    for step, x, y, color in processing_steps:
        rect = FancyBboxPatch((x-0.4, y-0.3), 0.8, 0.6, 
                             boxstyle="round,pad=0.1", 
                             facecolor=color, edgecolor='black', linewidth=2)
        ax6.add_patch(rect)
        ax6.text(x, y, step, ha='center', va='center', fontweight='bold', fontsize=9)
    
    # Arrows
    ax6.annotate('', xy=(4.5, 2), xytext=(3.5, 2), arrowprops=arrow_props)
    
    # Final dataset
    final_rect = FancyBboxPatch((5.5-0.6, 1.5-0.4), 1.2, 0.8, 
                               boxstyle="round,pad=0.1", 
                               facecolor='gold', edgecolor='black', linewidth=3)
    ax6.add_patch(final_rect)
    ax6.text(5.5, 1.5, 'Unified Dataset\n60,510 windows\n5 channels × 1000 samples\n@ 100 Hz', 
             ha='center', va='center', fontweight='bold', fontsize=11)
    
    # Arrows
    ax6.annotate('', xy=(6.5, 2), xytext=(5.5, 2), arrowprops=arrow_props)
    
    # Applications
    applications = [
        ('Activity\nRecognition', 7.5, 3, 'lightpink'),
        ('Stress\nDetection', 7.5, 2, 'lightcyan'),
        ('Arrhythmia\nDetection', 7.5, 1, 'lightsteelblue'),
    ]
    
    for app, x, y, color in applications:
        rect = FancyBboxPatch((x-0.4, y-0.3), 0.8, 0.6, 
                             boxstyle="round,pad=0.1", 
                             facecolor=color, edgecolor='black', linewidth=2)
        ax6.add_patch(rect)
        ax6.text(x, y, app, ha='center', va='center', fontweight='bold', fontsize=10)
    
    # Add title for pipeline
    ax6.text(5, 3.8, 'Dataset Processing Pipeline', ha='center', va='center', 
             fontsize=16, fontweight='bold')
    
    # Add summary statistics
    ax6.text(9, 3.5, f'Total Samples: {len(data):,}', ha='left', va='center', 
             fontsize=12, fontweight='bold')
    ax6.text(9, 3, f'Total Subjects: {sum(subject_counts):,}', ha='left', va='center', 
             fontsize=12, fontweight='bold')
    ax6.text(9, 2.5, f'Window Length: 10 seconds', ha='left', va='center', 
             fontsize=12, fontweight='bold')
    ax6.text(9, 2, f'Sampling Rate: 100 Hz', ha='left', va='center', 
             fontsize=12, fontweight='bold')
    ax6.text(9, 1.5, f'Channels: 5', ha='left', va='center', 
             fontsize=12, fontweight='bold')
    ax6.text(9, 1, f'Overlap: 50%', ha='left', va='center', 
             fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('plots/dataset_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Dataset overview plot saved to plots/dataset_overview.png")

if __name__ == "__main__":
    create_overview_plot()
