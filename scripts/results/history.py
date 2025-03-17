import json
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from collections import defaultdict
import argparse
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({
    'font.size': 15,           # Base font size
    'axes.labelsize': 15,      # Axis labels
    'axes.titlesize': 19,     # Title size
    'xtick.labelsize': 13,     # X tick labels
    'ytick.labelsize': 13,     # Y tick labels
    'legend.fontsize': 17,     # Legend text
    'legend.title_fontsize': 17 # Legend title
})

# Define method labels and colors
METRIC_NAMES = {
    'auc': 'AUC',
    'categ_div@5': 'Category diversity@5'
}

METHOD_LABELS = {
    "base": "Pretrained",
    "embed_4": "In-context TS",
    "resample_4": "Prior sampling"
}

COLORS = {
    'Pretrained': '#7f8c8d',              # Gray for baseline
    'Prior sampling': '#3498db',          # Blue for worse performance
    'In-context TS': '#2ecc71',           # Green for better performance
}

# Define colormaps for each method
METHOD_COLORMAPS = {
    'Pretrained': 'Greys',
    'Prior sampling': 'Blues',
    'In-context TS': 'Greens'
}

def get_method_label(method):
    return METHOD_LABELS.get(method, method)

def get_method_color(method):
    label = get_method_label(method)
    return COLORS.get(label, '#000000')

def extract_method_name(filepath):
    # Extract method name from path like 'logs/eval_scatter/runs/nrms_mindsmall_pretrainedemb_celoss_bertsent/embed_4/metrics.json'
    parts = filepath.split('/')
    return parts[-2]  # This should be the method name

def extract_metrics_by_method(json_files):
    metrics_by_method = defaultdict(dict)
    buckets = set()
    
    for file in json_files:
        method = extract_method_name(file)
        with open(file, 'r') as f:
            data = json.load(f)
            
        method_metrics = defaultdict(dict)
        
        # Extract metrics for each bucket
        for key, value in data.items():
            if 'hist_bucket' in key:
                bucket = key.split('/')[0].split('_')[-1]
                metric_type = key.split('/')[-1]
                
                if metric_type in ['auc', 'categ_div@5']:
                    method_metrics[bucket][metric_type] = value
                    buckets.add(bucket)
        
        metrics_by_method[method] = method_metrics
    
    # Sort buckets by their starting value
    sorted_buckets = sorted(list(buckets), key=lambda x: float(x.split('-')[0]))
    return metrics_by_method, sorted_buckets

def create_plot(metrics_by_method, buckets, metric_name, output_filename, save=True):
    plt.figure(figsize=(8, 6))
    
    # Convert bucket names to numeric values for x-axis
    bucket_values = [float(bucket.split('-')[0]) for bucket in buckets]
    
    for method, metrics in metrics_by_method.items():
        values = []
        for bucket in buckets:
            if bucket in metrics and metric_name in metrics[bucket]:
                values.append(metrics[bucket][metric_name])
            else:
                values.append(0)
        
        plt.plot(bucket_values, values, marker='o', markersize=15, linewidth=3, linestyle='-',
                label=get_method_label(method), color=get_method_color(method))
    
    plt.xlabel('History length buckets')
    plt.ylabel(METRIC_NAMES[metric_name])
    plt.title(f'{METRIC_NAMES[metric_name]} over history length buckets')
    
    # Set x-axis ticks to bucket names
    plt.xticks(bucket_values, buckets, rotation=45)
    
    # Add legend
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    if save:
        # Ensure figs directory exists
        os.makedirs('figs', exist_ok=True)
        plt.savefig(os.path.join('figs', output_filename), dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_joint_density(metrics_by_method, buckets, save=True):
    """
    Plots joint density estimation (KDE) between AUC and category diversity for each method.
    Creates separate plots for each method.
    """
    for method, metrics in metrics_by_method.items():
        plt.figure(figsize=(8, 6))
        
        # Extract AUC and diversity values for this method
        auc_values = []
        div_values = []
        for bucket in buckets:
            if bucket in metrics:
                if 'auc' in metrics[bucket] and 'categ_div@5' in metrics[bucket]:
                    auc_values.append(metrics[bucket]['auc'])
                    div_values.append(metrics[bucket]['categ_div@5'])
        
        # Calculate correlation measures
        auc_array = np.array(auc_values)
        div_array = np.array(div_values)
        
        # R-squared value
        pearson_corr = np.corrcoef(auc_array, div_array)[0, 1]
        r_squared = pearson_corr ** 2
        
        # Get method label and corresponding colormap
        method_label = get_method_label(method)
        colormap = METHOD_COLORMAPS[method_label]
        
        # Create joint density plot with predefined colormap and contour lines
        fig = plt.gcf()
        # First plot the filled contours
        sns.kdeplot(
            x=auc_values, y=div_values, 
            cmap=colormap, fill=True, 
            thresh=0.05, alpha=0.8
        )
        # Then overlay contour lines
        sns.kdeplot(
            x=auc_values, y=div_values,
            color=plt.cm.get_cmap(colormap)(0.7),  # Use a darker shade of the colormap
            levels=10,
            linewidths=1,
            alpha=0.5
        )
        
        # Keep current x limits but set y limit to start at 0
        current_ylim = plt.ylim()
        plt.ylim(0, current_ylim[1])
        
        # Add correlation information in text box with improved styling
        # Position text box relative to the plot limits
        x_pos = plt.xlim()[0] + (plt.xlim()[1] - plt.xlim()[0]) * 0.02
        y_pos = plt.ylim()[1] - (plt.ylim()[1] - plt.ylim()[0]) * 0.02
        stats_text = f"R² = {r_squared:.3f}"
        plt.text(x_pos, y_pos, stats_text, 
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', pad=5),
                fontsize=12, verticalalignment='top')
        
        # Set title using method label
        plt.title(f"AUC-diversity joint density ({get_method_label(method)})")
        plt.xlabel("AUC", fontsize=12)
        plt.ylabel("Category diversity@5", fontsize=12)
        
        # Remove horizontal grid lines and make vertical ones more subtle
        plt.grid(True, axis='x', linestyle='--', alpha=0.2)
        
        # Add a light border around the plot
        plt.gca().spines['top'].set_visible(True)
        plt.gca().spines['right'].set_visible(True)
        plt.gca().spines['bottom'].set_visible(True)
        plt.gca().spines['left'].set_visible(True)
        
        # Adjust layout
        plt.tight_layout()
        
        if save:
            # Ensure figs directory exists
            os.makedirs('figs', exist_ok=True)
            plt.savefig(os.path.join('figs', f'joint_density_{method}.png'), dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

def plot_bucket_sizes(bucket_sizes, bucket_ranges, save=True):
    """
    Plots a histogram of bucket sizes with their ranges.
    
    Parameters:
    - bucket_sizes: List of integers representing the size of each bucket
    - bucket_ranges: List of strings representing the range of each bucket
    - save: Whether to save the plot or display it
    """
    plt.figure(figsize=(8, 6))
    
    # Create bar plot
    bars = plt.bar(range(len(bucket_sizes)), bucket_sizes, color='#3498db', alpha=0.7)
    
    # Customize the plot
    plt.xlabel('History length buckets')
    plt.ylabel('Number of samples')
    plt.title('Distribution of history lengths')
    
    # Set x-axis ticks to bucket ranges
    plt.xticks(range(len(bucket_sizes)), bucket_ranges, rotation=45)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom')
    
    # Add grid for better readability
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    # Add a light border around the plot
    plt.gca().spines['top'].set_visible(True)
    plt.gca().spines['right'].set_visible(True)
    plt.gca().spines['bottom'].set_visible(True)
    plt.gca().spines['left'].set_visible(True)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    if save:
        # Ensure figs directory exists
        os.makedirs('figs', exist_ok=True)
        plt.savefig(os.path.join('figs', 'bucket_sizes.png'), dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate plots for metrics analysis')
    parser.add_argument('--save', action='store_true', help='Save plots instead of displaying them')
    args = parser.parse_args()

    # Find all relevant metric files
    path = '/Users/ryan/PycharmProjects/newsreclib/logs/eval_scatter/runs/nrms_mindsmall_pretrainedemb_celoss_bertsent'
    dirs = ["embed_4", "resample_4", "base"]
    fname = "metrics.json"
    metric_files = []
    for dir in dirs:
        for file in os.listdir(os.path.join(path, dir)):
            if file == fname:
                metric_files.append(os.path.join(path, dir, file))
    print(f"Found {len(metric_files)} metric files")
    
    # Extract metrics for all methods
    metrics_by_method, sorted_buckets = extract_metrics_by_method(metric_files)
    
    # Create AUC comparison plot with all three methods
    create_plot(metrics_by_method, sorted_buckets, 'auc', 'auc_comparison.png', save=args.save)
    
    # Create diversity comparison plot with all three methods
    create_plot(metrics_by_method, sorted_buckets, 'categ_div@5', 'diversity_comparison.png', save=args.save)
    
    # Create joint density plots
    plot_joint_density(metrics_by_method, sorted_buckets, save=args.save)
    
    # Plot bucket sizes
    bucket_sizes = [36203, 15666, 7914, 4222, 2622, 1435, 968, 672, 427, 809]
    bucket_ranges = [
        '0-20.9',
        '20.9-40.8',
        '40.8-60.7',
        '60.7-80.6',
        '80.6-100.5',
        '100.5-120.4',
        '120.4-140.3',
        '140.3-160.2',
        '160.2-180.1',
        '180.1-200.0'
    ]
    plot_bucket_sizes(bucket_sizes, bucket_ranges, save=args.save)

if __name__ == "__main__":
    main() 
