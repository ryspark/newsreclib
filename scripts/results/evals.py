import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate evaluation plots')
parser.add_argument('--save', action='store_true', help='Save plots instead of displaying them')
parser.add_argument('--scatter', action='store_true', help='Generate scatter plots')
parser.add_argument('--heatmap', action='store_true', help='Generate heatmap plots')
parser.add_argument('--no-bar', action='store_true', help='Disable bar plots (enabled by default)')
args = parser.parse_args()

# Set global font sizes for all plots
plt.rcParams.update({
    'font.size': 15,           # Base font size
    'axes.labelsize': 15,      # Axis labels
    'axes.titlesize': 19,     # Title size
    'xtick.labelsize': 13,     # X tick labels
    'ytick.labelsize': 13,     # Y tick labels
    'legend.fontsize': 17,     # Legend text
    'legend.title_fontsize': 17 # Legend title
})
# Configuration
root = "../../logs/eval_old/runs"
output_dir = "./figs"
if args.save:
    os.makedirs(output_dir, exist_ok=True)

# Metric definitions
PSEUDOCOUNT = 5
METRICS_TO_PLOT = [
    {
        'x': 'categ_div@5',
        'y': 'auc',
        'x_label': 'Category Diversity@5',
        'y_label': 'AUC',
        'title_template': '{}: Category Diversity@5 vs AUC Trade-off'
    },
    {
        'x': 'categ_div@10',
        'y': 'auc',
        'x_label': 'Category Diversity@10',
        'y_label': 'AUC',
        'title_template': '{}: Category Diversity@10 vs AUC Trade-off'
    },
]

# Method mappings
methods = {
    "base": "Pretrained",
    "embed": "In-context TS",
    "resample": "Prior sampling"
}
# Essential colors only
colors = {
    'Pretrained': '#7f8c8d',              # Gray for baseline
    'Prior sampling': '#3498db',          # Blue for worse performance
    'In-context TS': '#2ecc71',   # Green for better performance
}

def get_params(name):
    if name == "base":
        return "Pretrained", 0
    method, pseudocount = name.split("_")
    return methods[method], int(pseudocount)

# Load data
df_list = []
for model_dir in os.listdir(root):
    for sample_type in os.listdir(os.path.join(root, model_dir)):
        fp = os.path.join(root, model_dir, sample_type, "metrics.json")
        try:
            with open(fp, "r") as f:
                metrics = {k.replace("test/", ""): v for k, v in json.load(f).items()}
                metrics["base_model"] = model_dir.split("_")[0].upper()
                metrics["method"], metrics["counts"] = get_params(sample_type)
                df_list.append(metrics)
        except (FileNotFoundError, KeyError):
            print(f"skipping {fp}")
df = pd.DataFrame(df_list)

# Set style parameters
#plt.style.use('seaborn')

def get_dynamic_markers_and_sizes(unique_counts):
    """Create dynamic marker and size mappings based on count values"""
    # Sort counts in ascending order
    sorted_counts = sorted(unique_counts)
    
    # Define marker options
    marker_options = ['o', 's', 'd', 'p', 'h', 'v', '^', '<', '>', '8']
    
    # Create mappings
    markers = {}
    sizes = {}
    
    # Special case for base (count = 0)
    if 0 in sorted_counts:
        markers[0] = '*'
        sizes[0] = 150
        sorted_counts.remove(0)
    
    # Assign markers and sizes for other counts
    for i, count in enumerate(sorted_counts):
        markers[count] = marker_options[i % len(marker_options)]
        # Size decreases as count increases, but stays within reasonable bounds
        sizes[count] = max(60, 150 - (i * 10))
    
    return markers, sizes

# Plot scatter plots for each metric pair
if args.scatter:
    for metric_config in METRICS_TO_PLOT:
        for base_model, model_group in df.groupby('base_model'):
            fig, ax = plt.subplots(figsize=(5, 4))  # Increased figure size
            
            # Get dynamic markers and sizes for this group's counts
            unique_counts = sorted(model_group['counts'].unique())
            markers, sizes = get_dynamic_markers_and_sizes(unique_counts)
            
            # Plot each method
            for method in colors.keys():
                method_data = model_group[model_group['method'] == method]
                
                if not method_data.empty:
                    # Plot points with slightly smaller markers
                    for count, count_data in method_data.groupby('counts'):
                        label = f"{method}" if count == 0 else f"{method} (n={count})"
                        ax.scatter(count_data[metric_config['x']], 
                                 count_data[metric_config['y']],
                                 color=colors[method],
                                 marker=markers[count],
                                 s=sizes[count] * 0.8,  # Slightly reduce marker size
                                 label=label)
                    
                    # Connect points for the same method
                    if len(method_data) > 1:
                        method_points = method_data.sort_values('counts')
                        ax.plot(method_points[metric_config['x']], 
                               method_points[metric_config['y']],
                               color=colors[method],
                               linestyle='--',
                               alpha=0.6)

            ax.set_title(metric_config['title_template'].format(base_model))
            ax.set_xlabel(metric_config['x_label'])
            ax.set_ylabel(metric_config['y_label'])
            
            # Customize legend
            # ax.legend(title='Method & Sample Size', 
            #          bbox_to_anchor=(1.02, 1),
            #          loc='upper left')
            
            # Adjust layout to prevent legend cutoff
            plt.tight_layout()
            
            # Save or display figure
            if args.save:
                metric_name = metric_config['x'].replace('@', '_at_').replace('/', '_')
                filename = f"{base_model.lower()}_tradeoff_{metric_name}_vs_{metric_config['y']}.png"
                plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300)
            else:
                plt.show()
            plt.close()

# Create separate bar plots for metrics
if not args.no_bar:
    for metric_config in METRICS_TO_PLOT:
        for metric in [metric_config['x'], metric_config['y']]:
            fig, ax = plt.subplots(figsize=(8, 6))  # Increased figure size
            
            # Prepare data - filter for PSEUDOCOUNT except for Base model
            base_models = sorted(df['base_model'].unique())
            x = np.arange(len(base_models))
            width = 0.25
            
            # Plot bars for each method
            for i, (method, color) in enumerate(colors.items()):
                values = []
                errors = []
                for base_model in base_models:
                    # Filter for PSEUDOCOUNT only for non-base methods
                    if method == methods["base"]:
                        data = df[(df['base_model'] == base_model) & 
                                 (df['method'] == method)][metric]
                    else:
                        data = df[(df['base_model'] == base_model) & 
                                 (df['method'] == method) & 
                                 (df['counts'] == PSEUDOCOUNT)][metric]
                    values.append(data.mean() if not data.empty else np.nan)
                    errors.append(data.std() if not data.empty and len(data) > 1 else 0)
                
                # Skip plotting if all values are NaN
                if all(np.isnan(values)):
                    continue
                    
                # Plot bars for this method
                bars = ax.bar(x + i*width, values, width,
                            label=method,
                            color=color,
                            hatch='///' if method != 'In-context TS' else None,
                            #yerr=[0 if np.isnan(e) else e for e in errors],
                            capsize=3,
                            error_kw={'capthick': 1})
            
            metric_label = metric_config['x_label'] if metric == metric_config['x'] else metric_config['y_label']
            
            ax.set_title(f'{metric_label}')
            ax.set_xlabel('Base Model')
            ax.set_ylabel(metric_label)
            
            # Adjust x-axis positioning to center labels
            ax.set_xticks(x + width)
            ax.set_xticklabels(base_models)
            
            # Customize legend
            # ax.legend(title='Method',
            #          bbox_to_anchor=(1.02, 1),
            #          loc='upper left')
            
            plt.tight_layout()
            
            # Save or display figure
            if args.save:
                metric_name = metric.replace('@', '_at_').replace('/', '_')
                filename = f"barplot_{metric_name}_pseudocount_{PSEUDOCOUNT}.png"
                plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300)
            else:
                plt.show()
            plt.close()

# Create heatmap plots showing relative performance differences
if args.heatmap:
    for metric_config in METRICS_TO_PLOT:
        for metric in [metric_config['x'], metric_config['y']]:
            # Create figure with larger size
            fig, ax = plt.subplots(figsize=(5, 4))  # Increased figure size
            
            # Prepare data matrix
            base_models = sorted(df['base_model'].unique())
            method_order = list(colors.keys())
            
            # Create performance matrix
            perf_matrix = np.zeros((len(base_models), len(method_order)))
            for i, base_model in enumerate(base_models):
                for j, method in enumerate(method_order):
                    if method == methods["base"]:
                        data = df[(df['base_model'] == base_model) & 
                                 (df['method'] == method)][metric]
                    else:
                        data = df[(df['base_model'] == base_model) & 
                                 (df['method'] == method) & 
                                 (df['counts'] == PSEUDOCOUNT)][metric]
                    if not data.empty:
                        perf_matrix[i, j] = data.mean()
                    else:
                        perf_matrix[i, j] = np.nan
            
            # Calculate relative performance (difference from baseline)
            baseline_idx = method_order.index(methods["base"])
            relative_perf = perf_matrix - perf_matrix[:, baseline_idx:baseline_idx+1]
            
            # Create heatmap
            im = ax.imshow(relative_perf, cmap='RdYlGn', aspect='auto')
            
            # Customize the plot
            ax.set_xticks(np.arange(len(method_order)))
            ax.set_yticks(np.arange(len(base_models)))
            ax.set_xticklabels(method_order)
            ax.set_yticklabels(base_models)
            
            # Add value annotations
            for i in range(len(base_models)):
                for j in range(len(method_order)):
                    if not np.isnan(relative_perf[i, j]):
                        text = ax.text(j, i, f'{relative_perf[i, j]:.3f}',
                                     ha='center', va='center')
            
            # Add colorbar
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel('Relative Performance', rotation=-90, va='bottom')
            
            # Set title
            metric_label = metric_config['x_label'] if metric == metric_config['x'] else metric_config['y_label']
            ax.set_title(f'Relative Performance: {metric_label}')
            
            plt.tight_layout()
            
            # Save or display figure
            if args.save:
                metric_name = metric.replace('@', '_at_').replace('/', '_')
                filename = f"heatmap_{metric_name}_pseudocount_{PSEUDOCOUNT}.png"
                plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300)
            else:
                plt.show()
            plt.close()
