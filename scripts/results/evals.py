import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os

# Configuration
root = "../../logs/eval/runs"
output_dir = "./figs"
os.makedirs(output_dir, exist_ok=True)

# Metric definitions
METRICS_TO_PLOT = [
    {
        'x': 'categ_div@5',
        'y': 'auc',
        'x_label': 'Category Diversity@5',
        'y_label': 'AUC Score',
        'title_template': '{}: Category Diversity vs AUC Trade-off'
    },
    {
        'x': 'sent_div@5',
        'y': 'auc',
        'x_label': 'Sentence Diversity@5',
        'y_label': 'AUC Score',
        'title_template': '{}: Sentence Diversity vs AUC Trade-off'
    }
]

# Method mappings
methods = {
    "embed": "Similarity IC-TS",
    "category": "Category IC-TS",
    "resample": "Resampling"
}

# Essential colors only
colors = {
    'Base': '#7f8c8d',              # Gray for baseline
    'Similarity IC-TS': '#e74c3c',   # Red for worse performance
    'Category IC-TS': '#2ecc71',     # Green for better performance
    'Resampling': '#9467bd'     # Purple for resampling
}

def get_params(name):
    if name == "base":
        return "Base", 0
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
for metric_config in METRICS_TO_PLOT:
    for base_model, model_group in df.groupby('base_model'):
        plt.figure(figsize=(10, 6))
        
        # Get dynamic markers and sizes for this group's counts
        unique_counts = sorted(model_group['counts'].unique())
        markers, sizes = get_dynamic_markers_and_sizes(unique_counts)
        
        # Plot each method
        for method in colors.keys():
            method_data = model_group[model_group['method'] == method]
            
            if not method_data.empty:
                # Plot points
                for count, count_data in method_data.groupby('counts'):
                    label = f"{method}" if count == 0 else f"{method} (n={count})"
                    plt.scatter(count_data[metric_config['x']], 
                              count_data[metric_config['y']],
                              color=colors[method],
                              marker=markers[count],
                              s=sizes[count],
                              label=label,
                              zorder=3)
                
                # Connect points for the same method
                if len(method_data) > 1:
                    method_points = method_data.sort_values('counts')
                    plt.plot(method_points[metric_config['x']], 
                            method_points[metric_config['y']],
                            color=colors[method],
                            linestyle='--',
                            alpha=0.6,
                            zorder=2)

        plt.title(metric_config['title_template'].format(base_model), fontsize=12, pad=15)
        plt.xlabel(metric_config['x_label'], fontsize=11)
        plt.ylabel(metric_config['y_label'], fontsize=11)
        
        # Add grid but make it subtle
        plt.grid(True, linestyle='--', alpha=0.7, zorder=1)
        
        # Customize legend
        plt.legend(title='Method & Sample Size', 
                  title_fontsize=10,
                  bbox_to_anchor=(1.02, 1),
                  loc='upper left',
                  borderaxespad=0)
        
        # Adjust layout to prevent legend cutoff
        plt.tight_layout()
        
        # Save figure
        metric_name = metric_config['x'].replace('@', '_at_').replace('/', '_')
        filename = f"{base_model.lower()}_tradeoff_{metric_name}_vs_{metric_config['y']}.png"
        plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300)
        plt.close()

# Create separate bar plots for metrics
base_model_colors = ['#00B4D8', '#0077B6', '#023E8A']  # Blue gradient
method_order = list(colors.keys())

# Use a clean, modern style
plt.style.use('seaborn-v0_8-white')

for metric_config in METRICS_TO_PLOT:
    for metric in [metric_config['x'], metric_config['y']]:
        fig, ax = plt.subplots()
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#F8F9FA')
        
        # Prepare data
        base_models = sorted(df['base_model'].unique())
        x = np.arange(len(method_order))
        width = 0.22
        
        # Plot bars for each base model
        for i, (base_model, color) in enumerate(zip(base_models, base_model_colors)):
            values = []
            errors = []
            for method in method_order:
                data = df[(df['base_model'] == base_model) & (df['method'] == method)][metric]
                values.append(data.mean())
                errors.append(data.std())
            
            bars = ax.bar(x + i*width, values, width,
                         label=base_model,
                         color=color,
                         capsize=3,
                         error_kw={'elinewidth': 1.2,
                                  'capthick': 1.2,
                                  'alpha': 0.8},
                         zorder=3)
            
            # for bar, value in zip(bars, values):
            #     height = bar.get_height()
            #     ax.text(bar.get_x() + bar.get_width()/2., height + 0.002,
            #            f'{value:.3f}',
            #            ha='center', va='bottom',
            #            fontsize=8,
            #            color='#444444',
            #            zorder=4)
        
        metric_label = metric_config['x_label'] if metric == metric_config['x'] else metric_config['y_label']
        
        ax.set_title(f'Model Performance: {metric_label}', 
                     pad=20, 
                     fontsize=16, 
                     fontweight='bold',
                     color='#333333')
        
        ax.set_xlabel('Thompson Sampling Method', 
                     fontsize=12, 
                     labelpad=10,
                     color='#333333')
        
        ax.set_ylabel(metric_label, 
                     fontsize=12, 
                     labelpad=10,
                     color='#333333')
        
        ax.set_xticks(x + width)
        ax.set_xticklabels(method_order, 
                           rotation=25, 
                           ha='right',
                           fontsize=10)
        
        ax.yaxis.grid(True, linestyle='--', alpha=0.2, color='#666666', zorder=0)
        ax.set_axisbelow(True)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#666666')
        ax.spines['bottom'].set_color('#666666')
        ax.tick_params(colors='#666666')
        
        legend = ax.legend(title='Base Model',
                          title_fontsize=11,
                          fontsize=10,
                          bbox_to_anchor=(1.02, 1),
                          loc='upper left',
                          borderaxespad=0,
                          frameon=True,
                          edgecolor='#666666')
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)
        
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax + (ymax-ymin)*0.1)
        
        plt.tight_layout()
        
        # Save figure
        metric_name = metric.replace('@', '_at_').replace('/', '_')
        filename = f"barplot_{metric_name}.png"
        plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300)
        plt.close()
