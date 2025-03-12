import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os

root = "../../logs/eval/runs"
to_plot = [("categ_div@5", "auc"), ("sent_div@5", "auc")]
methods = {
    "iclm3": "Mean+1/sqrt(n) ICL TS",
    "iclmm": "Mean+1/n^2 ICL TS",
    "iclm": "Mean in-context TS",
    "icl": "In-context TS",
    "prior": "No-iteration TS"
}

def get_params(name):
    if name == "base":
        return "Base", 0
    method, pseudocount = name.split("_")
    return methods[method], int(pseudocount)

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
        except FileNotFoundError:
            print(f"{fp} not found")
df = pd.DataFrame(df_list)

# Set style parameters
plt.style.use('seaborn')
colors = {
    'Mean+1/sqrt(n) ICL TS': '#99d9ea', # Light blue
    'Mean+1/n^2 ICL TS': '#99d9ea', # Light blue
    'Mean in-context TS': '#99d9ea', # Light blue
    'In-context TS': '#2ecc71',    # Green for better performance
    'No-iteration TS': '#e74c3c',   # Red for worse performance
    'Base': '#7f8c8d'              # Gray for baseline
}
markers = {
    10000: 'd',  # Large circle for n=100
    1000: 'p',  # Large circle for n=100
    100: 'o',  # Large circle for n=100
    10: 's',   # Square for n=10
    5: 'x',    # Square for n=10
    0: '*'     # Star for base
}
sizes = {
    10000: 100,  # Larger markers for n=100
    1000: 100,  # Larger markers for n=100
    100: 100,  # Larger markers for n=100
    10: 80,    # Medium for n=10
    5: 60,     # Smaller for n=5
    0: 150     # Largest for base as it's a single important point
}

div_metric = "categ_div@5"
auc_metric = "auc"

for base_model, model_group in df.groupby('base_model'):
    plt.figure(figsize=(10, 6))
    
    # Plot each method
    for method in colors.keys():  # Specific order for legend
        method_data = model_group[model_group['method'] == method]
        
        # Plot points
        for count, count_data in method_data.groupby('counts'):
            label = f"{method}" if count == 0 else f"{method} (n={count})"
            plt.scatter(count_data[div_metric], count_data[auc_metric],
                       color=colors[method],
                       marker=markers[count],
                       s=sizes[count],
                       label=label,
                       zorder=3)
        
        # Connect points for the same method
        if len(method_data) > 1:
            method_points = method_data.sort_values('counts')
            plt.plot(method_points[div_metric], method_points[auc_metric],
                    color=colors[method],
                    linestyle='--',
                    alpha=0.6,
                    zorder=2)

    plt.title(f'{base_model}: Sentence Diversity vs AUC Trade-off', fontsize=12, pad=15)
    plt.xlabel('Sentence Diversity@5', fontsize=11)
    plt.ylabel('AUC', fontsize=11)
    
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
    plt.show()

# Create separate bar plots for AUC and Category Diversity
metrics_to_plot = [auc_metric, div_metric]
titles = {
    'auc': 'Model Performance (AUC)', 
    'ndcg@10': 'Model Performance (NDCG@10)',
    'categ_div@5': 'Category Diversity',
    'sent_div@5': 'Sentence Diversity'
}
ylabels = {
    'auc': 'AUC Score',
    'ndcg@10': 'NDCG@10 Score',
    'categ_div@5': 'Diversity Score',
    'sent_div@5': 'Diversity Score'
}

# Modern color palette
base_model_colors = ['#00B4D8', '#0077B6', '#023E8A']  # Blue gradient
method_order = ['Base', 'No-iteration TS', 'In-context TS', 'Mean in-context TS']

# Use a clean, modern style
plt.style.use('seaborn-v0_8-white')

for metric in metrics_to_plot:
    fig, ax = plt.subplots()  # Higher resolution
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#F8F9FA')
    
    # Prepare data
    base_models = sorted(df['base_model'].unique())
    x = np.arange(len(method_order))
    width = 0.22  # Adjusted bar width
    
    # Plot bars for each base model
    for i, (base_model, color) in enumerate(zip(base_models, base_model_colors)):
        values = []
        errors = []
        for method in method_order:
            data = df[(df['base_model'] == base_model) & (df['method'] == method)][metric]
            values.append(data.mean())
            errors.append(data.std())
        
        # Add bars with enhanced style
        bars = ax.bar(x + i*width, values, width,
                     label=base_model,
                     color=color,
                     # yerr=errors,
                     capsize=3,
                     error_kw={'elinewidth': 1.2,
                              'capthick': 1.2,
                              'alpha': 0.8},
                     zorder=3)
        
        # Add value labels with enhanced visibility
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                   f'{value:.3f}',
                   ha='center', va='bottom',
                   fontsize=8,
                   color='#444444',
                   zorder=4)
    
    # Enhance plot style
    ax.set_title(titles[metric], 
                 pad=20, 
                 fontsize=16, 
                 fontweight='bold',
                 color='#333333')
    
    ax.set_xlabel('Thompson Sampling Method', 
                 fontsize=12, 
                 labelpad=10,
                 color='#333333')
    
    ax.set_ylabel(ylabels[metric], 
                 fontsize=12, 
                 labelpad=10,
                 color='#333333')
    
    # Adjust x-axis with better positioning
    ax.set_xticks(x + width)
    ax.set_xticklabels(method_order, 
                       rotation=25, 
                       ha='right',
                       fontsize=10)
    
    # Enhance grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.2, color='#666666', zorder=0)
    ax.set_axisbelow(True)
    
    # Remove spines and ticks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#666666')
    ax.spines['bottom'].set_color('#666666')
    ax.tick_params(colors='#666666')
    
    # Enhanced legend
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
    
    # Set y-axis limits with some padding
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + (ymax-ymin)*0.1)
    
    # Adjust layout
    plt.tight_layout()
    plt.show()
