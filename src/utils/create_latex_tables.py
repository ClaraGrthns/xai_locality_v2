import numpy as np
import pandas as pd
from textwrap import dedent
import re

def model_accuracy_on_test_latex_table(data_model_performance, knn_results):
    # Extract datasets and models
    data_model_performance = convert_data(data_model_performance)
    knn_results = convert_data(knn_results)

    datasets = list(data_model_performance.keys())
    models = list(data_model_performance[datasets[0]].keys())
    distance_measures = list(knn_results[datasets[0]].keys())
    
    # Create a DataFrame for the model performances
    model_df = pd.DataFrame(index=models, columns=datasets)
    for dataset in datasets:
        for model in models:
            model_df.loc[model, dataset] = data_model_performance[dataset][model]
    
    # Create a DataFrame for the kNN results
    knn_df = pd.DataFrame(index=[f"kNN ({measure})" for measure in distance_measures], columns=datasets)
    for dataset in datasets:
        for i, measure in enumerate(distance_measures):
            out = knn_results[dataset][measure]
            accuracy, k = out if out is not np.nan else (np.nan, 0)
            knn_df.loc[f"kNN ({measure})", dataset] = accuracy
    
    # Combine the DataFrames
    combined_df = pd.concat([model_df, knn_df])
    
    # Create dataset names that are more readable
    dataset_display_names = {}
    for dataset in datasets:
        if "synthetic" in dataset:
            # Extract important information from synthetic dataset name
            n_feat_match = re.search(r'n_feat(\d+)', dataset)
            n_informative_match = re.search(r'n_informative(\d+)', dataset)
            
            n_feat = n_feat_match.group(1) if n_feat_match else "?"
            n_informative = n_informative_match.group(1) if n_informative_match else "?"
            
            display_name = f"Synthetic (d={n_feat}, i={n_informative})"
            dataset_display_names[dataset] = display_name
        else:
            # Capitalize real dataset names
            dataset_display_names[dataset] = dataset.capitalize()
    
    # Find highest accuracy for each dataset
    highest_accuracies = {}
    for dataset in datasets:
        highest_accuracies[dataset] = combined_df[dataset].max()
    
    # Generate LaTeX table
    latex_code = "\\begin{table}[htbp]\n\\small\n\\centering\n\\caption{Model Accuracy Across Datasets}\n"
    latex_code += "\\begin{tabular}{l" + "c" * len(datasets) + "}\n"
    latex_code += "\\toprule\n"
    
    # Header row
    latex_code += "Model & " + " & ".join([dataset_display_names[dataset] for dataset in datasets]) + " \\\\\n"
    latex_code += "\\midrule\n"
    
    # Model rows
    for model in models:
        row_values = []
        for dataset in datasets:
            value = combined_df.loc[model, dataset]
            if np.isclose(value, highest_accuracies[dataset]):
                row_values.append(f"\\textbf{{{value:.3f}}}")
            else:
                row_values.append(f"{value:.3f}")
        latex_code += f"{model} & " + " & ".join(row_values) + " \\\\\n"
    
    # Add thick separator before kNN results
    latex_code += "\\midrule[1.5pt]\n"
    
    # kNN rows
    for measure in distance_measures:
        model_name = f"kNN ({measure})"
        row_values = []
        for dataset in datasets:
            value = combined_df.loc[model_name, dataset]
            if np.isclose(value, highest_accuracies[dataset]):
                row_values.append(f"\\textbf{{{value:.3f}}}")
            else:
                row_values.append(f"{value:.3f}")
        latex_code += f"{model_name} & " + " & ".join(row_values) + " \\\\\n"
    
    # Close table
    latex_code += "\\bottomrule\n"
    latex_code += "\\end{tabular}\n"
    latex_code += "\\label{tab:model_accuracy}\n"
    latex_code += "\\end{table}"
    
    return latex_code

# Convert your data to the correct format (replace np.float64 with float, etc.)
def convert_data(data_dict):
    result = {}
    for dataset, values in data_dict.items():
        result[dataset] = {}
        for key, value in values.items():
            if isinstance(value, tuple) and len(value) == 2:
                # For kNN results (accuracy, k)
                result[dataset][key] = (float(value[0]), int(value[1])) if value[0] is not None else (None, None)
            else:
                # For model performance
                result[dataset][key] = float(value) if value is not None else None
    return result



def kNN_on_model_preds_latex_table(results_dict, models_to_include=None, datasets_to_include=None):
    """
    Create a LaTeX table with models as main rows and distance measures as sub-rows.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary with structure {model: {dataset: {distance_measure: (accuracy, k)}}}
    models_to_include : list, optional
        List of model names to include in the table. If None, includes all models.
    datasets_to_include : list, optional
        List of dataset names to include in the table. If None, includes all datasets.
    
    Returns:
    --------
    str
        LaTeX code for the table
    """
    # Define mapping for synthetic datasets
    # Create mapping for synthetic datasets
    SYNTHETIC_DATASET_MAPPING = {}
    
    # Function to parse synthetic dataset path and create a readable name
    def parse_synthetic_dataset(dataset_path):
        if not dataset_path.startswith('synthetic_data'):
            return None
            
        params = {}
        # Extract parameters using regex
        for param in ['n_feat', 'n_informative', 'n_redundant', 'n_repeated', 
                     'n_classes', 'n_samples', 'n_clusters_per_class', 
                     'class_sep', 'flip_y', 'random_state']:
            match = re.search(f"{param}(\\d+(?:\\.\\d+)?)", dataset_path)
            if match:
                params[param] = match.group(1)
        
        # Create readable name based on the most important parameters
        if 'n_feat' in params and 'n_informative' in params:
            return f"syn (d:{params['n_feat']}, inf feat.: {params['n_informative']}), sep: {params.get('class_sep', 'N/A')}, clusters: {params.get('n_clusters_per_class', 'N/A')}"
        else:
            return "synthetic (unknown params)"
    
    # Populate mapping for all synthetic datasets
    for model in results_dict:
        for dataset in results_dict[model]:
            if dataset.startswith('synthetic_data'):
                readable_name = parse_synthetic_dataset(dataset)
                if readable_name:
                    SYNTHETIC_DATASET_MAPPING[readable_name] = dataset
    
    # Reverse mapping for synthetic datasets
    REVERSE_MAPPING = {v: k for k, v in SYNTHETIC_DATASET_MAPPING.items()}
    
    # Set default datasets if not provided
    if datasets_to_include is None:
        all_datasets = set()
        for model in results_dict:
            all_datasets.update(results_dict[model].keys())
        
        regular_datasets = [d for d in all_datasets if not d.startswith('synthetic_data')]
        synthetic_datasets = [d for d in all_datasets if d.startswith('synthetic_data')]
        
        datasets_to_include = sorted(regular_datasets)
        
        synthetic_datasets_ordered = []
        for synth_path in sorted(synthetic_datasets):
            if "n_feat100" in synth_path:
                synthetic_datasets_ordered.append(synth_path)
        for synth_path in sorted(synthetic_datasets):
            if "n_feat50" in synth_path and "n_informative10" in synth_path:
                synthetic_datasets_ordered.append(synth_path)
        for synth_path in sorted(synthetic_datasets):
            if "n_feat50" in synth_path and "n_informative2" in synth_path:
                synthetic_datasets_ordered.append(synth_path)
        
        datasets_to_include.extend(synthetic_datasets_ordered)
    
    # Set default models if not provided
    if models_to_include is None:
        models_to_include = list(results_dict.keys())
    
    # Create short names for synthetic datasets for column headers
    dataset_display_names = []
    for dataset in datasets_to_include:
        if dataset in REVERSE_MAPPING:
            # For synthetic datasets, create a shorter name
            if "n_feat100_n_informative50" in dataset:
                short_name = "Synthetic (f=100, i=50)"
            elif "n_feat50_n_informative10" in dataset:
                short_name = "Synthetic (f=50, i=10)"
            elif "n_feat50_n_informative2" in dataset:
                short_name = "Synthetic (f=50, i=2)"
            else:
                short_name = dataset
            dataset_display_names.append(f"\\textbf{{{short_name}}}")
        else:
            # For regular datasets, capitalize the name
            dataset_display_names.append(f"\\textbf{{{dataset.capitalize()}}}")
    
    # Distance measures to include (with display names)
    distance_measures = ["euclidean", "cosine", "manhattan"]
    distance_display = {"euclidean": "Euclidean", "cosine": "Cosine", "manhattan": "Manhattan"}
    
    # Start LaTeX table
    column_spec = "l" + "c" * len(datasets_to_include)
    
    latex_lines = [
        r"\begin{table}[htbp]",
        r"\small",
        r"\centering",
        r"\caption{Accuracy of kNN-classifier trained on model's predictions on $D_\text{test}$}",
        r"\label{tab:knn_model_accuracy}",
        r"\begin{tabular}{l" + column_spec + "}",
        r"\toprule",
        r"Model & Distance & " + " & ".join(dataset_display_names) + r" \\",
        r"\midrule"
    ]
    
    # Process each model
    for model_idx, model in enumerate(models_to_include):
        # Check if model has any data
        if not any(dataset in results_dict.get(model, {}) for dataset in datasets_to_include):
            continue
        
        # Add rows for each distance measure
        for dist_idx, distance in enumerate(distance_measures):
            # Add model name only for first distance measure
            if dist_idx == 0:
                row = f"\\textbf{{{model}}}" + " & " + distance_display[distance] # 
            else:
                row = " & " + distance_display[distance]
            
            # Add data for each dataset
            for dataset in datasets_to_include:
                if dataset in results_dict.get(model, {}) and distance in results_dict[model][dataset]:
                    acc, k = results_dict[model][dataset][distance]
                    row += f" & {acc:.2f} ({int(k)})"
                else:
                    row += " & ---"
            
            row += r" \\"
            latex_lines.append(row)
        
        # Add a midrule between models (except after the last one)
        if model_idx < len(models_to_include) - 1:
            latex_lines.append(r"\midrule")
    
    # Complete the table
    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ])
    
    return "\n".join(latex_lines)
# Example usage:

