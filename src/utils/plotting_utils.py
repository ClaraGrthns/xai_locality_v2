import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib import gridspec, lines
from collections import defaultdict
import time
import colorcet as cc
import os
from src.utils.process_results import (get_knn_vs_diff_model_performance,
                                    get_performance_metrics_smpl_complex_models,
                                    get_kw_fp,
                                    get_fraction, 
                                    get_knn_vs_metric_data,
                                    extract_sort_keys,
                                    get_and_apply_filter,
                                    filter_best_performance_local_model,
                                    get_synthetic_dataset_friendly_name
                                    )
import matplotlib as mpl

font_path = "/home/grotehans/xai_locality/font/cmunbsr.ttf"
mpl.font_manager.fontManager.addfont(font_path)
font_path = "/home/grotehans/xai_locality/font/Computer Modern Roman.ttf"
mpl.font_manager.fontManager.addfont(font_path)
font_path = "/home/grotehans/xai_locality/font/Times New Roman.ttf"
mpl.font_manager.fontManager.addfont(font_path)
from matplotlib.font_manager import FontProperties

from src.utils.process_results import (get_results_files_dict, get_kernel_widths_to_filepaths, 
                                       get_random_seed_to_filepaths, get_str_cond_to_filepaths, 
                                       get_synthetic_dataset_mapping, 
                                       get_synthetic_dataset_friendly_name, 
                                       get_synthetic_dataset_friendly_name_regression,
                                       get_local_vs_constant_metric_data)
# Set global matplotlib style for all plotting functions
plt.style.use('seaborn-v0_8-ticks')
plt.rcParams.update({
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.axisbelow': True,
    'axes.grid': True,
    'grid.linestyle': ':',
    'axes.xmargin': 0,
    "font.family": "serif",
    "font.serif": "cmr10",
    "text.usetex": False,  # Optional: if you want real LaTeX rendering
    'axes.labelsize': 12,
    "mathtext.fontset": "cm",  # Use Computer Modern for math symbols
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'axes.titlesize': 14,
    'axes.unicode_minus': False,
    
    # Add these new settings:
    'axes.labelpad': 2,  # Decrease the distance between axis and label (default is ~4-5)
    'axes.titlepad': 15,  # Increase the distance between title and plot (default is ~6-7)
})


textwidth_pt = 426.79135
inches_per_pt = 1 / 72.27
TEXT_WIDTH = textwidth_pt * inches_per_pt  # in inches
fig_height = TEXT_WIDTH * 0.6  # for a 3:2 aspect ratio, adjust as needed

MARKERS = ['o', 's', '^', 'D', 'v', '<', 'p', '*','>'  ]
MODELS = [
"MLP",
"LightGBM",
"TabTransformer",
"TabNet",
"ResNet",
]
# Assign a unique color to each model using tab10 colormap for better distinction
colors_models = [plt.cm.tab10(i) for i in range(len(MODELS))]

MODEL_TO_COLOR = {model: color for model, color in zip(MODELS, colors_models)}
MODEL_TO_MARKER = {model: marker for model, marker in zip(MODELS, MARKERS)}
COLORMAP_CLF = cc.glasbey_light
COLORMAP_REG = cc.glasbey_dark
datasets_clf = [
    "diabetes130us",
    "credit",
    "jannis",
    "higgs",
    "MiniBooNE",
    "california",
    "bank_marketing",
    "magic_telescope",
    "house_16H",
]
categorical_datasets_clf = [
    "mushroom",
    "albert",
    "road_safety",
    "kdd_census_income",
    "electricity",
    "adult_census_income",
    "adult",
     "default_of_credit_card_clients",
    "eye_movements",
    "heloc",
    "pol"

]
real_world_clf = sorted(list(set(datasets_clf + categorical_datasets_clf)))

path = "/home/grotehans/xai_locality/results/knn_model_preds/LightGBM/synthetic_data"
synthetic_datasets = [get_synthetic_dataset_friendly_name(name) for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

CLF_DATASETS = real_world_clf + synthetic_datasets


real_world_reg= list(set(
    ["california_housing" ,
    "diamonds",
    "elevators" ,
    "medical_charges" ,
    "superconduct" ,
    "houses",
    "allstate_claims_severity",
    "sgemm_gpu_kernel_performance",
    "diamonds",
    "particulate_matter_ukair_2017",
    "seattlecrime6",
    "airlines_DepDelay_1M",
    "delays_zurich_transport",
    "nyc-taxi-green-dec-2016",
    "microsoft" ,
    "year",
    "bike_sharing_demand", 
    "brazilian_houses",
    "house_sales",
    "sulfur",
                            ]
    ))
synthetic_reg = [ 'syn linear \n(d:30, inf f.:20, noise:0.6)',
 'syn piecewise_linear \n(d:15, inf f.:10, noise:0.2)',
 'syn piecewise \n(d:60, inf f.:15, noise:0.25)',
 'syn polynomial \n(d:50, inf f.:25, noise:0.8)',
 'syn adv_polynomial \n(d:80, inf f.:40, noise:0.5)',
 'syn hierarchical \n(d:70, inf f.:25, noise:0.15)',
 'syn poly_interaction \n(d:90, inf f.:40, noise:0.1)',
 'syn exponential_interaction \n(d:50, inf f.:10, noise:0.2)',
 'syn polynomial \n(d:100, inf f.:10, noise:0.0)',
 'syn polynomial \n(d:100, inf f.:50, noise:0.0)',
 'syn interaction \n(d:50, inf f.:30, noise:0.1)',
 'syn poly_interaction \n(d:90, inf f.:70, noise:0.1)',
 'syn multiplicative_chain \n(d:70, inf f.:30, noise:0.5)',
 'syn sigmoid_mix \n(d:200, inf f.:80, noise:0.15)',
 'syn exponential_interaction \n(d:20, inf f.:10, noise:0.2)',
 'syn polynomial \n(d:20, inf f.:10, noise:0.4)',
 'syn polynomial \n(d:20, inf f.:5, noise:0.1)',
 'syn exponential_interaction \n(d:50, inf f.:2, noise:0.2)',
 'syn linear \n(d:50, inf f.:2, noise:0.6)',
 'syn friedman 1 (noise:0.1)', 'syn friedman 1 (noise:0.01)', 'syn friedman 3 (noise:0.1)', 'syn friedman 2 (noise:0.1)', 'syn friedman 1 (noise:0.5)', 'syn friedman 2 (noise:0.5)', 'syn friedman 3 (noise:0.5)'
 ]

REG_DATASETS = real_world_reg + synthetic_reg
SYN_COLOR_TO_CLF_DATASET = {dataset: "tab:blue" for dataset in synthetic_datasets}
SYN_COLOR_TO_REG_DATASET = {dataset: "tab:green" for dataset in synthetic_reg}
REAL_COLOR_TO_CLF_DATASET = {dataset: "tab:orange" for dataset in real_world_clf}
REAL_COLOR_TO_REG_DATASET = {dataset: "tab:purple" for dataset in real_world_reg}




COLOR_TO_REG_DATASET_DETAILLED = {
    dataset: color for dataset, color in zip(REG_DATASETS, COLORMAP_REG)
}
COLOR_TO_CLF_DATASET_DETAILLED = {
    dataset: color for dataset, color in zip(CLF_DATASETS, COLORMAP_CLF)
}

COLOR_TO_CLF_DATASET = SYN_COLOR_TO_CLF_DATASET | REAL_COLOR_TO_CLF_DATASET
COLOR_TO_REG_DATASET = SYN_COLOR_TO_REG_DATASET | REAL_COLOR_TO_REG_DATASET
# Constants
METRICS_TO_IDX_CLF = {
    "Accuracy $g_x$": 0,
    "Precision": 1,
    "Recall": 2,
    "F1": 3,
    "MSE prob.": 4,
    "MAE prob.": 5,
    "R2  prob.": 6,
    "MSE logit": 7,
    "MAE logit": 8,
    "R2 logit": 9,
    "Gini Impurity": 10,
    "Accuracy const. local model": 15,
    "Variance prob.": 12, 
    "Variance logit": 13,
    "Radius": 14,
    "Local Ratio All Ones": 11,
    "Accuracy $g_x$ - Accuracy const. local model": (0, 15),
}

METRICS_MAP_REG = {
    "MSE $g_x$": 0,
    "MAE $g_x$": 1,
    "R2 kNN": 2,
    "MSE const. local model": 3,
    "MAE const. local model": 4,
    "Variance $f(x)$": 5,
    "Radius": 6,
    "MSE const. local model - MSE $g_x$": (3, 0),
    "MAE const. local model - MAE $g_x$": (4, 1),
    "MSE $g_x$ / MSE const. local model": (0, 3),
    "MSE $g_x$ / Variance $f(x)$": (0, 5),
    "E(MSE $g_x$ / Variance $f(x)$)": (0, 5),
    "MSE const. local model / Variance $f(x)$": (3, 5),
}
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
LINESTYLES = [  "--","-", "-."]
KERNEL_LABELS = ["default/2", "default", "default*2"]

def fig_name_smpl_vs_complex_model(complexity_setting, regression):
    return f"{"reg_" if regression else "clf_"}smpl_vs_complex_model_regression_classificatio_smpl_models_{complexity_setting}.pdf"
def plot_metric(ax, values, neighbors, color, style, start_at_k = 0, max_neighbors=None):
    """Plot a single metric line."""
    if values is None:
        print("Warning: Missing values")
        return
    neighbors = neighbors[start_at_k:max_neighbors]
    values = values[start_at_k:max_neighbors]
    ax.plot(neighbors, values, color=color, 
           linestyle=style, linewidth=2)

def setup_plot(n_metrics):
    """Initialize figure with subplots and legend space."""
    fig = plt.figure(figsize=(6 * n_metrics + 2, 4))
    gs = gridspec.GridSpec(1, n_metrics + 1, width_ratios=[6] * n_metrics + [1.5])  # Increase legend space
    axes = [fig.add_subplot(gs[0, i]) for i in range(n_metrics)]
    legend_ax = fig.add_subplot(gs[0, -1])
    legend_ax.axis('off')  
    fig.subplots_adjust(wspace=0.4)  
    return fig, axes, legend_ax

def create_legend(models, colors, method, unique_kw_lines_idx=[]):
    """Generate legend handles and labels."""
    handles = []
    labels = []
    handles.append(lines.Line2D([], [], color='none'))
    labels.append('Explanation for Models:')
    for m in models:
        handles.append(lines.Line2D([], [], color=colors[m], linewidth=2))
        labels.append(m)
    if method == "lime":
        handles.append(lines.Line2D([], [], color='none'))
        labels.append('Kernel Width:')
        for i in unique_kw_lines_idx:
            handles.append(lines.Line2D([], [], color='k', linestyle=LINESTYLES[i], linewidth=2))
            labels.append(KERNEL_LABELS[i])
    return handles, labels

def create_model_data_legends(ax, 
                              models, 
                              markers, 
                              regression=False, 
                              models_legend_anchor=(1.02, 1.1), 
                              data_legend_anchor=(1.02, 0.4),
                              synthetic_only=False):
    """Create two legends side by side: models on the left, datasets on the right with column filling."""
    data_to_colors = {
        "Synthetic Data": "tab:green" if regression else "tab:blue",
        "Real Data": "tab:purple" if regression else "tab:orange",
    }
    
    # Create model legend elements
    model_handles = [
        plt.Line2D([], [], marker=markers[m], color='gray', label=m, linestyle='None')
        for m in models
    ]
    model_handles = [plt.Line2D([], [], color='none', label='Models:')] + model_handles
    
        # Create dataset legend elements
    if synthetic_only:
        dataset_handles = [
            plt.Line2D([], [], marker='o', color=data_to_colors[d],
                    label=d.replace("_", " "), linestyle='None')
            for d in ["Synthetic Data"]
        ]
    else:
        dataset_handles = [
            plt.Line2D([], [], marker='o', color=data_to_colors[d],
                    label=d.replace("_", " "), linestyle='None')
            for d in ["Real Data", "Synthetic Data"]
        ]
    dataset_handles = [plt.Line2D([], [], color='none', label='Datasets:')] + dataset_handles
    
    model_legend = ax.legend(
        handles=model_handles,
        loc='upper left',
        bbox_to_anchor=models_legend_anchor,  # Positioned at right edge of plot
        frameon=False,
        ncol=1,
        fontsize=12,
        handletextpad=0,
        columnspacing=2.0,
        borderpad=0.8
    )
    
    ax.add_artist(model_legend)
    dataset_legend = ax.legend(
        handles=dataset_handles,
        loc='upper left',
        bbox_to_anchor=data_legend_anchor,  # Positioned to the right of model legend
        frameon=False,
        ncol=1,
        fontsize=12,
        handletextpad=0,
        columnspacing=2.0,
        borderpad=0.8
    )
    for legend in [dataset_legend, model_legend]:
        for text in legend.get_texts():
            if text.get_text() in ["Datasets:", "Models:"]:
                text.set_weight('bold')
                text.set_fontsize(12)
    
    return model_legend, dataset_legend

def plot_dataset_metrics(models, 
                         datasets, 
                         method, 
                         metrics, 
                         distance="euclidean", 
                        max_neighbors=None, 
                        save=False, 
                        lime_features=10, 
                        regression=False, 
                        start_at_k = 0, 
                        scale_by_variance = False, 
                        random_seed=42, 
                        summarizing_statistics=None):
    """Main plotting function."""
    if regression:
        from src.utils.process_results import load_results_regression as load_results
        metrics_map = METRICS_MAP_REG
    else:
        from src.utils.process_results import load_results_clf as load_results
        metrics_map = METRICS_TO_IDX_CLF
    results = get_results_files_dict(method, models, datasets, distance, lime_features, random_seed=random_seed)
    colors = {m: plt.cm.tab10(i) for i, m in enumerate(models)}
    if summarizing_statistics is None:
        summarizing_statistics = lambda x, axis: np.mean(x, axis=axis)
    for dataset in set(d for model_data in results.values() for d in model_data):
        fig, axes, legend_ax = setup_plot(len(metrics))
        dataset_models = [m for m in results if dataset in results[m]]
        for ax_idx, (ax, metric) in enumerate(zip(axes, metrics)):
            metric_idx = metrics_map[metric]
            is_diff = "-" in metric
            is_ratio = "/" in metric
            for model in dataset_models:
                files = results[model][dataset]
                if isinstance(files, list) and len(files) == 0:
                    continue
                if method in ["lime", "lime_captum"] and metric not in ["Variance $f(x)$", "Radius", "Accuracy const. local model", "MSE const. local model", "MAE const. local model"]:
                    kw_to_pf = get_kernel_widths_to_filepaths(files)
                    path = get_kw_fp(kw_to_pf, kernel_width="default")
                else:
                    path = files[0] if isinstance(files, list) else files
                # print(model, dataset, path)
                data, neighbors = load_results(path)
                neighbors = np.arange(0, len(neighbors))
                neighbors = neighbors + 1 if neighbors[0] == 0 else neighbors
                if is_ratio:
                    metr0 = data[metric_idx[0]]
                    metr1 = data[metric_idx[1]]
                    vals = get_fraction(metr0, metr1)# if g_x = 0, then value as zero mistakes
                elif is_diff:
                    vals = data[metric_idx[0]] - data[metric_idx[1]]
                    if scale_by_variance and regression:
                        vals /= data[metrics_map["Variance $f(x)$"]]
                else:
                    vals = data[metric_idx]
                summary_vals = summarizing_statistics(vals, axis=1)
                plot_metric(ax, summary_vals, neighbors, colors[model], '-', start_at_k, max_neighbors)
            if is_ratio or is_diff:
                ax.axhline(0, color='k', alpha=0.8, linewidth=0.8)
            if is_diff and (not is_ratio) and scale_by_variance:
                ax.set_title(f"({metric})$\\times \\frac{{1}}{{\\text{{var}}(f(x'))}}$")
            else:
                ax.set_title(metric)
            ax.set_xlabel(f"Neighborhood size ({distance} distance)")
            ax.grid(True, linestyle=':')
        handles, labels = create_legend(dataset_models, colors, method, [])
        legend_ax.legend(handles, labels, frameon=True, fontsize=11)
        method_title = method.split("/")[-1]
        method_title = " ".join(method_title.split("_"))
        if method == "lime" and lime_features == "all":
            method_title = method_title + " (all features)"
        title = f"{method_title.capitalize()} on {dataset.capitalize()}"
        if method == "lime" and lime_features == "all":
            title += " (all features)"
        y_position = 1.04 if "syn" in dataset else 1.02
        fig.suptitle(title, y=y_position)
        if save:
            fig.savefig(
                f"graphics/knn_vs_metrics_{method.split('/')[-1]}_{dataset}.pdf",
                bbox_inches='tight',  # Only if needed
                dpi=150,              # Reduced from 300
                # optimize=True,         # Enable PDF optimizations
                metadata={'CreationDate': None}  # Disable timestamp
            )
        else:
            plt.show();



def create_model_legend(ax, models, markers, colors = False, bbox_to_anchor=(1.02, 0.5), plot_multiple=False):
    """Create a legend for models only, with specified markers."""
    if plot_multiple:
        ax.axis("off")
    if colors:
        model_handles = [
        plt.Line2D([], [], marker=markers[m], color=MODEL_TO_COLOR[m], label=m, linestyle='None')
        for m in models
    ]
    else:
        model_handles = [
            plt.Line2D([], [], marker=markers[m], color='gray', label=m, linestyle='None')
            for m in models
        ]
    handles = [plt.Line2D([], [], color='none', label='Models:')] + model_handles
    legend = ax.legend(
        handles=handles,
        loc='center left',
        bbox_to_anchor=bbox_to_anchor,
        frameon=False,
        ncol=1,
        fontsize=12,
        handletextpad=0.1,  # Decrease this value for smaller spacing
        columnspacing=1,
        borderpad=0.5
    )
    for text in legend.get_texts():
        if text.get_text() == "Models:":
            text.set_weight('bold')
            text.set_fontsize(14)
    return legend


def create_data_legend(ax, datasets, colors, bbox_to_anchor=(1.02, 0.5), plot_multiple=False):
    """Create a side-by-side legend with datasets on the left and models on the right."""
    if plot_multiple:
        ax.axis("off")
    dataset_handles = [
        plt.Line2D([], [], marker='o', color=colors.get(d, "black"), label=d.replace("_", " "), linestyle='None')
        for d in datasets
    ]
    handles = (
        [plt.Line2D([], [], color='none', label='Datasets:')] + dataset_handles 
    )
    legend = ax.legend(
        handles=handles,
        loc='center left',
        bbox_to_anchor=bbox_to_anchor, #(0.0, 0.5),
        frameon=False,
        ncol=2,
        fontsize=11,
        handletextpad=1,
        columnspacing=2.0,
        borderpad=0.8
    )
    for text in legend.get_texts():
        if text.get_text() in ["Datasets:"]:
            text.set_weight('bold')
            text.set_fontsize(14)
    return legend

def create_dual_legend(ax, datasets, colors, models, markers, bbox_to_anchor=(1.02, 0.5), plot_multiple=False):
    """Create a side-by-side legend with datasets on the left and models on the right."""
    if plot_multiple:
        ax.axis("off")
    dataset_handles = [
        plt.Line2D([], [], marker='o', color=colors.get(d, "black"), label=d, linestyle='None')
        for d in datasets
    ]
    model_handles = [
        plt.Line2D([], [], marker=markers[m], color='gray', label=m, linestyle='None')
        for i, m in enumerate(models)
    ]
    handles = (
        [plt.Line2D([], [], color='none', label='Datasets:')] + dataset_handles +
        [plt.Line2D([], [], color='none', label='Models:')] + model_handles
    )
    legend = ax.legend(
        handles=handles,
        loc='center left',
        bbox_to_anchor=bbox_to_anchor, #(0.0, 0.5),
        frameon=False,
        ncol=2,
        fontsize=12,
        handletextpad=1.5,
        columnspacing=2.0,
        borderpad=1.0
    )
    for text in legend.get_texts():
        if text.get_text() in ["Datasets:", "Models:"]:
            text.set_weight('bold')
            text.set_fontsize(14)
    return legend

def edit_ticks_regression(ticks, replace_with, label, exclude_lower= -np.inf, include_lower = -np.inf, exclude_upper=1.1):
    ticks = [t for t in ticks if exclude_lower<t<exclude_upper]  # filter out > 1.1
    if replace_with not in ticks:
        ticks.append(replace_with)
    if include_lower not in ticks:
        ticks.append(include_lower)
    ticks = sorted(set(ticks))
    ticks_labels = [label if np.isclose(t, replace_with) else t for t in ticks]
    ticks_labels = [str(round(t, 2)) if not isinstance(t, str) else t for t in ticks_labels]
    return ticks, ticks_labels

def edit_ticks(ticks, replace_with, label, exclude_lower = -np.inf, exclude_upper=1.1):
    ticks = [t for t in ticks if exclude_lower<=t<exclude_upper]  # filter out > 1.1
    if exclude_lower not in ticks:
        ticks.append(exclude_lower)
    ticks = sorted(set(ticks))
    ticks_labels = [label if np.isclose(t, replace_with) else str(round(t, 2)) for t in ticks]
    return ticks, ticks_labels

       
def get_y_axis_label(filter, metric_axis_label, is_diff, is_ratio, summary):
    if summary is np.nanmean:
        summary_label = "\\mathbb{E}_x"
    else:
        summary_label = "Median_x"
    if is_diff:
        if "Accuracy" in metric_axis_label:
            label = (
                f"$\\text{{{filter}}}_k\\{{{summary_label} [{{A}}_g(f, k; m) - {summary_label}{{A}}_t(f, k; m)]\\}}$"
                if isinstance(filter, str)
                else f"{summary_label} [${{A}}_g(f, {{{filter}}}; m) - {summary_label}{{A}}_t(f,{{{filter}}};m))$"
            )
        else:
            label = (
                f"$\\text{{{filter}}}_k \\{{{summary_label} [{{{metric_axis_label}}}_t(f, k; m) - {summary_label}{{{metric_axis_label}}}_g(f, k; m)]\\}}$"
                if isinstance(filter, str)
                else f"{summary_label} [${{{metric_axis_label}}}_t(f, {filter}; m) - {summary_label}{{{metric_axis_label}}}_g(f, {filter}; m)) $"
            )
    elif is_ratio:
        # label = (
        #     f"$\\text{{{filter}}}_k \\{{{summary_label} \\frac{{{{{metric_axis_label}}}_g(f, k; m)}}{{Var(f, k; m)}}\\}}$"
        #     if isinstance(filter, str)
        #     else f"${summary_label} \\frac{{{{{metric_axis_label}}}_g(f, {filter}; m)}}{{Var(f, {filter}; m)}} $"
        # )
        label = (
            f"$\\text{{{filter}}}_k \\{{{summary_label} [ \\text{{R}}^2(g_x, f, k; m) ]\\}} $"
            if isinstance(filter, str)
            else f"$\\{{{summary_label}[\\text{{R}}^2(g_x, f, {{{filter}}}; m)}}]$"
        )
    else:
        label = (f"${summary_label}$ [{metric_axis_label}](f, {filter}; m)" 
        if isinstance(filter, int) 
        else f"$\\text{{{filter}}}_k \\{{{summary_label} [ \\text{{Accuracy}}(g_x, f, k; m) ]\\}} $")
    return label

def plot_knn_metrics_vs_metric(models, 
                               method, 
                               datasets, 
                               distance="euclidean",
                               ax=None,
                               filter="max", 
                               metric="MSE $g_x$ / Variance $f(x)$", 
                               difference_to_constant_model=False,
                               regression=False, 
                               summarizing_statistics=None,
                               random_seed=42,
                               kernel_width="default",
                               plot_downsample_fraction=False,
                               plot_individual_random_seed=True,
                               complexity_regression = "best",
                               average_over_n_neighbors=200,
                                order_average_first=True,
                               cutoff_at = None,
                               cutoff_value_replaced = None,
                               width = TEXT_WIDTH, 
                               height = TEXT_WIDTH * 0.9,
                               save=False,
                               detailled_colors=False,):
    """Main plotting function comparing model complexity vs performance difference."""
    synthetic_dataset_name = get_synthetic_dataset_friendly_name_regression if regression else get_synthetic_dataset_friendly_name
    assert (plot_downsample_fraction and random_seed ==42) or (not plot_downsample_fraction), "Downsampled data is only available for random seed 42."
    create_legend_to_ax = ax is None
    if ax is None:
        ax = plt.subplots(figsize=(width, height))[1]
    if detailled_colors:
        cmap = COLOR_TO_CLF_DATASET_DETAILLED if not regression else COLOR_TO_REG_DATASET_DETAILLED
    else:
        cmap = COLOR_TO_CLF_DATASET if not regression else COLOR_TO_REG_DATASET
    res_dict = get_results_files_dict(method, 
                                      models, 
                                      datasets,
                                      distance, 
                                      random_seed=random_seed, 
                                      downsampled=plot_downsample_fraction,
                                      kernel_width=kernel_width)
    print(res_dict.get("MLP", {}))
    is_diff = "-" in metric
    is_ratio = "/" in metric    
    if not (cutoff_at and cutoff_value_replaced):
        cutoff_at = -3 if (regression and is_ratio) else 0.5
        cutoff_value_replaced = -3.5 if regression and is_ratio else 0.45
    if "-" in metric:
        cutoff_at = -3 
        cutoff_value_replaced = -3.5 
    cutoff_label = f"$<${cutoff_at}"
    all_results = defaultdict(list)
    all_results_std = defaultdict(list)
    
    if summarizing_statistics is None:
        summarizing_statistics = lambda x, axis: np.nanmean(x, axis=axis)
    
    for model in models:
        model_results = res_dict.get(model, None)
        if model_results is None or len(model_results) == 0:
            print(f"No results found for {model}.")
            continue
        if type(random_seed) == int:
            random_seeds = [random_seed]
        else: 
            all_random_seeds = set()
            for dataset in datasets:
                files = model_results.get(synthetic_dataset_name(dataset), None)
                if files is None or len(files) == 0: continue
                random_seed_to_fp = get_random_seed_to_filepaths(model_results.get(synthetic_dataset_name(dataset), None))
                random_seeds = [int(rs_fp[0]) for rs_fp in random_seed_to_fp]
                all_random_seeds.update(random_seeds)
            random_seeds = sorted(list(all_random_seeds))
        gather_res_per_dataset = defaultdict(list)
        if plot_downsample_fraction:
            random_seeds = [42]
            fractions_to_fp = get_str_cond_to_filepaths("downsample", model_results.get(synthetic_dataset_name(datasets[1]), None))
            downsample_fractions = [float(fp[0]) for fp in fractions_to_fp if fp[0] != np.inf]
        else:
            downsample_fractions = [None]
        for downsample_fraction in downsample_fractions:
            for rs in random_seeds: 
                mapping = get_synthetic_dataset_mapping(datasets, regression)
                points = get_knn_vs_metric_data(model_results, 
                                                model, 
                                                mapping, 
                                                filter, 
                                                metric, 
                                                distance, 
                                                regression, 
                                                order_average_first=order_average_first,
                                                summarizing_statistics=summarizing_statistics,
                                                average_over_n_neighbors=average_over_n_neighbors,
                                                kernel_width = kernel_width, 
                                                difference_to_constant_model=difference_to_constant_model,
                                                random_seed=rs,
                                                downsample_fraction=downsample_fraction,
                                                complexity_regression=complexity_regression,)
                if plot_individual_random_seed:
                    for dataset, complexity_res, best_avg_fidelity in points:
                        all_results[model].append((dataset, 1-complexity_res, best_avg_fidelity))
                else:
                    for dataset, complexity_res, best_avg_fidelity in points:
                        gather_res_per_dataset[dataset].append((1-complexity_res, best_avg_fidelity))
        for dataset, knn_metric_res_list in gather_res_per_dataset.items():
            complexity_res = np.nanmean([res[0] for res in knn_metric_res_list])
            best_avg_fidelity = np.nanmean([res[1] for res in knn_metric_res_list])
            knn_metric_res_std = np.nanstd([res[0] for res in knn_metric_res_list])
            filtered_res_std = np.nanstd([res[1] for res in knn_metric_res_list])
            all_results[model].append((dataset, complexity_res, best_avg_fidelity))
            all_results_std[model].append((dataset, knn_metric_res_std, filtered_res_std))
    unique_datasets = sorted({d for m in all_results for d, _, _ in all_results[m]}, key=lambda x: extract_sort_keys(x, regression))
    colors = {d: cmap[d] for i, d in enumerate(unique_datasets)}
    models_plotted = list(all_results.keys())
    markers_to_models = {m: MODEL_TO_MARKER[m] for i, m in enumerate(models_plotted)}
    current_min_x_y = 1
    current_min_y = 1
    for i, model in enumerate(models_plotted):
        x_vals, y_vals, colors_list = [], [], []
        for dataset, complexity_res, best_avg_fidelity in all_results[model]:
            x_vals.append(complexity_res)
            y_vals.append(best_avg_fidelity if best_avg_fidelity > cutoff_at else cutoff_value_replaced)
            # colors_list.append(MODEL_TO_COLOR[model])
            colors_list.append(colors[dataset])
        current_min_x_y = min(current_min_x_y, min(x_vals), min(y_vals))
        current_min_y = min(current_min_y, min(y_vals))
        ax.scatter(x_vals, y_vals, c=colors_list, marker=markers_to_models[model], 
               s=50, alpha=0.8, edgecolors="black", linewidths=0.2)
        if not plot_individual_random_seed and (type(random_seed) != int or plot_downsample_fraction):
            x_vals_std, y_vals_std = [], []
            for dataset, knn_metric_res_std, filtered_res_std in all_results_std[model]:
                x_vals_std.append(knn_metric_res_std)
                y_vals_std.append(filtered_res_std)
            for i, (xv, yv, xv_std, yv_std) in enumerate(zip(x_vals, y_vals, x_vals_std, y_vals_std)):
                if yv > cutoff_at:
                    yv_std = np.where(yv + yv_std > 200, 200-yv, yv_std)
                    yv_std = np.where(yv - yv_std < cutoff_at, yv-cutoff_at, yv_std)
                    ax.errorbar(xv, yv, yerr=yv_std, fmt='none', ecolor=colors_list[i], alpha=0.8, elinewidth=2, capsize=3, zorder=1)
                #     ellipse = plt.matplotlib.patches.Ellipse(
                #     (xv, yv), width=2*xv_std, height=2*yv_std,
                #     edgecolor='none', facecolor=colors_list[i], alpha=0.15, zorder=1
                #     )
                #     ax.add_patch(ellipse)
                # else:# Plot horizontal error bar for x-value standard deviation
                #     ax.errorbar(xv, yv, xerr=xv_std, fmt='none', ecolor=colors_list[i], alpha=0.3, capsize=3, zorder=1)
               
    # === Ticks ===
    if is_diff or "R2" in metric:
        ax.axhline(0, color='black', linestyle='--', alpha=0.7)
    elif is_ratio:
        ax.axhline(0, color='black', linestyle='--', alpha=0.7)
        # ax.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.7)
    elif "Accuracy" in metric and filter != "argmax":
        ax.axhline(0.5, color='black', linestyle='--', alpha=0.7)

    if regression:
        current_min_x_y = np.max([cutoff_value_replaced-0.2, current_min_x_y])
    else:
        current_min_x_y = np.max([cutoff_value_replaced, current_min_x_y])
        current_min_x_y = np.min([cutoff_at, current_min_x_y])
        current_min_y = np.max([cutoff_value_replaced, current_min_y])
        current_min_y = np.min([cutoff_at, current_min_y])
    # === Ticks ===
    ax.set_xlim((0, 1)) if regression else  ax.set_xlim((0, 0.5))
    if filter != "argmax":
        ax.set_ylim((current_min_x_y, 1))
        yticks = ax.get_yticks()
        if regression:
            yticks, ytick_labels = edit_ticks_regression(ticks = yticks, 
                                          replace_with=cutoff_value_replaced, 
                                          label=cutoff_label, 
                                          include_lower=cutoff_at,
                                          exclude_lower=current_min_x_y, 
                                          exclude_upper=1.1)
        else:
            yticks, ytick_labels = edit_ticks(yticks, cutoff_value_replaced, cutoff_label, exclude_lower=current_min_x_y, exclude_upper=1.1)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ytick_labels)
        if "Accuracy" in metric:
            print(current_min_y)
            ax.set_ylim((current_min_y, 1))

    # === Labels and Titles ===
    x_label = get_x_label(x_metrics="complexity", complexity_regression=complexity_regression, regression=regression, filter=filter)
    ax.set_xlabel(x_label)
    y_label = get_y_label(y_metrics="local_model", filter=filter, regression=regression)
    ax.set_ylabel(y_label, labelpad=-5)
    method_title = ' '.join(method.split('/')[-1].split('_')) 
    if method == "lime":
        method_title= f"LIME (sparse{"" if kernel_width == 'default' else f', {kernel_width} of default'})"
    if method == "lime_captum":
        method_title = f"LIME (cont.{"" if kernel_width == 'default' else f', {kernel_width} of default'})"
    if method == "gradient_methods/integrated_gradient":
        method_title = "Integrated Gradients"
    elif method == "gradient_methods/smooth_grad":
        method_title = "Smooth Grad+IG"
    elif method == "gradient_methods/saliency":
        method_title = "Saliency"
    title = f"{method_title}\nComplexity vs. max. avg. Local Fidelity"
    # title = get_title(method_title, is_diff, regression=regression, x_metrics="complexity", filter=filter, create_legend_to_ax=create_legend_to_ax)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if create_legend_to_ax:
        create_dual_legend(ax, unique_datasets, colors, models_plotted,markers_to_models, bbox_to_anchor=(1.02, 0.5))
    if save:
        plt.savefig(
            f"graphics/complexity_vs_{filter}_metrics_{method.split('/')[-1]}_{dataset}.pdf",
            bbox_inches='tight',
            dpi=100,
            metadata={'CreationDate': None}
        )
    return ax, unique_datasets, colors, models_plotted, markers_to_models

def get_smple_model_name(complexity_regression, regression=False):
    if complexity_regression == "best":
        # smpl_model = "one of: kNN-reg., linear reg. or decision tree\n" if regression else "one of: kNN-clf., logistic reg. or decision tree\n"
        smpl_model = "baseline models"
    elif complexity_regression == "kNN":
        smpl_model = "kNN clf." if not regression else "kNN reg."
    elif complexity_regression == "linear":
        smpl_model = "Lin. Reg" if regression else "Log. Reg"
    elif complexity_regression == "tree":
        smpl_model = "Decision Tree"
    return smpl_model

def get_x_label(x_metrics="complexity", complexity_regression="best", filter="max", regression=False):
    smpl_model = get_smple_model_name(complexity_regression, regression)
    if x_metrics =="complexity":
        x_label = "Model complexity $m^R$" if regression else "Model complexity $m^C$" #f"{"Lowest " if complexity_regression == "best" else ""}{"$1-R^2$" if regression else "error"} of {smpl_model}\non model predictions"
    elif x_metrics =="constant_model":
        if filter == "min":
            metric_axis_label = "$MSE$"
        else:
            metric_axis_label = "$\\bar{{F}}^{\;R}$" if regression else "$\\bar{{F}}^{\;C}$"
        x_label = f"Avg. {metric_axis_label} of\nconst. local model" if isinstance(filter, str) else f"Average {metric_axis_label} const. local model for {filter} closest neighbors"
    else:
        x_label = ""
    return x_label

def get_y_label(y_metrics="local_model", filter="max", regression=False):
    metric_axis_label = "$\\bar{{F}}^{\;R}$" if regression else "$\\bar{{F}}^{\;C}$"
    if filter == "min":
        metric_axis_label = "$MSE$"
    if filter != "argmax":
        y_axis_label = f"Avg. best {metric_axis_label} of $g_x$" #get_y_axis_label(filter, metric, is_diff, is_ratio, summary=summarizing_statistics)
    else:
        y_axis_label = f"Argmax k of avg. local {metric_axis_label}"
    return y_axis_label

def get_title(method_title, is_diff, regression =False, x_metrics="complexity", filter="max", create_legend_to_ax=False):
    metric_axis_label = "$\\bar{{F}}^{\;R}$" if regression else "$\\bar{{F}}^{\;C}$"
    if x_metrics =="complexity":
        if isinstance(filter, int):
            title = f"{method_title.capitalize()}-\n Complexity of f vs. {metric_axis_label} avg. {'improvement' if is_diff else 'of $g_x$'} within {filter} neighbors"
        elif filter == "argmax":
            title = f"{method_title.capitalize()}-\n Complexity of f vs. argmax k {metric_axis_label} avg. {metric_axis_label}"
        else:
            title = f"{method_title.capitalize()}\n Complexity of f vs. max avg. local fidelity"
    else:
        if isinstance(filter, int):
            title = f"{method_title.capitalize()+": " if create_legend_to_ax else ""}Average {metric_axis_label} of const. local expl. vs.\n {metric_axis_label} avg. of $g_x$ within {filter} neighbors"
        else:
            title = f"{method_title.capitalize()+": " if create_legend_to_ax else ""}Max. avg. {metric_axis_label} of const. local expl. vs.\n  max. avg. {metric_axis_label} of $g_x$"
    return title

def extract_random_seed(file_path):
    match = re.search(r'random_seed-(\d+)', file_path)
    return float(match.group(1)) if match else float('inf')

def order_by_seed(file_paths):
    """Order file paths by the downsampling fraction in the string."""
    return sorted(file_paths, key=extract_random_seed)



def plot_argmax_k_over_models_histograms(
    argmax_k_max_avg_stat_array,
    models_in_dict,
    figsize=(15, 5),
    max_k=10,
    tick_step=None,  # Parameter to control tick spacing
    color=None,
    alpha=0.7,
    regression=False,
    title="Distribution of argmax k values per model",
    save_path=None,
    fontsize_ticks=15,
    fontsize_ticks_title = 20, 
):
    """
    Plots a row of histograms with controlled x-tick spacing.
    Values above max_k are grouped into one bin.
    
    Args:
        argmax_k_max_avg_stat_array: Array of shape (n_models, n_datasets)
                                     Expected to contain integer k values
        models_in_dict: List of model names
        figsize: Figure size (width, height)
        max_k: Maximum k value to display individually; values above are grouped
        tick_step: Step size for x-axis ticks (default: max_k // 5)
        colors: List of colors for each model's histogram (if None, uses default colors)
        alpha: Transparency of histogram bars
        title: Figure title
        save_path: Path to save figure (if None, figure is not saved)
        
    Returns:
        Matplotlib figure object
    """
    n_models = len(models_in_dict)
    if tick_step is None:
        tick_step = max(1, max_k // 5)
    
    fig, axes = plt.subplots(1, n_models, figsize=figsize, sharey=True)
    
    if n_models == 1:
        axes = [axes]
    if color is None:
        color = "tab:blue"  # Default color if not provided
    bins = np.arange(0.5, max_k + 2.5)  # Creates bins centered on integer values
    bin_centers = np.arange(1, max_k + 2)  # Centers for ticks
    
    y_max = 0
    for j in range(n_models):
        data = argmax_k_max_avg_stat_array[j, :]
        if data is not None and len(data) > 0:
            clipped_data = np.where(data >= max_k, max_k + 1, data)
            counts, _ = np.histogram(clipped_data, bins=bins)
            y_max = max(y_max, np.max(counts))
    
    for j, ax in enumerate(axes):
        data = argmax_k_max_avg_stat_array[j, :]
        if data is not None and len(data) > 0:
            clipped_data = np.where(data > max_k, max_k + 1, data)
            
            counts, _, patches = ax.hist(
                clipped_data,
                bins=bins,
                alpha=alpha,
                edgecolor='black',
                color=color,
                align='mid'  # Ensure bars are centered on bins
            )
            
            ax.set_ylim(0, y_max * 1.01)
            tick_positions = np.arange(0, max_k, tick_step)
            tick_labels = [str(i) for i in range(0, max_k, tick_step)]
            if max_k + 1 not in tick_positions:
                tick_positions = np.append(tick_positions, max_k + 1)
                tick_labels.append(f"{max_k}+")
            else:
                tick_labels[-1] = f"{max_k}+"
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=45, fontsize=fontsize_ticks)
            
            mean_val = np.mean(data)
            median_val = np.median(data)
            ax.text(0.95, 0.95,
                  f"Mean: {mean_val:.1f}\nMedian: {median_val:.0f}",
                  transform=ax.transAxes,
                  va='top', ha='right',
                    fontsize=fontsize_ticks,
                  bbox=dict(facecolor='white', alpha=0.8))
        ax.set_title(models_in_dict[j], fontsize=fontsize_ticks+2)
        ax.set_xlabel("Number of Neighbors $k$")
        ax.set_yticks(np.arange(0, y_max*1.01, 2, dtype=int))
        ax.tick_params(axis='both', which='major', labelsize=fontsize_ticks)  # Make tick labels larger
        
        if j == 0: 
            ax.set_ylabel("Count", fontsize=fontsize_ticks)
    
    fig.suptitle(
        "Method: " + title+ f", Task: {"Regression" if regression else "Classification"}" + f"\nDistribution of argmax k of avg. local fidelity over {argmax_k_max_avg_stat_array.shape[1]} datasets",
        fontsize=fontsize_ticks_title, y=1,
        weight='bold'  
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    return fig

def plot_combined_argmax_k_histogram(
    argmax_k_max_avg_stat_array,
    models_in_dict,
    figsize=(10, 6),
    ax = None,
    max_k=10,
    tick_step=None,
    color=None,
    alpha=0.7,
    regression=False,
    title="Combined distribution of argmax k values across all models",
    save_path=None,
    fontsize_ticks=15,
    fontsize_ticks_title=20,
):
    """
    Plots a single histogram combining argmax k values from all models.
    Values above max_k are grouped into one bin.
    
    Args:
        argmax_k_max_avg_stat_array: Array of shape (n_models, n_datasets)
                                     Expected to contain integer k values
        models_in_dict: List of model names (for labeling)
        figsize: Figure size (width, height)
        max_k: Maximum k value to display individually; values above are grouped
        tick_step: Step size for x-axis ticks (default: max_k // 5)
        color: Color for the histogram (if None, uses default color)
        alpha: Transparency of histogram bars
        title: Figure title
        save_path: Path to save figure (if None, figure is not saved)
        fontsize_ticks: Font size for axis ticks
        fontsize_ticks_title: Font size for title
        
    Returns:
        Matplotlib figure object
    """
    if tick_step is None:
        tick_step = max(1, max_k // 5)
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    if color is None:
        color = "olive"  # Default color if not provided
    argmax_k_max_avg_stat_array = np.where(argmax_k_max_avg_stat_array == None, np.nan, argmax_k_max_avg_stat_array)
    # Flatten the array and remove any None values
    all_data = argmax_k_max_avg_stat_array.flatten()
    # all_data = all_data[~np.isnan(all_data)]  # Remove NaN values if any
    
    # Create bins centered on integer values
    bins = np.arange(0.5, max_k + 2.5)
    
    # Clip data and plot histogram
    clipped_data = np.where(all_data > max_k, max_k + 1, all_data)
    counts, _, patches = ax.hist(
        clipped_data,
        bins=bins,
        alpha=alpha,
        edgecolor='black',
        linewidth=0.8,
        color=color,
        align='mid'  # Ensure bars are centered on bins
    )
    
    # Set ticks and labels
    tick_positions = np.arange(0, max_k, tick_step)
    tick_labels = [str(i) for i in range(0, max_k, tick_step)]
    if max_k + 1 not in tick_positions:
        tick_positions = np.append(tick_positions, max_k + 1)
        tick_labels.append(f"{max_k}+")
    else:
        tick_labels[-1] = f"{max_k}+"
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, fontsize=fontsize_ticks)
    
    # Calculate and display statistics
    mean_val = np.mean(all_data)
    median_val = np.median(all_data)
    ax.text(0.95, 0.95,
          f"Mean: {mean_val:.1f}\nMedian: {median_val:.0f}",
          transform=ax.transAxes,
          va='top', ha='right',
          fontsize=fontsize_ticks-2,
          bbox=dict(facecolor='white', alpha=0.8))
    
    # Labeling
    ax.set_xlabel("Neighborhood size $k$", fontsize=fontsize_ticks)
    ax.set_ylabel("Count", fontsize=fontsize_ticks)
    ax.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
    
    # Set y-axis to integer steps
    y_max = np.max(counts)
    ax.set_yticks(np.arange(0, y_max * 1.1, max(1, int(y_max/5))))
    
    ax.set_title(
        f"{title}",
        # f"Models: {model_names}",
        fontsize=fontsize_ticks_title, y=0.95,
        weight='bold'
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    return ax


def plot_random_seeds_results_per_dataset(method, 
                                    models, 
                                    datasets, 
                                    regression=False):
    """
    Plots the model explanation accuracy vs. neighborhood size for given models and datasets.

    Parameters:
        method (str): The method to use ('lime' or 'gradient_methods/integrated_gradient').
        models (list): List of models to evaluate.
        datasets (list): List of datasets to evaluate.
        regression (bool): Whether the task is regression (True) or classification (False).
    """
    if regression:
        from src.utils.process_results import load_results_regression as load_results
        from src.utils.process_results import get_synthetic_dataset_friendly_name_regression as get_friendly_name
    else:
        from src.utils.process_results import load_results_clf as load_results
        from src.utils.process_results import get_synthetic_dataset_friendly_name as get_friendly_name
        
    colors = MODEL_TO_COLOR
    alpha = 0.2  
    mapping = get_synthetic_dataset_mapping(datasets, regression)
    results_dict = get_results_files_dict(method, models, datasets, downsampled=False, random_seed=True)
    cutoff = 0 if regression else 0.5
    for dataset_idx, dataset in enumerate(datasets):
        fig, ax = plt.subplots(figsize=(8, 5))
        max_y = 0
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False)
        ax.tick_params(axis='both', which='both', length=5)
        for model_idx, model in enumerate(models):
            data_set_name = get_friendly_name(dataset)
            res_fp_ls = results_dict[model].get(data_set_name, None)
            if res_fp_ls is None:
                continue
            res_fp_ls = order_by_seed(res_fp_ls)
            mean_metric_ls = []
            for i in range(len(res_fp_ls)):
                res, knn = load_results(res_fp_ls[i])
                metric_red = get_fraction(res[0], res[5]) if regression else res[0]
                mean_metric = np.nanmean(metric_red, axis=1)
                if len(mean_metric[:200]) != 200:
                    print(f"Warning: {model} {data_set_name} has {len(mean_metric[:200])} values instead of 200.")
                    continue
                mean_metric_ls.append(mean_metric[:200])
            mean_metric_ls = np.array(mean_metric_ls)
            mean_values = np.mean(mean_metric_ls, axis=0)
            mean_values = np.where(mean_values > cutoff, mean_values, cutoff)
            std_values = np.std(mean_metric_ls, axis=0)
            std_values = np.where(mean_values > cutoff, std_values, 0)
            max_y = max(max_y, np.max(mean_values + std_values))
            # mean_values = np.where(mean_values > cutoff, mean_values, cutoff)
            # std_values = np.where(std_values > cutoff, std_values, 0)
            ax.plot(knn[:200], mean_values, 
                    label=model,
                    color=colors[model],
                    linewidth=2)
            ax.fill_between(knn[:200],
                           mean_values - std_values,
                           mean_values + std_values,
                           color=colors[model],
                           alpha=alpha)
        ax.set_xlabel("Number of Nearest Neighbors (k)", fontsize=12)
        ax.set_ylim(cutoff, max_y+0.01)
        ax.set_ylabel(f"{'MSE' if regression else 'accuracy'} of $g_x$", fontsize=12)
        ax.set_title(f"{method.split("/")[-1].capitalize()}: Averaged over 5 random seeds, {'MSE' if regression else 'accuracy'} vs. Neighborhood Size\nDataset: {data_set_name}", 
                    fontsize=14, pad=20)
        ax.grid(True, linestyle='--', alpha=0.6)
        # ax.set_axisbelow(True)
        legend = ax.legend(title="Models", 
                          fontsize=10,
                          title_fontsize=11,
                          frameon=True,
                          framealpha=0.9,
                          loc='lower right')
        plt.tight_layout()
        plt.show();

def plot_local_metrics_vs_constant_metric(models, 
                               method, 
                               datasets, 
                               distance="euclidean",
                               ax=None,
                               filter="max", 
                               metric="MSE $g_x$ / Variance $f(x)$", 
                               regression=False, 
                               summarizing_statistics=None,
                               average_over_n_neighbors=200,
                               width = TEXT_WIDTH, 
                               kernel_width="default",
                               random_seed=42,
                               cut_off = None,
                               order_average_first = True,
                               replace_with = None,
                               plot_downsample_fraction=False,
                               plot_individual_random_seed=True,
                               plot_color_per_dataset=False,
                               height = TEXT_WIDTH * 0.6,
                               save=False):
    """Main plotting function comparing model complexity vs performance difference."""
    synthetic_dataset_name = get_synthetic_dataset_friendly_name_regression if regression else get_synthetic_dataset_friendly_name
    
    create_legend_to_ax = ax is None
    if ax is None:
        ax = plt.subplots(figsize=(width, height))[1]
    if plot_color_per_dataset:
        cmap = COLOR_TO_REG_DATASET_DETAILLED if regression else COLOR_TO_CLF_DATASET_DETAILLED
    else:
        cmap = COLOR_TO_REG_DATASET if regression else COLOR_TO_CLF_DATASET
    res_dict = get_results_files_dict(method, 
                                      models, 
                                      datasets,
                                      distance, 
                                      random_seed=random_seed, 
                                      downsampled=plot_downsample_fraction,
                                      kernel_width=kernel_width)
    is_diff = "-" in metric
    is_ratio = "/" in metric  
    if not(cut_off and replace_with):
        cut_off = -3 if regression else 0.5
        replace_with = -3.5 if regression else 0.45
    
    all_results = defaultdict(list)
    all_results_std = defaultdict(list)

    if summarizing_statistics is None:
        summarizing_statistics = lambda x, axis: np.nanmean(x, axis=axis)
    all_vals = []
    for model in models:
        model_results = res_dict.get(model, None)
        if model_results is None or len(model_results) == 0:
            print(f"No results found for {model}.")
            continue
        if type(random_seed) == int:
            random_seeds = [random_seed]
        else: 
            all_random_seeds = set()
            for dataset in datasets:
                files = model_results.get(synthetic_dataset_name(dataset), None)
                if files is None or len(files) == 0: continue
                random_seed_to_fp = get_random_seed_to_filepaths(model_results.get(synthetic_dataset_name(dataset), None))
                random_seeds = [int(rs_fp[0]) for rs_fp in random_seed_to_fp]
                all_random_seeds.update(random_seeds)
            random_seeds = sorted(list(all_random_seeds))
        gather_res_per_dataset = defaultdict(list)
        if plot_downsample_fraction:
            random_seeds = [42]
            fractions_to_fp = get_str_cond_to_filepaths("downsample", model_results.get(synthetic_dataset_name(datasets[0]), None))
            downsample_fractions = [float(fp[0]) for fp in fractions_to_fp if fp[0] != np.inf]
        else:
            downsample_fractions = [None]
        for downsample_fraction in downsample_fractions:
            for rs in random_seeds: 
                points = get_local_vs_constant_metric_data(res_model=model_results,
                            filter=filter,
                                metric=metric,
                                regression=regression, 
                                random_seed=rs,
                                kernel_width=kernel_width,
                                order_average_first=order_average_first,
                                downsample_fraction=downsample_fraction,
                                summarizing_statistics=summarizing_statistics,
                                average_over_n_neighbors=average_over_n_neighbors)
                if plot_individual_random_seed:
                    for dataset, constant_res, expl_res in points:
                        all_results[model].append((dataset, constant_res, expl_res))
                else:
                    for dataset, constant_res, expl_res in points:
                        gather_res_per_dataset[dataset].append((constant_res, expl_res))
        for dataset, knn_metric_res_list in gather_res_per_dataset.items():
            constant_res = np.nanmean([res[0] for res in knn_metric_res_list])
            expl_res = np.nanmean([res[1] for res in knn_metric_res_list])
            constant_res_std = np.nanstd([res[0] for res in knn_metric_res_list])
            expl_res_std = np.nanstd([res[1] for res in knn_metric_res_list])
            all_results[model].append((dataset, constant_res, expl_res))
            all_results_std[model].append((dataset, constant_res_std, expl_res_std))
            all_vals.append(expl_res)
    mean_over_all = np.nanmean(all_vals)
    std_over_all = np.nanstd(all_vals)
    unique_datasets = sorted({d for m in all_results for d, _, _ in all_results[m]}, key=lambda x: extract_sort_keys(x, regression))
    colors = {d: cmap[d] for i, d in enumerate(unique_datasets)}
    models_plotted = list(all_results.keys())
    markers_to_models = {m: MODEL_TO_MARKER[m] for i, m in enumerate(models_plotted)}
    current_min_x_y = 1
    current_max_x_y = -np.inf
    for i, model in enumerate(models_plotted):
        x_vals, y_vals, colors_list = [], [], []
        for dataset, metr_constant, metr_g in all_results[model]:
            if regression:
                if np.abs(metr_constant) >= mean_over_all + 2*std_over_all: 
                    print(f"dataset{dataset},{model} exceeds mean+2*std for constant model metric.")
                    metr_constant = mean_over_all + 2*std_over_all
                if np.abs(metr_g) >= mean_over_all + 2*std_over_all:
                    metr_g = mean_over_all + std_over_all
                    print(f"dataset{dataset},{model} exceeds mean+2*std for local model metric.")
                if "MSE $g_x$" == metric and metr_constant < 0:
                    print(f"Warning: {model} {dataset} has negative constant model metric.")
                if "MSE $g_x$" == metric and metr_g < 0:
                    print(f"Warning: {model} {dataset} has negative local model metric.")
            metr_constant = metr_constant if metr_constant >= cut_off else replace_with
            metr_g = metr_g if metr_g >= cut_off else replace_with
            x_vals.append(metr_constant)
            y_vals.append(metr_g)
            # colors_list.append(MODEL_TO_COLOR[model])
            colors_list.append(colors[dataset])
        current_min_x_y = min(current_min_x_y, min(x_vals), min(y_vals))
        current_max_x_y = max(current_max_x_y, max(x_vals), max(y_vals))
        ax.scatter(x_vals, y_vals, c=colors_list, marker=markers_to_models[model], 
                   s=50, alpha=0.8, edgecolors="black", linewidths=0.2)
        if not plot_individual_random_seed and (type(random_seed) != int or plot_downsample_fraction):
            x_vals_std, y_vals_std = [], []
            for dataset, constant_res_std, expl_res_std in all_results_std[model]:
                x_vals_std.append(constant_res_std)
                y_vals_std.append(expl_res_std)
            for i, (xv, yv, xv_std, yv_std) in enumerate(zip(x_vals, y_vals, x_vals_std, y_vals_std)):
                if yv > cut_off and xv > cut_off:
                    ellipse = plt.matplotlib.patches.Ellipse(
                    (xv, yv), width=2*xv_std, height=2*yv_std,
                    edgecolor='none', facecolor=colors_list[i], alpha=0.3, zorder=0
                    )
                    ax.add_patch(ellipse)
               
       
    # === Handle axes ===
    if is_diff or "R2" in metric:
        ax.axhline(0, color='black', linestyle='--', alpha=0.7)
        ax.set_ylim((replace_with, 1))
    elif "MSE $g_x$" == metric:
        ax.plot([0, current_max_x_y], [0, current_max_x_y], linestyle='--', color='gray', alpha=0.7)
        yticks = ax.get_yticks()
        yticks, ytick_labels = edit_ticks_regression(yticks,  mean_over_all+2*std_over_all, f"$>$mean+2*std", include_lower=0, exclude_upper=mean_over_all + 3*std_over_all+0.1)
        ax.set_yticks(sorted(set(yticks)))
        ax.set_yticklabels(ytick_labels)
        xticks = ax.get_xticks()
        xticks, xtick_labels = edit_ticks_regression(xticks,  mean_over_all+2*std_over_all, f"$>$mean+2*std", include_lower=0, exclude_upper=mean_over_all + 3*std_over_all+0.1)
        ax.set_xticks(sorted(set(xticks)))
        ax.set_xticklabels(xtick_labels)
        ax.set_ylim((0, current_max_x_y))
        ax.set_xlim((0, current_max_x_y))
    elif "Accuracy" in metric:
        current_min_x_y = np.max([replace_with, current_min_x_y])
        current_min_x_y = np.min([cut_off, current_min_x_y])
        ax.axhline(0.5, color='black', linestyle='--', alpha=0.7)
        ax.axvline(0.5, color='black', linestyle='--', alpha=0.7)
        exclude_lower = current_min_x_y
        yticks = ax.get_yticks()
        yticks, ytick_labels = edit_ticks(yticks, replace_with, f"$<${cut_off}",exclude_lower = current_min_x_y, exclude_upper=1.1)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ytick_labels)
        xticks = ax.get_xticks()
        xticks, xtick_labels = edit_ticks(xticks, replace_with, f"$<${cut_off}", exclude_lower = current_min_x_y, exclude_upper=1.1)
        ax.set_xticks(yticks)
    elif is_ratio: 
        ax.axhline(0, color='black', linestyle='--', alpha=0.7)
        exclude_lower = replace_with-0.2    
        current_min_x_y = np.max([replace_with, current_min_x_y])
        current_min_x_y = np.min([cut_off, current_min_x_y])
        yticks = ax.get_yticks()
        yticks, ytick_labels = edit_ticks_regression(yticks, replace_with, f"$<${cut_off}",exclude_lower = exclude_lower, include_lower=current_max_x_y, exclude_upper=1.1)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ytick_labels)
        xticks = ax.get_xticks()
        xticks, xtick_labels = edit_ticks_regression(xticks, replace_with, f"$<${cut_off}", exclude_lower = exclude_lower, include_lower=current_max_x_y, exclude_upper=1.1)
        ax.set_xticks(yticks)
        ax.set_xticklabels(ytick_labels)
    ax.plot([exclude_lower, 1], [exclude_lower, 1], linestyle='--', color='gray', alpha=0.7)
    ax.set_ylim((exclude_lower, 1))
    ax.set_xlim((exclude_lower, 1))
    # # === Labels ===
    y_axis_label = get_y_label(y_metrics="local_model", filter=filter, regression=regression)
    x_axis_label = get_x_label(x_metrics="constant_model", complexity_regression="best", regression=regression, filter=filter)
    ax.set_ylabel(y_axis_label, labelpad=-5)
    ax.set_xlabel(x_axis_label)
    method_title = ' '.join(method.split('/')[-1].split('_'))
    title = get_title(method_title, is_diff, regression=regression, x_metrics="constant_model", filter=filter, create_legend_to_ax=create_legend_to_ax)
    title = f"Comparison of local explanation\n vs. constant explanation"
    ax.set_title(title)

    ax.grid(True, alpha=0.3)
    if create_legend_to_ax:
        create_dual_legend(ax, unique_datasets, colors, models_plotted, markers_to_models)

    if save:
        plt.savefig(
            f"graphics/complexity_vs_{filter}_metrics_{method.split('/')[-1]}_{dataset}.pdf",
            bbox_inches='tight',
            dpi=100,
            metadata={'CreationDate': None}
        )
    return ax, unique_datasets, colors, models_plotted, markers_to_models
def plot_model_performances_scatter(models, 
                        datasets, 
                        distance="euclidean", 
                        regression=False, 
                        synthetic=False, 
                        random_seed=42,
                        complexity_regression="best", 
                        x_metrics = "baseline_preds",
                        y_metrics = "model_labels",
                        ax=None, 
                        width=TEXT_WIDTH,
                        height=TEXT_WIDTH * 0.6,
                        save=False,
                        detailled_colors=False):    
    """
    Scatter plot: x-axis = performance_smple_model_model_preds, y-axis = diff (model - simple model on true labels)
    Each point: (model, dataset). Color by dataset, marker by model.
    """
    
    create_legend_to_ax = ax is None
    
    if regression:
        from src.utils.process_results import get_synthetic_dataset_friendly_name_regression as get_friendly_name
    else:
        from src.utils.process_results import get_synthetic_dataset_friendly_name as get_friendly_name
    if ax is None:
        fig, ax = plt.subplots(figsize=(width, height))
    else:
        fig = ax.figure
    if detailled_colors:
        cmap = COLOR_TO_REG_DATASET_DETAILLED if regression else COLOR_TO_CLF_DATASET_DETAILLED
    else:
        cmap = COLOR_TO_REG_DATASET if regression else COLOR_TO_CLF_DATASET
    cutoff = 1 if regression else 0.5
    minimum_performance = 0.2 if regression else 0.6
    if y_metrics == "difference":
        minimum_performance = -np.inf
    results = []
    for model in models:
        for dataset in datasets:
            try:
                max_complexity_performance, max_baseline_true_labels, performance_model = get_performance_metrics_smpl_complex_models(model, 
                                                                      dataset, 
                                                                      distance=distance, 
                                                                      regression=regression,
                                                                      synthetic=synthetic, 
                                                                      random_seed=random_seed, 
                                                                      complexity_regression=complexity_regression)
                if x_metrics == "baseline_preds":
                    x = max_baseline_true_labels
                elif x_metrics == "complexity":
                    x = 1-max_complexity_performance
                if y_metrics == "model_labels":
                    y = performance_model
                elif y_metrics == "difference":
                    y = performance_model - max_baseline_true_labels
                results.append((model, get_friendly_name(dataset), x, y))
            except Exception as e:
                print(f"Skipping {model} on {dataset}: {e}")

    unique_datasets = sorted({d for _, d, _, _ in results}, key=lambda x: extract_sort_keys(x, regression))
    print(f"Unique datasets: {unique_datasets}")
    colors = {d: cmap[d] for d in unique_datasets}
    models_plotted = sorted({m for m, _, _, _ in results})
    markers_to_models = {m: MODEL_TO_MARKER[m] for i, m in enumerate(models_plotted)}
    dataset_to_exclude = []
    for model, dataset, x, y in results:
        if ("syn" in dataset) and (y < minimum_performance or dataset in dataset_to_exclude):
            dataset_to_exclude.append(dataset)
            if dataset in unique_datasets:
                unique_datasets.remove(dataset)
            continue
        ax.scatter(x, y, color=colors[dataset], marker=markers_to_models[model], s=50, alpha=0.8, edgecolors="black", linewidths=0.2)
    # smpl_model = "Lin. Reg" if regression and complexity_regression else ("kNN-reg." if regression else ("Log. Reg" if complexity_regression else "kNN-clf."))
    print(f"Excluded datasets: {set(dataset_to_exclude)}")
    smpl_model = get_smple_model_name(complexity_regression, regression)
    if x_metrics == "baseline_preds":
        ax.set_xlabel(f"{"Best " if complexity_regression == "best" else ""}{"$R^2$" if regression else "accuracy"} of {smpl_model}\n on true labels")
    elif x_metrics == "complexity":
        x_label = "Model complexity $m^R$" if regression else "Model complexity $m^C$" #f"{"Lowest " if complexity_regression == "best" else ""}{"$1-R^2$" if regression else "error"} of {smpl_model}\non model predictions"
        # ax.set_xlabel(f"{"Lowest " if complexity_regression == "best" else ""}{"$1-R^2$" if regression else "error"} of {smpl_model}\n on model predictions")
        ax.set_xlabel(x_label)
    if y_metrics == "model_labels":
        ax.set_ylabel(f"{"$R^2$" if regression else "Accuracy"} of complex models")
    elif y_metrics == "difference":
        ax.set_ylabel(f"Difference in {"$R^2$" if regression else "accuracy"}")
    task_title = "Regression: " if regression else "Classification: "

    if x_metrics == "baseline_preds" and y_metrics == "model_labels":
        # ax.set_title(f"{task_title}Baseline vs.\n complex models on true labels")
        ax.set_title(f"{task_title}\nPerformance comparison on test set")
    elif x_metrics == "complexity" and y_metrics == "difference":
        ax.set_title(f"Model complexity vs.\n Performance gain over baseline models")
    else:
        ax.set_title(f"{"Baseline on true labels" if x_metrics == "baseline_preds" else "Complexity of f"} vs. {"\nComplex models on true labels" if y_metrics == "model_labels" else "Difference of\n complex and baseline models on true labels" }")
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    if y_metrics == "model_labels":
        ax.set_ylim((0, 1)) if regression else ax.set_ylim((0.5, 1))
        if x_metrics == "baseline_preds":
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.7)
        ax.set_aspect('equal', adjustable='box')
    elif y_metrics =="difference":
        ax.axhline(0, color='black', linestyle='--', alpha=0.7)  
    if x_metrics == "complexity":
        ax.set_xlim((0, cutoff))
    else:
        ax.set_xlim((1-cutoff, 1))
    if create_legend_to_ax:
        create_dual_legend(ax, unique_datasets, colors, models_plotted, markers_to_models)
    plt.tight_layout()
    if save:
        plt.savefig("graphics/knn_vs_diff_scatter.pdf", bbox_inches='tight', dpi=100, metadata={'CreationDate': None})
    return ax, unique_datasets, colors, models_plotted, markers_to_models

