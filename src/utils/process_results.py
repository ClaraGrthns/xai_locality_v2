import os
import numpy as np
import re
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import os.path as osp
from pathlib import Path
from collections import Counter

DATASET_TO_NUM_FEATURES = {"higgs": 24,
                           "jannis": 54,
                           "synthetic_data/n_feat50_n_informative2_n_redundant30_n_repeated0_n_classes2_n_samples100000_n_clusters_per_class2_class_sep0.9_flip_y0.01_random_state{random_seed}": 50, 
            "synthetic_data/n_feat50_n_informative10_n_redundant30_n_repeated0_n_classes2_n_samples100000_n_clusters_per_class3_class_sep0.9_flip_y0.01_random_state{random_seed}": 50,
            "synthetic_data/n_feat100_n_informative50_n_redundant30_n_repeated0_n_classes2_n_samples100000_n_clusters_per_class3_class_sep0.9_flip_y0.01_random_state{random_seed}": 100,
}


def file_matching(file, distance_measure, condition=lambda x: True):
        return condition(file) and distance_measure in file
BASEDIR = str(Path(__file__).resolve().parent.parent.parent)
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
    "Accuracy const. local model": 11,
    "Variance prob.": 12, 
    "Variance logit": 13,
    "Radius": 14,
    "Local Ratio All Ones": 16,
    "Accuracy $g_x$ - Accuracy const. local model": (0, 11),
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

def get_fraction(metr0, metr1):
    min_dim = min(metr0.shape[0], metr1.shape[0])
    metr1 = np.where(np.isclose(metr1, 0), np.nan, metr1)  # Avoid division by zero
    return 1 - metr0[:min_dim]/metr1[499:500]

def get_filter(filter):
    filters = {
        'min': np.nanmin,
        'max': np.nanmax,
        'median': np.nanmedian,
        'mean': np.nanmean,
        'argmax': np.nanargmax,
        'argmin': np.nanargmin,
    }
    return filters.get(filter)

def get_and_apply_filter(mean_diffs, filter):
    """Apply filter to mean differences."""
    filters = {
        'min': np.nanmin,
        'max': np.nanmax,
        'median': np.nanmedian,
        'mean': np.nanmean,
        'argmax': np.nanargmax,
        'argmin': np.nanargmin,
    }
    return filters.get(filter, lambda x: x[filter])(mean_diffs) if isinstance(filter, str) else mean_diffs[filter]


def extract_sort_keys(dataset, regression = False):
    """Extract sorting keys from dataset name using regex."""
    if regression: 
        # Try to match polynomial regression pattern like: syn-reg polynomial (d:100, if:10, b: 1.0, ns:0.0, er:60)
        match = re.search(r'syn\s+\w+\s+\(d:(\d+),\s*inf f.:(\d+)', dataset)
        if match:
            return (int(match.group(1)), int(match.group(2)))
        return (int(match.group(1)), int(match.group(2))) if match else (0., 0.)
    else:
        pattern = r'd:(\d+).*?inf f.:(\d+).*?clust.:(\d+).*?sep.:([\d\.]+)' # inf f.:{i}, clust.:{c}, sep.:
        match = re.search(pattern, dataset)
        if match:
            n_feat = int(match.group(1))
            inf_feat = int(match.group(2))
            n_clusters = int(match.group(3))
            class_sep = float(match.group(4))
            return n_feat, inf_feat,  n_clusters, -class_sep # Negative class_sep for descending order
        return 0, 0, 0, 0  # Default for non-matching datasets


def filter_best_performance_local_model(filepath, 
                                        metric, 
                                        summarizing_statistics, 
                                        average_over_n_neighbors,
                                        filter, 
                                        order_average_first=True, 
                                        regression=False):
    if regression:
        load_results =  load_results_regression 
        metrics_map = METRICS_MAP_REG
    else:
        load_results = load_results_clf
        metrics_map = METRICS_TO_IDX_CLF
    data, _ = load_results(filepath)
    metric_idx = metrics_map[metric]
    is_diff = "-" in metric
    is_ratio = "/" in metric
    if is_ratio:
        vals = get_fraction(data[metric_idx[0]], data[metric_idx[1]])
    elif is_diff:
        min_dim = min(data[metric_idx[0]].shape[0],data[metric_idx[1]].shape[0])
        vals = data[metric_idx[0]][:min_dim] - data[metric_idx[1]][:min_dim]
        if regression:
            var = np.where(np.isclose(data[5], 0), np.nan, data[5])  # Avoid division by zero
            vals /= var[:min_dim]
    else:
        vals = data[metric_idx]
    # filtered_res = get_and_apply_filter(summary_vals, filter)
    if order_average_first:
        summary_vals = summarizing_statistics(vals, axis=1)
        filter_for_vals = get_filter(filter)
        summary_vals = filter_for_vals(summary_vals[:average_over_n_neighbors])

    else:
        filter_for_vals = get_filter(filter)
        filtered_vals = filter_for_vals(vals, axis=0)
        summary_vals = summarizing_statistics(filtered_vals)
    
    # if regression:
    #     mse_gx = np.nanmin(data[0], axis=0).mean()
    #     mse_constant = np.nanmin(data[3], axis=0).mean()
    #     r2_gx = np.nanmax(get_fraction(data[0], data[5]), axis=0).mean()
    #     r2_constant = np.nanmax(get_fraction(data[3], data[5]), axis=0).mean()
    #     if mse_constant > mse_gx and r2_constant > r2_gx:
    #         print(f"contradiction") 
    #     if mse_constant < mse_gx and r2_constant < r2_gx:
    #         print(f"contradiction 1")
    #     filtered_vals_aux = r2_constant if is_ratio else mse_constant
    #     summary_vals_aux = summarizing_statistics(filtered_vals_aux) 
    #     if summary_vals_aux != summary_vals and "const." in metric:
    #         print(f"contradiction 2")
    if "$g_x$" not in metric:
        if order_average_first:
            if regression:
                mse_gx_argmin = np.argmax(summarizing_statistics(data[0][:500], axis=1))
                r2_gx_argmax = np.argmax(summarizing_statistics(get_fraction(data[0], data[5])[:500], axis=1))
                argmax = r2_gx_argmax if is_ratio else mse_gx_argmin
                mse_constant = summarizing_statistics(data[3], axis=1)
                r2_constant = summarizing_statistics(get_fraction(data[3], data[5]), axis=1)
                summary_vals = r2_constant if is_ratio else mse_constant
                summary_vals = summary_vals[argmax]
            else:
                accuracy_gx_argmax = np.argmax(summarizing_statistics(data[0][:500], axis=1))
                accuracy_constant = summarizing_statistics(data[11], axis=1)
                summary_vals = accuracy_constant[accuracy_gx_argmax]
        else:
            if regression:
                mse_gx_argmin = np.argmin(data[0][:500], axis=0)
                r2_gx_argmax = np.argmax(get_fraction(data[0], data[5])[:500], axis=0)
                mse_constant = data[3][mse_gx_argmin, np.arange(0, 200)]
                r2_constant = get_fraction(data[3], data[5])[r2_gx_argmax, np.arange(0, 200)]
                filtered_vals = r2_constant if is_ratio else mse_constant
                summary_vals = summarizing_statistics(filtered_vals) 
            else:
                accuracy_gx_argmax = np.argmax(data[0], axis=0)
                accuracy_constant = data[11][accuracy_gx_argmax, np.arange(0, 200)]
                filtered_vals = accuracy_constant
                summary_vals = summarizing_statistics(filtered_vals)
    return summary_vals



#   if regression and "const." in metric:
#         mse_gx_argmin = np.argmin(data[0], axis=0)
#         r2_gx_argmax = np.argmax(get_fraction(data[0], data[5]), axis=0)
#         mse_constant = data[3][mse_gx_argmin, np.arange(0, 200)]
#         r2_constant = get_fraction(data[3], data[5])[r2_gx_argmax, np.arange(0, 200)]
#         filtered_vals = r2_constant if is_ratio else mse_constant
#     else:
#         filter_for_vals = get_filter(filter)
#         filtered_vals = filter_for_vals(vals, axis=0)
#     summary_vals = summarizing_statistics(filtered_vals)


def get_metric_vals(is_ratio, is_diff, metric_idx, data):
    if is_ratio:
        vals = get_fraction(data[metric_idx[0]], data[metric_idx[1]])
    elif is_diff:
        vals = data[metric_idx[0]] - data[metric_idx[1]]
    else:
        vals = data[metric_idx]
    return vals

def get_local_vs_constant_metric_data(res_model, 
                           filter, 
                           metric, 
                           regression=False, 
                           random_seed=42,
                           kernel_width="default",
                           order_average_first=True,
                           downsample_fraction = None, 
                           summarizing_statistics=None,
                           average_over_n_neighbors=200):
    """Extract kNN and performance difference data."""
    if regression:
        from src.utils.process_results import load_results_regression as load_results
        metrics_map = METRICS_MAP_REG
    else:
        from src.utils.process_results import load_results_clf as load_results
        metrics_map = METRICS_TO_IDX_CLF
    results = []
    if summarizing_statistics is None:
        summarizing_statistics = lambda x, axis: np.nanmedian(x, axis=axis)
    for dataset, files in res_model.items():
        if downsample_fraction is not None:
            files = get_downsample_fraction_to_filepaths(files, downsample_fraction)
            if files is None: continue
        rs_files = get_random_seed_to_filepaths(files)
        if len(rs_files) == 0: continue
        random_seeds = np.array([int(rs[0])for rs in rs_files])
        files_sorted_with_rs = [str(rs[1]) for rs in rs_files]
        try: 
            files_random_seed = files_sorted_with_rs[np.where(random_seeds==random_seed)[0][0]]
        except IndexError:
            print(f"{files}: Random seed {random_seed} not found in {dataset}.")
            continue
        file_path = get_kw_fp(files_random_seed, kernel_width)
        filtered_res_g = filter_best_performance_local_model(
            filepath=file_path,
            metric=metric,
            summarizing_statistics=summarizing_statistics,
            average_over_n_neighbors=average_over_n_neighbors,
            filter = filter, 
            order_average_first=order_average_first,
            regression=regression
        )
        # process constant xai model results =========
        metric_constant = metric.replace("$g_x$", "const. local model")
        filtered_res_constant = filter_best_performance_local_model(
            filepath=file_path,
            metric=metric_constant,
            summarizing_statistics=summarizing_statistics,
            average_over_n_neighbors=average_over_n_neighbors,
            order_average_first=order_average_first,
            filter = filter, 
            regression=regression
        )
        results.append((dataset, filtered_res_constant, filtered_res_g))
    return results


def get_knn_vs_metric_data(res_model, 
                           model_name, 
                           mapping, 
                           filter, 
                           metric, 
                           distance, 
                           regression=False, 
                           random_seed=42, 
                            order_average_first=True,
                           kernel_width="default",
                           difference_to_constant_model=False,
                           summarizing_statistics=None,
                           downsample_fraction = None, 
                           complexity_regression="best", 
                           complexity_model = "",
                           average_over_n_neighbors=200):
    """Extract kNN and performance difference data."""
    results = []
    if summarizing_statistics is None:
        summarizing_statistics = lambda x, axis: np.nanmean(x, axis=axis)
    for dataset, files in res_model.items():
        if downsample_fraction is not None:
            files = get_downsample_fraction_to_filepaths(files, downsample_fraction)
            if files is None: continue
        rs_files = get_random_seed_to_filepaths(files)
        if len(rs_files) == 0: continue
        random_seeds = np.array([int(rs[0])for rs in rs_files])
        files_sorted_with_rs = [str(rs[1]) for rs in rs_files]
        try: 
            files_random_seed = files_sorted_with_rs[np.where(random_seeds==random_seed)[0][0]]
        except IndexError:
            print(f"{model_name}: Random seed {random_seed} not found in {dataset}.")
            continue
        file_path = get_kw_fp(files_random_seed, kernel_width)
        filtered_res_g = filter_best_performance_local_model(
            filepath=file_path,
            metric=metric,
            order_average_first=order_average_first,
            summarizing_statistics=summarizing_statistics,
            average_over_n_neighbors=average_over_n_neighbors,
            filter = filter, 
            regression=regression
        )
        synthetic = "syn" in dataset
        dataset_name = mapping.get(dataset, dataset)
        res_complexity, _, _ = get_performance_metrics_smpl_complex_models(model_name,
                                                dataset_name,
                                                distance=distance,
                                                regression=regression,
                                                synthetic=synthetic,
                                                random_seed=random_seed,
                                                complexity_model=complexity_model,
                                                complexity_regression=complexity_regression)
        
        if res_complexity is None: continue
        compl_metr = max(res_complexity, 0) if regression else max(res_complexity, 0.5)
        results.append((dataset, compl_metr, filtered_res_g))
    return results

def get_kw_float(kw_to_files, kernel_width):
    """Get the kernel width and file path from a list of files."""
    if not isinstance(kw_to_files, list):
        return kw_to_files
    kw_to_files_wo_inf = [(kw, f) for kw, f in kw_to_files if kw != np.inf]
    if len(kw_to_files) > 0 and len(kw_to_files_wo_inf) >  0:
        kw_to_files = kw_to_files_wo_inf
    if len(kw_to_files) > 0:
        if kernel_width == "default":
            kw_idx = len(kw_to_files) // 2 # middle kernel width
        elif kernel_width == "half":
            kw_idx = 0
        elif kernel_width == "double":
            kw_idx = -1
        if len(kw_to_files) != 3:
            if kernel_width != "default": print(f"Warning: less than 3 kernel widths found: {kw_to_files}, using the {kw_idx}th.")
        return kw_to_files[kw_idx][0]
    else:
        print(f"Warning: no kernel widths found, using the first file.")
        return None

def get_kw_fp(kw_to_files, kernel_width):
    """Get the kernel width and file path from a list of files."""
    if not isinstance(kw_to_files, list):
        return kw_to_files
    kw_to_files_wo_inf = [(kw, f) for kw, f in kw_to_files if kw != np.inf]
    if len(kw_to_files) > 0 and len(kw_to_files_wo_inf) >  0:
        kw_to_files = kw_to_files_wo_inf
    if len(kw_to_files) > 0:
        if kernel_width == "default":
            kw_idx = len(kw_to_files) // 2 # middle kernel width
        elif kernel_width == "half":
            kw_idx = 0
        elif kernel_width == "double":
            kw_idx = -1
        if len(kw_to_files) != 3:
            if kernel_width != "default": print(f"Warning: less than 3 kernel widths found: {kw_to_files}, using the {kw_idx}th.")
        return kw_to_files[kw_idx][1]
    else:
        print(f"Warning: no kernel widths found, using the first file.")
        return None

def get_str_cond_to_filepaths(str_cond, files):
    """Extract kernel widths from file paths."""
    if not isinstance(files, list):
        files = [files]
    widths = []
    for f in files:
        match = re.search(rf'{str_cond}-(\d+\.?\d*)', str(f))
        widths.append((float(match.group(1)), f) if match else (np.inf, f))
    return sorted(widths, key=lambda x: x[0])

def get_kernel_widths_to_filepaths(files):
    return get_str_cond_to_filepaths("kernel_width", files)

def get_random_seed_to_filepaths(files):
    return get_str_cond_to_filepaths("random_seed", files)

def get_downsample_fraction_to_filepaths(files, downsample_fraction):
    res =  get_str_cond_to_filepaths("downsample", files)
    if len(res) == 0: 
        print(f"Warning: no downsampled files found, returning None.")
        return None
    downsample_fractions = np.array([rs[0]for rs in res])
    files_sorted_with_ds = [str(rs[1]) for rs in res]
    try: 
        file_ds = files_sorted_with_ds[np.where(downsample_fractions==downsample_fraction)[0][0]]
        return file_ds
    except IndexError:
        print(f"Fraction {downsample_fraction} not found in {files}.")
        return None
    
def get_synthetic_dataset_mapping(datasets, regression=False):
    """Generate a mapping between user-friendly names and full synthetic dataset names"""
    mapping = {}
    if type(datasets) == str:
        datasets = [datasets]
    for dataset in datasets:
        if 'syn' in dataset:
            if regression:
                friendly_name = get_synthetic_dataset_friendly_name_regression(dataset)
            else:
                friendly_name = get_synthetic_dataset_friendly_name(dataset)
            mapping[friendly_name] = dataset
    return mapping

def get_synthetic_dataset_friendly_name(dataset_name, pattern=None):
    """Generate a user-friendly name for synthetic datasets using regex to extract parameters"""
    if pattern is None:
        pattern = r'n_feat(\d+)_n_informative(\d+).*?n_clusters_per_class(\d+).*?class_sep([\d\.]+)'
    match = re.search(pattern, dataset_name.split("/")[-1])
    if match:
        d = match.group(1)       # number of features
        i = match.group(2)       # number of informative features
        c = match.group(3)       # clusters per class
        s = match.group(4)       # class separation
        hypercube_param = "hc: n" if "hypercubeFalse" in dataset_name else "hc: y"
        return f"syn \n(d:{d}, inf f.:{i}, clust.:{c}, sep.:{s}, {hypercube_param})"
    return dataset_name

def get_synthetic_dataset_friendly_name_regression(dataset_name, pattern=None):
    """Generate a user-friendly name for synthetic regression datasets using regex to extract parameters"""
    if pattern is None:
        pattern = r'regression_(\w+)_n_feat(\d+)_n_informative(\d+)_n_samples(\d+)_noise([\d\.]+)_bias([\d\.]+)(?:_random_state(\d+))?(?:_effective_rank(\d+)_tail_strength([\d\.]+))?'
    friedman_pattern = r"friedman(\d+)_n_samples200000_noise([\d\.]+)?" #f"friedman1_n_samples200000_noise0.1_random_state42"
    match = re.search(pattern, dataset_name.split("/")[-1])
    if match:
        mode = match.group(1)    # regression mode (e.g., 'linear', 'polynomial')
        d = match.group(2)       # number of features
        i = match.group(3)       # number of informative features
        n = match.group(4)       # number of samples
        noise = match.group(5)   # noise level
        bias = match.group(6)    # bias
        
        # Create a more readable version of the regression mode
        mode_display = {
            "polynomial": "polynomial",
            "interaction": "interaction",
            "poly_interaction": "poly_interaction",
            "multiplicative_chain": "multiplicative_chain",
            "exponential_interaction": "exponential_interaction",
            "sigmoid_mix": "sigmoid_mix",
            "hierarchical": "hierarchical",
            "piecewise": "piecewise",
            "advanced_polynomial": "adv_polynomial"
        }.get(mode, mode)
        
        # Extract additional parameters
        additional = ""
        # Extract effective rank number if present
        effective_rank_match = re.search(r'effective_rank(\d+)', dataset_name)
        if effective_rank_match:
            rank_num = effective_rank_match.group(1)
            additional += f", er:{rank_num}"
        return f"syn {mode_display} \n(d:{d}, inf f.:{i}, noise:{noise}{additional})"
    elif re.search(friedman_pattern, dataset_name.split("/")[-1]):
        match = re.search(friedman_pattern, dataset_name.split("/")[-1])
        d = match.group(1)
        noise = match.group(2)
        return f"syn friedman {d} (noise:{noise})"
    return dataset_name

def get_results_files_dict(explanation_method: str,
                        models: list[str], 
                        datasets: list[str], 
                        distance_measure: str = "euclidean", 
                        lime_features=10, 
                        sampled_around_instance=False, 
                        random_seed=42,
                        downsampled=False,
                        kernel_width = "default") -> dict:
    from pathlib import Path
    BASEDIR = str(Path(__file__).resolve().parent.parent.parent)
    results_folder = f"{BASEDIR}/results/{explanation_method}"
    results_files_dict = {}
    if type(models) == str:
        models = [models]
    if type(datasets) == str:
        datasets = [datasets]
    for model in models:
        results_files_dict[model] = {}
        for dataset in datasets:
            if downsampled:
                path_to_results = os.path.join(results_folder, model, dataset, "downsampled")
            else:
                path_to_results = os.path.join(results_folder, model, dataset)
            if not os.path.exists(path_to_results):
                continue
            if sampled_around_instance:
                condition = lambda x: x.startswith("sampled") and x.endswith("fraction.npz")
            else:
                if explanation_method == "lime":
                    if isinstance(lime_features, int) and lime_features != 10:
                        condition = lambda x: x.startswith("kNN") and f"num_features-{lime_features}.npz" in x
                    elif lime_features == "all":
                        num_feat = DATASET_TO_NUM_FEATURES[dataset]
                        condition = lambda x: (x.startswith("kNN") or x.startswith("regression")) and f"num_features-{num_feat}.npz" in x
                    else:
                        condition = lambda x: (x.startswith("kNN") or x.startswith("regression")) and x.endswith("fraction.npz")
                else:
                    condition = lambda x: (x.startswith("kNN") or x.startswith("regression")) and x.endswith("fraction.npz")
                
            def random_seed_condition(file):
                if type(random_seed) == int:
                    return f"random_seed-{random_seed}" in file
                elif type(random_seed) == bool:
                    return f"random_seed" in file
                return f"random_seed-42" in file
            def not_train_condition(file):
                return "dataset_test_val_trn" not in file
            # Combine conditions
            combined_condition = lambda x: file_matching(x, distance_measure, condition=condition) and random_seed_condition(x) and not_train_condition(x)
            files = [os.path.join(path_to_results, f) for f in os.listdir(path_to_results) if combined_condition(f)]
            if len(files) > 0 and not (isinstance(random_seed, int) and downsampled):
                upper_limits = []
                for f in files:
                    match = re.search(r'fractions-0-(\d+\.?\d*)_dataset', str(f))
                    if match:
                        upper_limits.append(float(match.group(1)))
                if len(upper_limits) > 0:
                    min_upper = min(upper_limits)
                    if random_seed == True:
                        # Take the upper limit in upper_limits that is the majority
                        counter = Counter(upper_limits)
                        min_upper = counter.most_common(1)[0][0]
                    files = [f for f in files if f"{min_upper}_dataset" in f]
            # Filter files with random_seed-42
            if explanation_method in ["lime", "lime_captum"]:
                kernel_widths = get_kernel_widths_to_filepaths(files)
                kw_float = get_kw_float(kernel_widths, kernel_width)
                for kw, f in kernel_widths:
                    if kw != kw_float:
                        files.remove(f)
                if len(files) !=5 and random_seed == True: 
                    print(f"Warning: {model} on {dataset} with distance measure {distance_measure} with {kernel_width} has {len(files)} files, expected 5.")
            # Filter files with random_seed-42  
            if len(files) == 0 and explanation_method in ["lime", "lime_captum"] :
                print(f"Warning: no files found for {model} on {dataset} with distance measure {distance_measure} with {kernel_width}.")
                continue
            # seed42_files = [f for f in files if "random_seed-42" in f or ("random_seed" not in f)]
            # if explanation_method in ["lime", "lime_captum"] and len(seed42_files)>0 and not downsampled and (random_seed!=42):
            #     kernel_widths = get_kernel_widths_to_filepaths(seed42_files)
            #     kernel_widths = [kw for kw in kernel_widths if kw[0] is not None]  # Filter out None kernel widths
            #     if len(kernel_widths) > 0:
            #         res = get_kw_fp(kernel_widths, kernel_width)
            #     else:
            #         res = seed42_files[0]
            #     if (type(random_seed) == bool) and random_seed:
            #         res = list(set(files)-set(seed42_files)| {res})
            # else:
            res = files
            if isinstance(res, list) and len(res) == 0:
                continue
            if len(res) > 0:
                results_files_dict[model][dataset] = res

    # Rename synthetic dataset keys
    for model in results_files_dict:
        keys = list(results_files_dict[model].keys())
        for key in keys:
            if "regression_synthetic_data" in key:
                friendly_name = get_synthetic_dataset_friendly_name_regression(key)
                results_files_dict[model][friendly_name] = results_files_dict[model].pop(key)
            elif "synthetic_data/" in key:
                friendly_name = get_synthetic_dataset_friendly_name(key)
                results_files_dict[model][friendly_name] = results_files_dict[model].pop(key)
    return results_files_dict

def get_non_zero_cols(array):
    return array.shape[1] - np.sum(np.all(array == 0, axis=0))


def load_results_clf(data_path):
    """
    Load results from a numpy file and extract non-zero columns for various metrics.

    Parameters:
    data_path (str): Path to the numpy file containing the results.

    Returns:
    tuple: A tuple containing:
        - A tuple of numpy arrays for the following metrics, each truncated to non-zero columns: shapes: kNN x testpoints
            - accuracy (0)
            - precision (1)
            - recall (2)
            - f1 (3)
            - mse_proba (4)
            - mae_proba (5)
            - r2_proba (6)
            - mse (7, if available else None)
            - mae (8, if available else None)
            - r2 (9, if available else None)
            - gini (10)
            - ratio_all_ones (11)
            - variance_proba (12)
            - variance_logit (13, if available else None)
            - radius (14)
            - accuraccy_constant_clf (15)
            - ratio_all_ones_local (16, if available else None)
        - n_points_in_ball (numpy array): Number of points in the ball.
    """
    results = np.load(data_path, allow_pickle=True)
    # nr_non_zero_columns = get_non_zero_cols(results['accuracy']) 
    n_points_in_ball = results['n_points_in_ball']
    # Common metrics for both methods
    accuraccy_constant_clf = results['accuraccy_constant_clf']
    accuracy = results['accuracy']
    precision = results['precision']
    recall = results['recall']
    f1 = results['f1']
    mse_proba = results['mse_proba']
    mae_proba = results['mae_proba']
    r2_proba = results['r2_proba']
    gini = results["gini"]
    ratio_all_ones = results['ratio_all_ones']
    radius = results['radius']
    variance_proba = results["variance_proba"]
    variance_logit = results.get('variance_logit', None)
    if variance_logit is not None:
        variance_logit = variance_logit
    mse = results.get('mse', None)
    if mse is not None:
        mse = mse
    mae = results.get('mae', None)
    if mae is not None:
        mae = mae
    r2 = results.get('r2', None)
    if r2 is not None:
        r2 = r2
    ratio_all_ones_local = results.get("ratio_all_ones_local", None)
    if ratio_all_ones_local is not None:
        ratio_all_ones_local = ratio_all_ones_local

    return (accuracy, precision, recall, f1,
            mse_proba, mae_proba, r2_proba,
            mse, mae, r2,
            gini, ratio_all_ones, variance_proba,
            variance_logit,
            radius,
            accuraccy_constant_clf,
            ratio_all_ones_local
            ), n_points_in_ball

def load_results_regression(data_path):
    """
    Load results from a numpy file and extract non-zero columns for various metrics.

    Parameters:
    data_path (str): Path to the numpy file containing the results.

    Returns:
    tuple: A tuple containing:
        - A tuple of numpy arrays for the following metrics, each truncated to non-zero columns: shapes: kNN x testpoints
            - mse (0)
            - mae (1)
            - r2 (2)
            - mse_constant_clf (3)
            - mae_constant_clf (4)
            - variance_logit (5)
            - radius (6)
        - n_points_in_ball (numpy array): Number of points in the ball.
    """
    results = np.load(data_path, allow_pickle=True)
    # nr_non_zero_columns = get_non_zero_cols(results['mse']) 
    n_points_in_ball = results['n_points_in_ball']
    # Extract metrics for regression tasks
    n_points_in_ball = results['n_points_in_ball']
    
    # Extract the metrics needed for regression analysis
    mse = results['mse']
    mae = results['mae']
    r2 = results['r2']
    mse_constant_clf = results['mse_constant_clf']
    mae_constant_clf = results['mae_constant_clf']
    variance_logit = results['variance_logit']
    if variance_logit is not None:
        variance_logit = variance_logit
    radius = results['radius']

    return (mse, mae, r2,
            mse_constant_clf, mae_constant_clf,
            variance_logit, radius), n_points_in_ball


def load_complexity_results(model, 
                     dataset, 
                     synthetic=False, 
                     distance_measure="euclidean", 
                     regression=False, 
                     random_seed=42, 
                    complexity_regression=False,
                    complexity_model = "", 
                     downsample_analysis=1.0):
    distance_measure = distance_measure.lower()
    suffix = "_regression" if regression else ""
    prefix = "regression_" if regression else ""
    results_folder = f"{BASEDIR}/results"
    random_seed_ending = ""
    if synthetic:
        # Get dataset name without synthetic_data/ prefix
        dataset_name = dataset.split('/')[-1]
        if complexity_regression:
            file_path = (f"{results_folder}/knn_model_preds/{model}/"
                    f"{prefix}synthetic_data/{dataset_name}/lr_on_model_preds{model}{random_seed_ending}{f"_{complexity_model}" if complexity_model != "" else complexity_model}.npz")
        else:
            file_path = (f"{results_folder}/knn_model_preds/{model}/"
                    f"{prefix}synthetic_data/{dataset_name}/kNN{suffix}_on_model_preds_{model}_"
                    f"dist_measure-{distance_measure}{random_seed_ending}{f"downsample-{np.round(downsample_analysis, 2)}" if downsample_analysis!=1.0 else ""}{f"_{complexity_model}" if complexity_model != "" else complexity_model}.npz")
    elif complexity_regression:
        file_path = (f"{results_folder}/knn_model_preds/{model}/"
                    f"{dataset}/lr_on_model_preds{model}{random_seed_ending}{f"_{complexity_model}" if complexity_model != "" else complexity_model}.npz")
    else:
        file_path = (f"{results_folder}/knn_model_preds/{model}/"
                    f"{dataset}/kNN{suffix}_on_model_preds_{model}_"
                    f"dist_measure-{distance_measure}{random_seed_ending}{f"downsample-{np.round(downsample_analysis, 2)}" if downsample_analysis!=1.0 else ""}{f"_{complexity_model}" if complexity_model != "" else complexity_model}.npz")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    try:
        res = np.load(file_path, allow_pickle=True)
        return res
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None   

def load_model_performance(model, dataset, synthetic=False, regression=False, random_seed=42, complexity_model=""):
    """Loads model performance metrics from a .npz file.

    Parameters
    ----------
    model : str
        Name of the model
    dataset : str
        Name of the dataset
    synthetic : bool, optional
        Whether the dataset is synthetic, by default False

    Returns
    -------
    numpy.lib.npyio.NpzFile
        NPZ file containing model performance metrics with key 'classification_model',
        where the array contains [accuracy, precision, recall, f1] scores
    """
    regression_str = "_regression" if regression else ""
    results_folder = f"{BASEDIR}/results"
    if synthetic:
        dataset_name = dataset.split('/')[-1]
        file_path = f"{results_folder}/knn_model_preds/{model}/{"regression_" if regression else ""}synthetic_data/{dataset_name}/model{regression_str}_performance_{model}{f"_{complexity_model}" if complexity_model != "" else complexity_model}.npz"
    else:
        file_path = f"{results_folder}/knn_model_preds/{model}/{dataset}/model{regression_str}_performance_{model}{f"_{complexity_model}" if complexity_model != "" else complexity_model}.npz"
    try:
        res = np.load(file_path, allow_pickle=True)
        return res
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None
    
def get_performance_metrics_models(model, dataset, synthetic=False, regression=False, random_seed=42, complexity_model=""):
    """Loads model performance metrics from a .npz file.
    classification output: auroc, accuracy, precision, recall, f1
    regression output: mse, mae, r2
    """
    res = load_model_performance(model, dataset, synthetic, regression, random_seed, complexity_model)
    if res is None:
        return None
    else:
        if regression:
            return float(res['regression_model'][0]), float(res['regression_model'][1]), float(res['regression_model'][2])
        else:
            return float(res['classification_model'][0]), float(res['classification_model'][1]), float(res['classification_model'][2]), float(res['classification_model'][3])
    
def get_performance_metrics_model(model, dataset, metric_str, synthetic=False, regression=False, random_seed=42):
    res = load_model_performance(model, dataset, synthetic, regression,  random_seed)
    if res is None:
        return np.nan
    else:
        metric_str_to_key_pair = {
            "AUROC": 0,
            "Accuracy": 1,
            "Precision": 2,
            "Recall": 3,
            "F1": 4
        }
        return float(res['classification_model'][metric_str_to_key_pair[metric_str]])

def get_performance_metrics_smpl_complex_models(model,
                                                dataset,
                                                distance="euclidean",
                                                regression=False,
                                                synthetic=False,
                                                random_seed=42,
                                                complexity_model="",
                                                complexity_regression="best"):
    """Get performance metrics for the model and dataset."""  
    if regression:
        from src.utils.process_results import get_best_metrics_of_complexity_of_f_regression as get_best_metrics
    else:
        from src.utils.process_results import get_best_metrics_of_complexity_of_f_clf as get_best_metrics
    if complexity_regression == "best":
        if regression:
            complexity_metrics = ["R2 kNN", "R2 Lin Reg", "R2 Decision Tree"]
        else:
            complexity_metrics = ["Accuracy kNN", "Accuracy Log Reg", "Accuracy Decision Tree"]
    elif complexity_regression in ["kNN", "knn"]:
        if regression:
            complexity_metrics = ["R2 kNN"]
        else:
            complexity_metrics = ["Accuracy kNN"]
    elif complexity_regression == "linear":
        if regression:
            complexity_metrics = ["R2 Lin Reg"]
        else:
            complexity_metrics = ["Accuracy Log Reg"]
    elif complexity_regression == "tree":
        if regression:
            complexity_metrics = ["R2 Decision Tree"]
        else:
            complexity_metrics = ["Accuracy Decision Tree"]

    max_complexity_performance = 0
    for complexity_metric in complexity_metrics:
        performance_smple_model_model_preds = get_best_metrics(model, 
                                        dataset, 
                                        complexity_metric, 
                                        synthetic=synthetic, 
                                        distance_measure=distance,
                                        complexity_regression=("kNN" not in complexity_metric),
                                        complexity_model=complexity_model,
                                        random_seed=random_seed)[0]
        max_complexity_performance = max(max_complexity_performance, performance_smple_model_model_preds)
    smple_model_true_labels = [complexity_metric + " true labels" for complexity_metric in complexity_metrics]
    max_performance_true_labels = 0
    for smple_model_true_label in smple_model_true_labels:
        performance_smpl_model_true_labels = get_best_metrics(model,
                                        dataset, 
                                        smple_model_true_label, 
                                        synthetic=synthetic, 
                                        distance_measure=distance,
                                        complexity_model=complexity_model,
                                        complexity_regression=("kNN" not in smple_model_true_label),
                                        random_seed=random_seed)[0]
        max_performance_true_labels = max(max_performance_true_labels, performance_smpl_model_true_labels)

    res = get_performance_metrics_models(model,
                                dataset,
                                synthetic=synthetic,
                                regression=regression,
                                complexity_model=complexity_model,
                                random_seed=random_seed,)
    if res is None: 
        print(f"Warning: no results found for {model} on {dataset}")
        performance_model = np.nan
    elif regression:
        performance_model = res[2]
    else:
        performance_model = res[1]
    return max_complexity_performance, max_performance_true_labels, performance_model

def get_knn_vs_diff_model_performance(model,
                                    dataset,
                                    distance="euclidean",
                                    regression=False,
                                    synthetic=False,
                                    random_seed=42,
                                    complexity_model="",
                                    complexity_regression="best"):
    performance_smple_model_model_preds, performance_smpl_model_true_labels, performance_model = get_performance_metrics_smpl_complex_models(
            model, dataset, distance=distance, regression=regression,
            synthetic=synthetic, random_seed=random_seed, complexity_regression=complexity_regression, complexity_model=complexity_model
        )
    diff = performance_model - performance_smpl_model_true_labels
    return performance_smple_model_model_preds, diff

def get_best_metrics_of_complexity_of_f_clf(model, 
                                dataset, 
                                metric_sr_ls, 
                                synthetic=False, 
                                distance_measure="euclidean", 
                                complexity_regression=False,
                                complexity_model = "", 
                                random_seed=42):
    distance_measure = distance_measure.lower()
    metric_str_to_key_pair = {
        "Accuracy kNN": ("classification", 0),
        "Precision": ("classification", 1),
        "Recall": ("classification", 2),
        "F1": ("classification", 3),
        "MSE prob.": ("proba_regression", 0),
        "MAE prob.": ("proba_regression", 1),
        "R2  prob.": ("proba_regression", 2),
        "MSE logit": ("logit_regression", 0),
        "MAE logit": ("logit_regresasion", 1),
        "R2 logit": ("logit_regression", 2),
        "Accuracy kNN true labels": ("classification_true_labels", 0),
        "Precision true labels": ("classification_true_labels", 1),
        "Recall true labels": ("classification_true_labels", 2),
        "F1 true labels": ("classification_true_labels", 3),
        "Accuracy Log Reg" : ("log_regression_res", 1),
        "Accuracy Log Reg true labels" : ("log_regression_true_y_res", 1),
        "Accuracy Decision Tree" : ("decision_tree_classification_res", 1),
        "Accuracy Decision Tree true labels" : ("decision_tree_classification_true_y_res", 1),
    }
    if type(metric_sr_ls) == str:
        metric_sr_ls = [metric_sr_ls]
    res = load_complexity_results(model = model, 
                           dataset = dataset, 
                           synthetic = synthetic, 
                           distance_measure= distance_measure, 
                           random_seed = random_seed, 
                           complexity_regression=complexity_regression,
                            complexity_model = complexity_model, 
                           regression=False)
    if res is None:
        return None
    metrics_res = []
    for metric_sr in metric_sr_ls:
        if metric_sr not in metric_str_to_key_pair:
            metrics_res.append((np.nan, np.nan))
            continue
        metric_key_pair = metric_str_to_key_pair[metric_sr]
        if complexity_regression:
            file = res.get(metric_key_pair[0], None)
            if file is None:
                print("simple model not found in file", random_seed, model, dataset, res.files, metric_sr_ls, metric_key_pair[0])
                metrics_res.append((np.nan, np.nan))
                continue
            best_metric = res[metric_key_pair[0]][metric_key_pair[1]]
            best_idx = 0
            metrics_res.append((best_metric, best_idx))
            continue
        if "MSE" in metric_sr or "MAE" in metric_sr:
            best_metric = np.min(res[metric_key_pair[0]][:, metric_key_pair[1]])
            best_idx = np.argmin(res[metric_key_pair[0]][:, metric_key_pair[1]])+1
        else:
            best_metric = np.max(res[metric_key_pair[0]][:, metric_key_pair[1]], axis = 0)
            best_idx = np.argmax(res[metric_key_pair[0]][:, metric_key_pair[1]])+1
        metrics_res.append((best_metric, best_idx))
    return metrics_res if len(metrics_res) > 1 else metrics_res[0]

def get_best_metrics_of_complexity_of_f_regression(model, 
                                       dataset, 
                                       metric_sr_ls, 
                                       synthetic=False, 
                                       distance_measure="euclidean", 
                                        complexity_regression=False,
                                        complexity_model = "", 
                                       random_seed=42):
    distance_measure = distance_measure.lower()
    metric_str_to_key_pair = {
        "MSE $g_x$": ("res_regression", 0),
        "RMSE $g_x$": ("res_regression", 0),
        "MAE $g_x$": ("res_regression", 1),
        "R2 kNN": ("res_regression", 2),
        "MSE true labels": ("res_regression_true_y", 0),
        "MAE true labels": ("res_regression_true_y", 1),
        "R2 kNN true lables": ("res_regression_true_y", 2),
        "MSE Lin Reg": ("linear_regression_res", 0),
        "MAE Lin Reg": ("linear_regression_res", 1),
        "R2 Lin Reg": ("linear_regression_res", 2),
        "R2 Lin Reg true labels": ("linear_regression_res_true_y", 2),
        "R2 Decision Tree": ("decision_tree_regression_res", 2),
        "R2 Decision Tree true labels": ("decision_tree_regression_res_true_y", 2),

    }
    if type(metric_sr_ls) == str:
        metric_sr_ls = [metric_sr_ls]

    res = load_complexity_results(model=model, 
                           dataset=dataset, synthetic = synthetic, 
                           distance_measure= distance_measure, 
                           random_seed = random_seed,  
                           regression = True,
                            complexity_model = complexity_model, 
                           complexity_regression=complexity_regression
                        )
    if res is None:
        return None
    metrics_res = []
    for metric_sr in metric_sr_ls:
        if metric_sr not in metric_str_to_key_pair:
            metrics_res.append((np.nan, np.nan))
            continue
        metric_key_pair = metric_str_to_key_pair[metric_sr]
        
        if complexity_regression:
            file = res.get(metric_key_pair[0], None)
            if file is None:
                print("simple model not found in file", model, random_seed, dataset, res.files, metric_sr_ls, metric_key_pair[0])
                metrics_res.append((np.nan, np.nan))
                continue
            best_metric = res[metric_key_pair[0]][metric_key_pair[1]]
            best_idx = 0
            metrics_res.append((best_metric, best_idx))
            continue
        if "R2" in metric_sr:
            best_metric = np.max(res[metric_key_pair[0]][:, metric_key_pair[1]])
            best_idx = np.argmax(res[metric_key_pair[0]][:, metric_key_pair[1]])+1
        else:
            best_metric = np.min(res[metric_key_pair[0]][:, metric_key_pair[1]])
            if metric_sr == "RMSE $g_x$":
                best_metric = np.sqrt(best_metric)
            best_idx = np.argmin(res[metric_key_pair[0]][:, metric_key_pair[1]])+1
        metrics_res.append((best_metric, best_idx))
    return metrics_res if len(metrics_res) > 1 else metrics_res[0]

def get_radii_for_dataset(
                        datasets,
                        regression, 
                        random_seed=True,
                        distance="euclidean",
                        summarizing_statistics=np.nanmean,
                        average_over_n_neighbors=200,
                     ):
    if regression:
        from src.utils.process_results import load_results_regression as load_results
        metrics_map = METRICS_MAP_REG
    else: 
        from src.utils.process_results import load_results_clf as load_results
        metrics_map = METRICS_TO_IDX_CLF
    res_dict = get_results_files_dict("lime", 
                                    "LightGBM", 
                                    datasets,
                                    distance, 
                                    random_seed=random_seed)
    datasets_in_dict = sorted(list(set([dataset for model, res_model in res_dict.items() for dataset in res_model.keys() if res_model is not None and len(res_model) > 0])))
    min_avg_radii_mean = np.empty((len(datasets_in_dict)), dtype=object)
    max_avg_radii_mean = np.empty((len(datasets_in_dict)), dtype=object)
    min_avg_radii_std = np.empty((len(datasets_in_dict)), dtype=object)
    max_avg_radii_std = np.empty((len(datasets_in_dict)), dtype=object)
    res_model = res_dict.get("LightGBM", None)
    for j, dataset in enumerate(datasets_in_dict):
        files = res_model.get(dataset, None)
        if files is None or len(files) == 0:
            print(f"No results found for {model} on {dataset}.")
            continue
        rs_files = [rs_fp for rs_fp in get_random_seed_to_filepaths(files)]
        random_seeds = np.array([int(rs[0])for rs in rs_files])
        files_sorted_with_rs = [str(rs[1]) for rs in rs_files]
        if random_seed == True and len(files_sorted_with_rs)!=5:
            print(files_sorted_with_rs)
        summary_radii_mins = []
        summary_radii_max = []
        for rs in random_seeds:
            try: 
                files_random_seed = files_sorted_with_rs[np.where(random_seeds==rs)[0][0]]
            except IndexError:
                print(f"{model}: Random seed {random_seed} not found in {dataset}.") 
            file_path = get_kw_fp(files_random_seed, kernel_width="default")
            data, kNN = load_results(file_path)
            radii = summarizing_statistics(data[metrics_map.get("Radius")], axis=1)[:average_over_n_neighbors]
            summary_radii_mins.append(np.nanmin(radii))
            summary_radii_max.append(np.nanmax(radii))
        min_avg_radii_mean[j] = np.nanmean(summary_radii_mins)
        max_avg_radii_mean[j] = np.nanmean(summary_radii_max)
        min_avg_radii_std[j] = np.nanstd(summary_radii_mins)
        max_avg_radii_std[j] = np.nanstd(summary_radii_max)

    return min_avg_radii_mean, max_avg_radii_mean, min_avg_radii_std, max_avg_radii_std, datasets_in_dict



def get_argmax_k_stats_for_model_dataset(method,
                                        models,
                                        datasets,
                                        regression, 
                                        random_seed=42,
                                        distance="euclidean",
                                        metric="Accuracy $g_x$",
                                        summarizing_statistics=np.nanmean,
                                        average_over_n_neighbors=200,
                                        take_max = True
                                        ):
    if regression:
        from src.utils.process_results import load_results_regression as load_results
        metrics_map = METRICS_MAP_REG
        synthetic_dataset_name = get_synthetic_dataset_friendly_name_regression
    else: 
        from src.utils.process_results import load_results_clf as load_results
        metrics_map = METRICS_TO_IDX_CLF
        synthetic_dataset_name = get_synthetic_dataset_friendly_name
    res_dict = get_results_files_dict(method, 
                                    models, 
                                    datasets,
                                    distance, 
                                    random_seed=random_seed)
    models_in_dict = sorted([model for model, res_model in res_dict.items() if res_model is not None and len(res_model) > 0])
    datasets_in_dict = sorted(list(set([dataset for model, res_model in res_dict.items() for dataset in res_model.keys() if res_model is not None and len(res_model) > 0])))
    cutoff=0 if regression else 0.5
    argmax_k_max_avg_stat_array = np.empty((len(models_in_dict), len(datasets_in_dict)), dtype=object)
    radius_of_argmax_k_max_avg_stat_array = np.empty((len(models_in_dict), len(datasets_in_dict)), dtype=object)
    for i, model in enumerate(models_in_dict):
        res_model = res_dict.get(model, None)
        for j, dataset in enumerate(datasets_in_dict):
            files = res_model.get(dataset, None)
            if files is None or len(files) == 0:
                print(f"No results found for {model} on {dataset}.")
                continue
            rs_files = [rs_fp for rs_fp in get_random_seed_to_filepaths(files)]
            random_seeds = np.array([int(rs[0])for rs in rs_files])
            files_sorted_with_rs = [str(rs[1]) for rs in rs_files]
            summary_radii = []
            for rs in random_seeds:
                try: 
                    files_random_seed = files_sorted_with_rs[np.where(random_seeds==rs)[0][0]]
                except IndexError:
                    print(f"{model}: Random seed {random_seed} not found in {dataset}.") 
                file_path = get_kw_fp(files_random_seed, kernel_width="default")
                data, kNN = load_results(file_path)
                metric_idx = metrics_map[metric]
                is_diff = "-" in metric
                is_ratio = "/" in metric
                if is_ratio:
                    vals = get_fraction(data[metric_idx[0]], data[metric_idx[1]])
                    summary_vals = summarizing_statistics(vals, axis=1)[:average_over_n_neighbors]
                elif is_diff:
                    vals = data[metric_idx[0]] - data[metric_idx[1]]
                    summary_vals = summarizing_statistics(vals, axis=1)[:average_over_n_neighbors]
                else:
                    vals = data[metric_idx]
                    summary_vals = summarizing_statistics(vals, axis=1)[:average_over_n_neighbors]
                vals = vals[:average_over_n_neighbors]
                summary_radii.append(summarizing_statistics(data[metrics_map.get("Radius")], axis=1)[:average_over_n_neighbors])
            summary_radii = np.array(summary_radii)
            vals = np.where(vals > cutoff, vals, 0)[:average_over_n_neighbors]
            summary_vals = np.where(summary_vals > cutoff, summary_vals, 0)
            avg_radii = np.mean(summary_radii, axis=0)
            if take_max:
                max_k_model_per_dataset = np.nanargmax(summary_vals)+1
            else:
                max_k_model_per_dataset = np.nanargmin(summary_vals)+1
            
            radius_of_argmax_k_max_avg_stat_array[i, j] = avg_radii[max_k_model_per_dataset-1]
            argmax_k_max_avg_stat_array[i, j] = max_k_model_per_dataset
    return argmax_k_max_avg_stat_array, None, None, radius_of_argmax_k_max_avg_stat_array, models_in_dict, datasets_in_dict

