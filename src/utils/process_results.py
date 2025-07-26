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

BASEDIR = str(Path(__file__).resolve().parent.parent.parent)
def random_seed_condition(file, random_seed):
        if type(random_seed) == int:
                return f"random_seed-{random_seed}" in file
        elif type(random_seed) == bool:
                return f"random_seed" in file
        return f"random_seed-42" in file

def get_condition(mode:str, random_seed=None, downsample_size:int=2000):
    if mode == "main_analysis":
        condition = lambda x: (x.startswith("kNN") and x.endswith(".npz")) and random_seed_condition(x, random_seed) and f"downsampled-{downsample_size}_" in x
    elif mode == "tree_depth":
        condition = lambda x: "depth" in x and x.endswith(".npz")
    elif mode == "accuracy_surrogate":
        condition = lambda x: "accuracy" in x and x.endswith(".npz") 
    else:
        raise ValueError("Warning: mode doesnt exist")
    return condition

def get_results_path(mode, explanation_method=None):
    BASEDIR = "/home/grotehans/xai_locality_v2/results" #"/Users/clara/Desktop/UNI/2024:25_WS/xai_locality_v2/results"#str(Path(__file__).resolve().parent.parent.parent)
    if mode == "main_analysis":
        results_path = osp.join(BASEDIR, f"{explanation_method}")
    elif mode == "model_complexity":
        results_path = osp.join(BASEDIR, "model_complexity")
    else:
        raise ValueError("Warning: mode doesnt exist")
    return results_path

def get_results_files_dict(explanation_method: str,
                        models: list[str], 
                        datasets: list[str], 
                        mode_results_path: str,
                        mode_file: str,
                        downsample_size:int=2000, 
                        random_seed=42) -> dict:
    from pathlib import Path
    results_folder = get_results_path(mode=mode_results_path, explanation_method=explanation_method)
    results_files_dict = {}
    if type(models) == str:
        models = [models]
    if type(datasets) == str:
        datasets = [datasets]
    for model in models:
        results_files_dict[model] = {}
        for dataset in datasets:
            path_to_results = os.path.join(results_folder, model, dataset)
            if not os.path.exists(path_to_results):
                continue
            condition = get_condition(mode=mode_file, random_seed=random_seed, downsample_size=downsample_size)
            files = [os.path.join(path_to_results, f) for f in os.listdir(path_to_results) if condition(f)]
            if len(files) == 0:
                print(f"Warning: no files found for {model} on {dataset}")
                continue
            else:
                results_files_dict[model][dataset] = files
    return results_files_dict

def get_main_analysis_results(explanation_method: str,
                        models: list[str], 
                        datasets: list[str], 
                        downsample_size:int=2000,
                        random_seed=42) -> dict:
    return get_results_files_dict(explanation_method=explanation_method,
                                models=models, 
                                datasets=datasets, 
                                mode_results_path="main_analysis",
                                mode_file="main_analysis",
                                downsample_size=downsample_size,
                                random_seed=random_seed)

def get_model_complexity_accuracy_results(models: list[str], 
                                          datasets: list[str], 
                                          ) -> dict:
    return get_results_files_dict(explanation_method=None,
                                models=models, 
                                datasets=datasets, 
                                mode_results_path="model_complexity",
                                mode_file="accuracy_surrogate",
                                )
def get_model_complexity_depth_results(models: list[str], 
                                       datasets: list[str], 
                                       ) -> dict:
    return get_results_files_dict(explanation_method=None,
                                models=models, 
                                datasets=datasets, 
                                mode_results_path="model_complexity",
                                mode_file="tree_depth",
                                )

def get_distances_mean_std(fp_main_analysis):
    if len(fp_main_analysis) == 0:
        return None
    mean_distances = {"l2_distance": [], "l1_distance": [], "cosine_distance": []}
    knn_mean_distances = []
    for fp in fp_main_analysis:
        res = np.load(fp, allow_pickle=True)
        mean_distances["l2_distance"].append(np.mean(res["l2_distance"], axis=-1))
        mean_distances["l1_distance"].append(np.mean(res["l1_distance"], axis=-1))
        mean_distances["cosine_distance"].append(np.mean(res["cosine_distance"], axis=-1))
        knn_mean_distances.append(np.mean(res["distances"], axis=-1))
    mean_l2_mean_distances = {key: np.array(mean_distances[key]).mean(axis=0) for key in mean_distances}
    std_l2_mean_distances = {key: np.array(mean_distances[key]).std(axis=0) for key in mean_distances}
    mean_knn_mean_distances = np.array(knn_mean_distances).mean(axis=0)
    std_knn_mean_distances = np.array(knn_mean_distances).std(axis=0)
    return (mean_l2_mean_distances, std_l2_mean_distances), (mean_knn_mean_distances, std_knn_mean_distances)

def get_model_complexity_accuracy(fp_complexity_accuracy):
    if isinstance(fp_complexity_accuracy, list):
        fp_complexity_accuracy = fp_complexity_accuracy[0]
    res = np.load(fp_complexity_accuracy, allow_pickle=True)
    dt_res = res['decision_tree_on_preds'][1]
    return dt_res

def get_model_complexity_depth(fp_complexity_depth):
    if isinstance(fp_complexity_depth, list):
        fp_complexity_depth = fp_complexity_depth[0]
    res = np.load(fp_complexity_depth, allow_pickle=True)
    accuracy = res.get('tree_depth_preds_fit', 0)[1]
    return res['tree_depth_preds'], accuracy


def get_x_y_points(res_dict, res_dict_accuracy, res_dict_depth, accuracy_depth_threshold=0):
    """
    Get x and y points for plotting the complexity vs closeness graph.
    Args:
        res_dict (dict): Dictionary containing the results of the main analysis.
        res_dict_accuracy (dict): Dictionary containing the results of the model complexity accuracy.
        res_dict_depth (dict): Dictionary containing the results of the model complexity depth.
        complexity (str): The type of complexity to use for the x-axis. Can be either "accuracy" or "depth".
    Returns:
        array: Complexity Accuracy, containing the lists of x points for plotting.
        array: Complexity Depth, containing the lists of x points for plotting.
        tuple: A tuple containing the arrays of mean and std of y points for plotting.
        tuple: A tuple containing the arrays of mean and std of distances between the neighbors for plotting.
        list: A list of tuples containing the model and dataset names for plotting.
    """
    main_results_means = {"l2_distance": [], "l1_distance": [], "cosine_distance": []}
    main_results_stds = {"l2_distance": [], "l1_distance": [], "cosine_distance": []}
    main_results_dist_means = []
    main_results_dist_stds = []
    complexity_results_accuracy = []
    complexity_results_depth = []

    model_data_list_for_plotting = []

    for model in res_dict:
        print(model)
        for dataset in res_dict[model]:
            ## Get similarity and distances (y axis)
            fp_main_analysis = res_dict[model][dataset]
            fp_complexity_accuracy = res_dict_accuracy.get(model, {}).get(dataset, None)
            fp_complexity_depth = res_dict_depth.get(model, {}).get(dataset, None)
            if fp_complexity_accuracy is None or fp_complexity_depth is None:
                print(f"Warning: no complexity results for {model} on {dataset}")
                continue   
            
            (y_mean, y_std), (dist_knn_mean, dist_knn_std) = get_distances_mean_std(fp_main_analysis)
            
            ## Get complexity results (x axis)
            best_accuracy = get_model_complexity_accuracy(fp_complexity_accuracy)
            fp_complexity_error = 1 - best_accuracy
            complexity_results_accuracy.append(fp_complexity_error)
            tree_depth, accuracy = get_model_complexity_depth(fp_complexity_depth)
            if accuracy < accuracy_depth_threshold:
                print(f"Warning: accuracy {accuracy} below threshold {accuracy_depth_threshold} for {model} on {dataset}. Skipping.")
                tree_depth = np.nan

            main_results_means["l2_distance"].append(y_mean["l2_distance"])
            main_results_stds["l2_distance"].append(y_std["l2_distance"])
            main_results_means["l1_distance"].append(y_mean["l1_distance"])
            main_results_stds["l1_distance"].append(y_std["l1_distance"])
            main_results_means["cosine_distance"].append(y_mean["cosine_distance"])
            main_results_stds["cosine_distance"].append(y_std["cosine_distance"])

            main_results_dist_means.append(dist_knn_mean)
            main_results_dist_stds.append(dist_knn_std)
            complexity_results_depth.append(tree_depth)
            
            model_data_list_for_plotting.append((model, dataset))
    complexity_results_accuracy = np.array(complexity_results_accuracy)
    complexity_results_depth = np.array(complexity_results_depth)
    main_results_means = {key: np.array(main_results_means[key]) for key in main_results_means}
    main_results_stds = {key: np.array(main_results_stds[key]) for key in main_results_stds}
    main_results_dist_means = np.array(main_results_dist_means)
    main_results_dist_stds = np.array(main_results_dist_stds)

    return  complexity_results_accuracy, \
            complexity_results_depth, \
            (main_results_means, main_results_stds), \
            (main_results_dist_means, main_results_dist_stds), \
            model_data_list_for_plotting
