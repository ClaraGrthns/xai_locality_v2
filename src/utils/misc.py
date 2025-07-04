import random
import numpy as np
import torch
import os.path as osp
def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def load_results(data_path):
    results = np.load(data_path, allow_pickle=True)
    accuracy_array = results['accuracy']
    fraction_array = results['fraction_points_in_ball']
    thresholds = results['thresholds']
    n_samples_in_ball = results['samples_in_ball']
    ratio_all_ones = results['ratio_all_ones']
    return accuracy_array, fraction_array, thresholds, n_samples_in_ball, ratio_all_ones

def get_non_zero_cols(array):
    return array.shape[1] - np.sum(np.all(array == 0, axis=0))

def load_and_get_non_zero_cols(data_path):
    results = np.load(data_path, allow_pickle=True)
    nr_non_zero_columns = get_non_zero_cols(results['accuracy']) 
    n_points_in_ball = results['n_points_in_ball']
    accuracy= results['accuracy'][:, :nr_non_zero_columns]
    precision= results['precision'][:, :nr_non_zero_columns]
    recall= results['recall'][:, :nr_non_zero_columns]
    f1= results['f1'][:, :nr_non_zero_columns]
    mse= results['mse'][:, :nr_non_zero_columns]
    mae= results['mae'][:, :nr_non_zero_columns]
    r2=results['r2'][:, :nr_non_zero_columns]
    gini= results['gini'][:, :nr_non_zero_columns]
    ratio_all_ones= results['ratio_all_ones'][:, :nr_non_zero_columns]
    variance_proba = results['variance'][:, :nr_non_zero_columns]
    if 'variance_logit' in results:
        variance_logit = results['variance_logit'][:, :nr_non_zero_columns]
    else:
        variance_logit = None
    radius= results['radius'][:, :nr_non_zero_columns]
    return (accuracy, precision, recall, f1, mse, mae, r2, gini, ratio_all_ones, variance_proba, variance_logit, radius), n_points_in_ball


def get_path(base_folder, base_path, setting, suffix=""):
    if base_path is not None:
        return base_path
    assert (base_folder is not None) and (setting is not None), "Setting must be specified if folder is provided"
    return osp.join(base_folder, f"{suffix}{setting}")
