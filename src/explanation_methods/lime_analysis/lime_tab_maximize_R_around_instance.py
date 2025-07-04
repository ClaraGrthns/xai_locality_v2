# import lime_pkg.lime.lime_tabular as lime.lime_tabular
import lime.lime_tabular
from torch_frame.gbdt import XGBoost
from torch_frame.typing import TaskType
from torch_frame.datasets import DataFrameBenchmark
import os
import os.path as osp
import xgboost
import torch
import numpy as np
from scipy import optimize


import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

from lime_analysis.lime_local_classifier import compute_lime_accuracy, get_sample_close_to_x


def predict_fn(X):
    if X.ndim == 1:
        X = X.reshape(1, -1)
    dummy_labels = np.zeros(X.shape[0])
    dtest = xgboost.DMatrix(X, label=dummy_labels,
                            feature_types=df_types,
                            enable_categorical=True)
    pred = model.model.predict(dtest)
    if model.task_type == TaskType.BINARY_CLASSIFICATION:
        pred = np.column_stack((1 - pred, pred))
    return pred


def objective(dist_threshold, x, dataset, dist_measure):
    """
    Objective function to maximize (the distance threshold itself).
    We want to maximize this value while maintaining constraints.
    """
    _, radius_ball = get_sample_close_to_x(x, dataset, dist_threshold, dist_measure)

    return -radius_ball  # Negative because scipy.minimize minimizes

def accuracy_constraint(dist_threshold, x, dataset, explainer, dist_measure, target_accuracy=0.8):
    """
    Constraint function: accuracy must be >= target_accuracy
    Returns: accuracy - target_accuracy (must be >= 0 to satisfy constraint)
    """
    accuracy, _, _, _ = compute_lime_accuracy(
        x=x,
        dataset=dataset,
        explainer=explainer,
        predict_fn=predict_fn,
        dist_measure=dist_measure,
        dist_threshold=dist_threshold[0] if isinstance(dist_threshold, np.ndarray) else dist_threshold
    )
    return accuracy - target_accuracy


def optimize_radius_ball_around_x(x, dataset, explainer, dist_measure="euclidean", target_accuracy=0.85, 
                           initial_threshold=100, bounds=(0.0, 200)):
    
    print("Starting optimization...")

    constraints = [{
        'type': 'ineq',
        'fun': accuracy_constraint,
        'args': (x, dataset, explainer, dist_measure, target_accuracy)
    }]
    
    result = optimize.minimize(
        objective,
        x0=[initial_threshold],
        args=(x, dataset, dist_measure),
        method='SLSQP',  # Sequential Least Squares Programming
        bounds=[bounds],
        constraints=constraints,
        options={'ftol': 1e-6}
    )
    
    # Get final accuracy with optimal threshold
    final_accuracy, final_ratio, final_radius, samples_within_ball = compute_lime_accuracy(
        x=x,
        dataset=dataset,
        explainer=explainer,
        predict_fn=predict_fn,
        dist_measure=dist_measure,
        dist_threshold=result.x[0]
    )
    if dist_measure == "cosine":
        final_distance = 1 - result.x[0]
    else:
        final_distance = result.x[0]
    
    return {
        'optimal_threshold': final_distance,
        'final_accuracy': final_accuracy,
        'final_radius': final_radius,
        'sample_ratio': final_ratio,
        'samples_within_ball': samples_within_ball,
        'success': result.success,
        'message': result.message,
        'iterations': result.nit,
        'optimization_result': result
    }

data_path = "/home/grotehans/pytorch-frame/data/"
model_path = '/home/grotehans/pytorch-frame/benchmark/results/xgboost_binary_jannis.pt'
distance_measure = "euclidean"
target_accuracy = 0.8
include_trn = True
include_val = True
n_experiment = 200
discretizer = 'quartile'
kernel_width=None
sampling_method = "observation_based"

model = XGBoost(task_type=TaskType.BINARY_CLASSIFICATION, num_classes=2 )
model.load(model_path)

dataset = DataFrameBenchmark(root=data_path, task_type=TaskType.BINARY_CLASSIFICATION,
                             scale='medium', idx=6)
dataset.materialize()
dataset = dataset.shuffle()
train_dataset, val_dataset, test_dataset = dataset.split()
train_tensor_frame = train_dataset.tensor_frame
test_tensor_frame = test_dataset.tensor_frame
val_tensor_frame = val_dataset.tensor_frame
tst_feat, tst_y, tst_types = model._to_xgboost_input(test_tensor_frame)
val_feat, val_y, val_types = model._to_xgboost_input(val_tensor_frame)
trn_feat, trn_y, trn_types = model._to_xgboost_input(train_tensor_frame)

df_feat = tst_feat
df_y = tst_y
df_types = tst_types
if include_trn:
    print("Including training data to compute estimates")
    df_feat = np.concatenate([trn_feat, df_feat], axis=0)
    df_y = np.concatenate([trn_y, df_y], axis=0)
if include_val:
    print("Including validation data to compute estimates")
    df_feat = np.concatenate([df_feat, val_feat], axis=0)
    df_y = np.concatenate([df_y, val_y], axis=0)


first_key = next(iter(train_tensor_frame.col_names_dict))
feature_names = train_tensor_frame.col_names_dict[first_key]
explainer = lime.lime_tabular.LimeTabularExplainer(trn_feat, 
                                                    feature_names=feature_names, 
                                                    class_names=[0,1], 
                                                    discretizer=discretizer,
                                                    kernel_width=kernel_width,
                                                    discretize_continuous=True)
# Choose a random instances from the test set
idxs = np.random.choice(len(tst_feat), n_experiment)
radiuss = []
final_accuracies = []
ratios = []
samples_within_balls = []
successes = []
for idx in idxs:
    x = tst_feat[idx]
    distances = pairwise_distances(x.reshape(1, -1), tst_feat, distance_measure)[0]
    if distance_measure == "cosine":
        distances = 1 - distances
    lower_bound = np.min(distances)
    upper_bound = np.mean(distances)+np.std(distances)
    initial_threshold = (upper_bound-lower_bound)/3
    result = optimize_radius_ball_around_x(x, 
                    tst_feat, 
                    explainer, 
                    dist_measure=distance_measure,
                    target_accuracy=target_accuracy,
                    initial_threshold=initial_threshold,
                    bounds=(lower_bound, upper_bound))
    radiuss.append(result['final_radius'])
    final_accuracies.append(result['final_accuracy'])
    ratios.append(result['sample_ratio'])
    successes.append(result['success'])
    samples_within_balls.append(result['samples_within_ball'])

radiuss = np.array(radiuss)
final_accuracies = np.array(final_accuracies)
ratios = np.array(ratios)
samples_within_balls = np.array(samples_within_balls)
# save the results
results = {
    'radiuss': radiuss,
    'final_accuracies': final_accuracies,
    'ratios': ratios,
    'samples_within_balls': samples_within_balls,
    'successes': successes

}
str_sampling = "trn" if include_trn else ""
str_sampling += "-val" if include_val else ""

results_path = f"/home/grotehans/xai_locality/results/XGBoost/Jannis/results_sampl-{sampling_method}_{str_sampling}_ac-{target_accuracy}_quantiles-{discretizer}.npy"
np.save(results_path, results)


