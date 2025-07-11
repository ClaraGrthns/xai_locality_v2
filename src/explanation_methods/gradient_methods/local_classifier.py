import numpy as np
from sklearn.neighbors import BallTree
import torch
import torch.nn.functional as F
from src.utils.metrics import binary_classification_metrics_per_row, regression_metrics_per_row, impurity_metrics_per_row
from src.utils.sampling import uniform_ball_sample

def compute_saliency_maps(explainer, predict_fn, data_loader_tst, is_smooth_grad):
    saliency_map = []
    for i, batch in enumerate(data_loader_tst):
        Xs = batch#[0]
        preds = predict_fn(Xs)
        if preds.ndim == 2 and preds.shape[1] == 1:
            if is_smooth_grad:
                saliency = explainer.attribute(Xs, stdevs=0.5).float()
            else:
                saliency = explainer.attribute(Xs).float()
        else:
            top_labels = torch.argmax(predict_fn(Xs), dim=1).tolist()
            if is_smooth_grad:
                saliency = explainer.attribute(Xs, target=top_labels, stdevs=0.5).float()
            else:
                saliency = explainer.attribute(Xs, target=top_labels).float()
        saliency_map.append(saliency)
        print("computed the first stack of saliency maps")
    return torch.cat(saliency_map, dim=0)

    #samples_reshaped = samples_in_ball.reshape(-1, *list(samples_in_ball.shape[2:])) # (num test samples * num closest points) x num features
