import numpy as np
from sklearn.neighbors import BallTree
import torch
import torch.nn.functional as F
from src.utils.metrics import binary_classification_metrics_per_row, regression_metrics_per_row, impurity_metrics_per_row
from src.utils.sampling import uniform_ball_sample

def linear_classifier(samples_in_ball, saliency_map):
    # samples in ball: kNN: (num_test_samples, num_closest_points,  num_feat) or R: (num_test_samples , num_closest_points , n_samples_around_instance, num_feat)
    if samples_in_ball.ndim == 4:
        return torch.einsum('bc, bksc -> bks', saliency_map.float(), samples_in_ball) # (num_test_samples, num_closest_points, n_samples_around_instance)
    else:
        return torch.einsum('bc, bkc -> bk', saliency_map.float(), samples_in_ball) # (num_test_samples, num_closest_points)

def compute_gradmethod_preds_for_all_kNN(
                tst_feat, 
                predictions_tst_feat, 
                predictions_baseline, 
                saliency_map, 
                predict_fn, 
                samples_in_ball,
                n_samples_around_instance,
                top_labels,
                is_integrated_grad=True,  
                pred_threshold=None,
                proba_output=False):
    sample_dim = samples_in_ball.ndim
    samples_reshaped = samples_in_ball.reshape(-1, samples_in_ball.shape[-1]) # kNN: (num_test_samples * num_kNN) x n_feat or R: (num_test_samples * num_kNN * n_samples_around_instance) x n_feat

    if sample_dim == 3:
        n_test_points, n_closest_points, n_features = samples_in_ball.shape
    elif sample_dim == 4:
        n_test_points, n_closest_points, n_samples_around_instance, n_features = samples_in_ball.shape
        samples_in_ball = samples_in_ball.reshape(-1, n_closest_points*n_samples_around_instance, n_features) # (num test samples * num closest points) x num features

    with torch.no_grad():
        model_preds = predict_fn(samples_reshaped)

    # 2. Predict labels of the kNN samples with the model
    num_classes = model_preds.shape[-1]
    model_preds = model_preds.reshape(n_test_points, -1 , num_classes)  # kNN:  (num_test_samples, num_closest_points, nclass) R: (num_test_samples, num_closest_points*n_samples_around_instance, nclass)

    if model_preds.shape[-1] == 1:
        if not proba_output:
            model_preds_sig = torch.sigmoid(model_preds)
            model_preds_softmaxed = torch.cat([1 - model_preds_sig, model_preds_sig], dim=-1)
            model_preds_top_label = (model_preds * (torch.tensor(top_labels)[:, None, None]*2-1)).squeeze(-1) # model act for top labels: -f(x) if top label was 0, f(x) if label was 1
        else:
            model_preds_softmaxed = torch.cat([1 - model_preds, model_preds], dim=-1)
            model_preds_top_label = model_preds_softmaxed[np.arange(len(model_preds)), :, top_labels].squeeze(-1)
    else:
        if not proba_output:
            model_preds_softmaxed = torch.softmax(model_preds, dim=-1)
        else:
            model_preds_softmaxed = model_preds
        model_preds_top_label = model_preds[torch.arange(len(model_preds)), :, top_labels].squeeze(-1)

    ## i) Get probability for the top label
    model_probs_top_label = model_preds_softmaxed[torch.arange(len(model_preds_softmaxed)), :, top_labels] # (num test samples,num_closest_points) or (num test samples, num_closest_points*n_samples_around_instance)
    
    ## ii) Get label predictions, is prediction the same as the top label?
    model_binary_pred_top_label = (torch.argmax(model_preds_softmaxed, dim=-1) == torch.tensor(top_labels)[:, None]).float() # (num test samples,num_closest_points) or (num test samples, num_closest_points*n_samples_around_instance)
    
    # 3. Predict labels of the kNN samples with the LIME explanation
    ## i)+ii) Get probability for top label, if prob > threshold, predict as top label
    if is_integrated_grad:
        local_preds = linear_classifier(samples_in_ball, saliency_map) # ∇fi(x)*x, i = argmax(f(x0)) if dim(f(x))>1 else: ∇f(x)*x
    else:
        local_preds = linear_classifier(samples_in_ball - tst_feat[:, None, :], saliency_map) # ∇fi(x)*x, i = argmax(f(x0)) if dim(f(x))>1 else: ∇f(x)*x

        
    if predictions_baseline.shape[-1] == 1:
        if is_integrated_grad:
            local_preds += predictions_baseline # ∇f(x)*x + f(0) OR: ∇sig(f(x))*x + sig(f(0))
        else:
            local_preds += predictions_tst_feat
        if not proba_output:
            local_preds *= (torch.tensor(top_labels)*2-1)[:, None] # -f(x) if argmax(f(x0)) was 0, f(x) if argmax(f(x0)) was 1
            local_probs_top_label = torch.sigmoid(local_preds)
            local_preds_top_label = local_preds
        else:
            local_probs_top_label = torch.stack([1-local_preds, local_preds], dim=-1)[torch.arange(len(top_labels)), :, top_labels] # f(x) for argmax(f(x0)), f(0) for argmax(f(0))
            local_preds_top_label = local_probs_top_label
    else:
        if is_integrated_grad:
            local_preds += predictions_baseline[torch.arange(len(top_labels)), top_labels] # # ∇fi(x)*x + fi(0) OR: ∇sig(fi(x))*x + sig(fi(0)), i = argmax(f(x)
        else:
            local_preds += predictions_tst_feat[torch.arange(len(top_labels)), top_labels] 
        if not proba_output:
            local_probs_top_label = torch.sigmoid(local_preds)
            local_preds_top_label = local_preds
        else:
            local_probs_top_label = local_preds
            local_preds_top_label = local_probs_top_label
    
    local_probs_top_label = local_probs_top_label.squeeze(-1)
    
    if pred_threshold is None:
        pred_threshold = 0.5
    local_binary_pred_top_labels = (local_probs_top_label >= pred_threshold).cpu().numpy().astype(int)

    return (model_preds_top_label.cpu().numpy(), model_binary_pred_top_label.cpu().numpy(), model_probs_top_label.cpu().numpy(),  \
            local_preds_top_label.cpu().numpy(), local_binary_pred_top_labels, local_probs_top_label.cpu().numpy())


def compute_gradmethod_local_regressionpreds_for_all_kNN(
                                        tst_feat, 
                                        predictions_tst_feat, 
                                        predictions_baseline, 
                                        saliency_map, 
                                        samples_in_ball,
                                        is_integrated_grad=True,  
                                        ):
    if predictions_tst_feat.ndim == 1:
        predictions_tst_feat = predictions_tst_feat.reshape(1, -1)#
    if predictions_baseline.ndim == 1:
        predictions_baseline = predictions_baseline.reshape(1, -1)
    # 2. Predict labels of the kNN samples with the model
    if is_integrated_grad:
        local_preds = linear_classifier(samples_in_ball, saliency_map) #  ∇f(x)*x, kNN: (num_test_samples, num_closest_points) R: (num_test_samples, num_closest_points, n_samples_around_instance)
        local_preds += predictions_baseline # ∇f(x)*x + f(0)
    else:
        local_preds = linear_classifier(samples_in_ball - tst_feat[:, None, :], saliency_map)
        local_preds += predictions_tst_feat
    return local_preds.cpu().numpy()


def compute_gradmethod_regressionpreds_for_all_kNN(
                                        tst_feat, 
                                        predictions_tst_feat, 
                                        predictions_baseline, 
                                        saliency_map, 
                                        predict_fn, 
                                        samples_in_ball,
                                        sample_around_instance, 
                                        is_integrated_grad=True,  
                                        ):
    if predictions_tst_feat.ndim == 1:
        predictions_tst_feat = predictions_tst_feat.reshape(1, -1)#
    # samples in ball: kNN: (num_test_samples, num_closest_points,  num_feat) or R: (num_test_samples , num_closest_points , n_samples_around_instance, num_feat)
    # samples_reshaped: kNN: (num_test_samples * num_closest_points), num_feat or R: (num_test_samples * num_closest_points * n_samples_around_instance, num_feat)
    samples_reshaped = samples_in_ball.reshape(-1, samples_in_ball.shape[-1]) 
    with torch.no_grad():
        # kNN: (num_test_samples * num_closest_points, 1) or R: (num_test_samples * num_closest_points * n_samples_around_instance, 1)
        model_preds = predict_fn(samples_reshaped)
    # 2. Predict labels of the kNN samples with the model
    model_preds = model_preds.reshape(*list(samples_in_ball.shape[:-1]))  # kNN: (num_test_samples, num_closest_points) R: (num_test_samples, num_closest_points, n_samples_around_instance)
    if is_integrated_grad:
        local_preds = linear_classifier(samples_in_ball, saliency_map) #  ∇f(x)*x, kNN: (num_test_samples, num_closest_points) R: (num_test_samples, num_closest_points, n_samples_around_instance)
        local_preds += predictions_baseline # ∇f(x)*x + f(0)
    else:
        local_preds = linear_classifier(samples_in_ball - tst_feat[:, None, :], saliency_map)
        local_preds += predictions_tst_feat
    return (model_preds.cpu().numpy(), local_preds.cpu().numpy())



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
