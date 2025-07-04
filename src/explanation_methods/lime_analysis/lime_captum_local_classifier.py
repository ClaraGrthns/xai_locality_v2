import numpy as np
import torch
from src.explanation_methods.lime_analysis.lime_local_classifier import cut_off_probability

def linear_classifier(samples_in_ball, saliency_map):
    # samples in ball: kNN: (num_test_samples, num_closest_points,  num_feat) or R: (num_test_samples , num_closest_points , n_samples_around_instance, num_feat)
    if samples_in_ball.ndim == 4:
        return torch.einsum('bc, bksc -> bks', saliency_map.float(), samples_in_ball) # (num_test_samples, num_closest_points, n_samples_around_instance)
    else:
        return torch.einsum('bc, bkc -> bk', saliency_map.float(), samples_in_ball) # (num_test_samples, num_closest_points)

def compute_lime_all_preds_for_all_kNN(
                explanation,
                predict_fn, 
                samples_in_ball,
                top_labels,
                pred_threshold=None,
                proba_output=False):
    coefficients, intercept = explanation
    sample_dim = samples_in_ball.ndim
    samples_reshaped = samples_in_ball.reshape(-1, samples_in_ball.shape[-1]) # kNN: (num_test_samples * num_kNN) x n_feat or R: (num_test_samples * num_kNN * n_samples_around_instance) x n_feat
    top_labels = torch.tensor(top_labels)
    if sample_dim == 3:
        n_test_points, n_closest_points, n_features = samples_in_ball.shape
    elif sample_dim == 4:
        n_test_points, n_closest_points, n_samples_around_instance, n_features = samples_in_ball.shape
        samples_in_ball = samples_in_ball.reshape(-1, n_closest_points*n_samples_around_instance, n_features) # (num test samples * num closest points) x num features
    with torch.no_grad():
        model_preds = predict_fn(samples_reshaped)

    # 2. Predict labels of the kNN samples with the model
    model_preds = predict_fn(samples_reshaped) #shape: (num_test_samples * num_kNN) x n classes
    num_classes = model_preds.shape[-1]
    model_preds = model_preds.reshape(n_test_points, -1 , num_classes)  # kNN:  (num_test_samples, num_closest_points, nclass) R: (num_test_samples, num_closest_points*n_samples_around_instance, nclass)
    ## i) Get probability for the top label
    model_prob_of_top_label = model_preds[np.arange(model_preds.shape[0]), :, top_labels]

    ## ii) Get label predictions, is prediction the same as the top label?
    labels_model_preds = np.argmax(model_preds, axis=-1) #shape: num_test_samples x num_kNN
    model_binary_preds_top_label = (labels_model_preds == top_labels[:, None]).int()

    # 3. Predict labels of the kNN samples with the LIME explanation
    ## i)+ii) Get probability for top label, if prob > threshold, predict as top label
    local_probs_top_label = linear_classifier(samples_in_ball, coefficients) # ∇fi(x)*x, i = argmax(f(x0)) if dim(f(x))>1 else: ∇f(x)*x
    local_probs_top_label += intercept[:, None] # ∇fi(x)*x + fi(0), i = argmax(f(x0)) if dim(f(x))>1 else: ∇f(x)*x + f(0)
    
    if pred_threshold is None:
        pred_threshold = 0.5
    local_binary_pred_top_labels = (local_probs_top_label >= pred_threshold).cpu().numpy().astype(int)

    return (model_binary_preds_top_label, model_prob_of_top_label, \
            local_binary_pred_top_labels, cut_off_probability(local_probs_top_label))


def compute_lime_only_local_preds_for_all_kNN(explanation,
                                              samples_in_ball):
    coefficients, intercept = explanation
    coefficients = coefficients.reshape(1, -1)
    local_preds = linear_classifier(samples_in_ball, coefficients)
    local_preds += intercept
    return local_preds
        
def compute_lime_all_regressionpreds_for_all_kNN(
                                        explanation, 
                                        predict_fn, 
                                        samples_in_ball,
                                        ):
    coefficients, intercept = explanation
    
    # samples in ball: kNN: (num_test_samples, num_closest_points,  num_feat) or R: (num_test_samples , num_closest_points , n_samples_around_instance, num_feat)
    # samples_reshaped: kNN: (num_test_samples * num_closest_points), num_feat or R: (num_test_samples * num_closest_points * n_samples_around_instance, num_feat)
    samples_reshaped = samples_in_ball.reshape(-1, samples_in_ball.shape[-1]) 
    with torch.no_grad():
        # kNN: (num_test_samples * num_closest_points, 1) or R: (num_test_samples * num_closest_points * n_samples_around_instance, 1)
        model_preds = predict_fn(samples_reshaped)

    # 2. Predict labels of the kNN samples with the model
    model_preds = model_preds.reshape(*list(samples_in_ball.shape[:-1]))  # kNN: (num_test_samples, num_closest_points) R: (num_test_samples, num_closest_points, n_samples_around_instance)
    local_preds = linear_classifier(samples_in_ball, coefficients)
    local_preds += intercept[:, None]
    return (model_preds.cpu().numpy(), local_preds.cpu().numpy())




def compute_feature_attributions(explainer, predict_fn, data_loader_tst, transform = None):
    bias_feature_attribution = []
    coefs_feature_attribution = []

    for i, batch in enumerate(data_loader_tst):
        Xs = batch#[0]
        preds = predict_fn(Xs)
        if preds.ndim == 1 or preds.shape[1] == 1:
            coefs, bias = explainer.attribute(Xs, return_input_shape=True)
            coefs = coefs.float()
            bias = bias.float()
        else:
            top_labels = torch.argmax(predict_fn(Xs), dim=1).tolist()
            coefs, bias = explainer.attribute(Xs, target=top_labels, return_input_shape=True)
            coefs = coefs.float()
            bias = bias.float()
        bias_feature_attribution.append(bias)
        coefs_feature_attribution.append(coefs)
        print("computed the first stack of feature_attribution maps")
    return torch.cat(coefs_feature_attribution, dim=0), torch.cat(bias_feature_attribution, dim=0)
