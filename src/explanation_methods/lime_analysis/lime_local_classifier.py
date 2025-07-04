import numpy as np
from joblib import Parallel, delayed
import torch
from tqdm import tqdm


def cut_off_probability(prob):
    prob = np.where(prob < 0, 0, prob)
    prob = np.where(prob > 1, 1, prob)
    return prob

def get_feat_coeff_intercept(exp, mode):
    """
    Extracts the feature IDs, coefficients, and intercept from a LIME explanation.

    Args:
        exp (lime.explanation.Explanation): The LIME explanation object.

    Returns:
        tuple: A tuple containing:
            - feat_ids (list): List of feature IDs.
            - coeffs (numpy.ndarray): Array of coefficients corresponding to the feature IDs.
            - intercept (float): The intercept of the local linear model.
    """
    if mode == "regression":
        top1_model = 1 # corresponds to "positive" explanation
    else:
        top1_model = exp.top_labels[0]
    feat_ids = []
    coeffs = []
    for feat_id, coeff in exp.local_exp[top1_model]:
        feat_ids.append(feat_id)
        coeffs.append(coeff)
    return np.array(feat_ids), np.array(coeffs), exp.intercept[top1_model]


def get_binary_vectorized(samples_around_xs:list , xs:np.array, explainer):
    """
    Converts the features of the instance and the explained instance into binary vectors.

    Args:
        samples_around_xs (list): samples around the instance, a list of 2D numpy arrays.
        xs (numpy.ndarray): The explained instance, a 1D numpy array.
        explainer (lime.lime_tabular.LimeTabularExplainer): The LIME explainer object.

    Returns:
        numpy.ndarray: A binary vector indicating which features match between the instance and the explained instance.
    """
    bins_sample = explainer.discretizer.discretize(samples_around_xs) # TODO: this could be a problem when np.nan is involved maybe change lime package
    bins_instance = explainer.discretizer.discretize(xs)
    binary = (bins_sample == bins_instance[:, None, :])
    return binary

def lime_pred_vectorized(binary_xs, exps, mode):
    """
    Computes the local prediction using the binary vector and the LIME explanation.

    Args:
        binary_x (numpy.ndarray): The binary vector representing the instance.
        exp (lime.explanation.Explanation): The LIME explanation object.

    Returns:
        numpy.ndarray: The local prediction for the instance.
    """
    feat_ids_array = []
    coeffs_array = []
    intercept_array = []

    for exp in exps:
        feat_ids, coeffs, intercept = get_feat_coeff_intercept(exp, mode)
        feat_ids_array.append(feat_ids)
        coeffs_array.append(coeffs)
        intercept_array.append(intercept)

    feat_ids_array = np.array(feat_ids_array)[:, None, :]
    coeffs_array = np.array(coeffs_array)[:, None, :]
    intercept_array = np.array(intercept_array)
        
    # Pre-select only needed features before dot product
    binary_x_selected = binary_xs[np.arange(binary_xs.shape[0])[:, None, None], np.arange(binary_xs.shape[1])[None, :, None], feat_ids_array]
    local_pred = intercept_array[:, None] + np.matmul(binary_x_selected, coeffs_array.transpose(0, 2, 1)).squeeze(-1)
    return local_pred

def compute_explanations(explainer, tst_feat, predict_fn, num_lime_features, distance_metric, sequential_computation=True):
    """
    Computes the LIME explanations for a set of instances.
    """

    if type(tst_feat) == torch.Tensor:
        tst_feat = tst_feat.numpy()
    with torch.no_grad():
        if not sequential_computation:
            explanations = Parallel(n_jobs=-1)(
            delayed(explainer.explain_instance)(instance, predict_fn, top_labels=1, num_features=num_lime_features, distance_metric=distance_metric)
            for instance in tst_feat
        )
        else:
            explanations = [explainer.explain_instance(instance, predict_fn, top_labels=1, num_features=num_lime_features, distance_metric=distance_metric) for instance in tqdm(tst_feat)]
        
    return explanations


def get_lime_preds_for_all_kNN(tst_set, 
                               explanations, 
                               explainer, 
                               predict_fn, 
                               samples_in_ball, 
                               sample_around_instance,
                               n_samples_around_instance,
                               distance_measure,
                               pred_threshold=0.5,
                               ):
    """
    Computes the accuracy of a LIME explanation by comparing the local model's predictions
    to the original model's predictions for samples within a ball around the instance.

    Args:
        tst_set (numpy.ndarray): The instances to explain, a 2D numpy array.
        dataset (numpy.ndarray): The dataset to look for the kNN samples, a 2D numpy array.
        explanations (list): The LIME explanations for the instances of tst_set.
        explainer (lime.lime_tabular.LimeTabularExplainer): The LIME explainer object.
        predict_fn (callable): The prediction function suitable for LIME.
        n_points_in_ball (numpy.ndarray): The number of points in the ball around the instance.
        tree (sklearn.neighbors.BallTree): The BallTree object for the dataset.
        pred_threshold (float, optional): The threshold for binary classification. Defaults to 0.5.

    Returns:
        tuple: A tuple containing: 
        - model_predicted_top_label (numpy.ndarray): Binary array indicating if the model predicted the same label for k neighbours of each instance in tst_set.
                shape: num_test_samples x num_kNN.
        - model_prob_of_top_label (numpy.ndarray): The model's softmaxed predictions: Probability for the top label.
                shape: num_test_samples x num_kNN.
        - local_preds_label (numpy.ndarray): The local model's predictions for the top label.
                shape: num_test_samples x num_kNN.
        - local_preds (numpy.ndarray): The local model's predictions.
                shape: num_test_samples x num_kNN.
        - dist (numpy.ndarray): The distances of the kNN samples from the instances of tst_set.
                shape: num_test_samples x num_kNN.
    """
    if tst_set.ndim == 1:
        tst_set = tst_set.reshape(1, -1)
    if type(tst_set) == torch.Tensor:
        tst_set = tst_set.numpy()
    top_labels = np.array([exp.top_labels[0] for exp in explanations])
    sample_dim = samples_in_ball.ndim
    samples_reshaped = samples_in_ball.reshape(-1, samples_in_ball.shape[-1]) # kNN: (num_test_samples * num_kNN) x n_feat or R:  (num_test_samples * num_kNN * n_samples_around_instance) x n_feat
    if sample_dim == 3:
        n_test_points, n_closest_points, n_features = samples_in_ball.shape
    elif sample_dim == 4:
        n_test_points, n_closest_points, n_samples_around_instance, n_features = samples_in_ball.shape
        samples_in_ball = samples_in_ball.reshape(-1, n_closest_points*n_samples_around_instance, n_features) # (num_test_samples * num_kNN) x n_feat
    binary_sample = get_binary_vectorized(samples_in_ball, tst_set, explainer) #binarize for lime prediction
    
    # 2. Predict labels of the kNN samples with the model
    model_preds = predict_fn(samples_reshaped) #shape: (num_test_samples * num_kNN) x n classes
    num_classes = model_preds.shape[-1]
    model_preds = model_preds.reshape(n_test_points, -1 , num_classes)  # kNN:  (num_test_samples, num_closest_points, nclass) R: (num_test_samples, num_closest_points*n_samples_around_instance, nclass)
    ## i) Get probability for the top label
    model_prob_of_top_label = model_preds[np.arange(model_preds.shape[0]), :, top_labels]

    ## ii) Get label predictions, is prediction the same as the top label?
    labels_model_preds = np.argmax(model_preds, axis=-1) #shape: num_test_samples x num_kNN
    model_binary_preds_top_label = (labels_model_preds == top_labels[:, None]).astype(int)

   
    # 3. Predict labels of the kNN samples with the LIME explanation
    ## i)+ii) Get probability for top label, if prob > threshold, predict as top label
    local_probs_top_label = lime_pred_vectorized(binary_sample, explanations, mode = "classification")
    if pred_threshold is None:
        pred_threshold = 0.5
    local_binary_pred_top_labels = (local_probs_top_label >= pred_threshold).astype(int)

    return (model_binary_preds_top_label, model_prob_of_top_label, \
            local_binary_pred_top_labels, cut_off_probability(local_probs_top_label))

def get_lime_local_rergression_preds_for_all_kNN(tst_set, 
                               explanations, 
                               explainer, 
                               samples_in_ball,
                ):
    
    if tst_set.ndim == 1:
        tst_set = tst_set.reshape(1, -1)
    if type(tst_set) == torch.Tensor:
        tst_set = tst_set.numpy()
    binary_sample = get_binary_vectorized(samples_in_ball, tst_set, explainer) #binarize for lime prediction
    local_preds = lime_pred_vectorized(binary_sample, explanations, mode="regression")
    return local_preds, binary_sample



def get_lime_rergression_preds_for_all_kNN(tst_set, 
                               explanations, 
                               explainer, 
                               predict_fn, 
                               samples_in_ball,
                               n_samples_around_instance,
                ):
    
    if tst_set.ndim == 1:
        tst_set = tst_set.reshape(1, -1)
    if type(tst_set) == torch.Tensor:
        tst_set = tst_set.numpy()

    sample_dim = samples_in_ball.ndim
    samples_reshaped = samples_in_ball.reshape(-1, samples_in_ball.shape[-1]) # kNN: (num_test_samples * num_kNN) x n_feat or R:  (num_test_samples * num_kNN * n_samples_around_instance) x n_feat

    if sample_dim == 3:
        n_test_points, n_closest_points, n_features = samples_in_ball.shape
    elif sample_dim == 4:
        n_test_points, n_closest_points, n_samples_around_instance, n_features = samples_in_ball.shape
        samples_in_ball = samples_in_ball.reshape(-1, n_closest_points*n_samples_around_instance, n_features) # (num_test_samples * num_kNN) x n_feat

    binary_sample = get_binary_vectorized(samples_in_ball, tst_set, explainer) #binarize for lime prediction
    
    # 2. Predict y of the kNN samples with the model
    model_preds = predict_fn(samples_reshaped) #shape: (num_test_samples * num_kNN)
    model_preds = model_preds.reshape(n_test_points, -1 ) #shape: (num_test_samples, num_kNN) or (num_test_samples, num_kNN x n_samples_around_instance)

    # 3. Predict labels of the kNN samples with the LIME explanation
    ## i)+ii) Get probability for top label, if prob > threshold, predict as top label
    local_preds = lime_pred_vectorized(binary_sample, explanations, mode="regression")
    
    return (model_preds, local_preds)

