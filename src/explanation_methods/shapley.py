from src.explanation_methods.base import BaseExplanationMethodHandler
import os.path as osp
import os
import numpy as np
from joblib import Parallel, delayed
import torch
from captum.attr import GradientShap, KernelShap, ShapleyValueSampling
from torch.utils.data import DataLoader
import h5py
import shap

class ShapleyHandler(BaseExplanationMethodHandler):
    def set_explainer(self, **kwargs):
        self.predict_fn = kwargs.get('model')
        train_data = kwargs.get('dataset', None)
        self.train_data = shap.sample(train_data, 100) 
        # pred_trn = model(self.train_data)
        self.baseline = torch.zeros((1, self.train_data.shape[1]), dtype=torch.float32)
        self.explainer = self._get_explainer(self.predict_fn)
        self.shap_variant = self._get_shap_variant()


    def get_experiment_setting(self, n_nearest_neighbors):
        df_setting = "dataset_test"
        df_setting += "_val" if self.args.include_val else ""
        df_setting += "_trn" if self.args.include_trn else ""
        setting = f"{self.args.method}-{self.shap_variant}-{df_setting}_model_type-{self.args.model_type}_dist_measure-{self.args.distance_measure}_random_seed-{self.args.random_seed}_difference_vs_kNN"
        setting = f"kNN-1-{n_nearest_neighbors}_"+setting   
        if self.args.regression:
            setting = setting + "_regression" 
        return setting
    
    def compute_explanations(self, results_path, predict_fn, tst_data, tst_set=True):
        batch_size = 1 if self._get_shap_variant() == "kernel_shap" else self.args.chunk_size

        tst_feat_for_expl_loader = DataLoader(tst_data, batch_size=batch_size, shuffle=False)
        device = torch.device("cpu")
        feature_attribution_folder = osp.join(results_path, 
                                        "feature_attributions")
        feature_attribution_file_path = osp.join(feature_attribution_folder, f"feature_attribution_{self.shap_variant}.h5")

        print("Looking for feature attributions in: ", feature_attribution_file_path)
        if osp.exists(feature_attribution_file_path): 
            print(f"Using precomputed feature attributions from: {feature_attribution_file_path}")
            with h5py.File(feature_attribution_file_path, "r") as f:
                feature_attributions = f["feature_attribution"][:]
            feature_attributions = torch.tensor(feature_attributions).float().to(device)
        else:
            print("Precomputed feature attributions not found. Computing feature attributions for the test set...")
            if not osp.exists(feature_attribution_folder):
                os.makedirs(feature_attribution_folder)
            feature_attributions = self.compute_feature_attributions(predict_fn, tst_feat_for_expl_loader)
            
            with h5py.File(feature_attribution_file_path, "w") as f:
                f.create_dataset("feature_attribution", data=feature_attributions.cpu().numpy())
        if feature_attributions.dim() == 3:
            feature_attributions = feature_attributions.squeeze()
        return feature_attributions
    
    def compute_feature_attributions(self, predict_fn, data_loader_tst):
        feature_attribution = []
        for i, batch in enumerate(data_loader_tst):
            Xs = batch
            feature_attr = self.explain_instance(input=Xs)
            if isinstance(feature_attr, torch.Tensor):
                feature_attr = feature_attr.float()
            elif isinstance(feature_attr, np.ndarray):
                feature_attr = torch.tensor(feature_attr, dtype=torch.float32)
            feature_attribution.append(feature_attr)
            print("computed the first stack of feature attributions")
        return torch.cat(feature_attribution, dim=0)

class GradientShapHandler(ShapleyHandler):
    def _get_explainer(self, predict_fn):
        return GradientShap(predict_fn)
    
    def _get_shap_variant(self):
        return "gradient_shap"

    def explain_instance(self, **kwargs):
        input_tensor = kwargs['input']
        target = kwargs.get('target', None)
        return self.explainer.attribute(input_tensor, baselines=self.baseline, target=target, n_samples=25)

class CaptumShapHandler(ShapleyHandler):
    def _get_explainer(self, predict_fn):
        return ShapleyValueSampling(predict_fn)

    def _get_shap_variant(self):
        return "captum_shap"

    def explain_instance(self, **kwargs):
        input_tensor = kwargs['input']
        if isinstance(input_tensor, np.ndarray):
            input_tensor = torch.tensor(input_tensor, dtype=torch.float32)
        try:
            return self.explainer.attribute(input_tensor, baselines=self.baseline)
        except AssertionError:
            print(f"Input tensor shape: {input_tensor.shape}")
            print(f"Input tensor has NaNs: {np.isnan(input_tensor).any()}")
            print(f"Output model: {self.model(input_tensor)}")
            raise



    
class TreeShapHandler(ShapleyHandler):
    def _get_explainer(self, predict_fn):
        model =  predict_fn.__self__.model.model #model.model is actual lightgbm object= 
        self.model = model
        model_output = "raw" if self.args.regression else "probability"
        return shap.TreeExplainer(model, data=self.train_data, model_output=model_output)
    def _get_shap_variant(self):
        return "tree_shap"
    def explain_instance(self, **kwargs):
        input_tensor = kwargs['input']
        if isinstance(input_tensor, torch.Tensor):
            input_tensor = input_tensor.numpy()
        try:
            return self.explainer.shap_values(input_tensor)
        except AssertionError:
            print(f"Input tensor shape: {input_tensor.shape}")
            print(f"Input tensor has NaNs: {np.isnan(input_tensor).any()}")
            print(f"Output model: {self.model.predict(input_tensor)}")
            print(f"Model output has NaNs: {np.isnan(self.model.predict(input_tensor)).any()}")
            raise

class KernelShapHandler(ShapleyHandler):
    def _get_explainer(self, predict_fn):
        shap_predict_fn = PredictWrapper(predict_fn)
        if isinstance(self.train_data, torch.Tensor):
            data = self.train_data.numpy()
            data = shap.sample(data, 100)
        return shap.KernelExplainer(shap_predict_fn, data=data, model_output="raw")
    def _get_shap_variant(self):
        return "kernel_shap"
    def explain_instance(self, **kwargs):
        input_tensor = kwargs['input']
        if isinstance(input_tensor, torch.Tensor):
            input_tensor = input_tensor.numpy()
        print(self.explainer.shap_values(input_tensor))
        return self.explainer.shap_values(input_tensor)

class CaptumKernelShapHandler(ShapleyHandler):
    def _get_explainer(self, predict_fn):
        return KernelShap(predict_fn)
    def _get_shap_variant(self):
        return "kernel_shap"
    def explain_instance(self, **kwargs):
        input_tensor = kwargs['input']
        target = kwargs.get('target', None)
        return self.explainer.attribute(input_tensor, baselines=self.baseline, n_samples=200)



class PredictWrapper:
    def __init__(self, predict_fn):
        self.predict_fn = predict_fn

    def __call__(self, X):
        """
        Wrapper function that SHAP can call
        X: numpy array of shape (n_samples, n_features)
        Returns: numpy array of probabilities
        """
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X_tensor = torch.FloatTensor(X)
            predictions = self.predict_fn(X_tensor)
            return predictions.numpy()

