from src.explanation_methods.base import BaseExplanationMethodHandler
from captum.attr import IntegratedGradients, NoiseTunnel, GuidedGradCam, Deconvolution, Saliency, GuidedBackprop
import torch
from src.explanation_methods.gradient_methods.local_classifier import (compute_gradmethod_preds_for_all_kNN, 
                                                                       compute_gradmethod_regressionpreds_for_all_kNN, 
                                                                       compute_gradmethod_local_regressionpreds_for_all_kNN, 
                                                                       compute_saliency_maps)
import os.path as osp
import os
import h5py
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import time 
from src.utils.sampling import uniform_ball_sample
from src.utils.metrics import binary_classification_metrics_per_row, regression_metrics_per_row, impurity_metrics_per_row


class GradientMethodHandler(BaseExplanationMethodHandler):
    def explain_instance(self, **kwargs):
        return self.explainer.attribute(kwargs["input"], target=kwargs["target"])
    
    
    def compute_explanations(self, results_path, predict_fn, tst_data):
        tst_feat_for_expl_loader = DataLoader(tst_data, batch_size=self.args.chunk_size, shuffle=False)
        device = torch.device("cpu")
        saliency_map_folder = osp.join(results_path, 
                                        "saliency_maps")
        if self.args.random_seed != 42: # not good style but too lazy to rename now.
            saliency_map_file_path = osp.join(saliency_map_folder, f"saliency_map_{self.args.gradient_method}_random_seed-{self.args.random_seed}.h5")
        else:
            saliency_map_file_path = osp.join(saliency_map_folder, f"saliency_map_{self.args.gradient_method}.h5")
        print("Looking for saliency maps in: ", saliency_map_file_path)
        if osp.exists(saliency_map_file_path) and (not self.args.force): # (not self.args.force or self.args.downsample_analysis != 1.0):
            print(f"Using precomputed saliency maps from: {saliency_map_file_path}")
            with h5py.File(saliency_map_file_path, "r") as f:
                saliency_maps = f["saliency_map"][:]
            saliency_maps = torch.tensor(saliency_maps).float().to(device)
        else:
            print("Precomputed saliency maps not found. Computing saliency maps for the test set...")
            if not osp.exists(saliency_map_folder):
                os.makedirs(saliency_map_folder)
            saliency_maps = compute_saliency_maps(self.explainer, predict_fn, tst_feat_for_expl_loader, self.is_smooth_grad)
            with h5py.File(saliency_map_file_path, "w") as f:
                f.create_dataset("saliency_map", data=saliency_maps.cpu().numpy())
        return saliency_maps
    
    def get_experiment_setting(self, fractions, max_radius):
        df_setting = "dataset_test"
        df_setting += "_val" if self.args.include_val else ""
        df_setting += "_trn" if self.args.include_trn else ""
        setting = f"{df_setting}_grad_method-{self.args.gradient_method}_model_type-{self.args.model_type}_dist_measure-{self.args.distance_measure}_random_seed-{self.args.random_seed}_accuracy_fraction"
        # else:
        #     setting = f"grad_method-{self.args.gradient_method}_model_type-{self.args.model_type}_dist_measure-{self.args.distance_measure}_accuracy_fraction"
        if self.args.create_additional_analysis_data:
            setting = f"downsample-{np.round(self.args.downsample_analysis, 2)}_" + setting
        if self.args.sample_around_instance:
            setting = f"sampled_at_point_max_R-{np.round(max_radius, 2)}_" + setting
        else:
            setting = f"fractions-0-{np.round(fractions, 2)}_"+setting   
        if self.args.regression:
            setting = "regression_" + setting
        return setting
    
    def _compute_local_preds(self, 
                             batch, 
                            df_feat_for_expl, 
                            explanation, 
                            predict_fn, ):
        with torch.no_grad():
            predictions = predict_fn(batch)
            predictions_baseline = predict_fn(torch.zeros_like(batch))
        explanation = explanation.reshape(1, -1)
        return compute_gradmethod_local_regressionpreds_for_all_kNN(
            tst_feat=batch,
            predictions_tst_feat=predictions,
            predictions_baseline=predictions_baseline,
            saliency_map=explanation,
            samples_in_ball=df_feat_for_expl,
            is_integrated_grad = self.is_integrated_gradients
        )

    def process_chunk(self, 
                      batch, 
                      tst_chunk_dist, 
                      df_feat_for_expl, 
                      explanations_chunk, 
                      predict_fn, 
                      n_points_in_ball, 
                      tree, 
                      max_radius):
        """
        Process a single chunk of data for gradient-based methods.
        """
        tst_chunk = batch  # For gradient methods, batch is already in the right format
        proba_output = self.args.model_type in ["LightGBM", "XGBoost", "LightGBM", "pt_frame_xgb", "LogReg"]
        
        with torch.no_grad():
            predictions = predict_fn(tst_chunk)
            predictions_baseline = predict_fn(torch.zeros_like(tst_chunk))

        if self.args.sample_around_instance:
            dist = np.linspace(0.001, max_radius, n_points_in_ball)   
            samples_in_ball = uniform_ball_sample(
                    centers = tst_chunk_dist,
                    R_list = dist,
                    N_per_center = self.args.n_samples_around_instance,
                    distance_measure = self.args.distance_measure,
                    random_seed=self.args.seed,)
            samples_in_ball = torch.Tensor(samples_in_ball)
            dist = np.repeat(dist[None, :], tst_chunk.shape[0], axis=0)
        else:  
            dist, idx = tree.query(tst_chunk_dist, k=n_points_in_ball, return_distance=True, sort_results=True)
            dist = np.array(dist)
            # 1. Get all the kNN samples from the analysis dataset
            samples_in_ball = [[df_feat_for_expl[idx] for idx in row] for row in idx]
            samples_in_ball = torch.stack([torch.stack(row, dim=0) for row in samples_in_ball], dim=0)  

        if self.args.regression:
            model_preds, local_preds = compute_gradmethod_regressionpreds_for_all_kNN(
                tst_feat = tst_chunk, 
                predictions_tst_feat= predictions,
                predictions_baseline = predictions_baseline,
                saliency_map = explanations_chunk, 
                predict_fn = predict_fn, 
                samples_in_ball = samples_in_ball,
                sample_around_instance = self.args.sample_around_instance,
                is_integrated_grad = self.is_integrated_gradients,
            )
            return model_preds, local_preds, dist
        else:
            if not proba_output:
                if predictions.shape[-1] == 1:
                    predictions_sm = torch.sigmoid(predictions)
                    predictions_sm = torch.cat([1 - predictions_sm, predictions_sm], dim=-1)
                else:
                    predictions_sm = torch.softmax(predictions, dim=-1)
            else:
                if predictions.shape[-1] == 1:
                    predictions_sm = torch.cat([1 - predictions, predictions], dim=-1)
                else:
                    predictions_sm = predictions
            top_labels = torch.argmax(predictions_sm, dim=1).tolist()
            model_preds, model_binary_preds, model_probs, local_preds, local_binary_preds, local_probs = compute_gradmethod_preds_for_all_kNN(
                tst_feat = tst_chunk, 
                predictions_tst_feat = predictions,
                predictions_baseline = predictions_baseline,
                saliency_map = explanations_chunk, 
                predict_fn = predict_fn, 
                samples_in_ball = samples_in_ball,
                top_labels = top_labels,
                is_integrated_grad = self.is_integrated_gradients,
                proba_output=proba_output,
                n_samples_around_instance = self.args.n_samples_around_instance,
            )
            if self.args.sample_around_instance:
                model_preds = model_preds.reshape(*list(samples_in_ball.shape[:-1]))
                model_binary_preds = model_binary_preds.reshape(*list(samples_in_ball.shape[:-1]))
                model_probs = model_probs.reshape(*list(samples_in_ball.shape[:-1]))
                local_preds = local_preds.reshape(*list(samples_in_ball.shape[:-1]))
                local_binary_preds = local_binary_preds.reshape(*list(samples_in_ball.shape[:-1]))
                local_probs = local_probs.reshape(*list(samples_in_ball.shape[:-1]))
            return model_preds, model_binary_preds, model_probs, local_preds, local_binary_preds, local_probs, dist
        
class SmoothGradHandler(GradientMethodHandler):
    def set_explainer(self, **kwargs):
        model = kwargs.get("model")
        self.explainer = NoiseTunnel(IntegratedGradients(model, multiply_by_inputs=False))
        self.is_integrated_gradients = True
        self.is_smooth_grad = True
        
    def explain_instance(self, **kwargs):
        return self.explainer.attribute(kwargs["input"], target=kwargs["target"])
    
class IntegratedGradientsHandler(GradientMethodHandler):
    def set_explainer(self, **kwargs):
        model = kwargs.get("model")
        self.explainer = IntegratedGradients(model, multiply_by_inputs=False)
        self.is_integrated_gradients = True
        
    def explain_instance(self, **kwargs):
        return self.explainer.attribute(kwargs["input"], target=kwargs["target"])

class GuidedGradCamHandler(GradientMethodHandler):
    def set_explainer(self, **kwargs):
        model = kwargs.get("model")
        self.explainer = GuidedGradCam(model, model.layer4[-1], model.layer4[-1])
        self.is_integrated_gradients = False        
        
    def explain_instance(self, **kwargs):
        return self.explainer.attribute(kwargs["input"], target=kwargs["target"])

class DeconvHandler(GradientMethodHandler):
    def set_explainer(self, **kwargs):
        model = kwargs.get("model")
        self.explainer = Deconvolution(model, model.layer4[-1], model.layer4[-1])
        self.is_integrated_gradients = False        
        
    def explain_instance(self, **kwargs):
        return self.explainer.attribute(kwargs["input"], target=kwargs["target"])
    
class SaliencyHandler(GradientMethodHandler):
    def set_explainer(self, **kwargs):
        model = kwargs.get("model")
        self.explainer = Saliency(model)
        self.is_integrated_gradients = False        
    
    def explain_instance(self, **kwargs):
        return self.explainer.attribute(kwargs["input"], target=kwargs["target"])
    
class GuidedBackpropHandler(GradientMethodHandler):
    def set_explainer(self, **kwargs):
        model = kwargs.get("model")
        self.explainer = GuidedBackprop(model)
        self.is_integrated_gradients = False        
        
    def explain_instance(self, **kwargs):
        return self.explainer.attribute(kwargs["input"], target=kwargs["target"])