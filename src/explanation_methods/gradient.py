from src.explanation_methods.base import BaseExplanationMethodHandler
from captum.attr import IntegratedGradients, NoiseTunnel, GuidedGradCam, Deconvolution, Saliency, GuidedBackprop
import torch
import os.path as osp
import os
import h5py
import numpy as np
from torch.utils.data import DataLoader


class GradientMethodHandler(BaseExplanationMethodHandler):
    def explain_instance(self, **kwargs):
        return self.explainer.attribute(kwargs["input"], target=kwargs["target"])
    
    
    def compute_explanations(self, results_path, predict_fn, tst_data, tst_set=True):
        tst_feat_for_expl_loader = DataLoader(tst_data, batch_size=self.args.chunk_size, shuffle=False)
        device = torch.device("cpu")
        saliency_map_folder = osp.join(results_path, 
                                        "saliency_maps")
        
        saliency_map_file_path = osp.join(saliency_map_folder, f"saliency_map_{self.args.gradient_method}.h5")

        print("Looking for saliency maps in: ", saliency_map_file_path)
        if osp.exists(saliency_map_file_path) and (not self.args.force): 
            print(f"Using precomputed saliency maps from: {saliency_map_file_path}")
            with h5py.File(saliency_map_file_path, "r") as f:
                saliency_maps = f["saliency_map"][:]
            saliency_maps = torch.tensor(saliency_maps).float().to(device)
        else:
            print("Precomputed saliency maps not found. Computing saliency maps for the test set...")
            if not osp.exists(saliency_map_folder):
                os.makedirs(saliency_map_folder)
            saliency_maps = self.compute_saliency_maps(self.explainer, predict_fn, tst_feat_for_expl_loader, self.is_smooth_grad)
            with h5py.File(saliency_map_file_path, "w") as f:
                f.create_dataset("saliency_map", data=saliency_maps.cpu().numpy())
        return saliency_maps
    
    def get_experiment_setting(self, n_nearest_neighbors):
        df_setting = "dataset_test"
        df_setting += "_val" if self.args.include_val else ""
        df_setting += "_trn" if self.args.include_trn else ""
        setting = f"{df_setting}_grad_method-{self.args.gradient_method}_model_type-{self.args.model_type}_dist_measure-{self.args.distance_measure}_random_seed-{self.args.random_seed}_difference_vs_kNN"
        setting = f"kNN-1-{np.round(n_nearest_neighbors, 2)}_"+ setting
        if self.args.regression:
            setting = setting + "_regression" 
        return setting
    
    def compute_saliency_maps(self, explainer, predict_fn, data_loader_tst, is_smooth_grad):
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