from src.explanation_methods.base import BaseExplanationMethodHandler
import lime.lime_tabular
from src.explanation_methods.lime_analysis.lime_local_classifier import get_feat_coeff_intercept
import os.path as osp
import os
import numpy as np
from joblib import Parallel, delayed
import torch

class LimeHandler(BaseExplanationMethodHandler):
    def set_explainer(self, **kwargs):
        args = self.args
        trn_feat = kwargs.get('dataset')
        if type(trn_feat)== torch.Tensor:
            trn_feat = trn_feat.numpy()
        class_names = kwargs.get('class_names')
        mode = "regression" if args.regression else "classification"
        self.explainer = lime.lime_tabular.LimeTabularExplainer(trn_feat, 
                                                    feature_names=np.arange(trn_feat.shape[1]),
                                                      class_names=class_names, 
                                                      discretize_continuous=True, 
                                                      mode=mode, 
                                                      random_state=args.random_seed_synthetic_data, 
                                                      kernel_width=args.kernel_width)
    
    def explain_instance(self, **kwargs):
        return self.explainer.explain_instance(**kwargs)

    def compute_lime_explanations(self, explainer, tst_feat, predict_fn, num_lime_features, distance_metric, sequential_computation=True):
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
                explanations = [explainer.explain_instance(instance, predict_fn, top_labels=1, num_features=num_lime_features, distance_metric=distance_metric) for instance in tst_feat]
        return explanations
    
    def lime_explanations_to_array(self, explanations):
        """
        Converts the LIME explanations to a numpy array.
        """
        coeffs_array = []
        mode = "regression" if self.args.regression else "classification"
        for exp in explanations:
            feat_ids, coeffs, intercept = get_feat_coeff_intercept(exp, mode)
            coeffs_array.append(coeffs)
        return np.array(coeffs_array)

    def compute_explanations(self, results_path, predict_fn, tst_data, tst_set=True):
        args = self.args
        # Construct the explanation file name and path
        explanation_file_name = f"explanations_kernel_width-{args.kernel_width}_model_regressor-{args.model_regressor}_distance_measure-{args.distance_measure}"
        explanations_dir = osp.join(results_path, "explanations")
        explanation_file_path = osp.join(explanations_dir, explanation_file_name)
        print(f"using explanation path: {explanation_file_path}")

        if not osp.exists(explanations_dir):
            os.makedirs(explanations_dir)
        
        if osp.exists(explanation_file_path+".npy"):
            print(f"Using precomputed explanations from: {explanation_file_path}")
            explanations = np.load(explanation_file_path+".npy", allow_pickle=True)
            print(f"{len(explanations)} explanations loaded")
        else:
            # raise FileNotFoundError(
            #     f"Precomputed explanations not found at {explanation_file_path}. "
            #     "Please run the explanation computation step or provide a precomputed file."
            # )
            tst_data = tst_data.features
            print("Precomputed explanations not found. Computing explanations for the test set...")
            explanations = self.compute_lime_explanations(self.explainer, tst_data, predict_fn, args.num_lime_features, sequential_computation=args.debug, distance_metric=args.distance_measure)
            
            # Save the explanations to the appropriate file
            np.save(explanation_file_path, explanations)
            print(f"Finished computing and saving explanations to: {explanation_file_path}")
            
        coeffs = self.lime_explanations_to_array(explanations)
        return coeffs
    
    
    def get_experiment_setting(self, n_nearest_neighbors):
        args = self.args
        df_setting = "dataset_test"
        # df_setting += "_val" if self.args.include_val else ""
        # df_setting += "_trn" if self.args.include_trn else ""
        df_setting += "_downsampled"
        experiment_setting = f"{df_setting}_kernel_width-{args.kernel_width}_model_regr-{args.model_regressor}_model_type-{args.model_type}_dist_measure-{args.distance_measure}_random_seed-{self.args.random_seed}_difference_vs_kNN"
        experiment_setting = f"kNN-1-{np.round(n_nearest_neighbors, 2)}_"+experiment_setting
        if self.args.regression:
            experiment_setting = "regression_" + experiment_setting
        return experiment_setting
    

