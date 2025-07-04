from src.explanation_methods.base import BaseExplanationMethodHandler
import lime.lime_tabular
from src.explanation_methods.lime_analysis.lime_local_classifier import compute_explanations, get_lime_preds_for_all_kNN, get_lime_rergression_preds_for_all_kNN, get_binary_vectorized, get_lime_local_rergression_preds_for_all_kNN
from src.utils.misc import get_path
import os.path as osp
import os
import numpy as np
from joblib import Parallel, delayed
from torch.utils.data import Subset, DataLoader
import time
import torch
from src.utils.metrics import binary_classification_metrics_per_row, regression_metrics_per_row, impurity_metrics_per_row
from src.utils.sampling import uniform_ball_sample

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

    
    def compute_explanations(self, results_path, predict_fn, tst_data):
        args = self.args
        # Construct the explanation file name and path
        if args.random_seed == 42:
            explanation_file_name = f"normalized_data_explanations_test_set_kernel_width-{args.kernel_width}_model_regressor-{args.model_regressor}_distance_measure-{args.distance_measure}"
        else:
            explanation_file_name = f"normalized_data_explanations_test_set_kernel_width-{args.kernel_width}_model_regressor-{args.model_regressor}_distance_measure-{args.distance_measure}_random_seed-{args.random_seed}"
        # if args.num_lime_features > 10 :
        #     explanation_file_name += f"_num_features-{args.num_lime_features}"
        # if args.num_lime_features > 10:
        #     explanation_file_name += f"_num_features-{args.num_lime_features}"
        # if args.num_test_splits > 1:
        #     explanation_file_name = f"split-{args.split_idx}_{explanation_file_name}"
        explanations_dir = osp.join(results_path, "explanations")
        explanation_file_path = osp.join(explanations_dir, explanation_file_name)
        print(f"using explanation path: {explanation_file_path}")

        if not osp.exists(explanations_dir):
            os.makedirs(explanations_dir)
        
        if osp.exists(explanation_file_path+".npy") and (not self.args.force):
            print(f"Using precomputed explanations from: {explanation_file_path}")
            explanations = np.load(explanation_file_path+".npy", allow_pickle=True)
            print(f"{len(explanations)} explanations loaded")
        else:
            tst_data = tst_data.features
            print("Precomputed explanations not found. Computing explanations for the test set...")
            explanations = compute_explanations(self.explainer, tst_data, predict_fn, args.num_lime_features, sequential_computation=args.debug, distance_metric=args.distance_measure)
            
            # Save the explanations to the appropriate file
            np.save(explanation_file_path, explanations)
            print(f"Finished computing and saving explanations to: {explanation_file_path}")
        return explanations
    
    
    def get_experiment_setting(self, fractions, max_radius=None):
        args = self.args
        df_setting = "dataset_test"
        df_setting += "_val" if args.include_val else ""
        df_setting += "_trn" if args.include_trn else ""
        experiment_setting = f"{df_setting}_kernel_width-{args.kernel_width}_model_regr-{args.model_regressor}_model_type-{args.model_type}_dist_measure-{args.distance_measure}_random_seed-{self.args.random_seed}_accuracy_fraction"
        # else:   
        #     experiment_setting = f"{df_setting}_kernel_width-{args.kernel_width}_model_regr-{args.model_regressor}_model_type-{args.model_type}_dist_measure-{args.distance_measure}_accuracy_fraction"
        if args.create_additional_analysis_data:
            experiment_setting = f"downsample-{np.round(args.downsample_analysis, 2)}_" + experiment_setting
        if self.args.sample_around_instance:
            experiment_setting = f"sampled_at_point_max_R-{np.round(max_radius, 2)}_" + experiment_setting
        else:
            experiment_setting = f"fractions-0-{np.round(fractions, 2)}_"+experiment_setting
        # if self.args.num_lime_features > 10:
        #     experiment_setting += f"_num_features-{self.args.num_lime_features}"
        if self.args.regression:
            experiment_setting = "regression_" + experiment_setting
        return experiment_setting
    
    def _compute_local_preds(self, batch, df_feat_for_expl, explanation, predict_fn):
        if isinstance(batch, torch.Tensor):
            batch = batch.cpu().numpy()
        if isinstance(df_feat_for_expl, torch.Tensor):
            df_feat_for_expl = df_feat_for_expl.cpu().numpy()
        if not isinstance(explanation, list):
            explanation = [explanation]
        return get_lime_local_rergression_preds_for_all_kNN(batch, 
                               explanation, 
                               self.explainer, 
                               df_feat_for_expl)
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
        Process a single chunk of data for LIME method.
        """
        tst_chunk = batch.numpy()  # For LIME method, convert batch to numpy
        predict_threshold = self.args.predict_threshold

        if self.args.sample_around_instance:
            dist = np.linspace(0.001, max_radius, n_points_in_ball) 
            samples_in_ball = uniform_ball_sample(
                centers = tst_chunk_dist,
                R_list = dist,
                N_per_center = self.args.n_samples_around_instance,
                distance_measure = self.args.distance_measure,)
            dist = np.repeat(dist[None, :], tst_chunk.shape[0], axis=0)
        else:
            dist, idx = tree.query(tst_chunk, k=n_points_in_ball, return_distance=True, sort_results=True)
            # 1. Get all the kNN samples from the analysis dataset
            samples_in_ball = [[df_feat_for_expl[idx] for idx in row] for row in idx]
            if type(samples_in_ball[0]) == torch.Tensor:
                samples_in_ball = [sample.numpy() for sample in samples_in_ball]
            samples_in_ball = np.array([np.array(row) for row in samples_in_ball])

        if self.args.regression:
            model_preds, local_preds = get_lime_rergression_preds_for_all_kNN(
                tst_set = tst_chunk, 
                explanations = explanations_chunk, 
                explainer = self.explainer, 
                predict_fn = predict_fn, 
                samples_in_ball = samples_in_ball,
                n_samples_around_instance = self.args.n_samples_around_instance,
            )
            if self.args.sample_around_instance:
                model_preds = model_preds.reshape(*list(samples_in_ball.shape[:-1]))
                local_preds = local_preds.reshape(*list(samples_in_ball.shape[:-1]))
            return model_preds, local_preds, dist
        else:
            res = get_lime_preds_for_all_kNN(
                tst_set = tst_chunk, 
                explanations = explanations_chunk, 
                explainer = self.explainer, 
                predict_fn = predict_fn, 
                samples_in_ball = samples_in_ball,
                pred_threshold = predict_threshold,
                sample_around_instance = self.args.sample_around_instance,
                n_samples_around_instance = self.args.n_samples_around_instance,
                distance_measure = self.args.distance_measure,

            )
            model_predicted_top_label, model_prob_of_top_label, local_preds_label, local_preds = res
            if self.args.sample_around_instance:
                model_predicted_top_label = model_predicted_top_label.reshape(*list(samples_in_ball.shape[:-1]))
                model_prob_of_top_label = model_prob_of_top_label.reshape(*list(samples_in_ball.shape[:-1]))
                local_preds_label = local_preds_label.reshape(*list(samples_in_ball.shape[:-1]))
                local_preds = local_preds.reshape(*list(samples_in_ball.shape[:-1]))
            # Reformat to match the expected output format of process_chunk
            return (
                model_prob_of_top_label,  
                model_predicted_top_label,
                model_prob_of_top_label,  
                local_preds, 
                local_preds_label, 
                local_preds,  
                dist  
            )



