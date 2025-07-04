from torch_frame.gbdt import XGBoost
from torch_frame.typing import TaskType
import os.path as osp
import torch
import numpy as np
import argparse
import lime.lime_tabular
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import BallTree
from joblib import Parallel, delayed
import random
from functools import partial
from src.utils.plotting_utils import plot_accuracy_vs_threshold, plot_accuracy_vs_radius, plot_3d_scatter
from src.utils.misc import get_non_zero_cols, set_random_seeds
from src.explanation_methods.lime_analysis.lime_local_classifier import compute_lime_accuracy_per_radius, compute_explanations
import os


def main(args):
    set_random_seeds(args.random_seed)
    

    include_trn = args.include_trn
    include_val = args.include_val

    def get_path(base_folder, base_path, setting, suffix=""):
        if base_folder is None:
            return base_path
        assert setting is not None, "Setting must be specified if folder is provided"
        return osp.join(base_folder, f"{suffix}{setting}")

    # Replace the original code with:
    data_path = get_path(args.data_folder, args.data_path, args.setting)
    model_path = get_path(args.model_folder, args.model_path, args.setting, suffix="final_model_")
    results_path = get_path(args.results_folder, args.results_path, args.setting)
    if args.num_test_splits > 1:
        results_path = osp.join(results_path, f"test_splits")
    
    if not osp.exists(results_path):
        os.makedirs(results_path)
    print("saving results to: ", results_path)

    # Add .npz extension specifically for data_path
    if args.data_folder is not None:
        data_path += ".npz"

    # Load model and data based on model type
    if args.model_type == "xgboost":
        from model.pytorch_frame_xgboost import load_model, load_data, predict_fn
    elif args.model_type == "LightGBM" and ("synthetic" in results_path):
        from model.lightgbm import load_model, load_data, predict_fn
    elif args.model_type == "LightGBM":
        from model.pytorch_frame_lgm import load_model, load_data, predict_fn
    elif args.model_type == "inception_v3":
        from model.tab_inception_v3 import load_model, load_data, predict_fn, get_class_names
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    model = load_model(model_path)
    tst_feat, _, val_feat, _, trn_feat, _ = load_data(model, data_path)

    # Handle test set splits
    if args.num_test_splits > 1:
        print(f"Splitting test set into {args.num_test_splits} chunks")
        print(f"Using split index {args.split_idx}")
        chunk_size = len(tst_feat) // args.num_test_splits
        assert (args.split_idx is not None), "split_idx must be specified if num_test_splits > 1"
        assert args.split_idx < args.num_test_splits, "split_idx must be less than num_test_splits"
        start = args.split_idx * chunk_size
        end = start + chunk_size
        tst_feat = tst_feat[start:end]

        # Split train and validation sets accordingly
        trn_chunk_size = len(trn_feat) // args.num_test_splits
        val_chunk_size = len(val_feat) // args.num_test_splits
        trn_start = args.split_idx * trn_chunk_size
        trn_end = trn_start + trn_chunk_size
        val_start = args.split_idx * val_chunk_size
        val_end = val_start + val_chunk_size
        trn_feat = trn_feat[trn_start:trn_end]
        val_feat = val_feat[val_start:val_end]

    feature_names = np.arange(trn_feat.shape[1])
    predict_fn_wrapped = partial(predict_fn, model=model)
    print("train, test, val feature shapes: ", trn_feat.shape, tst_feat.shape, val_feat.shape)
    df_feat = tst_feat

    
    if include_trn:
        df_feat = np.concatenate([trn_feat, df_feat], axis=0)
    if include_val:
        df_feat = np.concatenate([df_feat, val_feat], axis=0)

    
    # Compute pairwise distances on samples
    valid_distance_measures = BallTree.valid_metrics + ["cosine"]
    assert args.distance_measure in valid_distance_measures, f"Invalid distance measure: {args.distance_measure}. Valid options are: {valid_distance_measures}"
    distance_measure = args.distance_measure

    if args.distance_measure == "cosine":
        from sklearn.metrics.pairwise import cosine_similarity
        def cosine_distance(x, y):
            cosine_sim = cosine_similarity(x.reshape(1, -1), y.reshape(1, -1))[0, 0]
            return 1 - cosine_sim
        distance_measure = "pyfunc"
    
    # Sample 5000 points from each dataset to estimate min/max distances
    tst_sample_idx = np.random.choice(len(tst_feat), min(5000, len(tst_feat)), replace=False)
    df_sample_idx = np.random.choice(len(df_feat), min(5000, len(df_feat)), replace=False)
    tst_sample = tst_feat[tst_sample_idx]
    df_sample = df_feat[df_sample_idx]

    distances_pw = pairwise_distances(tst_sample, df_sample, metric=args.distance_measure)
    if args.distance_measure == "cosine":
        distances_pw = 1 - distances_pw

    max_dist = np.max(distances_pw)
    first_non_zero = np.min(distances_pw[np.round(distances_pw, 2) > 0])

    tree = BallTree(df_feat, metric=distance_measure) if args.distance_measure != "cosine" else BallTree(df_feat, metric=distance_measure, func=cosine_distance)

    num_tresh = args.num_tresh 
    thresholds = np.concatenate((np.array([1e-5]), np.linspace(first_non_zero, min(first_non_zero*5, max_dist/1.5), num_tresh)))

    if args.kernel_width is None:
        args.kernel_width = np.round(np.sqrt(trn_feat.shape[1]) * .75, 2)  # Default value

    df_setting = "complete_df" if include_trn and include_val else "only_test"
    experiment_setting = f"thresholds-0-{np.round(first_non_zero)}-max{np.round(thresholds[-1])}num_tresh-{num_tresh}_{df_setting}_kernel_width-{args.kernel_width}_model_regr-{args.model_regressor}_model_type-{args.model_type}_accuracy_fraction.npy"
    if args.num_lime_features > 10:
        experiment_setting = f"num_features-{args.num_lime_features}_{experiment_setting}"
    if args.num_test_splits > 1:
        experiment_setting = f"split-{args.split_idx}_{experiment_setting}"
    
    class_names = get_class_names() if args.model_type == "inception_v3" else [0, 1]

    explainer = lime.lime_tabular.LimeTabularExplainer(trn_feat,
                                                       feature_names=feature_names,
                                                       class_names=class_names,
                                                       discretize_continuous=True,
                                                       random_state=args.random_seed,
                                                       kernel_width=args.kernel_width)
    # Construct the explanation file name and path
    explanation_file_name = f"normalized_data_explanations_test_set_kernel_width-{args.kernel_width}_model_regressor-{args.model_regressor}"
    if args.num_lime_features > 10:
        explanation_file_name += f"_num_features-{args.num_lime_features}"
    if args.num_test_splits > 1:
        explanation_file_name = f"split-{args.split_idx}_{explanation_file_name}"
    explanations_dir = osp.join(results_path, "explanations")
    explanation_file_path = osp.join(explanations_dir, explanation_file_name)
    print(f"using explanation path: {explanation_file_path}")

    # Ensure the explanations directory exists
    if not osp.exists(explanations_dir):
        os.makedirs(explanations_dir)

    # Check if the explanation file already exists
    if osp.exists(explanation_file_path+".npy"):
        print(f"Using precomputed explanations from: {explanation_file_path}")
        explanations = np.load(explanation_file_path+".npy", allow_pickle=True)
        print(f"{len(explanations)} explanations loaded")
    else:
        print("Precomputed explanations not found. Computing explanations for the test set...")
        explanations = compute_explanations(explainer, tst_feat, predict_fn_wrapped, args.num_lime_features)
        
        # Save the explanations to the appropriate file
        np.save(explanation_file_path, explanations)
        print(f"Finished computing and saving explanations to: {explanation_file_path}")

    num_samples = len(tst_feat)
    num_thresholds = len(thresholds)
    results = {
        "accuracy": np.zeros((num_thresholds, num_samples)),
        "fraction_points_in_ball": np.zeros((num_thresholds, num_samples)),
        "samples_in_ball": np.zeros((num_thresholds, num_samples)),
        "ratio_all_ones": np.zeros((num_thresholds, num_samples)),
        "thresholds": thresholds
    }
    chunk_size = args.chunk_size
    predict_threshold = args.predict_threshold
    for i in range(0, len(tst_feat), chunk_size):
        if args.debug:
            # Normal for loop for easier debugging
            for dist_threshold in thresholds:
                print(f"Processing threshold {dist_threshold}")
                chunk_results = compute_lime_accuracy_per_radius(
                    tst_feat[i:i+chunk_size], df_feat, explanations, explainer, predict_fn_wrapped, dist_threshold, tree, predict_threshold
                )
                threshold_idx = np.where(thresholds == dist_threshold)[0][0]
                results["accuracy"][threshold_idx, i:i+chunk_size] = chunk_results[1]
                results["fraction_points_in_ball"][threshold_idx, i:i+chunk_size] = chunk_results[2]
                results["samples_in_ball"][threshold_idx, i:i+chunk_size] = chunk_results[3]
                results["ratio_all_ones"][threshold_idx, i:i+chunk_size] = chunk_results[4]
            np.savez(osp.join(results_path, experiment_setting), **results)
        else:
            # Parallel processing for normal execution
            chunk_end = min(i + chunk_size, len(tst_feat))
            tst_chunk = tst_feat[i:chunk_end]
            explanations_chunk = explanations[i:chunk_end]
            chunk_results = Parallel(n_jobs=-1)(
                    delayed(compute_lime_accuracy_per_radius)(
                        tst_chunk, df_feat, explanations_chunk, explainer, predict_fn_wrapped, dist_threshold, tree, predict_threshold
                    )
                    for dist_threshold in thresholds
                )
            # Unpack results directly into the correct positions in the arrays
            for threshold, acc, frac, samp, ratio in chunk_results:
                threshold_idx = np.where(thresholds == threshold)[0][0]
                results["accuracy"][threshold_idx, i:chunk_end] = acc
                results["fraction_points_in_ball"][threshold_idx, i:chunk_end] = frac
                results["samples_in_ball"][threshold_idx, i:chunk_end] = samp
                results["ratio_all_ones"][threshold_idx, i:chunk_end] = ratio
        
            # create graphs for the accuracy and fraction of points in the ball
            graphics_dir = osp.join(results_path, "graphics")
            if not osp.exists(graphics_dir):
                os.makedirs(graphics_dir)
            non_zero_cols = get_non_zero_cols(results["accuracy"])
            acc_vs_treshhold = f"accuracy_vs_threshold_kernel{args.kernel_width}.pdf" if args.num_test_splits == 1 else f"accuracy_vs_threshold_kernel{args.kernel_width}_split-{args.split_idx}.pdf"
            acc_vs_fraction = f"accuracy_vs_fraction_kernel{args.kernel_width}.pdf" if args.num_test_splits == 1 else f"accuracy_vs_fraction_kernel{args.kernel_width}_split-{args.split_idx}.pdf"
            plot_accuracy_vs_threshold(accuracy=results["accuracy"][:, :non_zero_cols], 
                                       thresholds=results["thresholds"], 
                                    #    model_predictions=model_predictions[:non_zero_cols], 
                                       save_path=osp.join(graphics_dir, acc_vs_treshhold))
            plot_accuracy_vs_radius(accuracy=results["accuracy"][:, :non_zero_cols], 
                                    fraction_points_in_ball=results["fraction_points_in_ball"][:, :non_zero_cols], 
                                    # model_predictions=model_predictions[:non_zero_cols], 
                                    save_path=osp.join(graphics_dir, acc_vs_fraction),
                                    )
        np.savez(osp.join(results_path, experiment_setting), **results)
        print(f"Processed chunk {i//chunk_size + 1}/{(len(tst_feat) + chunk_size - 1)//chunk_size}")
        if i + chunk_size >= 4000:
            break
    print("Finished computing LIME accuracy and fraction of points in the ball")

   
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Locality Analyzer")
    parser.add_argument("--data_folder", type=str, help="Path to the data folder")#, default = "/home/grotehans/xai_locality/data/synthetic_data")
    parser.add_argument("--model_folder", type=str, help="Path to the model folder")#, default = "/home/grotehans/xai_locality/pretrained_models/LightGBM/synthetic_data" )
    parser.add_argument("--results_folder", type=str, help="Path to the results folder")#, default="/home/grotehans/xai_locality/results/LightGBM/synthetic_data")
    parser.add_argument("--setting", type=str, help="Setting of the experiment")#, default= "n_feat50_n_informative20_n_redundant30_n_repeated0_n_classes2_n_samples100000_n_clusters_per_class5_class_sep0.9_flip_y0.01_random_state42")
    parser.add_argument("--data_path", type=str, help="Path to the data")
    parser.add_argument("--model_path", type=str, help="Path to the model")
    parser.add_argument("--model_type", type=str, default="LightGBM", help="Model type, so far only 'xgboost' and 'LightGBM' is supported")
    parser.add_argument("--results_path", type=str, help="Path to save results")#, default=" ")
    parser.add_argument("--distance_measure", type=str, default="euclidean", help="Distance measure")
    parser.add_argument("--num_tresh", type=int, default=50, help="Number of thresholds")
    parser.add_argument("--include_trn", action="store_true", help="Include training data")
    parser.add_argument("--include_val", action="store_true", help="Include validation data")
    parser.add_argument("--fraction_only", action="store_true", help="Compute only the fraction of points in the ball")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    parser.add_argument("--chunk_size", type=int, default=20, help="Chunk size of test set computed at once")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--kernel_width", type=float, default=None, help="Kernel size for the locality analysis")
    parser.add_argument("--model_regressor", type=str, default="ridge", help="Model regressor for LIME")
    parser.add_argument("--num_test_splits",  type=int, default = 10, help="Number of test splits for analysis")
    parser.add_argument("--split_idx", type=int, default = 0, help="Index of the test split")
    parser.add_argument("--num_lime_features", type=int, default = 10, help="Index of the test split")
    parser.add_argument("--predict_threshold", type=float, default = None, help="Threshold for classifying sample as top prediction")


                                    
    args = parser.parse_args()

    # Validate arguments
    if (args.data_folder and args.setting and args.model_folder and args.results_folder) or (args.data_path and args.model_path and args.results_path):
        print("Starting the experiment with the following arguments: ", args)
        main(args)
    else:
        parser.error("You must provide either data_folder, model_folder, results_folder, and setting or data_path, model_path, and results_path.")