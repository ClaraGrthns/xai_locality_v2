import argparse
import os
import os.path as osp
from pathlib import Path
import copy
import time
import numpy as np

from src.train.custom_data_frame_benchmark import main_deep_models, main_gbdt
from estimate_local_fidelity import main as main_knn_vs_accuracy
from model_complexity import main as main_knn_analyzer
from src.dataset.synthetic_data import get_setting_name_classification, get_setting_name_regression
from src.utils.misc import set_random_seeds

CUSTOM_MODELS = ["LogReg", "LinReg"]
GBT_MODELS = ["LightGBM", "XGBoost", "CatBoost"]
BASEDIR = str(Path(__file__).resolve().parent)
print(f"Base directory: {BASEDIR}")

def parse_args():
    parser = argparse.ArgumentParser(description="XAI Locality Experiment Suite")

    parser.add_argument("--config", type=str, 
                        help="Path to configuration file")

    # Basic configuration
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--random_seed_synthetic_data", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--downsample_analysis", action="store_true", help="Enable downsampling for analysis")    
    parser.add_argument("--create_additional_analysis_data", action="store_true", help="Create additional analysis data")    
    parser.add_argument("--data_folder", type=str,  
                        help="Path to the data folder")
    parser.add_argument("--model_folder", type=str, default=BASEDIR + "/pretrained_models",
                        help="Path to save/load models")
    parser.add_argument("--results_folder", type=str, default=BASEDIR + "/results",
                        help="Path to save results")
    parser.add_argument("--setting", type=str, help="Setting name for the experiment")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    # Model configuration
    parser.add_argument("--model_type", type=str, 
                        choices=["LightGBM", "XGBoost", "ExcelFormer", "MLP", "TabNet", "Trompt", "FTTransformer", "ResNet", "LogReg", "LinReg", "TabTransformer"],
                        help="Type of model to train/use")
    parser.add_argument("--regression", action="store_true",)
    parser.add_argument("--regression_model", type=str)
    parser.add_argument("--skip_training", action="store_true", 
                        help="Skip model training if the model already exists")
    parser.add_argument("--force_training", action="store_true", 
                        help="Force training even if the model exists")
    
    # Benchmark dataset configuration
    parser.add_argument("--use_benchmark", action="store_true", 
                        help="Use benchmark dataset instead of synthetic data")
    parser.add_argument("--task_type", type=str, 
                        choices=["binary_classification", "multiclass_classification", "regression"],
                        help="Task type for benchmark dataset")
    parser.add_argument("--scale", type=str, 
                        choices=["small", "medium", "large"],
                        help="Scale of benchmark dataset")
    parser.add_argument("--idx", type=int, default=0,
                        help="Index of benchmark dataset")
    parser.add_argument("--num_trials", type=int,  help="Number of trials for training", default=15)
    parser.add_argument("--num_repeats", type=int, help="Number of repeats for training", default = 1)
    parser.add_argument("--complexity_model", type=str,default="optimize",
                        choices=["simple", "complex", "optimize"],
                        help="Complexity of the model")

    # Train configuration
    parser.add_argument("--epochs", type=int, help="Number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument('--optimize', action='store_true', help='Use Optuna for hyperparameter optimization')
    
    # Synthetic data generation (if needed)
    parser.add_argument("--n_features", type=int, help="Number of features")
    parser.add_argument("--n_informative", type=int,  help="Number of informative features")
    parser.add_argument("--n_redundant", type=int,  help="Number of redundant features")
    parser.add_argument("--n_repeated", type=int,  help="Number of repeated features")
    parser.add_argument("--n_classes", type=int, default=2, help="Number of classes")
    parser.add_argument("--n_samples", type=int, default=100000, help="Number of samples")
    parser.add_argument("--n_clusters_per_class", type=int, default=2, help="Number of clusters per class")
    parser.add_argument("--class_sep", type=float,  help="Class separation")
    parser.add_argument("--flip_y", type=float, help="Fraction of samples with random labels")
    parser.add_argument("--hypercube", action="store_true", help="If True, the clusters are put on the vertices of a hypercube. If False, the clusters are put on the vertices of a random polytope.")
    parser.add_argument("--test_size", type=float, default=0.4, help="Test size for train-test split")
    parser.add_argument("--val_size", type=float, default=0.1, help="Validation size for train-validation split")
    parser.add_argument("--regression_mode", type=str)

    ## if regression:
    # Adding the missing arguments for regression dataset creation
    parser.add_argument("--noise", type=float, default=0.1, help="Standard deviation of the gaussian noise")
    parser.add_argument("--bias", type=float, default=0.0, help="Bias term in the underlying linear model")
    parser.add_argument("--tail_strength", type=float, default=0.5, help="Relative importance of the fat noisy tail of the covariance matrix")
    parser.add_argument("--effective_rank", type=int, default=None, help="Approximate number of singular vectors to use to generate the covariance matrix")
    
    # KNN analysis parameters
    parser.add_argument("--distance_measure", type=str, help="Distance measure for KNN")
    parser.add_argument("--distance_measures", nargs='+', default=["euclidean", "manhattan", "cosine"], 
                        help="List of distance measures to use")
    parser.add_argument("--min_k", type=int, default=2, help="Minimum k for KNN")
    parser.add_argument("--max_k", type=int, default=30, help="Maximum k for KNN")
    parser.add_argument("--k_step", type=int, default=2, help="Step size for k in KNN")
    parser.add_argument("--chunk_size", type=int, default=200, help="Chunk size for processing")
    parser.add_argument("--max_test_points", type=int, default=200, help="Maximum number of test points")
    parser.add_argument("--force_overwrite", action="store_true", help="Force overwrite existing results")
    
    # Explanation method parameters
    parser.add_argument("--method", type=str, choices=["lime", "gradient_methods", "lime_captum"], 
                        help="Explanation method to use")
    parser.add_argument("--gradient_method", type=str,
                        choices=["IG", "IG+SmoothGrad", "GuidedBackprob", "Deconv", "GuidedGradCam", "Saliency"], 
                        help="Gradient-based explanation method")
    parser.add_argument("--kernel_width", type=str, default="default", 
                        choices=["default", "double", "half"], 
                        help="Kernel width for LIME")
    parser.add_argument("--model_regressor", type=str, default="ridge", help="Model regressor for LIME")
    parser.add_argument("--num_lime_features", type=int, default=10, 
                        help="Number of features to use in LIME explanation")
    parser.add_argument("--predict_threshold", type=float,)
    parser.add_argument("--include_trn", action="store_true", help="Include training data")
    parser.add_argument("--include_val", action="store_true", help="Include validation data")
    parser.add_argument("--sample_around_instance", action="store_true", help="Sample around instance instead of kNN")
    parser.add_argument("--n_samples_around_instance",type=int, default = 100, help="Number of samples around instance instead of kNN")
    
    # Process steps control
    parser.add_argument("--skip_knn", action="store_true", help="Skip KNN analysis")
    parser.add_argument("--skip_fraction", action="store_true", help="Skip fraction vs accuracy analysis")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing results")
    
    # Add flag to choose whether to use custom or sklearn-based data generation
    parser.add_argument("--use_custom_generator", action="store_true", 
                        help="Use custom data generator instead of sklearn's make_regression")
    
    return parser.parse_args()

def check_model_exists(args):
    """Check if the model file exists."""
    if args.use_benchmark:# For benchmark datasets
        if args.setting is None:
            from src.train.data_frame_benchmark import get_dataset_name
            dataset_name = get_dataset_name(args.task_type, args.scale, args.idx)
            args.setting = dataset_name  # Use dataset name as setting
        else:
            from src.train.data_frame_benchmark import get_dataset_specs
            if not all([args.task_type, args.scale, args.idx]):
                args.task_type, args.scale, args.idx = get_dataset_specs(args.setting)
    elif args.regression:
        setting_name = get_setting_name_regression(
            n_features=args.n_features,
            regression_mode=args.regression_mode,
            n_informative=args.n_informative,
            n_samples=args.n_samples,
            noise=args.noise,
            bias=args.bias,
            tail_strength=args.tail_strength,
            coef=False,
            effective_rank=args.effective_rank,
            random_seed=args.random_seed_synthetic_data
        )
        args.setting = setting_name
    else:
        setting_name = get_setting_name_classification(
            n_features=args.n_features,
            n_informative=args.n_informative,
            n_redundant=args.n_redundant,
            n_repeated=args.n_repeated,
            n_classes=args.n_classes,
            n_samples=args.n_samples,
            n_clusters_per_class=args.n_clusters_per_class,
            class_sep=args.class_sep,
            flip_y=args.flip_y,
            hypercube=args.hypercube,
            random_seed=args.random_seed_synthetic_data
        )
        args.setting = setting_name
    model_path = get_results_path(args, "train")
    return osp.exists(model_path), model_path

def get_data_path(args):
    """Get data path based on model type and dataset."""
    if args.use_benchmark:
        if args.model_type in CUSTOM_MODELS:
            return osp.join(args.data_folder, f"LightGBM_{args.setting}_normalized_data.pt")
        return osp.join(args.data_folder, f"{args.model_type}_{args.setting}_normalized_data.pt")
    else:
        # For synthetic data, check if it's ExcelFormer which has a special path format
        is_ExcelFormer_str = "ExcelFormer_" if args.model_type == 'ExcelFormer' else ""
        return osp.join(args.data_folder,  
                      f"{is_ExcelFormer_str}{args.setting}_normalized_tensor_frame.pt")

def get_results_path(args, step):
    """Get results path for a specific step (train, knn, fraction)."""
    if args.use_benchmark:
        sub_directory = "" 
    elif args.regression:
        sub_directory = "regression_synthetic_data"
    else:
        sub_directory = "synthetic_data"
    if step == "knn":
        return osp.join(args.results_folder, 
                        "knn_model_preds", 
                        args.model_type, 
                        sub_directory, 
                        args.setting)
    elif step == "train" and args.use_benchmark:
        if args.regression:
            model_name = f"{args.model_type}_normalized_regression_{args.setting}_{f"{args.complexity_model}__" if args.complexity_model != "optimize" else ""}results.pt"
        else:
            model_name =f"{args.model_type}_normalized_binary_{args.setting}_{f"{args.complexity_model}__" if args.complexity_model != "optimize" else ""}results.pt"
        return osp.join(args.model_folder,
                        args.model_type, 
                        args.setting,
                        model_name
                        )
    elif step == "train":
        return osp.join(args.model_folder,
                        args.model_type, 
                        sub_directory , 
                        f"{args.model_type}_{args.setting}_results.pt")
    elif step == "fraction":
        # For fraction analysis, the path depends on the explanation method
        method_subdir = args.gradient_method if args.method == "gradient_methods" else ""
        if args.method == "gradient_methods":
            if args.gradient_method == "IG":
                method_subdir = "integrated_gradient"
            elif args.gradient_method == "IG+SmoothGrad":
                method_subdir = "smooth_grad"
        return osp.join(args.results_folder, args.method, 
                      method_subdir, args.model_type, sub_directory, args.setting)
    return None


def train_model(args):
    """Train model based on model type and dataset choice."""
    print(f"Training {args.model_type} model...")
    train_args = copy.deepcopy(args)
    train_args.results_path = get_results_path(args, "train")
    train_args.results_folder = osp.join(args.model_folder, args.model_type)
    if args.use_benchmark:
        from src.train.data_frame_benchmark import main_deep_models as benchmark_deep_models
        from src.train.data_frame_benchmark import main_gbdt as benchmark_gbdt
        from src.train.train_pytorch_model import main as benchmark_custom_models
        train_args.results_folder = osp.join(args.model_folder, args.model_type, args.setting)
        if args.model_type in GBT_MODELS:
            benchmark_gbdt(train_args)
        elif args.model_type in CUSTOM_MODELS:
            benchmark_custom_models(train_args)
        else:
            benchmark_deep_models(train_args)
    else:
        train_args.results_folder = osp.join(train_args.results_folder, "regression_synthetic_data" if args.regression else "synthetic_data")
        # For synthetic data
        if args.model_type in GBT_MODELS:
            main_gbdt(train_args)
        elif args.model_type in CUSTOM_MODELS:
            from src.train.train_pytorch_model import main as custom_models
            custom_models(train_args)
        else:
            main_deep_models(train_args)

def run_knn_analysis(args):
    """Run KNN analysis on model predictions."""
    print("Running KNN analysis on model predictions...")
    knn_args = copy.deepcopy(args)
    knn_args.results_path = get_results_path(args, "knn")
    main_knn_analyzer(knn_args)

def run_knn_vs_local_model_analysis(args):
    """Run fraction vs accuracy analysis."""
    print("Running fraction vs accuracy analysis...")
    fraction_args = copy.deepcopy(args)
    fraction_args.results_path = get_results_path(args, "fraction")
    if not hasattr(fraction_args, 'distance_measure') or not fraction_args.distance_measure:
        fraction_args.distance_measure = fraction_args.distance_measures[0] if fraction_args.distance_measures else "euclidean"
    main_knn_vs_accuracy(fraction_args)

def main():
    # Parse arguments and set random seed
    args = parse_args()
    set_random_seeds(args.random_seed)
    args.seed = args.random_seed_synthetic_data
    args.force = True #TODO: Delete
    args.force_overwrite = True #TODO: Delete
    args.use_custom_generator = True  # Default to using custom generator in debug mode
    args.include_trn = False
    
    if args.debug:
        args.model_type = "LightGBM"
        args.setting = "higgs"
        args.use_benchmark = True
        args.method = "lime"
        args.distance_measure = "euclidean"
        args.regression = False
        args.force = True
        args.random_seed = 42
        args.epochs = 30
        args.num_trials = 5
        args.num_repeats = 1
        args.kernel_width = "default"
        args.num_lime_features = 10

    if args.force_training:
        args.force_overwrite = True
        args.force = True
        args.force_create=True
    
    if args.model_folder is None:
        args.model_folder = os.path.join(BASEDIR, "pretrained_models")
    if args.data_folder is None:
        args.data_folder = os.path.join(BASEDIR, "data")
    if not args.use_benchmark:
        subdir = "regression_synthetic_data" if args.regression else "synthetic_data"
        args.data_folder = os.path.join(args.data_folder, subdir)

    if args.results_folder is None:
        args.results_folder = os.path.join(BASEDIR, "results")
    
    model_exists, model_path = check_model_exists(args)
    args.model_path = model_path
    args.data_path = get_data_path(args)
    args.coef = False
    args.skip_knn = True if not args.force_training or not model_exists else False
    args.skip_fraction = False #TODO: Delete

    print(args)

    if args.downsample_analysis:
        downsample_analysis_fractions = np.linspace(0.2, 1.0, 10)
    else:
        downsample_analysis_fractions = [1]
    
    if (not model_exists or args.force_training) and not args.skip_training:
        print("Starting with model training...")
        start_time = time.time()
        train_model(args)
        print(f"Model training completed in {(time.time() - start_time)/60:.2f} minutes.")
    else:
        print(f"Model already exists at {args.model_path}")
    

    if not args.skip_knn:
        print("Starting KNN analysis...")
        start_time = time.time()
        args.downsample_analysis = 1.0
        run_knn_analysis(args)
        print(f"KNN analysis completed in {(time.time() - start_time)/60:.2f} minutes.")

    if not args.skip_fraction:
        print("Starting fraction vs accuracy analysis...")
        start_time = time.time()
        for fraction in downsample_analysis_fractions:
            args.downsample_analysis = fraction
            run_knn_vs_local_model_analysis(args)
        print(f"Fraction vs accuracy analysis completed in {(time.time() - start_time):.2f} seconds.")
    print("Experiment complete!")

if __name__ == "__main__":
    main()