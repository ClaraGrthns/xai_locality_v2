import os
import argparse
from pathlib import Path
import sys
sys.path.append(os.path.join(os.getcwd(), '..'))
sys.path.append(os.path.join(os.getcwd(), 'src'))
# from src.dataset.synthetic_data import get_setting_name_classification, get_setting_name_regression


data_lookup = {
        "small": {
            0: "bike_sharing_demand",
            1: "brazilian_houses",
            2: "cpu_act",
            3: "elevators",
            4: "house_sales",
            5: "houses",
            6: "sulfur",
            7: "superconduct",
            8: "topo_2_1",
            9: "visualizing_soil",
            10: "wine_quality",
            11: "yprop_4_1",
            12: "california_housing",
        },
        "medium": {
            0: "allstate_claims_severity",
            1: "sgemm_gpu_kernel_performance",
            2: "diamonds",
            3: "medical_charges",
            4: "particulate_matter_ukair_2017",
            5: "seattlecrime6",
        },
        "large": {
            0: "airlines_DepDelay_1M",
            1: "delays_zurich_transport",
            2: "nyc-taxi-green-dec-2016",
            3: "microsoft",
            4: "yahoo",
            5: "year",
        }
}

def get_setting_name_regression(regression_mode,
                                n_features,
                                n_informative, 
                                n_samples, 
                                noise,
                                bias,
                                tail_strength,
                                effective_rank,
                                random_seed):
    print("regression_mode", regression_mode)
    if 'friedman' in regression_mode:
        return regression_mode + f'_n_samples{n_samples}_noise{noise}_random_state{random_seed}'
    setting_name = (f'regression_{regression_mode}_n_feat{n_features}_n_informative{n_informative}_n_samples{n_samples}_'
                    f'noise{noise}_bias{bias}_random_state{random_seed}')
    if effective_rank is not None:
        setting_name += f'_effective_rank{effective_rank}_tail_strength{tail_strength}'
    return setting_name

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate Python commands for running experiments")
    parser.add_argument('--output_dir', type=str, default='/home/grotehans/xai_locality_v2/commands_sbach_files/experiment_commands',
                        help='Directory to save the generated command files')
    parser.add_argument('--base_dir', type=str, default=str(Path(__file__).resolve().parent.parent),
                        help='Base directory for the experiment')
    parser.add_argument('--skip_training', action='store_true',
                        help='Add --skip_training flag to commands')
    parser.add_argument('--force_training', action='store_true',
                        help='Add --force_training flag to commands')
    parser.add_argument('--skip_knn', action='store_true',
                        help='Add --skip_knn flag to commands')
    parser.add_argument('--skip_fraction', action='store_true',
                        help='Add --skip_fraction flag to commands')
    parser.add_argument('--kernel_width', type=str, default="default",
                        help='Kernel width for LIME (default, double, half, or a float value)')
    parser.add_argument('--random_seed_synthetic_data', type=int, default=42,)
    parser.add_argument('--random_seed', type=int, default=42,)
    parser.add_argument('--gradient_method', type=str, default="Saliency",
                        help='Gradient method to use (IG or IG+SmoothGrad)')
    parser.add_argument("--create_additional_analysis_data", action="store_true",)
    return parser.parse_args()

def create_command_file(output_dir, model, setting, method, distance_measure, kernel_width, num_lime_features,
                        is_synthetic, skip_training, force_training, skip_knn, skip_fraction, create_additional_analysis_data, gradient_method=None,
                        synthetic_params=None, random_seed=42):
    """Create a file containing the Python command for a specific configuration"""
    
    # Create directory structure - organize by method first
    if method == "gradient_methods" and gradient_method:
        if gradient_method == "IG":
            method_dir = os.path.join(output_dir, method, "integrated_gradient")
        elif gradient_method =="Saliency":
            method_dir = os.path.join(output_dir, method, "saliency")
        else:
            method_dir = os.path.join(output_dir, method, "smooth_grad")
    else:
        method_dir = os.path.join(output_dir, method)
    
    # Then by model
    model_dir = os.path.join(method_dir, model)
    
    # Then by dataset
    if is_synthetic:
        dataset_dir = os.path.join(model_dir, "regression_custom_synthetic_data" ,  setting)
    else:
        dataset_dir = os.path.join(model_dir, setting)
    
    # Finally by distance measure
    model_dir = os.path.join(dataset_dir, distance_measure)
    
    os.makedirs(model_dir, exist_ok=True)
    epochs = 30
    # Format args for the command
    base_args = f"--model_type {model} --setting {setting} --method {method} --distance_measure {distance_measure} --regression --force --random_seed {random_seed}"
    
    if is_synthetic:
        # Use the synthetic parameters directly instead of parsing them from the setting name
        if synthetic_params is not None:
            # Extract all parameters with proper defaults
            n_features = synthetic_params.get('n_features', 100)
            n_informative = synthetic_params.get('n_informative', 10)
            n_samples = synthetic_params.get('n_samples', 100000)
            # Extract all parameters with proper defaults
            regression_mode = synthetic_params.get('regression_mode', "linear")
            n_features = synthetic_params.get('n_features', 100)
            n_informative = synthetic_params.get('n_informative', 10)
            n_samples = synthetic_params.get('n_samples', 100000)
            noise = synthetic_params.get('noise', 0.0)
            bias = synthetic_params.get('bias', 0.0)
            data_folder = synthetic_params.get('data_folder', 'data')
            test_size = synthetic_params.get('test_size', 0.4)
            val_size = synthetic_params.get('val_size', 0.1)
            tail_strength = synthetic_params.get('tail_strength', 0.5)
            effective_rank = synthetic_params.get('effective_rank', None)
            use_custom_generator = synthetic_params.get('use_custom_generator', None)
            random_seed_synthetic_data = synthetic_params.get('random_seed_synthetic_data', 42)
            
            # Add synthetic data parameters
            if 'friedman' in regression_mode:
                synthetic_args = (f" --regression_mode {regression_mode}"
                                f" --n_samples {n_samples}"
                                f" --noise {noise}"
                                f" --random_seed_synthetic_data {random_seed_synthetic_data}"
                                f" --epochs 30 --num_trials 10 --num_repeats 1 "
                                )
                base_args += synthetic_args
            else:
                synthetic_args = (f" --regression_mode {regression_mode}"
                                f" --n_features {n_features}"
                                f" --n_informative {n_informative}"
                                f" --n_samples {n_samples}"
                                f" --noise {noise}"
                                f" --bias {bias}"
                                f" --data_folder {data_folder}"
                                f" --test_size {test_size}"
                                f" --val_size {val_size}"
                                f" --random_seed_synthetic_data {random_seed_synthetic_data}"
                                f" --epochs {epochs} --num_trials 10 --num_repeats 1"
                                )
                if use_custom_generator is not None:
                    synthetic_args += f" --use_custom_generator"
                    
                base_args += synthetic_args
    # if setting == "airlines_DepDelay_1M":
    #     base_args += " --use_benchmark --task_type regression --scale large --idx 0 --num_trials 3 --num_repeats 1 --epochs 15"
    # elif setting == "delays_zurich_transport":
    #     base_args += " --use_benchmark --task_type regression --scale large --idx 1 --num_trials 3 --num_repeats 1 --epochs 15"
    # elif setting == "nyc-taxi-green-dec-2016":
    #     base_args += " --use_benchmark --task_type regression --scale large --idx 2 --num_trials 3 --num_repeats 1 --epochs 15"
    # elif setting == "microsoft":
    #     base_args += " --use_benchmark --task_type regression --scale large --idx 3 --num_trials 3 --num_repeats 1 --epochs 15"
    # elif setting == "yahoo":
    #     base_args += " --use_benchmark --task_type regression --scale large --idx 4 --num_trials 3 --num_repeats 1 --epochs 15"
    # elif setting == "year":
    #     base_args += " --use_benchmark --task_type regression --scale large --idx 5 --num_trials 3 --num_repeats 1 --epochs 15"
    # elif setting == "medical_charges":
    #     base_args += " --use_benchmark --task_type regression --scale medium --idx 3 --num_trials 5 --num_repeats 1 --epochs 25 --include_val "
    # elif setting == "california_housing":
    #     base_args += " --use_benchmark --task_type regression --scale small --idx 12 --num_trials 10 --num_repeats 1 --epochs 30 --include_val "
    # elif setting == "superconduct":
    #     base_args += " --use_benchmark --task_type regression --scale small --idx 7 --num_trials 10 --num_repeats 1 --epochs 30 --include_val "
    # elif setting == "houses":
    #     base_args += " --use_benchmark --task_type regression --scale small --idx 5 --num_trials 10 --num_repeats 1 --epochs 30 --include_val  "
    # elif setting == "elevators":
    #     base_args += " --use_benchmark --task_type regression --scale small --idx 3 --num_trials 10 --num_repeats 1 --epochs 30 --include_val   "
    # elif setting == "allstate_claims_severity":
    #     base_args += " --use_benchmark --task_type regression --scale medium --idx 0 --num_trials 5 --num_repeats 1 --epochs 25 --include_val"
    # elif setting == "sgemm_gpu_kernel_performance":
    #     base_args += " --use_benchmark --task_type regression --scale medium --idx 1 --num_trials 5 --num_repeats 1 --epochs 25 --include_val"
    # elif setting == "diamonds":
    #     base_args += " --use_benchmark --task_type regression --scale medium --idx 2 --num_trials 5 --num_repeats 1 --epochs 25 --include_val"
    # elif setting == "particulate_matter_ukair_2017":
    #     base_args += " --use_benchmark --task_type regression --scale medium --idx 4 --num_trials 5 --num_repeats 1 --epochs 25 --include_val"
    # elif setting == "seattlecrime6":
    #     base_args += " --use_benchmark --task_type regression --scale medium --idx 5 --num_trials 5 --num_repeats 1 --epochs 25 --include_val"
    else:
        dataset_scale = None
        idx = None
        for scale, datasets in data_lookup.items():
            if setting in [d for d in datasets.values()]:
                dataset_scale = scale
                break
        for scale, datasets in data_lookup.items():
            if setting in [d for d in datasets.values()]:
                idx = list(datasets.keys())[list(datasets.values()).index(setting)]
                break
        base_args += f" --scale {dataset_scale} --idx {idx} --use_benchmark --task_type regression --num_repeats 1 "
        if "medium" in base_args:
            base_args += " --epochs 30 --include_val --num_trials 10 "
        elif "large" in base_args:
            base_args += " --epochs 15 --num_trials 5 "
        else:
            base_args += " --epochs 50 --include_val --num_trials 15 "
        
    if method == "lime":
        base_args += f" --kernel_width {kernel_width} --num_lime_features {num_lime_features}"
    elif method == "gradient_methods" and gradient_method:
        # For IG, make sure to use the correct command line name
        base_args += f" --gradient_method {gradient_method}"
    
    # Add optional flags
    if skip_training:
        base_args += " --skip_training"
    if force_training:
        base_args += " --force_training"
    if skip_knn:
        base_args += " --skip_knn"
    if skip_fraction:
        base_args += " --skip_fraction"
    if create_additional_analysis_data:
        base_args += " --create_additional_analysis_data --downsample_analysis "
    # Create the full command
    command = f"/home/grotehans/miniconda3/envs/tab_models/bin/python -u run_experiment_setup.py {base_args}"
    
    # Define filename - include distance measure to distinguish files
    distance_suffix = f"_{distance_measure}"
    file_name_add_on = "_skip_kNN" if skip_knn else ""
    file_name_add_on += "_skip_fraction" if skip_fraction else ""
    file_name_add_on += f"_random_seed_{random_seed}"
    file_name_add_on += "_force_training" if force_training else ""
    file_name_add_on += f"_downsample_analysis" if create_additional_analysis_data else ""
    
    if method == "lime":
        filename = f"lime_{kernel_width}{distance_suffix}{file_name_add_on}.sh"
    elif method == "gradient_methods" and gradient_method:
        if gradient_method == "IG":
            filename = f"gradient_integrated_gradient{distance_suffix}{file_name_add_on}.sh"
        else:
            filename = f"gradient_{gradient_method}{distance_suffix}{file_name_add_on}.sh"
    else:
        filename = f"{method}{distance_suffix}{file_name_add_on}.sh"
    
    # Write command to file
    file_path = os.path.join(model_dir, filename)
    with open(file_path, "w") as f:
        f.write("#!/bin/bash\n\n")
        f.write(f"# Model: {model}, Dataset: {setting}, Method: {method}{' '+gradient_method if gradient_method else ''}\n")
        f.write(f"# Distance: {distance_measure}{', Kernel: '+kernel_width if kernel_width else ''}\n\n")
        f.write(f"{command}\n")
    
    # Make file executable
    os.chmod(file_path, 0o755)
    
    print(f"Created {file_path}")
    return file_path
def main():
    args = parse_arguments()
    output_dir = os.path.join(args.base_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Define synthetic configurations with explicit parameters
    synthetic_configs = [
        # Standard polynomial settings
        {
            'regression_mode': "exponential_interaction",
            'n_features': 20,
            'n_informative': 10,
            'n_samples': 200000,
            'bias': 0.0,
            "use_custom_generator": True,
            'noise': 0.2,
            'random_seed_synthetic_data': 42,
        },

        {
            'regression_mode': "piecewise_linear",
            'n_features': 15,
            'n_informative': 10,
            'n_samples': 200000,
            'bias': 0.0,
            "use_custom_generator": True,
            'noise': 0.2,
            'random_seed_synthetic_data': 42,
        },

        {
            'regression_mode': "polynomial",
            'n_features': 20,
            'n_informative': 5,
            'n_samples': 200000,
            'bias': 1.0,
            'noise': 0.1,
            'random_seed_synthetic_data': 42,
            'tail_strength': 0.5,
            "use_custom_generator": True,
            'effective_rank': 60
        },

        {
            'regression_mode': "polynomial",
            'n_features': 20,
            'n_informative': 10,
            'n_samples': 200000,
            'bias': 1.0,
            'noise': 0.4,
            'random_seed_synthetic_data': 42,
            'tail_strength': 0.5,
            "use_custom_generator": True,
            'effective_rank': 60
        },
        {
            'regression_mode': "linear",
            'n_features': 30,
            'n_informative': 20,
            'n_samples': 200000,
            'bias': 0.0,
            "use_custom_generator": True,
            'noise': 0.6,
            'random_seed_synthetic_data': 42,
            'tail_strength': 0.9,
            'effective_rank': 5
        },
### more difficult
        {
            'regression_mode': "polynomial",
            'n_features': 100,
            'n_informative': 10,
            'n_samples': 100000,
            'bias': 1.0,
            'noise': 0.0,
            'random_seed_synthetic_data': 42,
            'tail_strength': 0.5,
            "use_custom_generator": True,
            'effective_rank': 60
        },

        {
            'regression_mode': "polynomial",
            'n_features': 100,
            'n_informative': 50,
            'n_samples': 100000,
            'bias': 0.0,
            'noise': 0.0,
            'random_seed_synthetic_data': 42,
            'tail_strength': 0.5,
            "use_custom_generator": True,
            'effective_rank': 30
        },
        {
            'regression_mode': "polynomial",
            'n_features': 50,
            'n_informative': 25,
            'n_samples': 200000,
            'noise': 0.8,
            "use_custom_generator": True,
            'bias': 0.2,
            'random_seed_synthetic_data': 42
        },
        # Interaction-based settings
        {
            'regression_mode': "interaction",
            'n_features': 50,
            'n_informative': 30,
            'n_samples': 100000,
            'bias': 10.0,
            'noise': 0.1,
            "use_custom_generator": True,
            'random_seed_synthetic_data': 42,
            'tail_strength': 0.5,
            'effective_rank': 14
        },
        {
            'regression_mode': "poly_interaction",
            'n_features': 90,
            'n_informative': 70,
            'n_samples': 200000,
            'bias': 0.5,
            'noise': 0.1,
            'random_seed_synthetic_data': 42,
            "use_custom_generator": True,
            'tail_strength': 0.7,
            'effective_rank': 50
        },
        {
            'regression_mode': "poly_interaction",
            'n_features': 90,
            'n_informative': 40,
            'n_samples': 200000,
            'bias': 0.5,
            'noise': 0.1,
            'random_seed_synthetic_data': 42,
            "use_custom_generator": True,
            'tail_strength': None,
            'effective_rank': None
        },
        # Specialized regression modes
        {
            'regression_mode': "exponential_interaction",
            'n_features': 50,
            'n_informative': 10,
            'n_samples': 200000,
            'bias': 0.0,
            "use_custom_generator": True,
            'noise': 0.2,
            'random_seed_synthetic_data': 42,
            'tail_strength': 0.9,
            'effective_rank': 5
        },
        {
            'regression_mode': "piecewise",
            'n_features': 60,
            'n_informative': 15,
            'n_samples': 200000,
            'bias': 0.3,
            "use_custom_generator": True,
            'noise': 0.25,
            'random_seed_synthetic_data': 42,
            'tail_strength': 0.4,
            'effective_rank': 60
        },
        {
            'regression_mode': "multiplicative_chain",
            'n_features': 70,
            'n_informative': 30,
            'n_samples': 200000,
            "use_custom_generator": True,
            'bias': 0.0,
            'noise': 0.5,
            'random_seed_synthetic_data': 42,
            'tail_strength': 0.8,
            'effective_rank': 20
        },
        {
            'regression_mode': "sigmoid_mix",
            'n_features': 200,
            'n_informative': 80,
            'n_samples': 200000,
            "use_custom_generator": True,
            'bias': 0.1,
            'noise': 0.15,
            'random_seed_synthetic_data': 42,
            'tail_strength': 0.6,
            'effective_rank': 40
        },
        {
            'regression_mode': "advanced_polynomial",
            'n_features': 80,
            'n_informative': 40,
            'n_samples': 200000,
            "use_custom_generator": True,
            'bias': 0.2,
            'noise': 0.5,
            'random_seed_synthetic_data': 42,
            'tail_strength': None,
            'effective_rank': None
        },
        {
            'regression_mode': "hierarchical",
            'n_features': 70,
            'n_informative': 25,
            'n_samples': 200000,
            "use_custom_generator": True,
            'bias': 0.1,
            'noise': 0.15,
            'random_seed_synthetic_data': 42,
            'tail_strength': None,
            'effective_rank': None
        },
        {
            'regression_mode': "friedman1",
            'n_samples': 200000,
            'noise': 0.01,
            'random_seed_synthetic_data': 42,
        },

        {
            'regression_mode': "friedman1",
            'n_samples': 200000,
            'noise': 0.1,
            'random_seed_synthetic_data': 42,
        },
        {
            'regression_mode': "friedman1",
            'n_samples': 200000,
            'noise': 0.5,
            'random_seed_synthetic_data': 42,
        },

        {
            'regression_mode': "friedman2",
            'n_samples': 200000,
            'noise': 0.1,
            'random_seed_synthetic_data': 42,
        },
        {
            'regression_mode': "friedman2",
            'n_samples': 200000,
            'noise': 0.5,
            'random_seed_synthetic_data': 42,
        },
        {
            'regression_mode': "friedman3",
            'n_samples': 200000,
            'noise': 0.1,
            'random_seed_synthetic_data': 42,
        },
        {
            'regression_mode': "friedman3",
            'n_samples': 200000,
            'noise': 0.5,
            'random_seed_synthetic_data': 42,
        }



    ]
    
    # Generate setting names from configurations
    synthetic_settings = []
    for config in synthetic_configs:
        setting = get_setting_name_regression(regression_mode=config['regression_mode'],
                                              n_features=config.get('n_features', None),
                                              n_informative=config.get('n_informative', None),
                                              n_samples=config['n_samples'],
                                              noise=config['noise'],
                                              bias=config.get('bias', None),
                                              tail_strength=config.get('tail_strength', None),
                                              effective_rank=config.get('effective_rank', None),
                                              random_seed=config['random_seed_synthetic_data'])
        synthetic_settings.append((setting, config))
    
    models = ["LightGBM", "MLP", "TabNet", "ResNet", "TabTransformer"]
    standard_settings = ["airlines_DepDelay_1M", 
                         "delays_zurich_transport",
                           "nyc-taxi-green-dec-2016", 
                           "microsoft", 
                           "yahoo", 
                           "year",
                             "medical_charges",
                             "california_housing",
                             "superconduct",
                             "houses",
                             "elevators",
                            "allstate_claims_severity",
                            "sgemm_gpu_kernel_performance",
                            "diamonds",
                            "particulate_matter_ukair_2017",
                            "seattlecrime6",
                            "bike_sharing_demand", 
                            "brazilian_houses",
                            "house_sales",
                            "sulfur",
                             ]
    methods = ["gradient_methods", "lime", "lime_captum", "shap"] #
    distance_measures = ["euclidean"]
    
    # Generate all command files
    created_files = []
    for random_seed in [42, 54, 123, 999, 21]:
        # First, handle standard benchmark datasets
        for model in models:
            for setting in standard_settings:
                for method in methods:
                    if method == "gradient_methods":
                        if model =="LightGBM":
                            continue
                        for gradient_method in ["IG", "IG+SmoothGrad", "Saliency"]:
                            for distance_measure in distance_measures:
                                file = create_command_file(
                                    output_dir=output_dir,
                                    model=model,
                                    setting=setting,
                                    method=method,
                                    distance_measure=distance_measure,
                                    kernel_width=None,
                                    num_lime_features=10,
                                    is_synthetic=False,
                                    skip_training=args.skip_training,
                                    force_training=args.force_training,
                                    skip_knn=args.skip_knn,
                                    create_additional_analysis_data=args.create_additional_analysis_data,
                                    skip_fraction=args.skip_fraction,
                                    gradient_method=gradient_method,
                                    random_seed=random_seed,
                                )
                                created_files.append(file)
                    else:  # lime
                        kernel_width = args.kernel_width
                        
                        for distance_measure in distance_measures:
                            file = create_command_file(
                                output_dir=output_dir,
                                model=model,
                                setting=setting,
                                method=method,
                                distance_measure=distance_measure,
                                kernel_width=kernel_width,
                                num_lime_features=10,
                                is_synthetic=False,
                                skip_training=args.skip_training,
                                force_training=args.force_training,
                                create_additional_analysis_data=args.create_additional_analysis_data,
                                skip_knn=args.skip_knn,
                                skip_fraction=args.skip_fraction,
                                random_seed=random_seed,
                            )
                            created_files.append(file)
        
        # Then, handle synthetic datasets
        for model in models:
            for setting_info in synthetic_settings:
                setting, config = setting_info
                for method in methods:
                    if method == "gradient_methods":
                        for gradient_method in ["IG", "IG+SmoothGrad", "Saliency"]:
                            for distance_measure in distance_measures:
                                file = create_command_file(
                                    output_dir=output_dir,
                                    model=model,
                                    setting=setting,
                                    method=method,
                                    distance_measure=distance_measure,
                                    kernel_width=None,
                                    num_lime_features=10,
                                    is_synthetic=True,
                                    skip_training=args.skip_training,
                                    create_additional_analysis_data=args.create_additional_analysis_data,
                                    force_training=args.force_training,
                                    skip_knn=args.skip_knn,
                                    skip_fraction=args.skip_fraction,
                                    gradient_method=gradient_method,
                                    synthetic_params=config,
                                    random_seed=random_seed,
                                )
                                created_files.append(file)
                    else:  # lime
                        kernel_width = args.kernel_width
                        for distance_measure in distance_measures:
                            file = create_command_file(
                                output_dir=output_dir,
                                model=model,
                                setting=setting,
                                method=method,
                                distance_measure=distance_measure,
                                kernel_width=kernel_width,
                                create_additional_analysis_data=args.create_additional_analysis_data,
                                num_lime_features=10,
                                is_synthetic=True,
                                skip_training=args.skip_training,
                                force_training=args.force_training,
                                skip_knn=args.skip_knn,
                                skip_fraction=args.skip_fraction,
                                synthetic_params=config,
                                random_seed=random_seed,
                            )
                            created_files.append(file)
        
        

    print("\nOr run experiments for a specific method:")
    print(f"<method>/run_all.sh")
    print("\nOr for a specific model within a method:")
    print(f"<method>/<model>/run_all.sh")
if __name__ == "__main__":
    main()
