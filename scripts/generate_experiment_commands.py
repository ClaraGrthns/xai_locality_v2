import os
import argparse
from pathlib import Path

dataset_lookup = {
        "small": {
            0: "adult_census_income",
            1: "mushroom",
            2: "bank_marketing",
            3: "magic_telescope",
            4: "bank_marketing",
            5: "california",
            6: "credit",
            7: "default_of_credit_card_clients",
            8: "electricity",
            9: "eye_movements",
            10: "heloc",
            11: "house_16H",
            12: "pol",
            13: "adult",
        },
        "medium": {
            0: "dota2",
            1: "kdd_census_income",
            2: "diabetes130us",
            3: "MiniBooNE",
            4: "albert",
            5: "covertype",
            6: "jannis",
            7: "road_safety",
            8: "higgs_small",
        },
        "large": {
            0: "higgs",
        },
    }


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate Python commands for running experiments")
    parser.add_argument('--output_dir', type=str, default='commands_sbach_files/experiment_commands',
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
    parser.add_argument('--random_seed', type=int, default=42,)
    parser.add_argument('--random_seed_synthetic_data', type=int, default=42,)
    parser.add_argument('--gradient_method', type=str, default="IG",
                        help='Gradient method to use (IG or IG+SmoothGrad)')
    return parser.parse_args()

def create_command_file(output_dir, model, setting, method, distance_measure, kernel_width, num_lime_features,
                        is_synthetic, skip_training, force_training, skip_knn, skip_fraction, gradient_method=None,
                        synthetic_params=None, random_seed=42):
    """Create a file containing the Python command for a specific configuration"""
    
    # Create directory structure - organize by method first
    if method == "gradient_methods" and gradient_method:
        if gradient_method == "IG":
            method_dir = os.path.join(output_dir, method, "integrated_gradient")
        else:
            method_dir = os.path.join(output_dir, method, "smooth_grad")
    else:
        method_dir = os.path.join(output_dir, method)
    
    # Then by model
    model_dir = os.path.join(method_dir, model)
    
    # Then by dataset
    if is_synthetic:
        dataset_dir = os.path.join(model_dir, "synthetic_data", setting)
    else:
        dataset_dir = os.path.join(model_dir, setting)
    
    # Finally by distance measure
    model_dir = os.path.join(dataset_dir, distance_measure)
    
    os.makedirs(model_dir, exist_ok=True)
    
    # Format args for the command
    base_args = f"--model_type {model} --setting {setting} --method {method} --distance_measure {distance_measure} --random_seed {random_seed}"
    
    if is_synthetic:
        # Use the synthetic parameters directly instead of parsing them from the setting name
        if synthetic_params is not None:
            # Extract all parameters with proper defaults
            n_features = synthetic_params.get('n_features', 50)
            n_informative = synthetic_params.get('n_informative', 10)
            n_redundant = synthetic_params.get('n_redundant', 30)
            n_repeated = synthetic_params.get('n_repeated', 0)
            n_classes = synthetic_params.get('n_classes', 2)
            n_samples = synthetic_params.get('n_samples', 100000)
            n_clusters_per_class = synthetic_params.get('n_clusters_per_class', 3)
            class_sep = synthetic_params.get('class_sep', 0.9)
            flip_y = synthetic_params.get('flip_y', 0.05)
            random_seed_synthetic_data = synthetic_params.get('random_seed_synthetic_data', 42)
            hypercube = synthetic_params.get('hypercube', False)
            
            # Add synthetic data parameters
            synthetic_args = (f" --n_features {n_features}"
                             f" --n_informative {n_informative}"
                             f" --n_redundant {n_redundant}"
                             f" --n_repeated {n_repeated}"
                             f" --n_classes {n_classes}"
                             f" --n_samples {n_samples}"
                             f" --n_clusters_per_class {n_clusters_per_class}"
                             f" --class_sep {class_sep}"
                             f" --flip_y {flip_y}"
                             f" --random_seed_synthetic_data {random_seed_synthetic_data}"
                             )
            
            # Add hypercube flag ONLY if it's True
            if hypercube:
                synthetic_args += " --hypercube"
        else:
            # If no synthetic_params provided, parse from the setting name
            params = {}
            for param in setting.split('_'):
                if param.startswith('n_feat'):
                    params['n_features'] = param[6:]
                elif param.startswith('n_informative'):
                    params['n_informative'] = param[13:]
                elif param.startswith('n_redundant'):
                    params['n_redundant'] = param[11:]
                elif param.startswith('n_repeated'):
                    params['n_repeated'] = param[10:]
                elif param.startswith('n_classes'):
                    params['n_classes'] = param[9:]
                elif param.startswith('n_samples'):
                    params['n_samples'] = param[9:]
                elif param.startswith('n_clusters_per_class'):
                    params['n_clusters_per_class'] = param[20:]
                elif param.startswith('class_sep'):
                    params['class_sep'] = param[9:]
                elif param.startswith('flip_y'):
                    params['flip_y'] = param[6:]
                elif param.startswith('random_state'):
                    params['random_seed_synthetic_data'] = param[12:]
                elif param.startswith('hypercube'):
                    params['hypercube'] = param[9:]
            
            # Add synthetic data parameters from parsed setting
            synthetic_args = (f" --n_features {params.get('n_features', 50)}"
                             f" --n_informative {params.get('n_informative', 10)}"
                             f" --n_redundant {params.get('n_redundant', 30)}"
                             f" --n_repeated {params.get('n_repeated', 0)}"
                             f" --n_classes {params.get('n_classes', 2)}"
                             f" --n_samples {params.get('n_samples', 100000)}"
                             f" --n_clusters_per_class {params.get('n_clusters_per_class', 3)}"
                             f" --class_sep {params.get('class_sep', 0.9)}"
                             f" --flip_y {params.get('flip_y', 0.05)}"
                             f" --random_seed_synthetic_data {params.get('random_seed_synthetic_data', 42)}"
                             )
            
            if params.get('hypercube', "False").lower() == "true":
                synthetic_args += " --hypercube"
            
        base_args += synthetic_args
        base_args += " --num_trials 15 --num_repeats 1 --epochs 10 --optimize"
    else:
        # Determine dataset size scale from the dataset_lookup dictionary
        dataset_scale = None
        for scale, datasets in dataset_lookup.items():
            if setting in [d for d in datasets.values()]:
                dataset_scale = scale
                break
        base_args += f" --scale {dataset_scale}"  # Default to small if not found or explicitly small
        base_args += " --use_benchmark --task_type binary_classification --num_repeats 1 "
        if "medium" in base_args:
            base_args += " --epochs 25 --include_val --num_trials 5 "
        elif "large" in base_args:
            base_args += " --epochs 10 --num_trials 3 "
        else:
            base_args += " --epochs 40 --include_val --include_trn --num_trials 10 "

    # Method-specific parameters
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
    
    # Create the full command
    command = f"/home/grotehans/miniconda3/envs/tab_models/bin/python -u run_experiment_setup.py {base_args}"
    
    # Define filename - include distance measure to distinguish files
    distance_suffix = f"_{distance_measure}"
    file_name_add_on = "_skip_kNN" if skip_knn else ""
    file_name_add_on += "_skip_fraction" if skip_fraction else ""
    file_name_add_on += "_force_training" if force_training else ""
    file_name_add_on += f"_random_seed_{random_seed}"
    
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
        {
            'n_features': 50, 
            'n_informative': 2, 
            'n_redundant': 30, 
            'n_repeated': 0, 
            'n_classes': 2, 
            'n_samples': 100000, 
            'n_clusters_per_class': 2, 
            'class_sep': 0.9, 
            'flip_y': 0.01, 
            'random_seed_synthetic_data': 42,
            'hypercube': True
        },
        {
            'n_features': 50, 
            'n_informative': 10, 
            'n_redundant': 30, 
            'n_repeated': 0, 
            'n_classes': 2, 
            'n_samples': 100000, 
            'n_clusters_per_class': 3, 
            'class_sep': 0.9, 
            'flip_y': 0.01, 
            'random_seed_synthetic_data': 42,
            'hypercube': True
        },
        {
            'n_features': 100, 
            'n_informative': 50, 
            'n_redundant': 30, 
            'n_repeated': 0, 
            'n_classes': 2, 
            'n_samples': 100000, 
            'n_clusters_per_class': 3, 
            'class_sep': 0.9, 
            'flip_y': 0.01, 
            'random_seed_synthetic_data': 42,
            'hypercube': True
        },
        {
            'n_features': 25, 
            'n_informative': 5, 
            'n_redundant': 15, 
            'n_repeated': 2, 
            'n_classes': 2, 
            'n_samples': 100000, 
            'n_clusters_per_class': 4, 
            'class_sep': 0.8, 
            'flip_y': 0.05, 
            'random_seed_synthetic_data': 42,
            'hypercube': False
        },
        {
            'n_features': 25, 
            'n_informative': 10, 
            'n_redundant': 5, 
            'n_repeated': 2, 
            'n_classes': 2, 
            'n_samples': 100000, 
            'n_clusters_per_class': 4, 
            'class_sep': 0.8, 
            'flip_y': 0.05, 
            'random_seed_synthetic_data': 42,
            'hypercube': True
        },
        {
            'n_features': 55, 
            'n_informative': 30, 
            'n_redundant': 5, 
            'n_repeated': 5, 
            'n_classes': 2, 
            'n_samples': 100000, 
            'n_clusters_per_class': 10, 
            'class_sep': 0.6, 
            'flip_y': 0.05, 
            'random_seed_synthetic_data': 42,
            'hypercube': False
        },
        {
            'n_features': 55, 
            'n_informative': 30, 
            'n_redundant': 5, 
            'n_repeated': 5, 
            'n_classes': 2, 
            'n_samples': 100000, 
            'n_clusters_per_class': 10, 
            'class_sep': 0.7, 
            'flip_y': 0.05, 
            'random_seed_synthetic_data': 42,
            'hypercube': True
        },
        {
            'n_features': 55, 
            'n_informative': 30, 
            'n_redundant': 5, 
            'n_repeated': 5, 
            'n_classes': 2, 
            'n_samples': 100000, 
            'n_clusters_per_class': 5, 
            'class_sep': 0.8, 
            'flip_y': 0.05, 
            'random_seed_synthetic_data': 42,
            'hypercube': False
        },
        {
            'n_features': 40, 
            'n_informative': 20, 
            'n_redundant': 5, 
            'n_repeated': 5, 
            'n_classes': 2, 
            'n_samples': 100000, 
            'n_clusters_per_class': 10, 
            'class_sep': 0.7, 
            'flip_y': 0.05, 
            'random_seed_synthetic_data': 42,
            'hypercube': False
        },
        {
            'n_features': 40, 
            'n_informative': 20, 
            'n_redundant': 5, 
            'n_repeated': 5, 
            'n_classes': 2, 
            'n_samples': 100000, 
            'n_clusters_per_class': 10, 
            'class_sep': 0.7, 
            'flip_y': 0.05, 
            'random_seed_synthetic_data': 42,
            'hypercube': True
        },
        {
            'n_features': 55, 
            'n_informative': 20, 
            'n_redundant': 5, 
            'n_repeated': 5, 
            'n_classes': 2, 
            'n_samples': 100000, 
            'n_clusters_per_class': 20, 
            'class_sep': 0.3, 
            'flip_y': 0.05, 
            'random_seed_synthetic_data': 42,
            'hypercube': False
        },
        {
            'n_features': 100, 
            'n_informative': 60, 
            'n_redundant': 5, 
            'n_repeated': 5, 
            'n_classes': 2, 
            'n_samples': 100000, 
            'n_clusters_per_class': 50, 
            'class_sep': 0.3, 
            'flip_y': 0.1, 
            'random_seed_synthetic_data': 42,
            'hypercube': False
        },
         {
            'n_features': 100, 
            'n_informative': 10, 
            'n_redundant': 50, 
            'n_repeated': 5, 
            'n_classes': 2, 
            'n_samples': 100000, 
            'n_clusters_per_class': 50, 
            'class_sep': 0.8, 
            'flip_y': 0.1, 
            'random_seed_synthetic_data': 42,
            'hypercube': False
        },
        {
            'n_features': 200, 
            'n_informative': 20, 
            'n_redundant': 50, 
            'n_repeated': 0, 
            'n_classes': 2, 
            'n_samples': 100000, 
            'n_clusters_per_class': 50, 
            'class_sep': 0.5, 
            'flip_y': 0.1, 
            'random_seed_synthetic_data': 42,
            'hypercube': False
        },

         {
            'n_features': 100, 
            'n_informative': 60, 
            'n_redundant': 5, 
            'n_repeated': 5, 
            'n_classes': 2, 
            'n_samples': 100000, 
            'n_clusters_per_class': 20, 
            'class_sep': 0.2, 
            'flip_y': 0.1, 
            'random_seed_synthetic_data': 42,
            'hypercube': False
        },

        {
            'n_features': 50, 
            'n_informative':10, 
            'n_redundant': 10, 
            'n_repeated': 0, 
            'n_classes': 2, 
            'n_samples': 100000, 
            'n_clusters_per_class': 40, 
            'class_sep': 0.2, 
            'flip_y': 0.01, 
            'random_seed_synthetic_data': 42,
            'hypercube': False
        },
        {
            'n_features': 50, 
            'n_informative':10, 
            'n_redundant': 10, 
            'n_repeated': 0, 
            'n_classes': 2, 
            'n_samples': 100000, 
            'n_clusters_per_class': 10, 
            'class_sep': 0.2, 
            'flip_y': 0.1, 
            'random_seed_synthetic_data': 42,
            'hypercube': False
        },
    ]
    
    # Generate setting names from configurations
    synthetic_settings = []
    for config in synthetic_configs:
        setting = (f"n_feat{config['n_features']}_"
                   f"n_informative{config['n_informative']}_"
                   f"n_redundant{config['n_redundant']}_"
                   f"n_repeated{config['n_repeated']}_"
                   f"n_classes{config['n_classes']}_"
                   f"n_samples{config['n_samples']}_"
                   f"n_clusters_per_class{config['n_clusters_per_class']}_"
                   f"class_sep{config['class_sep']}_"
                   f"flip_y{config['flip_y']}_"
                   f"random_state{config['random_seed_synthetic_data']}")
        if not config['hypercube']:
            setting += f"_hypercube{config['hypercube']}"
        synthetic_settings.append((setting, config))
    
    models = ["LightGBM", "MLP", "LogReg",  "TabNet", "FTTransformer", "ResNet", "TabTransformer"]
    standard_settings = ["diabetes130us", "MiniBooNE", "credit", "california", "magic_telescope",  "house_16H", "higgs_small", "higgs", "jannis"]#["higgs", "jannis"] # "bank_marketing"
    standard_categorical = ['albert', 'road_safety', 'kdd_census_income', "electricity", "adult_census_income", "adult", "bank_marketing", "mushroom"]
    standard_settings += standard_categorical
    methods = ["lime", "gradient_methods"]
    distance_measures = ["euclidean", "manhattan", "cosine"]
    
    # Generate all command files
    created_files = []
    
    # First, handle standard benchmark datasets
    for model in models:
        for setting in standard_settings:
            for method in methods:
                if method == "gradient_methods":
                    if model =="LightGBM":
                        continue
                    gradient_method = args.gradient_method  # Integrated Gradient
                    
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
                            skip_fraction=args.skip_fraction,
                            gradient_method=gradient_method,
                            random_seed=args.random_seed,
                        )
                        created_files.append(file)
                else:  # lime
                    kernel_width = "default"
                    
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
                            skip_knn=args.skip_knn,
                            skip_fraction=args.skip_fraction,
                            random_seed=args.random_seed,

                        )
                        created_files.append(file)
    
    # Then, handle synthetic datasets
    for model in models:
        for setting_info in synthetic_settings:
            setting, config = setting_info
            
            for method in methods:
                if method == "gradient_methods":
                    gradient_method = args.gradient_method  # Integrated Gradient
                    
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
                            force_training=args.force_training,
                            skip_knn=args.skip_knn,
                            skip_fraction=args.skip_fraction,
                            gradient_method=gradient_method,
                            synthetic_params=config,
                            random_seed=args.random_seed,
                        )
                        created_files.append(file)
                else:  # lime
                    kernel_width = "default"
                    
                    for distance_measure in distance_measures:
                        file = create_command_file(
                            output_dir=output_dir,
                            model=model,
                            setting=setting,
                            method=method,
                            distance_measure=distance_measure,
                            kernel_width=kernel_width,
                            num_lime_features=10,
                            is_synthetic=True,
                            skip_training=args.skip_training,
                            force_training=args.force_training,
                            skip_knn=args.skip_knn,
                            skip_fraction=args.skip_fraction,
                            synthetic_params=config,
                            random_seed=args.random_seed,
                        )
                        created_files.append(file)
    
    
if __name__ == "__main__":
    main()
