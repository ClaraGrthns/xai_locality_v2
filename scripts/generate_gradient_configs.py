import os
import yaml
import pathlib

# Models that support gradient explanations
DEEP_MODELS = ['TabNet', 'FTTransformer', 'ResNet', 'MLP', 'TabTransformer',
               'Trompt', 'ExcelFormer', 'FTTransformerBucket']
ML_MODELS = ['LogReg']
folder_dir = str(pathlib.Path(__file__).parent.parent.absolute()) # Datasets#
print(folder_dir)
DATASETS = {
    'standard': ['higgs', 'jannis', 'MiniBooNE'],
    'synthetic': [
        'n_feat50_n_informative10_n_redundant30_n_repeated0_n_classes2_n_samples100000_n_clusters_per_class3_class_sep0.9_flip_y0.01_random_state42',
        'n_feat50_n_informative2_n_redundant30_n_repeated0_n_classes2_n_samples100000_n_clusters_per_class2_class_sep0.9_flip_y0.01_random_state42',
        'n_feat100_n_informative50_n_redundant30_n_repeated0_n_classes2_n_samples100000_n_clusters_per_class3_class_sep0.9_flip_y0.01_random_state42'
    ]
}

def create_config(model, dataset, is_synthetic=False):
    is_ExcelFormer_str = "ExcelFormer_" if model == 'ExcelFormer' else ""    
    if is_synthetic:
        model_path = f'{folder_dir}/pretrained_models/{model}/synthetic_data/{model}_{dataset}_results.pt'
        data_path = f'{folder_dir}/data/synthetic_data/{is_ExcelFormer_str}{dataset}_normalized_tensor_frame.pt'
    else:
        if model in ML_MODELS:        
            data_path = f'{folder_dir}/data/LightGBM_{dataset}_normalized_data.pt'
        else:
            data_path =  f'{folder_dir}/data/{model}_{dataset}_normalized_data.pt'
        model_path = f'{folder_dir}/pretrained_models/{model}/{dataset}/{model}_normalized_binary_{dataset}_results.pt'
        
    
    config = {
        'explanation_method': {
            'method': 'gradient'
        },
        'paths': {
            'results_path': f'{folder_dir}/results/gradient_methods/integrated_gradient/{model}/{"synthetic_data/" if is_synthetic else ""}{dataset}',
            'data_path': data_path,
            'model_path': model_path,
        },
        'model': {
            'model_type': model,
            'gradient_method': 'IG'
        },
        'analysis': {
            'distance_measure': 'euclidean',
            'max_frac': 0.05,
            'num_frac': 50,
            'random_seed': 42,
            'chunk_size': 100
        },
        'other': {
            'max_test_points': 200
        }
    }
    return config

def main():
    base_path = f'{folder_dir}/configs/gradient_methods/integrated_gradient'
    for model in DEEP_MODELS + ML_MODELS:
        # Standard datasets
        for dataset in DATASETS['standard']:
            path = f'{base_path}/{model}/{dataset}'
            os.makedirs(path, exist_ok=True)
            
            config = create_config(model, dataset)
            with open(f'{path}/config.yaml', 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        
        # Synthetic datasets
        for dataset in DATASETS['synthetic']:
            path = f'{base_path}/{model}/synthetic_data/{dataset}'
            os.makedirs(path, exist_ok=True)
            
            config = create_config(model, dataset, is_synthetic=True)
            with open(f'{path}/config.yaml', 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

if __name__ == '__main__':
    main()