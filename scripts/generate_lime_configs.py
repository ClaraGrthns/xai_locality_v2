import os
import yaml
import pathlib

folder_dir = str(pathlib.Path(__file__).parent.parent.absolute())
print(folder_dir)

DATASETS = {
    'standard': ['higgs', 'jannis', 'MiniBooNE'],
    'synthetic': [
        'n_feat50_n_informative10_n_redundant30_n_repeated0_n_classes2_n_samples100000_n_clusters_per_class3_class_sep0.9_flip_y0.01_random_state42',
        'n_feat50_n_informative2_n_redundant30_n_repeated0_n_classes2_n_samples100000_n_clusters_per_class2_class_sep0.9_flip_y0.01_random_state42',
        'n_feat100_n_informative50_n_redundant30_n_repeated0_n_classes2_n_samples100000_n_clusters_per_class3_class_sep0.9_flip_y0.01_random_state42'
    ]
}

MODELS = {
    'gbt': ['XGBoost', 'CatBoost', 'LightGBM'],
    'deep': ['TabNet', 'FTTransformer', 'ResNet', 'MLP', 'TabTransformer',
             'Trompt', 'ExcelFormer', 'FTTransformerBucket'],
    'ml' : ['LogReg']
}


def get_gbt_paths_synthetic_data(model, dataset):
    model_path = f'{folder_dir}/pretrained_models/{model}/synthetic_data/{model}_{dataset}.pt'
    return model_path

def create_lime_config(model, dataset, is_synthetic=False):
    # Determine if it's a GBT model
    # Get correct model type for GBT models
    if is_synthetic:
        is_ExcelFormer_str = "ExcelFormer_" if model == 'ExcelFormer' else ""
        model_path = f'{folder_dir}/pretrained_models/{model}/synthetic_data/{model}_{dataset}_results.pt'
        data_path = f'{folder_dir}/data/synthetic_data/{is_ExcelFormer_str}{dataset}_normalized_tensor_frame.pt'
    else:
        if model in MODELS['ml']:        
            data_path = f'{folder_dir}/data/LightGBM_{dataset}_normalized_data.pt'
        else:
            data_path =  f'{folder_dir}/data/{model}_{dataset}_normalized_data.pt'
        model_path = f'{folder_dir}/pretrained_models/{model}/{dataset}/{model}_normalized_binary_{dataset}_results.pt'
    config = {
        'explanation_method': {
            'method': 'lime'
        },
        'paths': {
            'results_path': f'{folder_dir}/results/lime/{model}/{"synthetic_data/" if is_synthetic else ""}{dataset}',
            'data_path': data_path,
            'model_path': model_path
        },
        'model': {
            'model_type': model
        },
        'analysis': {
            'num_features': 50,
            'num_samples': 1000,
            'random_seed': 42,
            'chunk_size': 20
        },
        'other': {
            'max_test_points': 200
        }
    }
    return config

def main():
    # base_path = '{folder_dir}/configs'
    base_path = f"{folder_dir}/configs"
    # Generate LIME configs for all models
    for model_type, models in MODELS.items():
        for model in models:
            # Standard datasets
            for dataset in DATASETS['standard']:
                path = f'{base_path}/lime/{model}/{dataset}'
                os.makedirs(path, exist_ok=True)
                
                config = create_lime_config(model, dataset)
                with open(f'{path}/config.yaml', 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
            
            # Synthetic datasets
            for dataset in DATASETS['synthetic']:
                path = f'{base_path}/lime/{model}/synthetic_data/{dataset}'
                os.makedirs(path, exist_ok=True)
                
                config = create_lime_config(model, dataset, is_synthetic=True)
                with open(f'{path}/config.yaml', 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)

if __name__ == '__main__':
    main()