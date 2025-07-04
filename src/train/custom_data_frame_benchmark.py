# This script is adapted from PyG (PyTorch Geometric).
# Original source: https://github.com/pyg-team/pytorch-frame/benchmark/data_frame_benchmark.py
# 
# Modifications:
# - Adapted for custom dataset
# - Normalize datasets before tuning
# - Saving models and data for later analysis
# - Change execution of mains only when script is run directly
#
# The original license follows:
#
# Copyright (c) 2023 PyG Team <team@pyg.org>
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy...

from src.utils.metrics import binary_classification_metrics, regression_metrics
from torch_frame import Metric
import argparse
import math
import os
import os.path as osp
import time
from typing import Any, Optional
from pathlib import Path

import numpy as np
import optuna
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, Module, MSELoss
from torch.optim.lr_scheduler import ExponentialLR
from torchmetrics import AUROC, Accuracy, MeanSquaredError
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import pandas as pd
from torch_frame.data import Dataset

import torch_frame
from torch_frame import stype
from torch_frame.data import DataLoader
from torch_frame.datasets import DataFrameBenchmark
from torch_frame.gbdt import CatBoost, LightGBM, XGBoost
from torch_frame.nn.encoder import EmbeddingEncoder, LinearBucketEncoder
from torch_frame.nn.models import (
    MLP,
    ExcelFormer,
    FTTransformer,
    ResNet,
    TabNet,
    TabTransformer,
    Trompt,
)
from torch_frame.typing import TaskType
import random 
import sys
sys.path.append(osp.join(os.getcwd(), '..'))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset.synthetic_data import create_synthetic_classification_data_sklearn, create_custom_synthetic_regression_data #create_synthetic_regression_data_sklearn

def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def normalize_features(train_features, val_features, test_features):
    """Normalize numerical features using StandardScaler."""
    # Fit scaler on training data only
    scaler = StandardScaler()
    scaler.fit(train_features)
    
    # Transform all datasets
    train_features_norm = scaler.transform(train_features)
    val_features_norm = scaler.transform(val_features)
    test_features_norm = scaler.transform(test_features)
    
    # Save scaler for later use
    return train_features_norm, val_features_norm, test_features_norm, scaler

def normalize_target(train_y, val_y, test_y):
    """Normalize target values."""
    # Fit scaler on training data only
    scaler = StandardScaler()
    scaler.fit(train_y.reshape(-1, 1))
    
    # Transform all datasets
    train_y_norm = scaler.transform(train_y.reshape(-1, 1)).flatten()
    val_y_norm = scaler.transform(val_y.reshape(-1, 1)).flatten()
    test_y_norm = scaler.transform(test_y.reshape(-1, 1)).flatten()
    
    return train_y_norm, val_y_norm, test_y_norm

def normalize_tensor_frame(train_tf, val_tf, test_tf):
    """Normalize numerical features and return normalized TensorFrames."""
    if stype.numerical not in train_tf.feat_dict:
        return train_tf, val_tf, test_tf
    
    # Get numerical features
    train_num = train_tf.feat_dict[stype.numerical]
    val_num = val_tf.feat_dict[stype.numerical]
    test_num = test_tf.feat_dict[stype.numerical]
    
    # Calculate mean and std from training data
    mean = train_num.mean(dim=0, keepdim=True)
    std = train_num.std(dim=0, keepdim=True)
    
    # Normalize all sets
    train_num_norm = (train_num - mean) / std
    val_num_norm = (val_num - mean) / std
    test_num_norm = (test_num - mean) / std
    
    # Update numerical features with normalized values
    train_tf.feat_dict[stype.numerical] = train_num_norm
    val_tf.feat_dict[stype.numerical] = val_num_norm
    test_tf.feat_dict[stype.numerical] = test_num_norm
    
    return train_tf, val_tf, test_tf



TRAIN_CONFIG_KEYS = ["batch_size", "gamma_rate", "base_lr"]
GBDT_MODELS = ["XGBoost", "CatBoost", "LightGBM"]

parser = argparse.ArgumentParser()
parser.add_argument(
    '--task_type', type=str, choices=[
        'binary_classification',
        'multiclass_classification',
        'regression',
    ], default='binary_classification')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--regression', action='store_true')
parser.add_argument('--num_trials', type=int, default=20,
                    help='Number of Optuna-based hyper-parameter tuning.')
parser.add_argument(
    '--num_repeats', type=int, default=5,
    help='Number of repeated training and eval on the best config.')
parser.add_argument(
    '--model_type', type=str, default='TabNet', choices=[
        'TabNet', 'FTTransformer', 'ResNet', 'MLP', 'TabTransformer', 'Trompt',
        'ExcelFormer', 'FTTransformerBucket', 'XGBoost', 'CatBoost', 'LightGBM'
    ])
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--results_folder', type=str, default='')
parser.add_argument('--data_path', type=str)
parser.add_argument("--n_features", type=int, default=20)
parser.add_argument("--n_informative", type=int, default=15)
parser.add_argument("--n_clusters_per_class", type=int, default=2)
parser.add_argument("--class_sep", type=float, default=1.0)
parser.add_argument("--flip_y", type=float, default=0.01)
parser.add_argument("--hypercube", action="store_true", help="If True, the clusters are put on the vertices of a hypercube. If False, the clusters are put on the vertices of a random polytope.")
parser.add_argument("--n_redundant", type=int, default=5)
parser.add_argument("--n_repeated", type=int, default=0)
parser.add_argument("--n_classes", type=int, default=2)
parser.add_argument("--n_samples", type=int, default=1000000)
parser.add_argument("--n_trials", type=int, default=100)
parser.add_argument("--timeout", type=int, default=3600)
parser.add_argument("--data_folder", type=str, default="/home/grotehans/xai_locality/data/synthetic_data")
parser.add_argument("--test_size", type=float, default=0.4)
parser.add_argument("--val_size", type=float, default=0.1)
# Adding the missing arguments for regression dataset creation
parser.add_argument("--noise", type=float, default=0.1, help="Standard deviation of the gaussian noise")
parser.add_argument("--bias", type=float, default=0.0, help="Bias term in the underlying linear model")
parser.add_argument("--tail_strength", type=float, default=0.5, help="Relative importance of the fat noisy tail of the covariance matrix")
parser.add_argument("--coef", type=bool, default=False, help="Whether to return the coefficients of the underlying linear model")
parser.add_argument("--effective_rank", type=int, default=None, help="Approximate number of singular vectors to use to generate the covariance matrix")
parser.add_argument("--setting", type=str, default="", help="Name of the experimental setting")

def prepare_data_and_models(args):
    """Prepare data and initialize model configurations based on provided arguments."""
    # Set random seeds for reproducibility
    set_random_seeds(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    torch.manual_seed(args.seed)
    
    # Prepare datasets
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')

    # if args.data_path:
    #     data_path_wo_file_ending = Path(args.data_path).stem
    #     data_folder = os.path.dirname(args.data_path)
    #     data_path = osp.join(data_folder, args.setting + ".npz")
    #     data_path_wo_file_ending = Path(data_path).stem
    #     print(data_path)
    #     data = np.load(data_path)
    #     print(args.data_path)
    #     print(data.files)
    #     tst_feat, val_feat, trn_feat = data['X_test'], data['X_val'], data['X_train'] 
    #     y_test, y_val, y_train = data['y_test'], data['y_val'], data['y_train']
    if args.regression:
        data_path_wo_file_ending, trn_feat, val_feat, tst_feat, y_train, y_val, y_test, col_indices = create_custom_synthetic_regression_data(
            regression_mode = args.regression_mode,
            n_features=args.n_features,
            n_informative=args.n_informative,
            n_samples=args.n_samples, 
            noise=args.noise,
            bias=args.bias,
            random_seed=args.seed,
            data_folder=args.data_folder,
            test_size=args.test_size,
            val_size=args.val_size,
            tail_strength=args.tail_strength,
            effective_rank=args.effective_rank,
            force_create = args.force_training
            )
        data_folder = args.data_folder
    else: 
        data_path_wo_file_ending, trn_feat, val_feat, tst_feat, y_train, y_val, y_test = create_synthetic_classification_data_sklearn(
            n_features=args.n_features, 
            n_informative=args.n_informative, 
            n_redundant=args.n_redundant, 
            n_repeated=args.n_repeated,
            n_classes=args.n_classes, 
            n_samples=args.n_samples,
            n_clusters_per_class=args.n_clusters_per_class, 
            class_sep=args.class_sep, 
            flip_y=args.flip_y, 
            random_seed=args.seed, 
            data_folder=args.data_folder,
            hypercube = args.hypercube,
            test_size=args.test_size, 
            val_size=args.val_size,
            force_create = args.force_training)
        data_folder = args.data_folder
        
    trn_feat_norm, val_feat_norm, tst_feat_norm, scaler = normalize_features(trn_feat, val_feat, tst_feat)
    if args.regression:
        y_train, y_val, y_test = normalize_target(y_train, y_val, y_test)

    df_trn = pd.DataFrame(trn_feat_norm)
    df_trn['y'] = y_train
    df_val = pd.DataFrame(val_feat_norm)
    df_val['y'] = y_val
    df_tst = pd.DataFrame(tst_feat_norm)
    df_tst['y'] = y_test
    col_to_stype = {col: torch_frame.numerical for col in df_trn.columns}
    col_to_stype['y'] = torch_frame.numerical if args.regression else torch_frame.categorical

    os.makedirs(args.results_folder, exist_ok=True)

    train_dataset = Dataset(df_trn, col_to_stype=col_to_stype, target_col='y')
    val_dataset = Dataset(df_val, col_to_stype=col_to_stype, target_col='y')
    test_dataset = Dataset(df_tst, col_to_stype=col_to_stype, target_col='y')

    train_dataset.materialize()
    val_dataset.materialize()
    test_dataset.materialize()

    dataset = train_dataset

    train_tensor_frame = train_dataset.tensor_frame
    val_tensor_frame = val_dataset.tensor_frame
    test_tensor_frame = test_dataset.tensor_frame

    print(f"Train: {len(train_tensor_frame)}, Val: {len(val_tensor_frame)}, "
          f"Test: {len(test_tensor_frame)}")
          
    # Initialize model classes based on model type
    if args.model_type in GBDT_MODELS:
        gbdt_cls_dict = {
            'XGBoost': XGBoost,
            'CatBoost': CatBoost,
            'LightGBM': LightGBM
        }
        model_cls = gbdt_cls_dict[args.model_type]

        normalized_data = {
            'train': train_tensor_frame,
            'val': val_tensor_frame,
            'test': test_tensor_frame
        }
        if args.model_type != 'ExcelFormer':
            data_path_wo_file_ending_data = data_path_wo_file_ending
        else:
            data_path_wo_file_ending_data = f'ExcelFormer_{data_path_wo_file_ending}'
        tensorframe_file_name = os.path.join(data_folder, data_path_wo_file_ending_data + "_normalized_tensor_frame.pt")
        torch.save(normalized_data, tensorframe_file_name)
                   
        return {
            'model_cls': model_cls,
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'test_dataset': test_dataset,
            'data_path_wo_file_ending': data_path_wo_file_ending,
            'data_folder': data_folder,
            'device': device,
            'train_tensor_frame': train_tensor_frame,
            'val_tensor_frame': val_tensor_frame,
            'test_tensor_frame': test_tensor_frame,
            'dataset': dataset
        }
    else:
        # Initialize for deep learning models
        if dataset.task_type == TaskType.BINARY_CLASSIFICATION:
            out_channels = 1
            loss_fun = BCEWithLogitsLoss()
            metric_computer = AUROC(task='binary').to(device)
            higher_is_better = True
        elif dataset.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            out_channels = dataset.num_classes
            loss_fun = CrossEntropyLoss()
            metric_computer = Accuracy(task='multiclass',
                                      num_classes=dataset.num_classes).to(device)
            higher_is_better = True
        elif dataset.task_type == TaskType.REGRESSION:
            out_channels = 1
            loss_fun = MSELoss()
            metric_computer = MeanSquaredError(squared=False).to(device)
            higher_is_better = False
            
        # Initialize model-specific configurations
        # To be set for each model
        model_cls = None
        col_stats = None
        model_search_space = {}
        train_search_space = {}

        # Set up model specific search space
        if args.model_type == 'TabNet':
            model_search_space = {
                'split_attn_channels': [64, 128, 256],
                'split_feat_channels': [64, 128, 256],
                'gamma': [1., 1.2, 1.5],
                'num_layers': [4, 6, 8],
            }
            if args.complexity_model == "simple":
                model_search_space = {
                    'split_attn_channels': [64],  # Smallest attention channels
                    'split_feat_channels': [64],  # Smallest feature channels
                    'gamma': [1.0],               # Less feature reuse
                    'num_layers': [2],            # Fewest layers
                }
            elif args.complexity_model == "complex":
                model_search_space ={
                'split_attn_channels': [256],  # Largest attention channels
                'split_feat_channels': [256],  # Largest feature channels
                'gamma': [1.5],               # Encourages more feature reuse
                'num_layers': [8],            # Deepest architecture

                }
            train_search_space = {
                'batch_size': [2048, 4096],
                'base_lr': [0.001, 0.01],
                'gamma_rate': [0.9, 0.95, 1.],
            }
            model_cls = TabNet
            col_stats = dataset.col_stats
        elif args.model_type == 'FTTransformer':
            model_search_space = {
                'channels': [64, 128, 256],
                'num_layers': [4, 6, 8],
            }
            train_search_space = {
                'batch_size': [256, 512],
                'base_lr': [0.0001, 0.001],
                'gamma_rate': [0.9, 0.95, 1.],
            }
            model_cls = FTTransformer
            col_stats = dataset.col_stats
        elif args.model_type == 'FTTransformerBucket':
            model_search_space = {
                'channels': [64, 128, 256],
                'num_layers': [4, 6, 8],
            }
            train_search_space = {
                'batch_size': [256, 512],
                'base_lr': [0.0001, 0.001],
                'gamma_rate': [0.9, 0.95, 1.],
            }
            model_cls = FTTransformer
            col_stats = dataset.col_stats
        elif args.model_type == 'ResNet':
            model_search_space = {
                'channels': [64, 128, 256],
                'num_layers': [4, 6, 8],
            }
            if args.complexity_model == "simple":
                model_search_space['num_layers'] = [4]
                model_search_space['channels'] = [64]
            elif args.complexity_model == "complex":
                model_search_space['num_layers'] = [8]
                model_search_space['channels'] = [256]
            
            train_search_space = {
                'batch_size': [256, 512],
                'base_lr': [0.0001, 0.001],
                'gamma_rate': [0.9, 0.95, 1.],
            }
            model_cls = ResNet
            col_stats = dataset.col_stats
        elif args.model_type == 'MLP':
            model_search_space = {
                'channels': [64, 128, 256],
                'num_layers': [1, 2, 4],
            }
            if args.complexity_model == "simple":
                model_search_space['num_layers'] = [1]
                model_search_space['channels'] = [64]
            elif args.complexity_model == "complex":
                model_search_space['num_layers'] = [8]
                model_search_space['channels'] = [256]

            train_search_space = {
                'batch_size': [256, 512],
                'base_lr': [0.0001, 0.001],
                'gamma_rate': [0.9, 0.95, 1.],
            }
            model_cls = MLP
            col_stats = dataset.col_stats
        elif args.model_type == 'TabTransformer':
            model_search_space = {
                'channels': [16, 32, 64, 128],
                'num_layers': [4, 6, 8],
                'num_heads': [4, 8],
                'encoder_pad_size': [2, 4],
                'attn_dropout': [0, 0.2],
                'ffn_dropout': [0, 0.2],
            }
            train_search_space = {
                'batch_size': [128, 256],
                'base_lr': [0.0001, 0.001],
                'gamma_rate': [0.9, 0.95, 1.],
            }
            model_cls = TabTransformer
            col_stats = dataset.col_stats
        elif args.model_type == 'Trompt':
            model_search_space = {
                'channels': [64, 128, 192],
                'num_layers': [4, 6, 8],
                'num_prompts': [64, 128, 192],
            }
            if args.complexity_model == "simple":
                model_search_space['num_layers'] = [4]
                model_search_space['channels'] = [64]
                model_search_space['num_prompts'] = [64]
            elif args.complexity_model == "complex":
                model_search_space['num_layers'] = [8]
                model_search_space['channels'] = [256]
                model_search_space['num_prompts'] = [256]
            train_search_space = {
                'batch_size': [128],
                'base_lr': [0.01, 0.001],
                'gamma_rate': [0.9, 0.95, 1.],
            }
            if train_tensor_frame.num_cols > 20:
                # Reducing the model size to avoid GPU OOM
                model_search_space['channels'] = [64, 128]
                model_search_space['num_prompts'] = [64, 128]
            elif train_tensor_frame.num_cols > 50:
                model_search_space['channels'] = [64]
                model_search_space['num_prompts'] = [64]
            model_cls = Trompt
            col_stats = dataset.col_stats
        elif args.model_type == 'ExcelFormer':
            from torch_frame.transforms import (
                CatToNumTransform,
                MutualInformationSort,
            )
            categorical_transform = CatToNumTransform()
            categorical_transform.fit(train_dataset.tensor_frame,
                                    dataset.col_stats)
            train_tensor_frame = categorical_transform(train_tensor_frame)
            val_tensor_frame = categorical_transform(val_tensor_frame)
            test_tensor_frame = categorical_transform(test_tensor_frame)
            col_stats = categorical_transform.transformed_stats

            mutual_info_sort = MutualInformationSort(task_type=dataset.task_type)
            mutual_info_sort.fit(train_tensor_frame, col_stats)
            train_tensor_frame = mutual_info_sort(train_tensor_frame)
            val_tensor_frame = mutual_info_sort(val_tensor_frame)
            test_tensor_frame = mutual_info_sort(test_tensor_frame)

            model_search_space = {
                'in_channels': [128, 256],
                'num_heads': [8, 16, 32],
                'num_layers': [4, 6, 8],
                'diam_dropout': [0, 0.2],
                'residual_dropout': [0, 0.2],
                'aium_dropout': [0, 0.2],
                'mixup': [None, 'feature', 'hidden'],
                'beta': [0.5],
                'num_cols': [train_tensor_frame.num_cols],
            }
            train_search_space = {
                'batch_size': [256, 512] if train_tensor_frame.num_cols < 100 else [128, 256],
                'base_lr': [0.001],
                'gamma_rate': [0.9, 0.95, 1.],
            }
            model_cls = ExcelFormer

        assert model_cls is not None
        assert col_stats is not None
        assert set(train_search_space.keys()) == set(TRAIN_CONFIG_KEYS)
        col_names_dict = train_tensor_frame.col_names_dict
        if args.model_type != 'ExcelFormer':
            data_path_wo_file_ending_col = data_path_wo_file_ending
        else:
            data_path_wo_file_ending_col = f'ExcelFormer_{data_path_wo_file_ending}'
        print("saving col names and stats to", os.path.join(data_folder, data_path_wo_file_ending_col + "_normalized_tensor_frame_col_names_dict.pt"))
        torch.save(col_names_dict, os.path.join(data_folder, data_path_wo_file_ending_col + "_normalized_tensor_frame_col_names_dict.pt"))
        torch.save(col_stats, os.path.join(data_folder, data_path_wo_file_ending_col + "_normalized_tensor_frame_col_stats.pt"))

        normalized_data = {
            'train': train_tensor_frame,
            'val': val_tensor_frame,
            'test': test_tensor_frame
        }
        if args.model_type != 'ExcelFormer':
            data_path_wo_file_ending_data = data_path_wo_file_ending
        else:
            data_path_wo_file_ending_data = f'ExcelFormer_{data_path_wo_file_ending}'
        tensorframe_file_name = os.path.join(data_folder, data_path_wo_file_ending_data + "_normalized_tensor_frame.pt")
        torch.save(normalized_data, tensorframe_file_name)
        
        return {
            'model_cls': model_cls,
            'model_search_space': model_search_space,
            'train_search_space': train_search_space,
            'col_stats': col_stats,
            'col_names_dict': col_names_dict,
            'out_channels': out_channels,
            'loss_fun': loss_fun,
            'metric_computer': metric_computer,
            'higher_is_better': higher_is_better,
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'test_dataset': test_dataset,
            'data_path_wo_file_ending': data_path_wo_file_ending,
            'data_folder': data_folder,
            'device': device,
            'train_tensor_frame': train_tensor_frame,
            'val_tensor_frame': val_tensor_frame,
            'test_tensor_frame': test_tensor_frame,
            'dataset': dataset
        }

def main():
    """Main function to execute the script."""
    args = parser.parse_args()
    print(args)
    
    if args.model_type in GBDT_MODELS:
        main_gbdt(args)
    else:
        main_deep_models(args)

def main_deep_models(args=None):
    """Execute deep learning model training and evaluation."""
    if args is None:
        args = parser.parse_args()
        
    # Get prepared data and model configurations
    config = prepare_data_and_models(args)
    
    # Extract needed variables from config
    model_cls = config['model_cls']
    model_search_space = config['model_search_space']
    train_search_space = config['train_search_space']
    col_stats = config['col_stats']
    col_names_dict = config['col_names_dict']
    out_channels = config['out_channels']
    loss_fun = config['loss_fun']
    metric_computer = config['metric_computer']
    higher_is_better = config['higher_is_better']
    data_path_wo_file_ending = config['data_path_wo_file_ending']
    device = config['device']
    
    # Hyper-parameter optimization with Optuna
    print("Hyper-parameter search via Optuna")
    start_time = time.time()
    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(),
        direction="maximize" if higher_is_better else "minimize",
    )
    
    # Create a wrapper for the objective function that has access to the config
    def objective_wrapper(trial):
        model_cfg = {}
        for name, search_list in model_search_space.items():
            model_cfg[name] = trial.suggest_categorical(name, search_list)
        train_cfg = {}
        for name, search_list in train_search_space.items():
            train_cfg[name] = trial.suggest_categorical(name, search_list)

        model, best_val_metric, _ = train_and_eval_with_cfg(
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            trial=trial,
            config=config,
            args = args
        )
        return best_val_metric
    
    # Pass the wrapped objective function to optimize
    study.optimize(objective_wrapper, n_trials=args.num_trials)
    end_time = time.time()
    search_time = end_time - start_time
    print("Hyper-parameter search done. Found the best config.")
    params = study.best_params
    best_train_cfg = {}
    for train_cfg_key in TRAIN_CONFIG_KEYS:
        best_train_cfg[train_cfg_key] = params.pop(train_cfg_key)
    best_model_cfg = params

    print(f"Repeat experiments {args.num_repeats} times with the best train "
          f"config {best_train_cfg} and model config {best_model_cfg}.")
    start_time = time.time()
    best_val_metrics = []
    best_test_metrics = []
    for _ in range(args.num_repeats):
        best_model_state_dict, best_val_metric, best_test_metric = train_and_eval_with_cfg(
            best_model_cfg, best_train_cfg, config=config, args=args)
        best_val_metrics.append(best_val_metric)
        best_test_metrics.append(best_test_metric)
    end_time = time.time()
    final_model_time = (end_time - start_time) / args.num_repeats
    best_val_metrics = np.array(best_val_metrics)
    best_test_metrics = np.array(best_test_metrics)

    result_dict = {
        'args': args.__dict__,
        'best_val_metrics': best_val_metrics,
        'best_test_metrics': best_test_metrics,
        'best_val_metric': best_val_metrics.mean(),
        'best_test_metric': best_test_metrics.mean(),
        'best_train_cfg': best_train_cfg,
        'best_model_cfg': best_model_cfg,
        # 'search_time': search_time,
        'final_model_time': final_model_time,
        # 'total_time': search_time + final_model_time,
    }
    print(result_dict)
    # Save results
    results_file_path = os.path.join(args.results_folder, f'{args.model_type}_{data_path_wo_file_ending}_results.pt')
    os.makedirs(args.results_folder, exist_ok=True)
    torch.save({'model_state_dict': best_model_state_dict, **result_dict},
                results_file_path)

def train(
    model: Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss_fun: Module = BCEWithLogitsLoss(),
    dataset: Optional[Dataset] = None,
    metric_computer: Optional[Module] = None,
    device: Optional[torch.device] = None,
    out_channels: Optional[int] = None,
) -> float:
    model.train()
    loss_accum = total_count = 0
    print("start training")

    for tf in tqdm(loader, desc=f'Epoch: {epoch}'):
        tf = tf.to(device)
        y = tf.y
        if isinstance(model, ExcelFormer):
            # Train with FEAT-MIX or HIDDEN-MIX
            pred, y = model(tf, mixup_encoded=True)
        elif isinstance(model, Trompt):
            # Trompt uses the layer-wise loss
            pred = model.forward_stacked(tf)
            num_layers = pred.size(1)
            # [batch_size * num_layers, num_classes]
            pred = pred.view(-1, out_channels)
            y = tf.y.repeat_interleave(num_layers)
        else:
            pred = model(tf)

        if pred.size(1) == 1:
            pred = pred.view(-1, )
        if dataset.task_type == TaskType.BINARY_CLASSIFICATION:
            y = y.to(torch.float)
        loss = loss_fun(pred, y)
        optimizer.zero_grad()
        loss.backward()
        loss_accum += float(loss) * len(tf.y)
        total_count += len(tf.y)
        optimizer.step()
    return loss_accum / total_count


@torch.no_grad()
def test(
    model: Module,
    loader: DataLoader,
    dataset: Optional[Dataset] = None,
    metric_computer: Optional[Module] = None,
    device: Optional[torch.device] = None,
) -> float:
    model.eval()
    metric_computer.reset()
    for tf in loader:
        tf = tf.to(device)
        pred = model(tf)
        if dataset.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            pred = pred.argmax(dim=-1)
        elif dataset.task_type == TaskType.REGRESSION:
            pred = pred.view(-1, )
        metric_computer.update(pred, tf.y)
    return metric_computer.compute().item()

def train_and_eval_with_cfg(
    model_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    args,
    trial: Optional[optuna.trial.Trial] = None,
    config: Optional[dict] = None,
) -> tuple[float, float]:
    """Train and evaluate a model with the given configuration."""
    if config is None:
        # If called directly without config, parse args and prepare data
        args = parser.parse_args()
        config = prepare_data_and_models(args)
    
    # Extract needed variables from config
    model_cls = config['model_cls']
    col_stats = config['col_stats']
    col_names_dict = config['col_names_dict']
    out_channels = config['out_channels'] 
    dataset = config['dataset']
    loss_fun = config['loss_fun']
    metric_computer = config['metric_computer']
    higher_is_better = config['higher_is_better']
    train_tensor_frame = config['train_tensor_frame']
    val_tensor_frame = config['val_tensor_frame']
    test_tensor_frame = config['test_tensor_frame']
    device = config['device']
    data_path_wo_file_ending = config['data_path_wo_file_ending']
    
    # Use model_cfg to set up training procedure
    if args.model_type == 'FTTransformerBucket':
        # Use LinearBucketEncoder instead
        stype_encoder_dict = {
            stype.categorical: EmbeddingEncoder(),
            stype.numerical: LinearBucketEncoder(),
        }
        model_cfg['stype_encoder_dict'] = stype_encoder_dict
    model = model_cls(
        **model_cfg,
        out_channels=out_channels,
        col_stats=col_stats,
        col_names_dict=col_names_dict,
    ).to(device)
    model.reset_parameters()
    # Use train_cfg to set up training procedure
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg['base_lr'])
    lr_scheduler = ExponentialLR(optimizer, gamma=train_cfg['gamma_rate'])
    train_loader = DataLoader(train_tensor_frame,
                              batch_size=train_cfg['batch_size'], shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_tensor_frame,
                            batch_size=train_cfg['batch_size'])
    test_loader = DataLoader(test_tensor_frame,
                             batch_size=train_cfg['batch_size'])

    if higher_is_better:
        best_val_metric = 0
    else:
        best_val_metric = math.inf

    for epoch in range(1, args.epochs + 1):
        train_loss = train(model=model,
                           loader=train_loader,
                           optimizer=optimizer,
                            epoch=epoch,
                            loss_fun=loss_fun,
                            dataset=dataset,
                            metric_computer=metric_computer,
                            device=device,
                            out_channels=out_channels)

        val_metric = test(model=model,
                          loader=val_loader,
                          dataset=dataset,
                          metric_computer=metric_computer,
                          device=device)
        if higher_is_better:
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                best_test_metric = test(model, 
                                        test_loader,
                                        dataset=dataset,
                                        metric_computer=metric_computer,
                                        device=device)
                # save new best model
                best_model_state_dict = model.state_dict()
                model_path = os.path.join(args.results_folder,  f'{args.model_type}_{data_path_wo_file_ending}_best_model.pt')
                torch.save(model.state_dict(), model_path)
        else:
            if val_metric < best_val_metric:
                best_val_metric = val_metric
                best_test_metric = test(model, 
                                        test_loader,
                                        dataset=dataset,
                                        metric_computer=metric_computer,
                                        device=device)
                best_model_state_dict = model.state_dict()
                model_path = os.path.join(args.results_folder, f'{args.model_type}_{data_path_wo_file_ending}_best_model.pt')
                torch.save(model.state_dict(), model_path)
        lr_scheduler.step()
        print(f'Train Loss: {train_loss:.4f}, Val: {val_metric:.4f}')

        if trial is not None:
            trial.report(val_metric, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    print(
        f'Best val: {best_val_metric:.4f}, Best test: {best_test_metric:.4f}')
    return best_model_state_dict, best_val_metric, best_test_metric

def main_gbdt(args=None):
    """Execute GBDT model training and evaluation."""
    if args is None:
        args = parser.parse_args()
    
    # Get prepared data and model configurations
    config = prepare_data_and_models(args)
    
    # Extract needed variables from config
    model_cls = config['model_cls']
    train_dataset = config['train_dataset']
    val_dataset = config['val_dataset']
    test_dataset = config['test_dataset']
    data_path_wo_file_ending = config['data_path_wo_file_ending']
    dataset = config['dataset']
    
    if dataset.task_type.is_classification:
        num_classes = dataset.num_classes
    else:
        num_classes = None
    metric_task = Metric.ACCURACY if dataset.task_type.is_classification else Metric.RMSE
    
    model = model_cls(task_type=dataset.task_type, num_classes=num_classes, metric=metric_task)

    import time
    start_time = time.time()
    model.tune(tf_train=train_dataset.tensor_frame,
               tf_val=val_dataset.tensor_frame, num_trials=args.num_trials)
    val_pred = model.predict(tf_test=val_dataset.tensor_frame)
    val_metric = model.compute_metric(val_dataset.tensor_frame.y, val_pred)
    test_pred = model.predict(tf_test=test_dataset.tensor_frame)
    test_metric = model.compute_metric(test_dataset.tensor_frame.y, test_pred)
    if not args.regression:
        print(binary_classification_metrics(
            test_dataset.tensor_frame.y, (test_pred>0.5).int(), test_pred))
    end_time = time.time()
    result_dict = {
        'args': args.__dict__,
        'best_val_metric': val_metric,
        'best_test_metric': test_metric,
        'best_cfg': model.params,
        'total_time': end_time - start_time,
    }
    print(result_dict)
    # Save results
    os.makedirs(args.results_folder, exist_ok=True)
    file_path_wo_ending = os.path.join(args.results_folder, f"{args.model_type}_{data_path_wo_file_ending}_results")
    torch.save(result_dict, file_path_wo_ending + "_args.pt")
    model.save(file_path_wo_ending+ ".pt")

if __name__ == '__main__':
    main()
