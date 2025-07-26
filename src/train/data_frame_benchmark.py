# This script is adapted from PyG (PyTorch Geometric).
# Original source: https://github.com/pyg-team/pytorch-frame/benchmark/data_frame_benchmark.py
# 
# Modifications:
# - Normalize datasets before tuning
# - Saving models and data for later analysis
#
# The original license follows:
#
# Copyright (c) 2023 PyG Team <team@pyg.org>
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy...
from torch_frame import Metric

# Add this import near the top of the file
from torch.utils.tensorboard import SummaryWriter
import datetime
from torch_frame.data.stats import StatType, compute_col_stats

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

from src.utils.preprocessing import CatToOneHotTransform


# Constants and configuration
TRAIN_CONFIG_KEYS = ["batch_size", "gamma_rate", "base_lr"]
GBDT_MODELS = ["XGBoost", "CatBoost", "LightGBM"]
BASEDIR = str(Path(__file__).resolve().parent)

# Lookup table for datasets
dataset_lookup = {
    "binary_classification": {
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
    },
    "multiclass_classification": {
        "small": {
            # Added small section for multiclass classification
        },
        "medium": {
            0: "aloi",
            1: "helena",
            2: "jannis",
        },
        "large": {
            0: "forest_cover_type",
            1: "poker_hand",
            2: "covtype",
        },
    },
    "regression": {
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
        },
    },
}

# Reverse lookup for datasets
def get_dataset_specs(dataset_name):
    """
    Get the task type, scale, and index for a given dataset name.
    
    Args:
        dataset_name (str): The name of the dataset
        
    Returns:
        tuple: (task_type, scale, index) or None if not found
    """
    for task_type, scales in dataset_lookup.items():
        for scale, indices in scales.items():
            for idx, name in indices.items():
                if name == dataset_name:
                    return (task_type, scale, idx)
    return None

# Create argument parser
def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task_type', type=str, choices=[
            'binary_classification',
            'multiclass_classification',
            'regression',
        ], default='binary_classification')
    parser.add_argument('--scale', type=str, choices=['small', 'medium', 'large'],
                        default='small')
    parser.add_argument('--idx', type=int, default=0,
                        help='The index of the dataset within DataFrameBenchmark')
    parser.add_argument('--epochs', type=int, default=10)
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
    parser.add_argument('--data_folder', type=str, default='')
    return parser

parser = create_parser()

# Function to get dataset name
def get_dataset_name(classification_type, scale, index):
    return dataset_lookup.get(classification_type, {}).get(scale, {}).get(index, "Dataset not found")

def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# After the dataset splits and before model setup, add:
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

    transformed_col_stats = dict()
    transformed_df = pd.DataFrame(
            train_tf.feat_dict[stype.numerical].detach().cpu().numpy(),
            columns=train_tf.col_names_dict[stype.numerical],
        )
    for col in train_tf.col_names_dict[stype.numerical]:
        transformed_col_stats[col] = compute_col_stats(
            transformed_df[col], stype.numerical)
        
    return train_tf, val_tf, test_tf, transformed_col_stats

def normalize_target(train_y, val_y, test_y):
    """Normalize target values."""
    train_y_mean = torch.mean(train_y)
    train_y_std = torch.std(train_y)
    train_y = (train_y - train_y_mean) / train_y_std
    val_y = (val_y - train_y_mean) / train_y_std
    test_y = (test_y - train_y_mean) / train_y_std
    return train_y, val_y, test_y

def prepare_data_and_models(args):
    """Prepare data and initialize model configurations based on provided arguments."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    torch.manual_seed(args.seed)
    set_random_seeds(args.seed)
    
    # Prepare datasets
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
    os.makedirs(args.results_folder, exist_ok=True)
    dataset_name = get_dataset_name(args.task_type, args.scale, args.idx)
    print(f"Dataset: {dataset_name}")

    dataset = DataFrameBenchmark(root=path, 
                                 task_type=TaskType(args.task_type),
                                 scale=args.scale, 
                                 idx=args.idx,
                                )
    dataset.materialize()
    dataset = dataset.shuffle()
    train_dataset, val_dataset, test_dataset = dataset.split()

    train_tensor_frame = train_dataset.tensor_frame
    val_tensor_frame = val_dataset.tensor_frame
    test_tensor_frame = test_dataset.tensor_frame
    labels_tst = test_tensor_frame.y.to(device)

    print(f"Train: {len(train_tensor_frame)}, Val: {len(val_tensor_frame)}, "
          f"Test: {len(test_tensor_frame)}")
    
    if stype.categorical in train_tensor_frame.col_names_dict:
        unique_cat_per_col = (torch.unique(train_tensor_frame.feat_dict[stype.categorical], dim = 0).sum(axis=0) != 0)
        train_tensor_frame.feat_dict[stype.categorical] = train_tensor_frame.feat_dict[stype.categorical][:, unique_cat_per_col]
        val_tensor_frame.feat_dict[stype.categorical] = val_tensor_frame.feat_dict[stype.categorical][:, unique_cat_per_col]
        test_tensor_frame.feat_dict[stype.categorical] = test_tensor_frame.feat_dict[stype.categorical][:, unique_cat_per_col]
        col_names_dict_train = train_tensor_frame.col_names_dict[stype.categorical]
        cols_names_updated = [col for col, keep in zip(col_names_dict_train, unique_cat_per_col) if keep]
        train_tensor_frame.col_names_dict[stype.categorical] = cols_names_updated
        val_tensor_frame.col_names_dict[stype.categorical] = cols_names_updated
        test_tensor_frame.col_names_dict[stype.categorical] = cols_names_updated

        print(train_tensor_frame, "before transform")
        categorical_transform = CatToOneHotTransform()
        categorical_transform.fit(train_tensor_frame,
                                train_dataset.col_stats)
        train_tensor_frame = categorical_transform(train_tensor_frame)
        print(train_tensor_frame, "after transform")
        val_tensor_frame = categorical_transform(val_tensor_frame)
        test_tensor_frame = categorical_transform(test_tensor_frame)
        col_stats = categorical_transform.transformed_stats
    else:
        col_stats = dataset.col_stats

    train_tensor_frame, val_tensor_frame, test_tensor_frame, col_stats = normalize_tensor_frame(
        train_tensor_frame, val_tensor_frame, test_tensor_frame
    )

    if args.task_type == 'regression':
        # Normalize target values
        train_tensor_frame.y, val_tensor_frame.y, test_tensor_frame.y = normalize_target(
            train_tensor_frame.y, val_tensor_frame.y, test_tensor_frame.y
        )
        
    col_names_dict = train_tensor_frame.col_names_dict
    if not osp.exists(args.data_folder):
        os.makedirs(args.data_folder)
    print(f"save data under: {os.path.join(args.data_folder, f'{dataset_name}_normalized_data_col_names_dict.pt')}")
    torch.save(col_names_dict, 
            os.path.join(args.data_folder, f"{dataset_name}_normalized_data_col_names_dict.pt"))
    torch.save(col_stats, 
            os.path.join(args.data_folder,f"{dataset_name}_normalized_data_col_stats.pt"))


    normalized_data = {
        'train': train_tensor_frame,
        'val': val_tensor_frame,
        'test': test_tensor_frame
    }
    norm_path = os.path.join(args.data_folder, f'{args.model_type}_{dataset_name}_normalized_data.pt')
    torch.save(normalized_data, norm_path)

    # Initialize model classes based on model type
    if args.model_type in GBDT_MODELS:
        gbdt_cls_dict = {
            'XGBoost': XGBoost,
            'CatBoost': CatBoost,
            'LightGBM': LightGBM
        }
        model_cls = gbdt_cls_dict[args.model_type]
        return {
            'model_cls': model_cls,
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'test_dataset': test_dataset,
            'dataset_name': dataset_name,
            'device': device,
            'train_tensor_frame': train_tensor_frame,
            'val_tensor_frame': val_tensor_frame, 
            'test_tensor_frame': test_tensor_frame,
            'dataset': dataset,
            'labels_tst': labels_tst
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
        model_cls = None
        # col_stats = None
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
            
        elif args.model_type == 'MLP':
            model_search_space = {
                'channels': [64, 128, 256],
                'num_layers': [1, 2, 4, 8],
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
            
        elif args.model_type == 'ExcelFormer':
            from torch_frame.transforms import (
                CatToNumTransform,
                MutualInformationSort,
            )

            categorical_transform = CatToNumTransform()
            categorical_transform.fit(train_tensor_frame,
                                    train_dataset.col_stats)
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
                'batch_size': [256, 512],
                'base_lr': [0.001],
                'gamma_rate': [0.9, 0.95, 1.],
            }
            model_cls = ExcelFormer
            print("ExcelFormer is being optimized for the current dataset.")

        assert model_cls is not None
        assert col_stats is not None
        assert set(train_search_space.keys()) == set(TRAIN_CONFIG_KEYS)
        
        
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
            'dataset_name': dataset_name,
            'device': device,
            'train_tensor_frame': train_tensor_frame,
            'val_tensor_frame': val_tensor_frame,
            'test_tensor_frame': test_tensor_frame,
            'dataset': dataset
        }

def train(
    model: Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    device: torch.device,
    dataset: Any,
    out_channels: int,
    loss_fun: Module,
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
    device: torch.device,
    dataset: Any,
    metric_computer: Module,
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
    final_training: bool = False,
) -> tuple[torch.nn.Module, float, float]:
    """Train and evaluate a model with the given configuration."""
    if config is None:
        # If called directly without config, parse args and prepare data
        args = parser().parse_args()
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
    dataset_name = config['dataset_name']
    
    # Set up TensorBoard logging
    log_dir = osp.join(BASEDIR, f"tensorboard_logs/{args.model_type}/{dataset_name}/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")
    
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

    # Log model hyperparameters
    writer.add_text('Model Configuration', str(model_cfg))
    writer.add_text('Training Configuration', str(train_cfg))
    
    if higher_is_better:
        best_val_metric = 0
    else:
        best_val_metric = math.inf

    if final_training:
        epochs = np.max(args.epochs, 40)
    else:
        epochs = args.epochs
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer, epoch, device, dataset, out_channels, loss_fun)
        val_metric = test(model, val_loader, device, dataset, metric_computer)
        
        # Log metrics to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Metric/validation', val_metric, epoch)
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        if higher_is_better:
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                best_test_metric = test(model, test_loader, device, dataset, metric_computer)
                # Log best test metric
                writer.add_scalar('Metric/test_best', best_test_metric, epoch)
                # save new best model
                best_model_state_dict = model.state_dict()
                norm_path = os.path.join(args.results_folder, f'{args.model_type}_{dataset_name}_{f"{args.complexity_model}_" if args.complexity_model != "optimize" else ""}best_model.pt')
                torch.save(model.state_dict(), norm_path)
        else:
            if val_metric < best_val_metric:
                best_val_metric = val_metric
                best_test_metric = test(model, test_loader, device, dataset, metric_computer)
                # Log best test metric
                writer.add_scalar('Metric/test_best', best_test_metric, epoch)
                best_model_state_dict = model.state_dict()
                norm_path = os.path.join(args.results_folder, f'{args.model_type}_{dataset_name}_{f"{args.complexity_model}_" if args.complexity_model != "optimize" else ""}best_model.pt')
                torch.save(model.state_dict(), norm_path)
        lr_scheduler.step()
        print(f'Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}, Val: {val_metric:.4f}')

        if trial is not None:
            trial.report(val_metric, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
    
    # Final evaluation on test set
    final_test_metric = test(model, test_loader, device, dataset, metric_computer)
    writer.add_scalar('Metric/test_final', final_test_metric, epochs)
    
    # Log hyperparameters and final metrics together
    hparam_dict = {**model_cfg, **train_cfg}
    metric_dict = {
        'hparam/best_val_metric': best_val_metric,
        'hparam/best_test_metric': best_test_metric,
        'hparam/final_test_metric': final_test_metric
    }
    writer.add_hparams(hparam_dict, metric_dict)
    
    # Close the writer
    writer.close()

    print(f'Best val: {best_val_metric:.4f}, Best test: {best_test_metric:.4f}')
    return best_model_state_dict, best_val_metric, best_test_metric

def main_deep_models(args=None):
    """Execute deep learning model training and evaluation."""
    if args is None:
        args = parser.parse_args()
        
    # Get prepared data and model configurations
    config = prepare_data_and_models(args)
    
    # Extract needed variables from config
    model_search_space = config['model_search_space']
    train_search_space = config['train_search_space']
    higher_is_better = config['higher_is_better']
    dataset_name = config['dataset_name']
    
    # Hyper-parameter optimization with Optuna
    print("Hyper-parameter search via Optuna")
    start_time = time.time()
    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(),
        direction="maximize" if higher_is_better else "minimize",
    )
    
    # Create a wrapper for the objective function
    def objective_wrapper(trial):
        model_cfg = {}
        for name, search_list in model_search_space.items():
            model_cfg[name] = trial.suggest_categorical(name, search_list)
        train_cfg = {}
        for name, search_list in train_search_space.items():
            train_cfg[name] = trial.suggest_categorical(name, search_list)

        best_model_state_dict, best_val_metric, _ = train_and_eval_with_cfg(
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            trial=trial,
            args=args,
            config=config
        )
        return best_val_metric
    
    # Optimize with the wrapped objective function
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
            best_model_cfg, best_train_cfg, args=args,config=config)
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
        'search_time': search_time,
        'final_model_time': final_model_time,
        'total_time': search_time + final_model_time,
    }
    print(result_dict)
    # Save results
    
    os.makedirs(args.results_folder, exist_ok=True)
    if args.regression:
        torch.save({'model_state_dict': best_model_state_dict, **result_dict},
                    os.path.join(args.results_folder, f'{args.model_type}_normalized_regression_{dataset_name}_{f"{args.complexity_model}_" if args.complexity_model != "optimize" else ""}results.pt'))
    else:
        torch.save({'model_state_dict': best_model_state_dict, **result_dict},
                    os.path.join(args.results_folder, f'{args.model_type}_normalized_binary_{dataset_name}_{f"{args.complexity_model}_" if args.complexity_model != "optimize" else ""}results.pt'))


def main_gbdt(args=None):
    """Execute GBDT model training and evaluation."""
    if args is None:
        args = parser.parse_args()
    
    # Get prepared data and model configurations
    config = prepare_data_and_models(args)
    
    # Extract needed variables from config
    model_cls = config['model_cls']
    train_tensor_frame = config['train_tensor_frame']
    val_tensor_frame = config['val_tensor_frame']
    test_tensor_frame = config['test_tensor_frame']
    dataset = config['dataset']
    dataset_name = config['dataset_name']
    
    if dataset.task_type.is_classification:
        num_classes = dataset.num_classes
    else:
        num_classes = None

    metric_task = Metric.ACCURACY if dataset.task_type.is_classification else Metric.RMSE
    model = model_cls(task_type=dataset.task_type, num_classes=num_classes, metric=metric_task)
    
    import time
    start_time = time.time()
    model.tune(tf_train=train_tensor_frame,
               tf_val=val_tensor_frame, 
               num_trials=args.num_trials,)
    val_pred = model.predict(tf_test=val_tensor_frame)
    val_metric = model.compute_metric(val_tensor_frame.y, val_pred)
    test_pred = model.predict(tf_test=test_tensor_frame)
    test_metric = model.compute_metric(test_tensor_frame.y, test_pred)
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
    if args.regression:
        torch.save(result_dict, os.path.join(args.results_folder, f'{args.model_type}_normalized_regression_{dataset_name}_results.pt'))
        model.save(os.path.join(args.results_folder, f'{args.model_type}_normalized_regression_{dataset_name}_results.pt'))
    else:
        torch.save(result_dict, os.path.join(args.results_folder, f'{args.model_type}_normalized_binary_{dataset_name}_results.pt'))
    
        model.save(os.path.join(args.results_folder, f'{args.model_type}_normalized_binary_{dataset_name}_results.pt'))


def main():
    """Main function to execute the script."""
    parser = create_parser()
    args = parser.parse_args()
    print(args)
    
    if args.model_type in GBDT_MODELS:
        main_gbdt(args)
    else:
        main_deep_models(args)


if __name__ == '__main__':
    main()

