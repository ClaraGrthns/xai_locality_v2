
# model/base.py
import numpy as np
from src.dataset.tab_data import TabularDataset
from src.utils.misc import get_path
import torch
from sklearn.datasets import make_classification
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from src.utils.pytorch_frame_utils import (
    tensorframe_to_tensor,
    load_dataframes, 
    ) 
from src.dataset.synthetic_data import create_synthetic_classification_data_sklearn, create_custom_synthetic_regression_data
from torch_frame.data import Dataset
import torch_frame
import pandas as pd
import os

class BaseModelHandler:
    def __init__(self, args):
        self.args = args
        if (args.data_path or args.data_folder) is not None:
            self.data_path = self.get_data_path()
            print(f"Loading data from: {self.data_path}")
        if (args.model_path or args.model_folder) is not None:
            self.model_path = self.get_model_path()
            print(f"Loading model from: {self.model_path}")
        self.model = self.load_model()

    def get_model_path(self):
        model_path = get_path(self.args.model_folder, self.args.model_path, self.args.setting, suffix=self.args.model_type + "_")
        return model_path
    
    def get_data_path(self):
        data_path = get_path(self.args.data_folder, self.args.data_path, self.args.setting)
        return data_path
    
    def _get_split_indices(self, whole_tst_feat):
        indices = np.random.permutation(len(whole_tst_feat))
        tst_indices, analysis_indices = np.split(indices, [self.args.max_test_points])
        print("using the following indices for testing: ", tst_indices)
        return tst_indices, analysis_indices
    

    def _transform_materialize_data(self, X, y):
        df = pd.DataFrame(X)
        df['y'] = y
        col_to_stype = {col: torch_frame.numerical for col in df.columns}
        col_to_stype['y'] = torch_frame.numerical if self.args.regression else torch_frame.categorical
        dataset = Dataset(df, col_to_stype=col_to_stype, target_col='y')
        dataset.materialize()
        tensorframe = dataset.tensor_frame
        return tensorframe_to_tensor(tensorframe)
    
    def get_col_indices_informative_features(self):
        args = self.args
        if args.regression:
            # For regression, we need to create synthetic data
            _, _, _, _, _, _, _, col_indices = create_custom_synthetic_regression_data(
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
                force_create=True
            )
        else:
            col_indices = np.arange(args.n_features)
        return col_indices
    
    def load_data_for_kNN(self):
        """
        Loads and processes data for k-Nearest Neighbors (kNN) analysis.

        This method loads data from a specified path, which can be either a PyTorch
        tensor file (with a .pt extension) or a NumPy file. It then processes the data
        to extract features and labels for training and testing.

        Returns:
            tuple: A tuple containing the following elements:
                - trn_feat (numpy.ndarray): Training features.
                - analysis_feat (numpy.ndarray): Features for analysis.
                - tst_feat (numpy.ndarray): Testing features.
                - y_trn (numpy.ndarray): Training labels.
                - analysis_y (numpy.ndarray): Labels for analysis.
                - tst_y (numpy.ndarray): Testing labels.
        """
        if self.data_path.endswith(".pt"):
            data = torch.load(self.data_path, weights_only=False)
            test_tensor_frame = data["test"]
            whole_tst_feat = tensorframe_to_tensor(test_tensor_frame).numpy()
            y_test = test_tensor_frame.y.numpy()
            trn_tensor_frame = data["train"]
            trn_feat = tensorframe_to_tensor(trn_tensor_frame).numpy()
            y_trn = trn_tensor_frame.y.numpy()

            if "val" in data:
                val_tensor_frame = data["val"]
                val_feat = tensorframe_to_tensor(val_tensor_frame).numpy()
                y_val = val_tensor_frame.y.numpy()
        else:
            data = np.load(self.data_path)
            whole_tst_feat = data['X_test']
            y_test = data['y_test']
            trn_feat = data['X_train']
            y_trn = data['y_train']
            if 'X_val' in data:
                val_feat = data['X_val']
                y_val = data['y_val']

        return trn_feat, whole_tst_feat, y_trn, y_test, val_feat, y_val

    def _check_for_dublicates(self, dataset):
        if isinstance(dataset, np.ndarray):
            unique_rows, unique_indices, counts = np.unique(dataset, axis=0, return_index=True, return_counts=True)
            duplicate_mask = counts > 1
            if np.any(duplicate_mask):
                duplicate_count = np.sum(counts[duplicate_mask] - 1)
                print(f"WARNING: {duplicate_count} duplicate samples found in test data and removed.")
                dataset = dataset[unique_indices]
        else:
            whole_tst_feat_np = dataset.cpu().numpy()
            unique_rows, unique_indices, counts = np.unique(whole_tst_feat_np, axis=0, return_index=True, return_counts=True)
            duplicate_mask = counts > 1
            if np.any(duplicate_mask):
                duplicate_count = np.sum(counts[duplicate_mask] - 1)
                print(f"WARNING: {duplicate_count} duplicate samples found in test data and removed.")
                dataset = dataset[torch.tensor(unique_indices)]
        return dataset
    

    def split_data_in_tst_analysis(self, whole_tst_feat, val_feat):
        # Check for duplicates in whole_tst_feat
        tst_indices, _ = self._get_split_indices(whole_tst_feat)
        analysis_feat = whole_tst_feat
        tst_feat = whole_tst_feat[tst_indices]
        if self.args.include_val:
            if isinstance(val_feat, np.ndarray):
                analysis_feat = np.concatenate([analysis_feat, val_feat], axis=0)
            else:
                analysis_feat = torch.cat([analysis_feat, val_feat], dim=0)
        print("Length of data set for analysis", len(analysis_feat))
        print("Length of test set", len(tst_feat)) 
        return tst_feat, analysis_feat, tst_indices



    def load_model(self):
        """Load model from path"""
        raise NotImplementedError
    
    def load_feature_vectors(self):
        return None

    def load_data(self):
        '''
        Load data from path
        to be overwritten, if does not work for the model
        '''
        trn_feat, val_feat, whole_tst_feat = load_dataframes(self.data_path)
        whole_tst_feat = self._check_for_dublicates(whole_tst_feat)
        return trn_feat, val_feat, whole_tst_feat
        
    def predict_fn(self, X):
        """Run predictions"""
        raise NotImplementedError

    def get_feature_names(self, trn_feat):
        """Return feature names (if applicable)"""
        return np.arange(trn_feat.shape[1])  # Default behavior
    
    def get_class_names(self):
        """Return class names (if applicable)"""
        return np.arange(2)
