from __future__ import annotations

import copy
import logging
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F

from torch_frame import NAStrategy, TensorFrame, stype
from torch_frame.data.stats import StatType, compute_col_stats
from torch_frame.transforms import FittableBaseTransform

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms

def get_pil_transform(): 
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224)
    ])    
    return transf

def get_preprocess_transform():
    transf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]) 
    ])    
    return transf 

def grid_segments(image, n_segments = 150):
    """Creates a grid segmentation of an image.
    
    Args:
        image: numpy array of shape [H, W, C] or [H, W]
        n_segments: int, number of grid segments (will be squared to get total segments)
    
    Returns:
        segments: numpy array of shape [H, W] where each unique value represents a segment
    """

    if len(image.shape) == 3 and type(image)==np.ndarray:
        height, width, _ = image.shape
    elif len(image.shape) == 3 and type(image)==torch.Tensor:
        _, height, width = image.shape
        image = image.numpy()
    else:
        height, width = image.shape
        
    # Calculate grid size
    n_segments_side = int(np.sqrt(n_segments))
    h_step = height // n_segments_side
    w_step = width // n_segments_side
    
    # Create segment labels
    segments = np.zeros((height, width), dtype=np.int64)
    for i in range(n_segments_side):
        for j in range(n_segments_side):
            h_start = i * h_step
            h_end = (i + 1) * h_step if i < n_segments_side - 1 else height
            w_start = j * w_step
            w_end = (j + 1) * w_step if j < n_segments_side - 1 else width
            segments[h_start:h_end, w_start:w_end] = i * n_segments_side + j
            
    return segments

def grid_wrapper(n_segments):
    def wrapper(image):
        return grid_segments(image, n_segments)
    return wrapper




class CatToOneHotTransform(FittableBaseTransform):
    r"""A transform that one-hot encodes the categorical features of
    the :class:`TensorFrame` object, converting them into numerical features.
    Each categorical feature with k categories is transformed into k binary
    numerical features (one-hot encoded).
    """
    def _fit(
        self,
        tf_train: TensorFrame,
        col_stats: dict[str, dict[StatType, Any]],
    ):
        if stype.categorical not in tf_train.col_names_dict:
            logging.info(
                "The input TensorFrame does not contain any categorical "
                "columns. No fitting will be performed.")
            self._transformed_stats = col_stats
            return
            
        tensor = tf_train.feat_dict[stype.categorical]
        self.col_stats = col_stats
        self.categories_per_col = {}
        self.new_columns = []
        
        # Determine the number of categories for each categorical column
        for i, col_name in enumerate(tf_train.col_names_dict[stype.categorical]):
            # Get the count of each category from the stats
            counts = col_stats[col_name][StatType.COUNT][1]
            num_categories = len(counts)
            self.categories_per_col[col_name] = num_categories
            
            # Generate new column names for the one-hot encoded features
            self.new_columns.extend(
                [f"{col_name}_{cat_idx}" for cat_idx in range(num_categories)]
            )

        # Compute stats for the transformed columns
        transformed_col_stats = dict()
        
        # Keep existing numerical columns
        if stype.numerical in tf_train.col_names_dict:
            for col in tf_train.col_names_dict[stype.numerical]:
                transformed_col_stats[col] = copy.copy(col_stats[col])
        
        # Create dummy transformed tensor to compute stats
        # (actual transformation happens in _forward)
        total_one_hot_features = sum(self.categories_per_col.values())
        dummy_transformed = torch.zeros(
            (len(tensor), total_one_hot_features), 
            dtype=torch.float32
        )
        transformed_df = pd.DataFrame(
            dummy_transformed.cpu().numpy(),
            columns=self.new_columns
        )
        
        # Compute stats for each one-hot column (will be 0/1)
        for col in self.new_columns:
            transformed_col_stats[col] = compute_col_stats(
                transformed_df[col], stype.numerical)
        
        self._transformed_stats = transformed_col_stats

    def _forward(self, tf: TensorFrame) -> TensorFrame:
        if stype.categorical not in tf.col_names_dict:
            logging.info(
                "The input TensorFrame does not contain any categorical "
                "columns. The original TensorFrame will be returned.")
            return tf
            
        tensor = tf.feat_dict[stype.categorical]
        batch_size = tensor.size(0)
        
        # Initialize tensor for one-hot encoded features
        total_one_hot_features = sum(self.categories_per_col.values())
        one_hot_tensor = torch.zeros(
            (batch_size, total_one_hot_features),
            dtype=torch.float32,
            device=tf.device
        )
        
        # One-hot encode each categorical column
        current_idx = 0
        for i, col_name in enumerate(tf.col_names_dict[stype.categorical]):
            num_categories = self.categories_per_col[col_name]
            col_data = tensor[:, i]
            
            # Validate that no new categories are present
            max_cat = col_data.max()
            if max_cat >= num_categories:
                raise RuntimeError(
                    f"{col_name} contains new category {max_cat} not seen "
                    f"during fit stage.")
            
            # One-hot encode this column
            one_hot = F.one_hot(col_data.long(), num_classes=num_categories)
            one_hot_tensor[:, current_idx:current_idx + num_categories] = one_hot.float()
            current_idx += num_categories
        
        # Combine with existing numerical features if they exist
        if stype.numerical in tf.feat_dict:
            tf.feat_dict[stype.numerical] = torch.cat(
                [tf.feat_dict[stype.numerical], one_hot_tensor],
                dim=1
            )
            tf.col_names_dict[stype.numerical] = (
                tf.col_names_dict[stype.numerical] + self.new_columns
            )
        else:
            tf.feat_dict[stype.numerical] = one_hot_tensor
            tf.col_names_dict[stype.numerical] = self.new_columns
        
        # Remove the categorical features
        tf.col_names_dict.pop(stype.categorical)
        tf.feat_dict.pop(stype.categorical)

        return tf