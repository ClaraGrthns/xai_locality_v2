# model/xgboost_handler.py
import numpy as np
import torch
import xgboost
from torch_frame.typing import TaskType
from torch_frame.gbdt import XGBoost
from torch.utils.data import DataLoader
from src.model.base import BaseModelHandler
from src.dataset.tab_data import TabularDataset

class PTFrame_XGBoostHandler(BaseModelHandler):
    def load_model(self):
        """Load an XGBoost model using the TorchFrame wrapper."""
        train_tensor_frame = torch.load(self.data_path)["train"]
        y = train_tensor_frame.y.numpy()
        num_classes = len(np.unique(y))
        if num_classes == 2:
            task_type = TaskType.BINARY_CLASSIFICATION
        else:
            task_type = TaskType.MULTICLASS_CLASSIFICATION
        model = XGBoost(task_type=task_type, num_classes=num_classes)
        model.load(self.model_path)
        return model

    def load_data(self):
        """Load train, validation, and test datasets from a Torch tensor frame."""
        data = torch.load(self.data_path)
        train_tensor_frame, val_tensor_frame, test_tensor_frame = data["train"], data["val"], data["test"]

        tst_feat, _, _ = self.model._to_xgboost_input(test_tensor_frame)
        val_feat, _, _ = self.model._to_xgboost_input(val_tensor_frame)
        trn_feat, _, _ = self.model._to_xgboost_input(train_tensor_frame)
        tst_feat, analysis_feat, tst_dataset, analysis_dataset = self._split_data_in_tst_analysis(tst_feat,
                                                                                                val_feat,
                                                                                                trn_feat)
        return trn_feat, tst_feat, analysis_feat, tst_dataset, analysis_dataset
    
    def predict_fn(self, X):
        """Perform inference using the XGBoost model."""
        types = ["q"] * X.shape[1]
        if X.ndim == 1:
            X = X.reshape(1, -1)

        dummy_labels = np.zeros(X.shape[0])
        dtest = xgboost.DMatrix(X, label=dummy_labels, feature_types=types, enable_categorical=True)
        pred = self.model.model.predict(dtest)
        if self.args.regression:
            return pred
        if self.model.task_type == TaskType.BINARY_CLASSIFICATION and not self.args.method == "tree_shap":
            pred = np.column_stack((1 - pred, pred))
        return pred

    def get_feature_names(self, tensor_frame):
        """Extract feature names from the Torch tensor frame."""
        first_key = next(iter(tensor_frame.col_names_dict))
        return tensor_frame.col_names_dict[first_key]

