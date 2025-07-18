import numpy as np
from torch_frame.typing import TaskType
from torch_frame.gbdt import LightGBM
import torch
from src.model.base import BaseModelHandler
from src.dataset.tab_data import TabularDataset
from torch.utils.data import DataLoader

class PTFrame_LightGBMHandler(BaseModelHandler):
    def load_model(self):
        train_tensor_frame = torch.load(self.data_path, weights_only=False)["train"]
        y = train_tensor_frame.y.numpy()
        num_classes = len(np.unique(y))
        if self.args.regression:
            task_type = TaskType.REGRESSION
        elif num_classes == 2:
            task_type = TaskType.BINARY_CLASSIFICATION
        else:
            task_type = TaskType.MULTICLASS_CLASSIFICATION
        model = LightGBM(task_type=task_type, num_classes=num_classes)
        model.load(self.model_path)
        return model

    def load_data(self):
        data = torch.load(self.data_path, weights_only=False)
        train_tensor_frame, val_tensor_frame, test_tensor_frame = data["train"], data["val"], data["test"]
        tst_feat, _ , _ = self.model._to_lightgbm_input(test_tensor_frame)
        val_feat, _, _ = self.model._to_lightgbm_input(val_tensor_frame)
        trn_feat, _, _ = self.model._to_lightgbm_input(train_tensor_frame)
        tst_feat = np.array(tst_feat)
        val_feat = np.array(val_feat)
        trn_feat = np.array(trn_feat)
        return trn_feat, val_feat, tst_feat

    def predict_fn(self, X):
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        pred = self.model.model.predict(X)
        if not self.args.regression and self.args.method != "tree_shap":
            pred = np.column_stack((1 - pred, pred))
        if self.args.method == "lime_captum":
            pred = torch.tensor(pred)
        return pred