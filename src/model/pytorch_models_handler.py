# Define the logistic regression model
import torch
import torch.nn as nn
import torch.optim as optim

import os
from pathlib import Path
import torch
from functools import partial
from src.model.base import BaseModelHandler
from src.utils.pytorch_frame_utils import (
    load_dataframes, 
)
import numpy as np

class LogReg(nn.Module):
    def __init__(self, input_size, output_size):
        super(LogReg, self).__init__()
        self.linear = nn.Linear(input_size, output_size)  # One output neuron
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return self.sigmoid(self.linear(x))
    
class LinReg(nn.Module):
    def __init__(self, input_size, output_size=1):
        super(LinReg, self).__init__()
        self.linear = nn.Linear(input_size, 1)  # One output neuron
    
    def forward(self, x):
        return self.linear(x)

class PytorchHandler(BaseModelHandler):
    def __init__(self, args, model_class):
        self.model_class = model_class
        super().__init__(args)

    def load_model(self):
        trn_feat, _, _ = load_dataframes(self.data_path)
        input_size = trn_feat.shape[1]
        model = self.model_class(input_size=input_size, output_size=1)
        model.load_state_dict(torch.load(self.model_path))
        model.eval()
        return model
    
    def predict_fn(self, X):
        if isinstance(X, np.ndarray):
            X = torch.Tensor(X)
        if X.dtype == torch.double:
            X = X.float()
        if self.args.method == "lime" or self.args.method == "lime_captum":
            with torch.no_grad():
                preds = self.model(X)
            if self.args.regression:
                return preds 
            elif self.args.method == "lime": 
                preds = preds.numpy()
                return np.column_stack((1 - preds, preds)) if preds.shape[1] == 1 else preds
            elif self.args.method == "lime_captum":
                return torch.cat((1 - preds, preds), dim=1) if preds.shape[1] == 1 else preds
        else:
            return self.model(X) # gradient methods with torch grad 
