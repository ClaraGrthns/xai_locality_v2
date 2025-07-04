import torch
from torch.utils.data import Dataset
import pandas as pd

class TabularDataset(Dataset):
    def __init__(self, features, target=None, train=False):
        self.features = features
        self.train = train
        if train:
            self.targets = target

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = self.features[idx]
        return (features, self.targets[idx]) if self.train else features
    

class TabularTensorDataset(Dataset):
    def __init__(self, features, target=None, train=False, transform=None):
        self.features = features
        self.train = train
        self.transform = transform
        if train:
            self.targets = target

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = self.features[idx]
        tf_features = self.transform(features)
        return features, tf_features, self.targets[idx] if self.train else features, tf_features