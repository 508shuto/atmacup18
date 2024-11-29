import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class MLPDataset(Dataset):
    def __init__(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        transforms=None,
    ):
        self.scaler = StandardScaler()
        self.features = self._preprocess(features.values)
        self.targets = targets.values

        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        features = self.features[index]
        targets = self.targets[index]

        return (
            torch.tensor(features, dtype=torch.float),
            torch.tensor(targets, dtype=torch.float),
        )

    def _preprocess(self, X):
        X = self.scaler.fit_transform(X)
        return X
