from enum import Enum
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset


class BicepFeatures(Enum):
    height = 0
    weight = 1
    age = 2


class BicepCDataset(Dataset):
    """ BicepC dataset """

    def __init__(self, csv_file: str, transform: bool = None) -> None:
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied 
        """
        self.df = pd.read_csv(csv_file, index_col='ID')
        assert list(self.df.columns) == [
            "height", "weight", "age", "BicepC"], "column name not as expected"

        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.df.iloc[idx, [0, 1, 2]]
        y = self.df.iloc[idx, 3]

        x = np.array(x)
        y = np.array(y)

        if self.transform:
            # attributes combination
            x_BMI = self.df.iloc[idx, BicepFeatures.weight.value] / \
                self.df.iloc[idx, BicepFeatures.height.value]**2
            x_weight_per_age = self.df.iloc[idx,
                                            BicepFeatures.weight.value]/self.df.iloc[idx, BicepFeatures.age.value]

            x_transform = np.array([x_BMI, x_weight_per_age])

            x = np.concatenate([x, x_transform])

        return x, y
