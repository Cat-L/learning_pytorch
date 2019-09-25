import torch
from torch.utils.data import Dataset
import  pandas as pd

class BulldozerDataset(Dataset):
    def __init__(self,csv_file):
        self.df=pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx].SalePrice


ds_demo=BulldozerDataset('median_benchmark.csv')
