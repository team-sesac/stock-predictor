import torch
from torch.utils.data import TensorDataset, DataLoader
from scaler import Scaler
import numpy as np
import pandas as pd

class FeedForwardDataLoader():
    
    def __init__(self, train, target, scaler: Scaler):

        self.train = train
        self.length_series = len(self.train)
        self.target = target
        self.scaler = scaler
    
    def make_dataset(self, train_size: float, train_batch_size: int, valid_batch_size: int):

        self.scaler.fit_x(self.train.iloc[:, 1:])
        self.scaler.fit_y(self.target)
        self.train.iloc[:, 1:] = self.scaler.transform_x(self.train.iloc[:, 1:])
        self.target.iloc[:] = self.scaler.transform_y(self.target)

        concat = pd.concat([self.train, self.target], axis=1)
        
        grouped_data = concat.groupby("stock_id")
        train_x_array, train_y_array = [], []
        test_x_array, test_y_array = [], []
        for _, data in grouped_data:
            train_size = int(len(data) * 0.9)
            train = data[:train_size]
            test = data[train_size:]
            train_x_array.append(train.iloc[:, :-1])
            train_y_array.append(train.iloc[:, [-1]])
            test_x_array.append(test.iloc[:, :-1])
            test_y_array.append(test.iloc[:, [-1]])
        
        train_x, train_y = pd.concat(train_x_array, axis=0, ignore_index=True), pd.concat(train_y_array, axis=0, ignore_index=True)
        test_x, test_y = pd.concat(test_x_array, axis=0, ignore_index=True), pd.concat(test_y_array, axis=0, ignore_index=True)

        train_dataset = TensorDataset(torch.FloatTensor(np.array(train_x)), torch.FloatTensor(np.array(train_y)))
        valid_dataset = TensorDataset(torch.FloatTensor(np.array(test_x)), torch.FloatTensor(np.array(test_y)))
        
        train_data_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=False)
        valid_data_loader = DataLoader(dataset=valid_dataset, batch_size=valid_batch_size, shuffle=False)
        return (train_data_loader, valid_data_loader), (train_dataset, valid_dataset)
    