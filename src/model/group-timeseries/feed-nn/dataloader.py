import torch
from torch.utils.data import TensorDataset, DataLoader
from scaler import Scaler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class FeedForwardDataLoader():
    
    def __init__(self, train, target, scaler: Scaler):

        self.train = train
        self.length_series = len(self.train)
        self.target = target
        self.scaler = scaler
        self.seed = 42
    
    def make_dataset(self, sample_size: float, test_size: float, train_batch_size: int, valid_batch_size: int):
        
        self.scaler.fit_x(self.train.iloc[:, 1:])
        self.scaler.fit_y(self.target)
        self.train.iloc[:, 1:] = self.scaler.transform_x(self.train.iloc[:, 1:])
        self.target.iloc[:] = self.scaler.transform_y(self.target)

        concat = pd.concat([self.train, self.target], axis=1)
        grouped_data = concat.groupby("stock_id")
        train_x_array, train_y_array = [], []
        test_x_array, test_y_array = [], []
        for _, data in grouped_data:
            X = data.iloc[:, :-1].sample(frac=sample_size, random_state=self.seed)
            y = data.iloc[:, [-1]].sample(frac=sample_size, random_state=self.seed)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=self.seed, shuffle=True)
            
            train_x_array.append(X_train)
            train_y_array.append(y_train)
            test_x_array.append(X_test)
            test_y_array.append(y_test)
        
        train_x, train_y = pd.concat(train_x_array, axis=0, ignore_index=True), pd.concat(train_y_array, axis=0, ignore_index=True)
        test_x, test_y = pd.concat(test_x_array, axis=0, ignore_index=True), pd.concat(test_y_array, axis=0, ignore_index=True)
        train_dataset = TensorDataset(torch.FloatTensor(np.array(train_x)), torch.FloatTensor(np.array(train_y)))
        valid_dataset = TensorDataset(torch.FloatTensor(np.array(test_x)), torch.FloatTensor(np.array(test_y)))
        
        train_data_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=False)
        valid_data_loader = DataLoader(dataset=valid_dataset, batch_size=valid_batch_size, shuffle=False)
        return (train_data_loader, valid_data_loader), (train_dataset, valid_dataset)
    