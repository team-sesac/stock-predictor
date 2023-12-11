import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from scaler import Scaler

class RecurrentDataLoader():
    
    def __init__(self, data, target: str, scaler: Scaler):
        data['Y'] = data.loc[:, [target]]
        data = data.drop(labels=[target], axis=1)
        self.time_series = data
        self.length_series = len(self.time_series)
        self.target = target
        self.scaler = scaler
    
    def make_dataset(self, train_size: float, train_batch_size: int, valid_batch_size: int, seq_length: int) -> tuple[DataLoader, torch.FloatTensor, torch.FloatTensor]:
        train_size = int(self.length_series * train_size)
        train_set = self.time_series[0:train_size]
        test_set = self.time_series[train_size-seq_length:]
        
        # scaling
        self.scaler.fit_x(train_set.iloc[:, :-1])
        train_set.iloc[:, :-1] = self.scaler.transform_x(train_set.iloc[:, :-1])
        test_set.iloc[:, :-1] = self.scaler.transform_x(test_set.iloc[:, :-1])
        self.scaler.fit_y(train_set.iloc[:, [-1]])
        train_set.iloc[:, [-1]] = self.scaler.transform_y(train_set.iloc[:, [-1]])
        test_set.iloc[:, [-1]] = self.scaler.transform_y(test_set.iloc[:, [-1]])
        
        train_x, train_y = self._build_dataset(time_series=train_set, seq_length=seq_length)
        test_x, test_y = self._build_dataset(time_series=test_set, seq_length=seq_length)
        
        train_dataset = TensorDataset(torch.FloatTensor(train_x), torch.FloatTensor(train_y))
        valid_dataset = TensorDataset(torch.FloatTensor(test_x), torch.FloatTensor(test_y))
        
        train_data_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=False, drop_last=True)
        valid_data_loader = DataLoader(dataset=valid_dataset, batch_size=valid_batch_size, shuffle=False) # drop_last=True
        return (train_data_loader, valid_data_loader), (train_dataset, valid_dataset)
    
    def _build_dataset(self, time_series, seq_length: int) -> tuple[np.ndarray, np.ndarray]:
        data_x, data_y = [], []
        time_series = time_series.reset_index(drop=True)
        for i in range(len(time_series)-seq_length):
            _x = time_series.iloc[i:i+seq_length, :]
            _y = time_series.iloc[i+seq_length, [-1]]
            data_x.append(_x)
            data_y.append(_y)
        return np.array(data_x), np.array(data_y)
    
if __name__ == '__main__':
    pass