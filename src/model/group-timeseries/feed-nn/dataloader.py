import torch
from torch.utils.data import TensorDataset, DataLoader
from scaler import Scaler
import numpy as np
import pandas as pd

class FeedForwardDataLoader():
    
    def __init__(self, data, target: str, scaler: Scaler):
        # 그룹핑 후 일정 비율로 나눈 후 각각 합치기
        # Trainer에 최종 비교 함수도 구현(kaggle_test.csv 예측 비교 후 MAE 평가)
        data['Y'] = data.loc[:, [target]]
        data = data.drop(labels=[target], axis=1)
        self.data = data
        self.length_series = len(self.data)
        self.target = target
        self.scaler = scaler
    
    def make_dataset(self, train_size: float, train_batch_size: int, valid_batch_size: int):

        self.scaler.fit_x(self.data.iloc[:, 1:-1])
        self.scaler.fit_y(self.data.iloc[:, [-1]])
        self.data.iloc[:, 1:-1] = self.scaler.transform_x(self.data.iloc[:, 1:-1])
        self.data.iloc[:, [-1]] = self.scaler.transform_y(self.data.iloc[:, [-1]])

        grouped_data = self.data.groupby("stock_id")
        train_x_array, train_y_array = [], []
        test_x_array, test_y_array = [], []
        for _, data in grouped_data:
            train_size = int(len(data) * 0.9)
            train_set = data[:train_size]
            test_set = data[train_size:]
            train_x_array.append(train_set.iloc[:, :-1])
            train_y_array.append(train_set.iloc[:, [-1]])
            test_x_array.append(test_set.iloc[:, :-1])
            test_y_array.append(test_set.iloc[:, [-1]])
        
        train_x, train_y = pd.concat(train_x_array, axis=0, ignore_index=True), pd.concat(train_y_array, axis=0, ignore_index=True)
        test_x, test_y = pd.concat(test_x_array, axis=0, ignore_index=True), pd.concat(test_y_array, axis=0, ignore_index=True)
        
        train_dataset = TensorDataset(torch.FloatTensor(np.array(train_x)), torch.FloatTensor(np.array(train_y)))
        valid_dataset = TensorDataset(torch.FloatTensor(np.array(test_x)), torch.FloatTensor(np.array(test_y)))
        
        train_data_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=False)
        valid_data_loader = DataLoader(dataset=valid_dataset, batch_size=valid_batch_size, shuffle=False) # drop_last=True
        return (train_data_loader, valid_data_loader), (train_dataset, valid_dataset)
