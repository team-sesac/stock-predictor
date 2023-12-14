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
    
    def make_dataset(self, train_size: float, train_batch_size: int, valid_batch_size: int) -> tuple[DataLoader, torch.FloatTensor, torch.FloatTensor]:

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
        
        # # train_size = int(self.length_series * train_size)
        # train_set = self.data
        # test_set = self.time_series_test
        
        # # scaling
        # self.scaler.fit_x(train_set.iloc[:, :-1])
        # train_set.iloc[:, :-1] = self.scaler.transform_x(train_set.iloc[:, :-1])
        # test_set.iloc[:, :-1] = self.scaler.transform_x(test_set.iloc[:, :-1])
        # self.scaler.fit_y(train_set.iloc[:, [-1]])
        # train_set.iloc[:, [-1]] = self.scaler.transform_y(train_set.iloc[:, [-1]])
        # test_set.iloc[:, [-1]] = self.scaler.transform_y(test_set.iloc[:, [-1]])
        
        # train_x, train_y = train_set.iloc[:, :-1], train_set.iloc[:, [-1]]
        # test_x, test_y = test_set.iloc[:, :-1], test_set.iloc[:, [-1]]
        
        train_dataset = TensorDataset(torch.FloatTensor(np.array(train_x)), torch.FloatTensor(np.array(train_y)))
        valid_dataset = TensorDataset(torch.FloatTensor(np.array(test_x)), torch.FloatTensor(np.array(test_y)))
        
        train_data_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=False)
        valid_data_loader = DataLoader(dataset=valid_dataset, batch_size=valid_batch_size, shuffle=False) # drop_last=True
        return (train_data_loader, valid_data_loader), (train_dataset, valid_dataset)
    
if __name__ == '__main__':
    
    data = pd.read_csv("data/temp/kaggle_train.csv")
    # data = data[data["date_id"] < 478]
    data = data.fillna(method="ffill").fillna(0)
    data = data.drop(labels=["row_id"], axis=1)
    # data['Y'] = data.loc[:, ["target"]]
    # data = data.drop(labels=["target"], axis=1)
    
    # grouped_data = data.groupby("stock_id")
    # train_x_array, train_y_array = [], []
    # test_x_array, test_y_array = [], []
    # for idx, data in grouped_data:
        
    #     train_size = int(len(data) * 0.9)
    #     train_set = data[:train_size]
    #     test_set = data[train_size:]
    #     train_x_array.append(train_set.iloc[:, :-1])
    #     train_y_array.append(train_set.iloc[:, [-1]])
    #     test_x_array.append(test_set.iloc[:, :-1])
    #     test_y_array.append(test_set.iloc[:, [-1]])
    
    # train_x, train_y = pd.concat(train_x_array, axis=0, ignore_index=True), pd.concat(train_y_array, axis=0, ignore_index=True)
    # test_x, test_y = pd.concat(test_x_array, axis=0, ignore_index=True), pd.concat(test_y_array, axis=0, ignore_index=True)
    
    # print(len(train_x))
    # print(len(test_x))
    
    scaler = Scaler(scale_type="min_max")
    data_loader = FeedForwardDataLoader(data=data, target="target", scaler=scaler)
    dataloaders, datasets = data_loader.make_dataset(train_size=0.9, train_batch_size=1000, valid_batch_size=10000)

    train_data_sets, valid_data_sets = datasets
    
    train_x, train_y = train_data_sets.tensors
    valid_x, valid_y = valid_data_sets.tensors
    
    print(len(train_x))
    print(len(valid_x))
    
    train_l, test_l = dataloaders
    print(len(train_l))
    print(len(test_l))
    
    