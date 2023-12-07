import torch
import pandas as pd
from dataloader import LSTMDataLoader
from model import LSTMNet
from trainer import LSTMTrainer
from scaler import Scaler
from hyperparameter import HyperParameter
from utils import *

seed = 42
torch.manual_seed(seed)

path = "data/"
filename = "005930_2023.csv"
data = pd.read_csv(path + filename)
num_features = len(data.columns)

# 데이터 파라미터
train_size = 0.8
train_batch_size = 20
valid_batch_size = 100
target = "Close"
scale_type = "min_max"

# 하이퍼 파라미터
hyper_parameter = HyperParameter(seq_length = 14,
                                    lr = 0.01,
                                    epochs = 100,
                                    num_layers = 1,
                                    drop_out = 0)

# 1. DataLodaer
scaler = Scaler(scale_type=scale_type)
dataloader = LSTMDataLoader(data=data, target=target, scaler=scaler)
dataloaders, datasets = dataloader.make_dataset(
    train_size=train_size,
    train_batch_size=train_batch_size,
    valid_batch_size=valid_batch_size,
    seq_length=hyper_parameter.get_seq_length()
)

# 2. Model
model = LSTMNet(input_size=num_features, hidden_size=num_features*2,
                        output_size=1, hyper_parameter=hyper_parameter)

# 3. Trainer
trainer = LSTMTrainer(model=model, scaler=scaler, 
                        data_loaders=dataloaders, datasets=datasets, 
                        hyper_parameter=hyper_parameter)
trainer.train()
trainer.visualization()
