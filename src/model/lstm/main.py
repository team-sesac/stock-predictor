import pandas as pd
from dataloader import LSTMDataLoader
from model import LSTMNet
from trainer import LSTMTrainer
from utils import *
from scaler import Scaler
import torch

seed = 42
torch.manual_seed(seed)

path = "data/"
filename = "005930_2023.csv"
data = pd.read_csv(path + filename)
num_features = len(data.columns)

# 데이터 파라미터
train_size = 0.8
train_batch_size = 100
valid_batch_size = 100

# 하이퍼 파라미터
SEQ_LENGTH = 7
DEVICE = get_device()
LR = 0.01
EPOCHS = 100
NUM_LAYERS = 2
DROP_OUT = 0.1

# 1. DataLodaer
scaler = Scaler(scale_type="standard")
dataloader = LSTMDataLoader(data=data, target="Close", scaler=scaler)
dataloaders, datasets = dataloader.make_dataset(
    train_size=train_size,
    train_batch_size=train_batch_size,
    valid_batch_size=valid_batch_size,
    seq_length=SEQ_LENGTH
)

# 2. Model
model = LSTMNet(input_size=num_features, hidden_size=num_features*2,
                        seq_length=SEQ_LENGTH, output_size=1,
                        layers=NUM_LAYERS, drop_out=DROP_OUT)

# 3. Trainer
trainer = LSTMTrainer(model=model, scaler=scaler, 
                        data_loaders=dataloaders, datasets=datasets, 
                        lr=LR, num_epochs=EPOCHS, device=DEVICE)
trainer.train()
trainer.performance_eval_metrics()
trainer.visualization()
