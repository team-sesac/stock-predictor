import pandas as pd
from dataloader import RecurrentDataLoader
from model import RecurrentNN
from trainer import Trainer
from scaler import Scaler
from hyperparameter import HyperParameter
from initializr import *
from utils import *

set_torch_seed(seed=42)

settings = load_setting(path="src/model/train_settings.json")
init(settings)
set_data, set_hyper = settings["data"], settings["hyper"]

data = pd.read_csv(set_data["path"] + set_data["filename"])
num_features = len(data.columns)

# 하이퍼 파라미터
hyper_parameter = HyperParameter(seq_length=set_hyper["seq_length"],
                                    lr=set_hyper["lr"],
                                    epochs=set_hyper["epochs"],
                                    num_layers=set_hyper["num_layers"],
                                    drop_out=set_hyper["drop_out"])

# 1. DataLodaer
scaler = Scaler(scale_type=set_data["scale_type"])
dataloader = RecurrentDataLoader(data=data, target=set_data["target"], scaler=scaler)
dataloaders, datasets = dataloader.make_dataset(
    train_size=set_hyper["train_size"],
    train_batch_size=set_hyper["train_batch_size"],
    valid_batch_size=set_hyper["valid_batch_size"],
    seq_length=hyper_parameter.get_seq_length()
)

# 2. Model
model = RecurrentNN(model=settings["model"], input_size=num_features, hidden_size=num_features*2,
                        output_size=1, hyper_parameter=hyper_parameter)

# 3. Trainer
trainer = Trainer(model=model, scaler=scaler, 
                        data_loaders=dataloaders, datasets=datasets, 
                        hyper_parameter=hyper_parameter, loss=settings["loss"], huber_beta=set_hyper["huber_beta"])
trainer.train()
trainer.save_result(model=settings["model"], loss=settings["loss"], learn_topic=set_data["learn_topic"], 
                    path=set_data["result_path"],
                    description=set_data["description"])
trainer.visualization()
