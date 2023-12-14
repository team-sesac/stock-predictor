import pandas as pd
from dataloader import FeedForwardDataLoader
from model import FeedForwardNN
from trainer import Trainer
from scaler import Scaler
from hyperparameter import HyperParameter
from initializr import *
from utils import *
import numpy as np
from processor.preprocessor import DefaultPreprocessor

set_torch_seed(seed=42)

settings = load_setting(path="src/model/group-timeseries/feed-nn/train_settings.json")
init(settings)
set_data, set_hyper = settings["data"], settings["hyper"]
models, losses = settings["model"], settings["loss"]
models, losses = models if is_iter(models) else [models], losses if is_iter(losses) else [losses]

data_origin = pd.read_csv(set_data["path"] + set_data["filename"])
preprocessor = DefaultPreprocessor()
data_origin_ = preprocessor.execute(data_origin)
data_origin_ = data_origin_.ffill().fillna(0)
data_origin_ = data_origin_.drop(labels=set_data["remove_features"], axis=1)

date_flag = 478
data = data_origin_[data_origin_["date_id"] < date_flag]

# 하이퍼 파라미터
hyper_parameter = HyperParameter(lr=set_hyper["lr"],
                                    epochs=set_hyper["epochs"],
                                    hidden_units=set_hyper["hidden_units"],
                                    embedding_dims=set_hyper["embedding_dims"],
                                    drop_outs=set_hyper["drop_outs"])

num_features = len(data.columns) - 1

num_categorical_features = [ len(data_origin[col].unique()) for col in set_data["categorical_features"] ]
num_continuous_features = num_features - len(num_categorical_features)

# 1. DataLodaer
scaler = Scaler(scale_type=set_data["scale_type"])
dataloader = FeedForwardDataLoader(data=data, target=set_data["target"], scaler=scaler)
dataloaders, datasets = dataloader.make_dataset(
    train_size=set_hyper["train_size"],
    train_batch_size=set_hyper["train_batch_size"],
    valid_batch_size=set_hyper["valid_batch_size"]
)

test_data = data_origin_[data_origin_["date_id"] >= date_flag]
test_data['Y'] = test_data.loc[:, [set_data["target"]]]
test_data = test_data.drop(labels=[set_data["target"]], axis=1)
test_data.iloc[:, 1:-1] = scaler.transform_x(test_data.iloc[:, 1:-1])
test_data.iloc[:, [-1]] = scaler.transform_y(test_data.iloc[:, [-1]])
test_data = test_data.reset_index(drop=True)

def learn(model_type, loss_type, dataloaders, datasets, set_data, set_hyper, hyper_parameter):
    
    # 2. Model
    model = FeedForwardNN(num_continuous_features=num_continuous_features, num_categorical_features=num_categorical_features, hyper_parameter=hyper_parameter)

    # 3. Trainer
    trainer = Trainer(model=model, scaler=scaler, 
                            data_loaders=dataloaders, datasets=datasets, 
                            hyper_parameter=hyper_parameter, loss=loss, huber_beta=set_hyper["huber_beta"])
    trainer.train()
    trainer.save_result(model_type=model_type, loss_type=loss_type, learn_topic=set_data["learn_topic"], 
                        path=set_data["result_path"],
                        description=set_data["description"], test_set=test_data)
    #trainer.visualization()
    
for model in models:
    for loss in losses:
        learn(model_type=model, loss_type=loss,
                dataloaders=dataloaders, datasets=datasets, set_data=set_data, 
                set_hyper=set_hyper, hyper_parameter=hyper_parameter)
        

