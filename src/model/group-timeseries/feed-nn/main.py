import pandas as pd
from dataloader import FeedForwardDataLoader
from model import FeedForwardNN
from trainer import Trainer
from scaler import Scaler
from hyperparameter import HyperParameter
from initializr import *
from utils import *
from processor.default_processor import DefaultPreprocessor
from processor.explained_optiver_processor import ExplainedOptiverProcessor

set_torch_seed(seed=42)

try: 
    settings = load_setting(path="src/model/group-timeseries/feed-nn/train_settings.json")
except FileNotFoundError as e:
    settings = load_setting(path="./train_settings.json")
platform, base_path, result_path = init(settings)
set_data, set_hyper = settings["data"], settings["hyper"]
model_type, loss_type = settings["model"], settings["loss"]

data_origin = pd.read_csv(base_path + set_data["filename"])
preprocessor = DefaultPreprocessor()
df_train_x = preprocessor.execute_x(data_origin) # without target
df_train_x = reduce_mem_usage(df_train_x)
df_train_y = preprocessor.execute_y(data_origin, target=set_data["target"]) # without target
df_train_y = reduce_mem_usage(df_train_y)
feature_names = df_train_x.columns.values

# 하이퍼 파라미터
hyper_parameter = HyperParameter(lr=set_hyper["lr"],
                                    epochs=set_hyper["epochs"],
                                    hidden_units=set_hyper["hidden_units"],
                                    embedding_dims=set_hyper["embedding_dims"],
                                    drop_outs=set_hyper["drop_outs"],
                                    train_batch_size=set_hyper["train_batch_size"],
                                    valid_batch_size=set_hyper["valid_batch_size"])

num_features = len(df_train_x.columns)

num_categorical_features = [ len(data_origin[col].unique()) for col in set_data["categorical_features"] ]
num_continuous_features = num_features - len(num_categorical_features)

# 1. DataLodaer
scaler = Scaler(scale_type=set_data["scale_type"])
dataloader = FeedForwardDataLoader(train=df_train_x, target=df_train_y, scaler=scaler)
dataloaders, datasets = dataloader.make_dataset(
    train_size=set_hyper["train_size"],
    train_batch_size=set_hyper["train_batch_size"],
    valid_batch_size=set_hyper["valid_batch_size"]
)

test_revealed = pd.read_csv(base_path + "revealed_targets.csv")
test_data = pd.read_csv(base_path + set_data["test_filename"])
test_data = test_data.drop(labels=["date_id", "time_id", "row_id", "currently_scored"], axis=1)
test_data[set_data["target"]] = test_revealed["revealed_target"]
test_data = test_data.ffill().fillna(0)
test_data.iloc[:, 1:-1] = scaler.transform_x(test_data.iloc[:, 1:-1])
test_data.iloc[:, [-1]] = scaler.transform_y(test_data.iloc[:, [-1]])
test_data = test_data.reset_index(drop=True)

def learn(model_type, loss_type, dataloaders, datasets, set_data, set_hyper, hyper_parameter):
    
    # 2. Model
    model = FeedForwardNN(num_continuous_features=num_continuous_features, num_categorical_features=num_categorical_features, hyper_parameter=hyper_parameter)

    # 3. Trainer
    trainer = Trainer(model=model, scaler=scaler, 
                            data_loaders=dataloaders, datasets=datasets, 
                            hyper_parameter=hyper_parameter, loss=loss_type, huber_beta=set_hyper["huber_beta"])
    trainer.train()
    trainer.save_result(platform=platform, model_type=model_type, loss_type=loss_type, learn_topic=set_data["learn_topic"], 
                        path=result_path,
                        description=set_data["description"], test_set=test_data, feature_names=feature_names)
    #trainer.visualization()
    
    if settings["is_infer"]:
        import optiver2023
        env = optiver2023.make_env()
        iter_test = env.iter_test()
        
        for (test, revealed_targets, sample_prediction) in iter_test:
            pred = model(test)
            sample_prediction['target'] = scaler.inverse_y(pred)
    
learn(model_type=model_type, loss_type=loss_type,
        dataloaders=dataloaders, datasets=datasets, set_data=set_data, 
        set_hyper=set_hyper, hyper_parameter=hyper_parameter)
    