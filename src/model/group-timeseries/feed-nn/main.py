import pandas as pd
import torch
from dataloader import FeedForwardDataLoader
from model import FeedForwardNN
from trainer import Trainer
from scaler import Scaler
from hyperparameter import HyperParameter
from initializr import *
from utils import *
from processor.default_processor import DefaultPreprocessor
from processor.explained_optiver_processor import ExplainedOptiverProcessor
from tqdm import tqdm

set_torch_seed(seed=42)

try: 
    settings = load_setting(path="src/model/group-timeseries/feed-nn/train_settings.json")
except FileNotFoundError as e:
    settings = load_setting(path="./train_settings.json")
platform, result_path, data_list = init(settings)
train_data_set, test_data_set, target_data_set = data_list
set_data, set_hyper = settings["data"], settings["hyper"]
model_type, loss_type = settings["model"], settings["loss"]

data_origin = pd.read_csv(train_data_set)
preprocessor = ExplainedOptiverProcessor()
df_train_x = preprocessor.execute_x(data=data_origin, target=set_data["target"]) # without target
df_train_x = reduce_mem_usage(df_train_x)
df_train_y = preprocessor.execute_y(data=data_origin, target=set_data["target"])
df_train_y = reduce_mem_usage(df_train_y)
feature_names = df_train_x.columns.values

# 하이퍼 파라미터
hyper_parameter = HyperParameter(lr=set_hyper["lr"],
                                    epochs=set_hyper["epochs"],
                                    hidden_units=set_hyper["hidden_units"],
                                    embedding_dims=set_hyper["embedding_dims"],
                                    drop_outs=set_hyper["drop_outs"],
                                    train_batch_size=set_hyper["train_batch_size"],
                                    valid_batch_size=set_hyper["valid_batch_size"],
                                    sample_size=set_hyper["sample_size"])

num_features = len(df_train_x.columns)

num_categorical_features = [ len(data_origin[col].unique()) for col in set_data["categorical_features"] ]

num_continuous_features = num_features - len(num_categorical_features)

scaler = Scaler(scale_type=set_data["scale_type"])
dataloader = FeedForwardDataLoader(train=df_train_x, target=df_train_y, scaler=scaler)
dataloaders, datasets = dataloader.make_dataset(
    sample_size=hyper_parameter.get_sample_size(),
    test_size=set_hyper["test_size"],
    train_batch_size=hyper_parameter.get_train_batch_size(),
    valid_batch_size=hyper_parameter.get_valid_batch_size()
)

test_revealed = pd.read_csv(target_data_set)
test_data = pd.read_csv(test_data_set)
test_data = test_data.drop(labels=["currently_scored"], axis=1)

df_test_x = preprocessor.execute_x(test_data)
df_test_x = reduce_mem_usage(df_test_x)

df_test_y = test_revealed["revealed_target"].to_frame()
df_test_y.columns = [set_data["target"]]
df_test_y = preprocessor.execute_y(df_test_y, target=set_data["target"])
df_test_y = reduce_mem_usage(df_test_y)

df_test_x = np.hstack([test_data.iloc[:, [0]], scaler.transform_x(df_test_x.iloc[:, 1:])])
df_test_y = scaler.transform_y(df_test_y.iloc[:])

test_set = np.hstack([df_test_x, df_test_y])

model = FeedForwardNN(num_continuous_features=num_continuous_features,
                            num_categorical_features=num_categorical_features,
                            hyper_parameter=hyper_parameter).to(hyper_parameter.get_device())

trainer = Trainer(model=model, scaler=scaler, 
                    data_loaders=dataloaders, datasets=datasets, 
                    hyper_parameter=hyper_parameter, loss=loss_type, huber_beta=set_hyper["huber_beta"])
trainer.train()
        
trainer.save_result(platform=platform, model_type=model_type, loss_type=loss_type, learn_topic=set_data["learn_topic"], 
                    path=result_path, description=set_data["description"], test_set=test_set, feature_names=feature_names,
                    processor_name=preprocessor.name())
#trainer.visualization()

if settings["is_infer"]:
    import optiver2023
    env = optiver2023.make_env()
    iter_test = env.iter_test()
    print(f"Submit inference")
    for (test, revealed_targets, sample_prediction) in tqdm(iter_test, desc="진행중"):
        X = preprocessor.execute_x(data=test)
        X = X.drop(labels=["currently_scored"], axis=1).astype(float)
        X = np.hstack([X.iloc[:, [0]], scaler.transform_x(X.iloc[:, 1:])])
        X = torch.FloatTensor(X).to(hyper_parameter.get_device())
        with torch.no_grad():
            pred = model(X[:, 1:], X[:, [0]])
            sample_prediction["target"] = scaler.inverse_y(pred.detach())
            env.predict(sample_prediction)