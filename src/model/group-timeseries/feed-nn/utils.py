import torch
import json
from datetime import datetime
import os
from collections.abc import Iterable
import numpy as np
import pandas as pd

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    # elif torch.backends.mps.is_available():
    #     return torch.device("mps")
    else:
        return torch.device("cpu")
    
def load_setting(path):
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)
    
def set_torch_seed(seed):
    torch.manual_seed(seed)
    
def get_current_time() -> str:
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H.%M.%S")
    return formatted_time

def mkdir(dir):
    os_dir = os.path.join(dir)
    if not os.path.exists(os_dir):
        os.makedirs(os_dir)
        
def write_text(path, text):
    with open(path, 'w') as file:
        file.write(text)

def is_iter(obj):
    return not isinstance(obj, str) and isinstance(obj, Iterable)

def reduce_mem_usage(df, verbose=0):
    if isinstance(df, pd.Series):
        df = df.to_frame()
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float32)
    if verbose:
        print(f"Memory usage of dataframe is {start_mem:.2f} MB")
        end_mem = df.memory_usage().sum() / 1024**2
        print(f"Memory usage after optimization is: {end_mem:.2f} MB")
        decrease = 100 * (start_mem - end_mem) / start_mem
        print(f"Decreased by {decrease:.2f}%")
    return df