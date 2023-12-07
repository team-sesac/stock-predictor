import torch
import json
from datetime import datetime
import os

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
    
def load_setting(path):
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)
    
def set_torch_seed(seed):
    torch.manual_seed(seed)
    
def get_current_time() -> str:
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H•%M•%S")
    return formatted_time

def mkdir(dir):
    os_dir = os.path.join(dir)
    if not os.path.exists(os_dir):
        os.makedirs(os_dir)
        
def write_text(path, text):
    with open(path, 'w') as file:
        file.write(text)