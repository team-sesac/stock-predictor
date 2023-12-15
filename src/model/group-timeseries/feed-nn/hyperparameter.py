from utils import *

class HyperParameter():
    
    def __init__(self,
                    lr = 0.01,
                    epochs = 100,
                    hidden_units = [128, 128],
                    embedding_dims = [20],
                    drop_outs = [0]):
    
        self._lr = lr
        self._epochs = epochs
        self.hidden_units = hidden_units
        self.embedding_dims = embedding_dims
        self._drop_outs = drop_outs
        self._device = get_device()
        print(f"device - {self._device}")
        print(f"count - {torch.cuda.device_count()}")
        print(f"name - {torch.cuda.get_device_name(0)}")
        
    def get_lr(self):
        return self._lr
    
    def get_epochs(self):
        return self._epochs
    
    def get_hidden_units(self):
        return self.hidden_units
    
    def get_embedding_dims(self):
        return self.embedding_dims
    
    def get_drop_outs(self):
        return self._drop_outs
    
    def get_device(self):
        return self._device