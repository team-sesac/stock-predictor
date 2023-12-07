from utils import *

class HyperParameter():
    
    def __init__(self,
                    seq_length = 7,
                    lr = 0.01,
                    epochs = 100,
                    num_layers = 1,
                    drop_out = 0):
    
        self._seq_length = seq_length
        self._lr = lr
        self._epochs = epochs
        self._num_layers = num_layers
        self._drop_out = drop_out
        self._device = get_device()
        
    def get_seq_length(self):
        return self._seq_length

    def get_lr(self):
        return self._lr
    
    def get_epochs(self):
        return self._epochs
    
    def get_num_layers(self):
        return self._num_layers
    
    def get_drop_out(self):
        return self._drop_out
    
    def get_device(self):
        return self._device