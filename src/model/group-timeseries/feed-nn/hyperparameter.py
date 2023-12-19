from utils import *

class HyperParameter():
    
    def __init__(self,
                    lr,
                    epochs,
                    hidden_units,
                    embedding_dims,
                    drop_outs,
                    train_batch_size,
                    valid_batch_size,
                    sample_size):
    
        self._lr = lr
        self._epochs = epochs
        self.hidden_units = hidden_units
        self.embedding_dims = embedding_dims
        self._drop_outs = drop_outs
        self._device = get_device()
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.sample_size = sample_size
        print(f"device - {self._device}")
        print(f"count - {torch.cuda.device_count()}")
        if torch.cuda.device_count() > 0:
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
    
    def get_train_batch_size(self):
        return self.train_batch_size
    
    def get_valid_batch_size(self):
        return self.valid_batch_size
    
    def get_sample_size(self):
        return self.sample_size