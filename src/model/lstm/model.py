import torch
import torch.nn as nn

class LSTMNet(nn.Module):
    
    def __init__(self, input_size: int, 
                hidden_size: int,
                seq_length: int, 
                output_size: int,
                layers: int):
        
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.output_size = output_size
        self.layers = layers 
        
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size,
                            num_layers=layers,
                            dropout=0,
                            batch_first=True)
        self.fc = nn.Linear(in_features=hidden_size, out_features=output_size, bais=True)
        
    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.layers, self.seq_length, self.hidden_size),
            torch.zeros(self.layers, self.seq_length, self.hidden_size)
        )
        
    def forward(self, x):
        x, _status = self.lstm(x)
        x = self.fc(x[:, -1])
        return x

