import torch
import torch.nn as nn
from hyperparameter import HyperParameter

class RecurrentNN(nn.Module):
    
    def __init__(self, 
                    model,
                    input_size: int, 
                    hidden_size: int,
                    output_size: int,
                    hyper_parameter: HyperParameter):
        
        super(RecurrentNN, self).__init__()
        self.model = model.upper()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = hyper_parameter.get_seq_length()
        self.output_size = output_size
        self.layers = hyper_parameter.get_num_layers()
        self.drop_out = hyper_parameter.get_drop_out()
        
        self.net = self._get_model()
        self.fc = nn.Linear(in_features=hidden_size, out_features=output_size, bias=True)
    
    def _get_model(self):
        if self.model == "LSTM":
            return nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.layers,
                            dropout=self.drop_out,
                            batch_first=True)
        elif self.model == "RNN":
            return nn.RNN(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.layers,
                            dropout=self.drop_out,
                            batch_first=True)
        elif self.model == "GRU":
            return nn.GRU(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.layers,
                            dropout=self.drop_out,
                            batch_first=True)
        else:
            raise ValueError("model은 'rnn', 'lstm', 'gru' 중 하나여야합니다.")
        
    def reset_hidden_state(self):
        if self.model == "lstm":
            self.hidden = (
                torch.zeros(self.layers, self.seq_length, self.hidden_size),
                torch.zeros(self.layers, self.seq_length, self.hidden_size)
            )
        else:
            self.hidden = torch.zeros(self.layers, self.seq_length, self.hidden_size)
        
    def forward(self, x):
        x, _status = self.net(x)
        x = self.fc(x[:, -1])
        return x

