import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class Trainer():
    
    def __init__(self, model, data_loaders, datasets, lr, num_epochs, device):
        train_data_loader, valid_data_loader = data_loaders
        train_data_sets, valid_data_sets = datasets
        self.model = model
        self.loss_fn = nn.MESLoss().to(device)
        self.optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
        self.epochs = num_epochs
        self.train_epoch_losses = [] # epoch마다 loss 저장
        self.valid_epoch_losses = []
        self.mae_scores = []
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.train_data_sets = train_data_sets
        self.valid_data_sets = valid_data_sets
        self.device = device
        
    def train(self, verbose=10, patience=10):

        for epoch in tqdm(range(self.epochs), desc="진행중"):
            train_avg_cost = 0
            train_total_batch = len(self.train_data_loader)
            
            for x_train, y_train in self.train_data_loader:

                # seq별 hidden state reset
                self.model.reset_hidden_state()
                
                # H(x) 계산
                outputs = self.model(x_train)
                    
                # cost 계산
                loss = self.loss_fn(outputs, y_train)                    
                
                # cost로 H(x) 개선
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_avg_cost += loss / train_total_batch
                
            self.train_epoch_losses.append(train_avg_cost)
            
            # validation loss
            valid_avg_cost = 0
            valid_total_batch = len(self.valid_data_loader)
            with torch.no_grad():
                for x_valid, y_valid in self.valid_data_loader:
                    self.model.reset_hidden_state()
                    outputs = self.model(x_valid)
                    loss = self.loss_fn(outputs, y_valid)               
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    valid_avg_cost += loss / valid_total_batch
                self.valid_epoch_losses.append(valid_avg_cost)
            
            if epoch % verbose == 0:
                print('Epoch:', '%04d' % (epoch), 'train loss :', '{:.4f}'.format(train_avg_cost))
                
            # patience번째 마다 early stopping 여부 확인
            if (epoch % patience == 0) & (epoch != 0):
                
                # loss가 커졌다면 early stop
                if self.train_epoch_losses[epoch-patience] < self.train_epoch_losses[epoch]:
                    print('\n Early Stopping')
                    break
                
        return self.model.eval(), self.train_epoch_losses
    
    def _inverse_transform(self, inverse):
        with torch.no_grad():
            pred = []
            test_x, test_y = self.valid_data_sets
            
            for i in range(len(test_x)):
                self.model.reset_hidden_state()
                
                predicated = self.model(torch.unsqueeze(test_x[i], 0))
                predicated = torch.flatten(predicated).item()
                pred.append(predicated)
                
        pred_inverse = inverse(np.array(pred).reshape(-1, 1))
        test_y_inverse = inverse(test_y)
        return pred_inverse, test_y_inverse
    
    def performance_eval_metrics(self):
        # mae
        # mse
        # rmse
        # mape
        # mpe
        pass
    
    def visualization(self, inverse):
        # loss graph
        # pred graph
        epochs = self.train_epoch_losses
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))
        axe_loss = axes[0]
        axe_loss.plot(range(epochs), self.train_epoch_losses, label="train", color="blue")
        axe_loss.plot(range(epochs), self.valid_epoch_losses, label="valid", color="orange")
        axe_loss.set_ylabel("MSE Loss")
        
        axe_pred = axes[1]
        pred, test_y = self._inverse_transform(inverse)
        axe_pred.plot(range(len(pred)), pred, label="model", color="orange")
        axe_pred.plot(range(len(test_y)), test_y, label="test", color="orange")
        axe_pred.set_ylabel("Close")
        axe_pred.legend()
        plt.title("Predict Graph")
        plt.show()