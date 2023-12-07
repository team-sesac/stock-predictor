import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scaler import Scaler
from earlystop import EarlyStopping
from hyperparameter import HyperParameter

class LSTMTrainer():
    
    def __init__(self, model, scaler: Scaler, data_loaders, datasets, hyper_parameter: HyperParameter):
        train_data_loader, valid_data_loader = data_loaders
        train_data_sets, valid_data_sets = datasets
        self.model = model
        self.loss_fn = nn.MSELoss().to(hyper_parameter.get_device())
        self.optimizer = torch.optim.Adam(params=model.parameters(), lr=hyper_parameter.get_lr())
        self.epochs = hyper_parameter.get_epochs()
        self.train_epoch_losses = [] # epoch마다 loss 저장
        self.valid_epoch_losses = []
        self.mae_scores = []
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.train_data_sets = train_data_sets
        self.valid_data_sets = valid_data_sets
        self.device = hyper_parameter.get_device()
        self.scaler = scaler
        
    def train(self, verbose=10, patience=10):

        ealry_stop = EarlyStopping(patience=patience)

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
                
                train_avg_cost += loss.item() / train_total_batch
                
            self.train_epoch_losses.append(train_avg_cost)
            
            # validation loss
            valid_avg_cost = 0
            valid_total_batch = len(self.valid_data_loader)
            with torch.no_grad():
                for x_valid, y_valid in self.valid_data_loader:
                    self.model.reset_hidden_state()
                    outputs = self.model(x_valid)
                    loss = self.loss_fn(outputs, y_valid)               
                    valid_avg_cost += loss.item() / valid_total_batch
                self.valid_epoch_losses.append(valid_avg_cost)
            
            # if epoch % verbose == 0:
            #     print('Epoch:', '%03d' % (epoch), ' / train loss :', '{:.4f}'.format(train_avg_cost), ' / valid loss :', '{:.10f}'.format(valid_avg_cost))
            
            # early stop 
            if ealry_stop(valid_avg_cost):
                break
            
        return self.model.eval(), self.train_epoch_losses
    
    def _inverse_transform(self):
        with torch.no_grad():
            pred = []
            test_x, test_y = self.valid_data_sets.tensors
            
            for i in range(len(test_x)):
                self.model.reset_hidden_state()
                
                predicated = self.model(torch.unsqueeze(test_x[i], 0))
                predicated = torch.flatten(predicated).item()
                pred.append(predicated)
                
        pred_inverse = self.scaler.inverse_y(np.array(pred).reshape(-1, 1))
        test_y_inverse = self.scaler.inverse_y(test_y)
        return pred_inverse, test_y_inverse
    
    def _perform_eval_metrics(self, pred, y):
        eval = {}
        print(len(y))
        mae = np.mean(np.abs(y-pred))
        eval["MAE(\u2193)"] = mae
        mape = np.mean(np.abs((y-pred) / y) * 100)
        eval["MAPE(\u2193)"] = mape
        mse = np.mean(np.square(y-pred))
        eval["MSE(\u2193)"] = mse
        rmse = np.sqrt(mse)
        eval["RMSE(\u2193)"] = rmse
        msle = np.mean(np.sum(np.square(np.log(y+1) - np.log(pred+1))))
        eval["MSLE(\u2193)"] = msle
        rmsle = np.sqrt(msle)
        eval["RMSLE(\u2193)"] = rmsle
        r2 =  1 - (np.sum(np.square(y-pred)) / np.sum(np.square(y - np.mean(y))))
        eval["R2(\u2191)"] = r2
        return eval
    
    def visualization(self):
        epochs = len(self.train_epoch_losses)
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(9, 8))
        # loss graph
        axe_loss = axes[0]
        axe_loss.plot(range(epochs), self.train_epoch_losses, label="train", color="blue")
        axe_loss.plot(range(epochs), self.valid_epoch_losses, label="valid", color="orange")
        axe_loss.set_xlabel("Epoch")
        axe_loss.legend()
        axe_loss.set_ylabel("MSE Loss")
        
        # pred graph
        axe_pred = axes[1]
        pred, test_y = self._inverse_transform()
        axe_pred.plot(range(len(pred)), pred, label="pred", color="orange")
        axe_pred.plot(range(len(test_y)), test_y, label="test", color="blue")
        axe_pred.set_ylabel("Close")
        axe_pred.legend()
        
        eval = self._perform_eval_metrics(pred=pred, y=test_y)
        axe_eval = axes[2]
        evals = eval.items()
        x = 0.0
        y = 0.9
        for k, v in evals:
            str: str = f"{'%-8s' % k} : {round(v, 2)}"
            axe_eval.text(x, y, str, fontsize=12, ha='left', va='center', family='monospace')
            y -= 0.12
        axe_eval.axis("off")
        
        plt.tight_layout()
        plt.show()