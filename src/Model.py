import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import mean_squared_error
from src.util import training_progressbar

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ModelCNN(nn.Module):
    def __init__(self, data, plotter=None, lr=1e-3, max_epoch=300):
        nn.Module.__init__(self)

        self.lr = lr
        self.max_epoch = max_epoch
        self.plotter = plotter

        self.train_set = data['train']
        self.val_set = data['validation']
        self.test_set = data['test']

        # CNN layers
        self.conv1 = nn.Conv2d(3, 3, (5, 5))
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(3, 6, 5)
        self.pool2 = nn.MaxPool2d(2)
        
        self.dropout = nn.Dropout(0.4)

        # FC layers
        self.fc1 = nn.Linear(5046, 120)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(120, 1)

    def forward(self, X):
        '''Forward propagation'''
        # CNN layers
        h = self.pool1(self.conv1(X))
        h = self.pool2(self.conv2(h))

        # FC layers
        h = torch.flatten(h)
        h = self.act1(self.fc1(h))
        h = self.dropout(h)
        y_pred = self.fc2(h)
        return y_pred

    def forward_all(self, Xs):
        '''Wrapper for Full batch Forward propagation'''
        y_pred = [self.forward(X.to(device)) for X in Xs]
        return torch.stack(y_pred)

    def train(self):
        if self.plotter == None:
            raise RuntimeWarning("Plotter not defined.")

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        loss_func = nn.MSELoss()

        y_train = torch.FloatTensor(self.train_set['y']).unsqueeze(1).to(device)

        print("Start Training.")
        for epoch in range(self.max_epoch):
            optimizer.zero_grad()

            y_pred = self.forward_all(self.train_set['X'])
            loss = loss_func(y_pred, y_train)

            loss.backward()
            optimizer.step()

            train_loss = loss.item()
            val_loss = self.validate()
            
            if self.plotter:
                self.plotter.train_loss.append(train_loss)
                self.plotter.val_loss.append(val_loss)

            training_progressbar(epoch, self.max_epoch, round(train_loss, 3))

    def validate(self):
        y_val = torch.FloatTensor(self.val_set['y']).unsqueeze(1)

        with torch.no_grad():
            y_pred = self.forward_all(self.val_set['X']).cpu().detach().numpy()
            return mean_squared_error(y_val, y_pred)

    def test(self):
        y_test = torch.FloatTensor(self.test_set['y']).unsqueeze(1)
        y_pred = self.forward_all(self.test_set['X']).cpu().detach().numpy()
        return mean_squared_error(y_test, y_pred)
