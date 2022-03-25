import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import mean_absolute_error

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("training on: " + device.type)


class ModelCNN(nn.Module):
    def __init__(self, lr=1e-3, max_epoch=300):
        nn.Module.__init__(self)
        
        self.lr = lr
        self.max_epoch = max_epoch
        
        self.conv1 = nn.Conv2d(3, 3, (5, 5))
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(3, 6, 5)
        self.pool2 = nn.MaxPool2d(2)
        
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
        y_pred = self.fc2(h)
        return y_pred
    
    def train(self, data):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        loss_func = nn.MSELoss()
        
        X_train = torch.stack(data['train']['X'])
        y_train = torch.FloatTensor(data['train']['y']).unsqueeze(1)
        y_pred = []
        
        for epoch in range(self.max_epoch):
            optimizer.zero_grad()
            
            y_pred = [self.forward(X) for X in data['train']['X']]
            y_pred = torch.stack(y_pred)
            
            loss = loss_func(y_pred, y_train)
            loss.backward()
            optimizer.step()
            
            if epoch % 100 == 0:
                print(
                    'Epoch', epoch,
                    'Loss', loss.item(),
                    'MSE', mean_absolute_error(y_train, y_pred.detach().numpy())
                )
    
        
        
        
