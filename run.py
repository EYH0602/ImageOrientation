from src.DatasetLoader import DatasetLoader
from src.Model import ModelCNN
from src.Plotter import Plotter
import matplotlib.pyplot as plt
import numpy as np
import torch

if __name__ == '__main__':
    SEED = 1
    np.random.seed(SEED)
    dataset = DatasetLoader('./data/img_align_celeba/', total_num=100, seed=SEED)
    dataset.load()
    
    plotter = Plotter("Oritentation Detection Plot", "CNNRegression.png", plot_acc=False)
    
    model = ModelCNN(dataset.data, plotter) 
    model.train()
    
    print("Test Accuracy:", model.test())
    plotter.plot()
    
    