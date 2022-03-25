from src.DatasetLoader import DatasetLoader
from src.Model import ModelCNN
from src.Plotter import Plotter
import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle

if __name__ == '__main__':
    SEED = 1
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    # dataset = DatasetLoader('./data/img_align_celeba/', total_num=100, seed=SEED)
    # dataset.load()
    f = open('save/data.pkl', 'rb')
    dataset = pickle.load(f)
    f.close()
    
    
    plotter = Plotter("Oritentation Detection Plot", "CNNRegression.png", plot_acc=False)
    
    model = ModelCNN(dataset.data, plotter)
    if torch.cuda.is_available():
        print("training on: cuda")
        model = model.cuda()

    model.train()
    
    print("Test Accuracy:", model.test())
    plotter.plot()
    
    