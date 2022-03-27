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
    dataset = DatasetLoader('./data/img_align_celeba/', total_num=20000, seed=SEED)
    dataset.load()
    
    
    plotter = Plotter("Orientation Detection Plot", "CNNRegression.png", plot_acc=False)
    
    model = ModelCNN(dataset.data, plotter, max_epoch=500)
    if torch.cuda.is_available():
        print("training on: cuda")
        model = model.cuda()

    model.train()
    
    print("Test Accuracy:", model.test())
    plotter.plot()

    print(model.predict(dataset.data['test']['X'][0]))

