from src.DatasetLoader import DatasetLoader
from src.Model import ModelCNN
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    SEED = 1
    np.random.seed(SEED)
    dataset = DatasetLoader('./data/img_align_celeba/', total_num=100, seed=SEED)
    dataset.load()
    
    model = ModelCNN(dataset.data) 
    model.train()
    
    print("Test Accuracy:", model.test())
    
    