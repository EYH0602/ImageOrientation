import numpy as np
import torch
import os
from PIL import Image
import math
from torchvision.transforms.functional import rotate, center_crop
from sklearn.model_selection import train_test_split
from src.util import progressbar


class DatasetLoader:
    def __init__(self, directory, test_size=0.3, validation_size=0.2, total_num=10000, seed=123):
        self.dir = directory
        self.cap = total_num
        self.test_size = test_size
        self.val_size = validation_size
        self.seed = seed

        self.data = {}

    def load(self):
        print('Loading Data from', self.dir)
        images, labels = [], []
        np.random.seed(self.seed)
         
        for file in os.listdir(self.dir):
            img = torch.FloatTensor(self.load_img(file))
            # reshape into (Channel, Height, Weight) as torch required
            img = img.permute(2, 0, 1) 
            # normalize
            img /= 255
            img = img.unsqueeze(0)
            
            deg = np.random.uniform(-60, 60, 1)[0]
            
            # rotate the image and take the center 130 x 130
            # to avoid the network learn from unfilled coners after rotation
            images.append(center_crop(rotate(img, deg), output_size=130))
            labels.append(deg)
            
            if self.cap and len(images) >= self.cap:
                break
            progressbar(len(images), min(len(os.listdir(self.dir)), self.cap))
        
        X_train, X_test, y_train, y_test = train_test_split(
            images, labels, 
            test_size=self.test_size, 
            random_state=self.seed
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=self.test_size, 
            random_state=self.seed
        )
        
        self.data['train'] = {'X': X_train, 'y': y_train}
        self.data['validation'] = {'X': X_val, 'y': y_val}
        self.data['test'] = {'X': X_test, 'y': y_test}
        

    def load_img(self, file_name):
        '''
        load a image file into numpy array
        '''
        img = Image.open(self.dir + file_name)
        img.load()
        return np.asarray(img, dtype="float32")
    