
import matplotlib.pyplot as plt
import numpy as np

class Plotter:
    def __init__(self, plot_name, file_name, plot_acc=True, plot_val=True):
        self.title = plot_name
        self.path = file_name
        self.plot_acc = plot_acc
        self.plot_val = plot_val
        
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        
        
    def plot(self):
        if self.plot_acc:
            self.plot_two()
        else:
            self.plot_one()
            
    def plot_one(self):
        '''
        plot train and/or validation loss
        '''
        fig = plt.figure()
        plt.plot(self.train_loss, color='#EFAEA4', label = 'Training Loss')
        if self.plot_val:
            plt.plot(self.val_loss, color='#B2D7D0', label = 'Validation Loss')

        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        fig.suptitle(self.title, fontsize=24)
        fig.savefig(self.path)
        
    def plot_two(self):
        '''
        plot train and/or validation loss
        and train and/or validation accuracy if task is classification
        '''
        fig, ax = plt.subplots(1,2,figsize = (16,4))
        ax[0].plot(self.train_loss, color='#EFAEA4', label = 'Training Loss')
        ax[1].plot(self.train_acc, color='#EFAEA4',label = 'Training Accuracy')
        if self.plot_val:
            ax[0].plot(self.val_loss, color='#B2D7D0', label = 'Validation Loss')
            ax[1].plot(self.val_acc, color='#B2D7D0', label = 'Validation Accuracy')

        ax[0].legend()
        ax[1].legend()
        ax[0].set_xlabel('Epochs')
        ax[1].set_xlabel('Epochs');
        ax[0].set_ylabel('Loss')
        ax[1].set_ylabel('Accuracy %');
        fig.suptitle(self.title, fontsize = 24)
        
        plt.savefig(self.path)