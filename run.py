from src.DatasetLoader import DatasetLoader
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dataset = DatasetLoader('./data/img_align_celeba/', total_num=10)
    dataset.load()
    
    train = dataset.data['train']
    
    print(train['X'][0].shape)
    print(train['y'][0])
    p = plt.imshow(train['X'][0].squeeze(0).permute(1,2,0))
    plt.show()
    
    