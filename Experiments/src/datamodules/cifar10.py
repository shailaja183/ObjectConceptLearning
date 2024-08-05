import os
import scipy.io
from os.path import join
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url, list_dir
import torch
import torchvision
import torchvision.transforms as transforms

class Cifar10(Dataset):
    def __init__(self, train=False, download=False):

        self.test_dataset = torchvision.datasets.CIFAR10(root='./data', train=train,
                                       download=download, transform=transforms.Compose([transforms.ToTensor(), transforms.ToPILImage()]))
        self.length = len(self.test_dataset)
        self.test_dataset = iter(self.test_dataset)
        
        self.classes = ('airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image, label = next(self.test_dataset)
        return image, self.classes[label]



if __name__ == '__main__':
    test_dataset = Cifar10(train=False, download=True)
    for sample in test_dataset:
        print(sample[-1])
        break