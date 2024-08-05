import os
import scipy.io
from os.path import join
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url, list_dir
import torch
import torchvision
import torchvision.transforms as transforms

class OxfordPets(Dataset):
    def __init__(self, split="test", download=False, classes=None):

        self.test_dataset = torchvision.datasets.OxfordIIITPet(root='./data', split=split,
                                       download=download, transform=transforms.Compose([transforms.ToTensor(), transforms.ToPILImage()]))
        self.classes = classes
        if self.classes is None:
            self.classes = self.test_dataset.classes
        self.length = len(self.test_dataset)
        self.test_dataset = iter(self.test_dataset)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image, label = next(self.test_dataset)
        return image, self.classes[label].lower()


if __name__ == '__main__':
    test_dataset = OxfordPets(split="test", download=True)
    for sample in test_dataset:
        print(sample[-1])
        break