import torch
from sklearn import preprocessing
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self,dataset,Class,transform=None):
        self.data=dataset
        self.data_class=Class
        self.transform=transform
    def __len__(self):
        return len(self.data_class)
    def __getitem__(self, item):
        label=self.data_class[item]
        if self.transform is not None:
            input=self.transform( self.data[item])
        return input,label

