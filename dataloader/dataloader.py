from cProfile import label
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset

from config import *

# transform
img_transform = {
    'train': transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
}


class CropsDataset(Dataset):
    def __init__(self, mode):
        self.mode = mode
        crops_list = pd.read_csv(f'./csv/{mode}.csv')
        self.img_path = crops_list['img_path']
        self.labels = crops_list['label']
        self.transforms = img_transform[mode]

    def __getitem__(self, idx):
        img = Image.open(f'.{self.img_path[idx]}').convert('RGB')
        img = self.transforms(img)
        label = torch.tensor(self.labels[idx]) 
        return img, label

    def __len__(self):
        return len(self.labels)
