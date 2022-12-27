import glob
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from config import *

# transform
img_transform = {
    'test': transforms.Compose([
        transforms.Resize(size=(380, 380)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
}


class CropsDataset(Dataset):
    def __init__(self):
        self.transforms = img_transform["test"]
        images_list = list(glob.glob(
            '/home/rorubyy/Documents/PythonWorkspace/ai_cup/testDataset/0/*.jpg'))
        images_list_str = [str(x) for x in images_list]
        self.images = images_list_str

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path)  # 读取到的是RGB， W, H, C
        image = self.transforms(image)
        return image

    def __len__(self):
        return len(self.images)
