# from sympy import Mod
import torch
from torch.cuda.amp import autocast as autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from termcolor import colored
from dataloader.test_dataloader import CropsDataset
from model.model import EfNetModel
from config import *


def test(model, test_data):
    model.eval()
    with torch.no_grad():
        for imgs in tqdm(test_data):
            imgs = imgs.to(device)  # tensor
            logits = model(imgs)
            print(logits.cpu().argmax(dim=-1))
            # print(logits)
            # logits_numpy = logits.cpu().numpy()
            # print(logits_numpy.shape)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(colored(f'Using Device : {device}', 'yellow'))

    # Data
    test_dataset = CropsDataset()
    print(len(test_dataset))  # 713

    test_loader = DataLoader(test_dataset,
                             shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    # Model
    model = EfNetModel(num_classes=NUM_CLASS)
    model.load_state_dict(torch.load(
        '/home/rorubyy/Documents/PythonWorkspace/ai_cup/ckpt/NLL_EffModel_0.865_0.863.ckpt'))
    model = model.to(device)
    print("Model loaded success")
    test(model, test_loader)
    # create_csv('test_data')
