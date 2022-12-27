import logging
from matplotlib.pyplot import plot
# from sympy import Mod
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from termcolor import colored
from dataloader.dataloader import CropsDataset
from model.model import EfNetModel
from model.ViT import VIT
import utils
from config import *
from torch.utils.tensorboard import SummaryWriter

# https://github.com/jacobgil/pytorch-grad-cam
'''
Label: {
    'longan': 0,
    'custardapple': 1,
    'roseapple': 2,
    'sweetpotato': 3,
    'others': 4,
    'grape': 5,
    'asparagus': 6,
    'mango': 7,
    'greenonion': 8,
    'chinesechives': 9,
    'tea': 10,
    'onion': 11,
    'papaya': 12,
    'waterbamboo': 13,
    'bambooshoots': 14,
    'sesbania': 15,
    'broccoli': 16,
    'loofah': 17,
    'lettuce': 18,
    'pear': 19,
    'sunhemp': 20,
    'betel': 21,
    'soybeans': 22,
    'chinesecabbage': 23,
    'passionfruit': 24,
    'lemon': 25,
    'kale': 26,
    'cauliflower': 27,
    'redbeans': 28,
    'litchi': 29,
    'pennisetum': 30,
    'taro': 31,
    'greenhouse': 32
}
'''


def train(model, train_data, vali_data, criterion, optimizer, scheduler):
    best_acc = 0.0
    stop_count = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = []
        train_pred = torch.tensor([])
        train_label = torch.tensor([])

        correct = 0
        total = 0

        # progress_bar = tqdm(train_data)
        for i, (datas, labels) in enumerate(tqdm(train_data)):
            optimizer.zero_grad()
            imgs = datas
            with autocast():
                imgs = imgs.to(device)
                labels = labels.to(device)
                # forward
                logits = model(imgs)
                # compute loss
                loss = criterion(logits, labels)

                """
                if len(train_pred) == 0:
                    train_pred = logits.cpu()
                    train_label = labels.cpu()
                else:
                    train_pred = torch.cat((train_pred, logits.cpu()), 0)
                    train_label = torch.cat((train_label, labels.cpu()), 0)

                acc = (logits.cpu().argmax(dim=-1) ==
                       labels.cpu()).float().mean()
                """
                # f1_score = utils.cal_f1_score(train_pred, train_label)
                """correct += sum(logits.cpu().argmax(dim=-1)
                               == labels.cpu()).item()"""
                # TOBY
                _, preds = torch.max(logits, 1)
                correct += torch.sum(preds == labels.data)

                total += labels.shape[0]
                train_loss.append(loss.item())

                '''
                # set tqdm
                progress_bar.set_description(
                    f"[ Train | {epoch + 1:03d}/{EPOCHS:03d} ] {colored(f'Loss = {loss.item():.5f}', 'red')}, {colored(f'ACC = {acc:.5f}', 'green')}, {colored(f'F1-Score = {f1_score:.5f}', 'yellow')}")
                progress_bar.update()
                '''

            loss.backward()
            optimizer.step()
            scheduler.step()

        train_loss = sum(train_loss) / len(train_loss)
        writer.add_scalar('train_loss', train_loss, epoch)
        # train_f1 = utils.cal_f1_score(train_pred, train_label)
        train_acc = correct / total
        writer.add_scalar('train_acc', train_acc, epoch)

        # Print the infor
        logging.info(
            f"[ Train | {epoch + 1:03d}/{EPOCHS:03d} ] Loss = {train_loss:.5f}, Acc = {train_acc:.5f}")
        print(colored(
            f'[ Train | {epoch +1 :03d}/{EPOCHS :03d} ] Loss = {train_loss:.5f}, Acc = {train_acc:.5f}', 'green'))

        # ---------- Validation ----------
        model.eval()
        valid_loss = []
        valid_pred = torch.tensor([])
        valid_label = torch.tensor([])
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in tqdm(vali_data):
                imgs = imgs.to(device)
                labels = labels.to(device)
                logits = model(imgs)

                loss = criterion(logits, labels)
                # loss = criterion(torch.log(torch.softmax(logits, dim=-1)), labels)

                # Compute the Accuracy
                # acc = (logits.cpu().argmax(dim=-1) ==
                #        labels.cpu()).float().mean()
                correct += sum(logits.cpu().argmax(dim=-1)
                               == labels.cpu()).item()  
                total += labels.shape[0]
                # Record the loss and accuracy
                valid_loss.append(loss.item())
                if len(valid_pred) == 0:
                    valid_pred = logits.cpu()
                    valid_label = labels.cpu()
                else:
                    valid_pred = torch.cat((valid_pred, logits.cpu()), 0)
                    valid_label = torch.cat((valid_label, labels.cpu()), 0)

        valid_loss = sum(valid_loss) / len(valid_loss)
        writer.add_scalar('valid_loss', valid_loss, epoch)

        valid_f1 = utils.cal_f1_score(valid_pred, valid_label)
        writer.add_scalar('valid_f1', valid_f1, epoch)

        valid_acc = correct / total
        writer.add_scalar('valid_acc', valid_acc, epoch)

        # Print the info
        logging.info(
            f"[ Valid | {epoch + 1:03d}/{EPOCHS:03d} ] Loss = {valid_loss:.5f}, Acc = {valid_acc:.5f}, Macro-F1 = {valid_f1:.5f}")
        print(colored(
            f'[ Valid | {epoch +1 :03d}/{EPOCHS :03d} ] Loss = {valid_loss:.5f}, Acc = {valid_acc:.5f}, Macro-F1 = {valid_f1:.5f}', 'red'))

        # Save Best Model
        if valid_acc >= best_acc:
            model_path = f'{MODEL_PATH}NLL_EffModel_{valid_acc:.3f}_{valid_f1:.3f}.ckpt'
            torch.save(model.state_dict(), model_path)
            logging.info(f'Save Model With Acc: {valid_acc:.5f}')
            print(f'Save Model With Acc: {valid_acc:.5f}')
            best_acc = valid_acc
            # stop_count = 0
        # else:
        #     stop_count += 1
        #     if stop_count > EARLY_STOP:
        #         break


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(colored('=' * 15, 'blue'))
    print(colored(f'Using Device : {device}', 'blue'))
    print(colored(f'Batch Size : {BATCH_SIZE}', 'blue'))
    print(colored(f'Total Epochs : {EPOCHS}', 'blue'))
    print(colored(f'Early Stopping : {EARLY_STOP}', 'blue'))
    print(colored('=' * 15, 'blue'), end="\n\n")

    utils.set_seeds(42)
    writer = SummaryWriter('./log')
    logging.basicConfig(level=logging.INFO, filename='./log/log.txt', filemode='w',
                        format='[%(asctime)s %(levelname)-8s] %(message)s',
                        datefmt='%Y%m%d %H:%M:%S',
                        )

    # Data
    train_dataset = CropsDataset(mode='train')
    valid_dataset = CropsDataset(mode='valid')

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)

    # Model
    model = EfNetModel(num_classes=NUM_CLASS) #EffNet
    # model = VIT(num_classes=NUM_CLASS)
    
    model = model.to(device)

    # Loss function
    # criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=T_MAX)

    # Call trainer
    train(model, train_loader, val_loader, criterion, optimizer, scheduler)

    # def show_train_history(train_model, train, validation):
    #     plot.plot(train_model.history[train])
    #     plot.plot(train_model.history[validation])
    #     plot.title('Train History')
    #     plot.ylabel(train)  
    #     plot.xlabel('Epochs')  ㄎ
    #     plot.legend(['train', 'validation'], loc='upper left')  
    #     plot.show()
    
    # show_train_history(train, 'train_acc', 'valid_acc')
    # show_train_history(train, 'train_loss', 'valid_loss')