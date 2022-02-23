#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 20:22:26 2020

@author: krishna
"""
import torch
import numpy as np
from torch.utils.data import DataLoader
from SpeechDataGenerator import SpeechDataGenerator
import torch.nn as nn
import os
from torch import optim
import argparse
from models.x_vector_Indian_LID import X_vector
from sklearn.metrics import accuracy_score
from utils.utils import speech_collate
import matplotlib.pyplot as plt

def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_acc['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_acc['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig(os.path.join('meta/lossGraphs', 'train.jpg'))

torch.multiprocessing.set_sharing_strategy('file_system')

#Plot
y_loss = {'train': [], 'val': []}  # loss history
y_acc = {'train': [], 'val': []}
x_epoch = []

fig = plt.figure()
ax0 = fig.add_subplot(121, title="mean loss")
ax1 = fig.add_subplot(122, title="mean accuracy")

# Argument parser
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-training_filepath', type=str, default='meta/bea/speakers')
parser.add_argument('-testing_filepath', type=str, default='meta/bea/speakers')
parser.add_argument('-validation_filepath', type=str, default='meta/bea/speakers')

parser.add_argument('-audio_files', type=str, default='meta/bea/speakers')

parser.add_argument('-input_dim', action="store_true", default=120)
parser.add_argument('-num_classes', action="store_true", default=89)
parser.add_argument('-lamda_val', action="store_true", default=0.1)
parser.add_argument('-batch_size', action="store_true", default=64)
parser.add_argument('-use_gpu', action="store_true", default=True)
parser.add_argument('-num_epochs', action="store_true", default=12)
args = parser.parse_args()

# Data related
SHUFFLE_SEED = 45  # random seed
dataset = SpeechDataGenerator(dataset_audio_path=args.audio_files, mode='train', shuffle_seed=SHUFFLE_SEED)

args.num_classes = len(set(dataset.labels))

train_size = int(0.80 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

print("All data:" + str(len(dataset)))
print("Training on " + str(len(train_dataset)))
print("Validating on " + str(len(test_dataset)))

dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=speech_collate, pin_memory=False)
dataloader_val = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=speech_collate, pin_memory=False)

# dataset_test = SpeechDataGenerator(dataset_audio_path=args.training_filepath, mode='test', shuffle_seed=SHUFFLE_SEED)
# dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True, collate_fn=speech_collate)

# Model related
use_cuda = True  # torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = X_vector(args.input_dim, args.num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8)
loss_fun = nn.CrossEntropyLoss()


def train(dataloader_train, epoch):
    train_loss_list = []
    full_preds = []
    full_gts = []
    model.train()
    for i_batch, sample_batched in enumerate(dataloader_train):
        features = torch.from_numpy(np.asarray([torch_tensor.numpy().T for torch_tensor in sample_batched[0]])).float()
        labels = torch.from_numpy(np.asarray([torch_tensor[0].numpy() for torch_tensor in sample_batched[1]]))
        features, labels = features.to(device), labels.to(device)
        features.requires_grad = True
        optimizer.zero_grad()
        pred_logits, x_vec = model(features)

        # CE loss
        loss = loss_fun(pred_logits, labels.long())
        loss.backward()
        optimizer.step()
        train_loss_list.append(loss.item())
        if i_batch % 10 == 0:
            print('Loss {} after {} iteration'.format(np.mean(np.asarray(train_loss_list)), i_batch))

        predictions = np.argmax(pred_logits.detach().cpu().numpy(), axis=1)
        for pred in predictions:
            full_preds.append(pred)
        for lab in labels.detach().cpu().numpy():
            full_gts.append(lab)

    mean_acc = accuracy_score(full_gts, full_preds)
    mean_loss = np.mean(np.asarray(train_loss_list))
    y_loss['train'].append(mean_loss)
    y_acc['train'].append(mean_acc)

    print('Total training loss {} and training Accuracy {} after {} epochs'.format(mean_loss, mean_acc, epoch))


def validation(dataloader_val, epoch):
    model.eval()
    with torch.no_grad():
        val_loss_list = []
        full_preds = []
        full_gts = []
        for i_batch, sample_batched in enumerate(dataloader_val):
            features = torch.from_numpy(
                np.asarray([torch_tensor.numpy().T for torch_tensor in sample_batched[0]])).float()
            labels = torch.from_numpy(np.asarray([torch_tensor[0].numpy() for torch_tensor in sample_batched[1]]))
            features, labels = features.to(device), labels.to(device)
            pred_logits, x_vec = model(features)

            # CE loss
            loss = loss_fun(pred_logits, labels.long())
            val_loss_list.append(loss.item())

            predictions = np.argmax(pred_logits.detach().cpu().numpy(), axis=1)
            for pred in predictions:
                full_preds.append(pred)
            for lab in labels.detach().cpu().numpy():
                full_gts.append(lab)

        mean_acc = accuracy_score(full_gts, full_preds)
        mean_loss = np.mean(np.asarray(val_loss_list))
        y_loss['val'].append(mean_loss)
        y_acc['val'].append(mean_acc)

        print('Total validation loss {} and Validation accuracy {} after {} epochs'.format(mean_loss, mean_acc, epoch))
        model_save_path = os.path.join('save_model_2', 'best_check_point_' + str(epoch) + '_' + str(mean_loss))
        state_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state_dict, model_save_path)


if __name__ == '__main__':
    if not os.path.exists("meta/FeatureExt"):
        os.makedirs("meta/FeatureExt")

    for epoch in range(args.num_epochs):
        train(dataloader_train, epoch)
        validation(dataloader_val, epoch)
