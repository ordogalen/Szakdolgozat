from collections import OrderedDict

from torch.utils.data import DataLoader
import os
import numpy as np
from pathlib import Path

from SpeechDataGenerator import SpeechDataGenerator
from utils import utils
import torch
import torchvision
from models import x_vector

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook



model = x_vector.X_vector(120,15)
model.segment6.register_forward_hook(get_activation('segment6'))
id = 1
npy = utils.load_npy_data("D:/Szakdoga/pythonProject/meta/Features/test/bea036f021_0001_001.npy")
npy = torch.from_numpy(np.asarray(npy)).float()
npy = npy.to("cuda")
npy.requires_grad = True
print(npy)

x,output = model(npy)
print(activation['segment6'])

