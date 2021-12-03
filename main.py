from collections import OrderedDict

from torch import optim
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
import os
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchaudio.kaldi_io import read_mat_scp

from SpeechDataGenerator import SpeechDataGenerator
from utils import utils
import torch
import torchvision
from models.x_vector_Indian_LID import X_vector
import feature_extraction
from utils.utils import speech_collate

PATH = "./save_model/best_check_point_15_0.3372245247165362"




net = X_vector(120,15)


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model=X_vector(120,15).to(device)
checkpoint = torch.load(os.path.join(PATH))
model.load_state_dict(checkpoint['model'])
layer = model._modules.get('segment6')


def get_vector(path):
    asd = SpeechDataGenerator(dataset_audio_path=path, mode='val', shuffle_seed=42)
    dataloader_train = DataLoader(asd, batch_size=32, shuffle=True, collate_fn=speech_collate)
    model.eval()
    my_embedding = torch.zeros(1,512)
    def copy_data(m, i, o):
        my_embedding.copy_(o.data)
    h = layer.register_forward_hook(copy_data)

    features = ""
    for i_batch, sample_batched in enumerate(dataloader_train):
        for i in sample_batched[0]:
            print("asd")
            print(i)

        print("asd2")
        features = torch.from_numpy(np.asarray([torch_tensor.numpy().T for torch_tensor in sample_batched[0]]))
    model(features.to(device))
    h.remove()
    return my_embedding

a = get_vector(path = "./meta/Feature1")
print(a)


def get_vector2(path):
    spec = utils.load_data(path, mode="train")
    sample = {'features': torch.from_numpy(np.ascontiguousarray(spec)),
              'labels': torch.from_numpy(np.ascontiguousarray(0))}

    model.eval()
    my_embedding = torch.zeros(1,512)
    def copy_data(m, i, o):
        my_embedding.copy_(o.data)
    h = layer.register_forward_hook(copy_data)

    features = torch.from_numpy(np.asarray([sample['features'].numpy().T]))
    model(features.to(device))
    h.remove()
    return my_embedding

a = get_vector2(path = "./meta/Feature1/001/bea001f001_0002_001.wav")
print(a)
