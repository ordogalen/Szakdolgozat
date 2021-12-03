from torch import optim
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
import os
import numpy as np

import torch
from models.x_vector_Indian_LID import X_vector
from utils import utils

PATH = "./save_model/best_check_point_15_0.3372245247165362"

net = X_vector(120, 15)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = X_vector(120, 15).to(device)
checkpoint = torch.load(os.path.join(PATH))
model.load_state_dict(checkpoint['model'])
layer = model.modules.get('segment6')


def get_vector(raw_wav):
    spec = utils.load_data(raw_wav, mode="train")
    sample = {'features': torch.from_numpy(np.ascontiguousarray(spec)),}

    model.eval()
    my_embedding = torch.zeros(1,512)
    def copy_data(m, i, o):
        my_embedding.copy_(o.data)
    h = layer.register_forward_hook(copy_data)

    features = torch.from_numpy(np.asarray([sample['features'].numpy().T]))
    model(features.to(device))
    h.remove()
    return my_embedding

wav_aibo = "./meta/aibo/wav-aibo/"

def extract_features_into_file(raw, label, mode):
    filesDone = 0
    files = open(raw, "r")
    labels = open(label, "r")
    dir = ""
    if (mode == 'train'):
        dir = "./meta/aibo/train/"
    if(mode == "dev"):
        dir = "./meta/aibo/dev/"
    if(mode == "test"):
        dir = "./meta/aibo/test/"
    for i, (key1, key2) in enumerate(zip(files, labels)):
        feature = get_vector(wav_aibo+key1.strip()+".wav")
        dirWithLabel = dir+key2.strip()

        if not os.path.exists(dirWithLabel):
            os.makedirs(dirWithLabel)
        dest_filepath = dirWithLabel + '/'
        np.save(dest_filepath+key1.strip()+"_"+key2.strip(), feature)
        filesDone+=1
        print(filesDone)

    files.close()
    labels.close()


extract_features_into_file("./meta/aibo/filelist.raw.test.txt","./meta/aibo/labels.test.txt","test")
extract_features_into_file("./meta/aibo/filelist.raw.dev.txt","./meta/aibo/labels.dev.txt","dev")






