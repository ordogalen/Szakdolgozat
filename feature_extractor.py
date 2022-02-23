import os
import numpy as np

import torch
from models.x_vector_Indian_LID import X_vector
from utils import utils

#PATH = "./save_model/best_check_point_15_0.3372245247165362"
PATH = "save_model_2/best_check_point_29_1.2584018047819747"

net = X_vector(120, 89)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = X_vector(120, 89).to(device)
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
        dir = "./meta/aibo/train_downsample2/"
    if(mode == "dev"):
        dir = "./meta/aibo/dev2/"
    if(mode == "test"):
        dir = "./meta/aibo/test2/"
    for i, (key1, key2) in enumerate(zip(files, labels)):
        feature = get_vector(wav_aibo+key1.strip()+".wav")

        np.save(dir+key1.strip()+"_#"+key2.strip(), feature)
        filesDone += 1
        if(filesDone % 100 == 0):
            print(filesDone)

    files.close()
    labels.close()


extract_features_into_file("./meta/aibo/filelist.raw.test.txt","./meta/aibo/labels.test.txt","test")
extract_features_into_file("./meta/aibo/filelist.raw.dev.txt","./meta/aibo/labels.dev.txt","dev")
extract_features_into_file("./meta/aibo/filelist.raw.train.downsample.txt","./meta/aibo/labels.train.downsample.txt","train")





