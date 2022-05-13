import os
import numpy as np

import torch
from models.x_vector import X_vector
from utils import utils

# Paths for check points
# PATH = "./save_model/best_check_point_15_0.3372245247165362"
PATH = "save_model/best_check_point_55_0.8169845943339169"
PATH_A = "save_model/best_check_point_32_0.7193116955459118"
wav_aibo = "./meta/aibo/wav-aibo/"

#Settins for x_Vector
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = X_vector(257, 12).to(device)
checkpoint = torch.load(os.path.join(PATH))
model.load_state_dict(checkpoint['model'])
layer = model.modules.get('segment6')


def get_vector(raw_wav, mode):
    """
    :param raw_wav: A single audio wav files
    :param mode: 'train' or 'test', it is needed to determine how the wav should be process
    :return: the feature x-vector which is (1, 512) size
    """
    spec = utils.load_data(raw_wav, mode=mode)
    sample = {'features': torch.from_numpy(np.ascontiguousarray(spec)), }

    model.eval()
    my_embedding = torch.zeros(1, 512)

    def copy_data(m, i, o):
        my_embedding.copy_(o.data)

    h = layer.register_forward_hook(copy_data)

    features = torch.from_numpy(np.asarray([sample['features'].numpy().T]))
    model(features.to(device))

    h.remove()
    return my_embedding


def extract_features_into_file(raw, label, mode, spec_mode):
    """
    :param raw: The '*.txt' file which contains the audio path's
    :param label: The '*.txt' file which contains the label for the audio's
    :param mode: 'train' or 'test' or 'dev' for making the .npy files into these directories
    :param spec_mode: 'train' or 'test', it is needed to determine how the wav should be process
    """
    filesDone = 0
    files = open(raw, "r")
    labels = open(label, "r")
    dir = ""
    if (mode == 'train'):
        dir = "./meta/aibo/train_downsample/"
    if (mode == "dev"):
        dir = "./meta/aibo/dev/"
    if (mode == "test"):
        dir = "./meta/aibo/test/"
    for i, (key1, key2) in enumerate(zip(files, labels)):
        feature = get_vector(wav_aibo + key1.strip() + ".wav", spec_mode)

        np.save(dir + key1.strip() + "_#" + key2.strip(), feature)
        filesDone += 1
        if (filesDone % 100 == 0):
            print(filesDone)

    files.close()
    labels.close()


extract_features_into_file("./meta/aibo/filelist.raw.test.txt", "./meta/aibo/labels.test.txt", "test", "test")
extract_features_into_file("./meta/aibo/filelist.raw.dev.txt", "./meta/aibo/labels.dev.txt", "dev", "train")
extract_features_into_file("./meta/aibo/filelist.raw.train.downsample.txt", "./meta/aibo/labels.train.downsample.txt", "train", "train")
