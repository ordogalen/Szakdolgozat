from torch import optim
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
import os
import numpy as np

from SpeechDataGenerator import SpeechDataGenerator2
import torch
from models.x_vector_Indian_LID import X_vector
from utils import utils
from utils.utils import speech_collate

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


a = get_vector(raw_wav="./meta/Feature1/001/bea001f001_0002_001.wav")
print(a)


