

from torch.utils.data import DataLoader
import os
import numpy as np
from pathlib import Path
from utils import utils
import torch

class SpeechDataGenerator():
    """Speech dataset."""

    def __init__(self, class_names, mode, shuffle_seed):
        self.mode = mode
        self.audio_links = []
        self.labels = []
        for label, name in enumerate(class_names):
            dir_path = Path(DATASET_AUDIO_PATH) / name
            speaker_sample_paths = [
                os.path.join(dir_path, filepath)
                for filepath in os.listdir(dir_path)
                if filepath.endswith(".wav")
            ]
            self.audio_links += speaker_sample_paths
            self.labels += [label] * len(speaker_sample_paths)
            rng = np.random.RandomState(shuffle_seed)
            rng.shuffle(self.audio_links)
            rng = np.random.RandomState(shuffle_seed)
            rng.shuffle(self.labels)

    def __len__(self):
        return len(self.audio_links)

    def __getitem__(self, idx):
        audio_link = self.audio_links[idx]
        class_id = self.labels[idx]
        # lang_label=lang_id[self.audio_links[idx].split('/')[-2]]
        spec = utils.load_data(audio_link, mode=self.mode)
        sample = {'features': torch.from_numpy(np.ascontiguousarray(spec)),
                  'labels': torch.from_numpy(np.ascontiguousarray(class_id))}
        return sample


DATASET_AUDIO_PATH = "./meta/speakers"
class_names = os.listdir(DATASET_AUDIO_PATH)
a = SpeechDataGenerator(class_names,"train",41)
print(a[5])






