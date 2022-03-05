import numpy as np
import torch
from utils import utils
import os
from pathlib import Path
import random


class SpeechDataGenerator():
    """Speech dataset."""

    def __init__(self, dataset_audio_path, mode):
        self.mode = mode
        self.audio_links = []
        self.labels = []
        self.i = 0
        self.class_name = 0

        class_names = os.listdir(dataset_audio_path)
        for label, name in enumerate(class_names):
            dir_path = Path(dataset_audio_path) / name
            speaker_sample_paths = [
                os.path.join(dir_path, filepath)
                for filepath in os.listdir(dir_path)
                if filepath.endswith(".wav")
            ]
            self.audio_links += speaker_sample_paths
            self.labels += [self.class_name] * len(speaker_sample_paths)
            self.class_name += 1

        print("Classes: " + str(len(set(self.labels))))

    def __len__(self):
        return len(self.audio_links)

    def __getitem__(self, idx):
        audio_link = self.audio_links[idx]
        class_id = self.labels[idx]
        spec = utils.load_data(audio_link, mode=self.mode)
        sample = {'features': torch.from_numpy(np.ascontiguousarray(spec)),
                  'labels': torch.from_numpy(np.ascontiguousarray(class_id))}
        return sample
