#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 14:09:31 2019

@author: Krishna
"""
import numpy as np
import torch
from utils import utils
import os
from pathlib import Path

from torch.utils.data import DataLoader, Dataset, random_split


#class SpeechDataGenerator():
#    """Speech dataset."""
#
#    def __init__(self, manifest, mode):
#        """
#        Read the textfile and get the paths
#        """
#        self.mode = mode
#        self.audio_links = [line.rstrip('\n').split(' ')[0] for line in open(manifest)]
#        self.labels = [int(line.rstrip('\n').split(' ')[1]) for line in open(manifest)]
#
#    def __len__(self):
#        return len(self.audio_links)
#
#    def __getitem__(self, idx):
#        audio_link = self.audio_links[idx]
#        class_id = self.labels[idx]
#        # lang_label=lang_id[self.audio_links[idx].split('/')[-2]]
#        spec = utils.load_data(audio_link, mode=self.mode)
#        sample = {'features': torch.from_numpy(np.ascontiguousarray(spec)),
#                  'labels': torch.from_numpy(np.ascontiguousarray(class_id))}
#        return sample


class SpeechDataGenerator():
    """Speech dataset."""

    def __init__(self, dataset_audio_path, mode, shuffle_seed):
        self.mode = mode
        self.audio_links = []
        self.labels = []
        class_names = os.listdir(dataset_audio_path)
        for label, name in enumerate(class_names):
            dir_path = Path(dataset_audio_path) / name
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
        print("called")
        audio_link = self.audio_links[idx]
        class_id = self.labels[idx]
        spec = utils.load_data(audio_link, mode=self.mode)
        sample = {'features': torch.from_numpy(np.ascontiguousarray(spec)),
                  'labels': torch.from_numpy(np.ascontiguousarray(class_id))}
        return sample

