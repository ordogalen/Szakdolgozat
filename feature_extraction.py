# -*- coding: utf-8 -*-
"""
Created on Sun May 31 11:15:47 2020
@author: krishna
"""

import os
import numpy as np
from utils import utils


def extract_features(audio_filepath):
    features = utils.feature_extraction(audio_filepath)
    return features


def FE_pipeline(feature_list, store_loc, mode):
    global fid
    create_root = os.path.join(store_loc, mode)
    if not os.path.exists(create_root):
        os.makedirs(create_root)
    if mode == 'train':
        fid = open('meta/bea/training_feat.txt', 'w')
    elif mode == 'test':
        fid = open('meta/bea/testing_feat.txt', 'w')
    elif mode == 'validation':
        fid = open('meta/bea/validation_feat.txt', 'w')
    else:
        print('Unknown mode')

    for row in feature_list:
        filepath = row.split(' ')[0]
        speaker_id = row.split(' ')[1]
        filename = filepath.split('\\')[2]

        create_folders = os.path.join(create_root)
        if not os.path.exists(create_folders):
            os.makedirs(create_folders)
        extract_feats = extract_features(filepath)
        dest_filepath = create_folders + '/' + filename[:-4] + '.npy'
        np.save(dest_filepath, extract_feats)
        to_write = dest_filepath + ' ' + speaker_id
        fid.write(to_write + '\n')
    fid.close()


if __name__ == '__main__':
    store_loc = 'meta/Features'
    read_train = [line.rstrip('\n') for line in open('meta/bea/training.txt')]
    FE_pipeline(read_train, store_loc, mode='train')

    read_test = [line.rstrip('\n') for line in open('meta/bea/testing.txt')]
    FE_pipeline(read_test, store_loc, mode='test')

    read_val = [line.rstrip('\n') for line in open('meta/bea/validation.txt')]
    FE_pipeline(read_val, store_loc, mode='validation')

