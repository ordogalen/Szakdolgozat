#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 19:09:44 2020
@author: krishna
"""

import os
import glob
import shutil
import random


import numpy


def unison_shuffled_copies(a, b):
    temp = list(zip(a, b))
    random.shuffle(temp)
    res1, res2 = zip(*temp)
    return res1, res2

def list_files_into_txt(files_list, txt):
    with open(file=txt, mode="w") as file:
        file.write(os.listdir(files_list))

def a(files_list, mode):
    wav_files = os.listdir(files_list)
    for file in wav_files:
        audio_link = files_list+"/" + file
        label = file.split("_")[0][7:10]
        if not os.path.exists("meta/bea/speakers/"+label):
            os.makedirs("meta/bea/speakers/"+label)
        shutil.copy(audio_link, "meta/bea/speakers/"+label)


def create_meta(files_list, store_loc, mode='train'):
    if not os.path.exists(store_loc):
        os.makedirs(store_loc)
    print(store_loc)
    print(files_list)
    if mode == 'train':
        print("asd")
        meta_store = store_loc + '/training.txt'
        fid = open(meta_store, 'w')
        for filepath in files_list:
            print(filepath)
            fid.write(filepath + '\n')
        fid.close()
    elif mode == 'test':
        meta_store = store_loc + '/testing.txt'
        fid = open(meta_store, 'w')
        for filepath in files_list:
            fid.write(filepath + '\n')
        fid.close()
    elif mode == 'validation':
        meta_store = store_loc + '/validation.txt'
        fid = open(meta_store, 'w')
        for filepath in files_list:
            fid.write(filepath + '\n')
        fid.close()
    else:
        print('Error in creating meta files')


def extract_files(folder_path):
    all_lang_folders = sorted(glob.glob(folder_path + '/*/'))
    train_lists = []
    test_lists = []
    train_nums = int(len(all_lang_folders)*0.8)

    for i in range(train_nums):
        speaker_id = all_lang_folders[i].split('\\')[1]
        all_files = sorted(glob.glob(all_lang_folders[i]+'\*'))
        for audio_filepath in all_files:
            to_write = audio_filepath + ' ' + str(speaker_id)
            train_lists.append(to_write)


    for i in range(train_nums, train_nums + int(len(all_lang_folders)*0.2)):
        speaker_id = all_lang_folders[i].split('\\')[1]
        all_files = sorted(glob.glob(all_lang_folders[i]+'\*'))
        for audio_filepath in all_files:
            to_write = audio_filepath + ' ' + str(speaker_id)
            test_lists.append(to_write)

    return train_lists, test_lists

if __name__ == '__main__':
    #parser = argparse.ArgumentParser("Configuration for data preparation")
    #parser.add_argument("--processed_data", default="meta/bea/bea_files_2", type=str,
    #                    help='Dataset path')
    #parser.add_argument("--meta_store_path", default="meta/bea", type=str, help='Save directory after processing')
    #config = parser.parse_args()
    #train_list, test_list, val_lists = extract_files(config.processed_data)
    #create_meta(train_list, config.meta_store_path, mode='train')
    #create_meta(test_list, config.meta_store_path, mode='test')
    #create_meta(val_lists, config.meta_store_path, mode='validation')
    #create_txt_files("meta/bea/bea_files_2","train")
    # #a("meta/bea/bea_files_2","meta/bea")
    # print(os.listdir("meta/bea/speakers"))
    # c = 0
    # for i in os.listdir("meta/bea/speakers"):
    #     c += 1
    # print(c)
    a = [1,2,3,4]
    b = [1,2,3,4]
    a,b = unison_shuffled_copies(a,b)
    print(a)
    print(b)


