"""
@author: ordogalen
"""

import os
import glob
import shutil


def create_directories(files_list, mode):
    wav_files = os.listdir(files_list)
    for file in wav_files:
        audio_link = files_list + "/" + file
        label = file.split("_")[0][7:10]
        if not os.path.exists("meta/bea/speakers_2/" + label):
            os.makedirs("meta/bea/speakers_2/" + label)
        shutil.copy(audio_link, "meta/bea/speakers_2/" + label)


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
    train_nums = int(len(all_lang_folders) * 0.8)

    for i in range(train_nums):
        speaker_id = all_lang_folders[i].split('\\')[1]
        all_files = sorted(glob.glob(all_lang_folders[i] + '\*'))
        for audio_filepath in all_files:
            to_write = audio_filepath + ' ' + str(speaker_id)
            train_lists.append(to_write)

    for i in range(train_nums, train_nums + int(len(all_lang_folders) * 0.2)):
        speaker_id = all_lang_folders[i].split('\\')[1]
        all_files = sorted(glob.glob(all_lang_folders[i] + '\*'))
        for audio_filepath in all_files:
            to_write = audio_filepath + ' ' + str(speaker_id)
            test_lists.append(to_write)

    return train_lists, test_lists


if __name__ == '__main__':
    create_directories("meta/bea/bea_files_1", "meta/bea")
