import librosa
import numpy as np
import os, sys
import pyworld
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from utils import *
from tqdm import tqdm
from collections import defaultdict
from collections import namedtuple
from sklearn.model_selection import train_test_split
import glob
from os.path import join, basename


def split_data(paths):
    indices = np.arange(len(paths))
    test_size = 0.1
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=1234)
    train_paths = list(np.array(paths)[train_indices])
    test_paths = list(np.array(paths)[test_indices])
    return train_paths, test_paths

def get_spk_world_feats(spk_fold_path):
    paths = glob.glob(join(spk_fold_path, '*.wav'))
    spk_name = basename(spk_fold_path)
    train_paths, test_paths = split_data(paths)
    f0s = []
    coded_sps = []
    for wav_file in train_paths:
        f0, _, _, _, coded_sp = world_encode_wav(wav_file, fs=sample_rate)
        f0s.append(f0)
        coded_sps.append(coded_sp)
    log_f0s_mean, log_f0s_std = logf0_statistics(f0s)
    coded_sps_mean, coded_sps_std = coded_sp_statistics(coded_sps)
    np.savez(join(mc_dir_train, spk_name+'_stats.npz'), 
            log_f0s_mean=log_f0s_mean,
            log_f0s_std=log_f0s_std,
            coded_sps_mean=coded_sps_mean,
            coded_sps_std=coded_sps_std)
    
    for wav_file in train_paths:
        wav_nam = basename(wav_file)
        f0, timeaxis, sp, ap, coded_sp = world_encode_wav(wav_file, fs=sample_rate)
        normed_coded_sp = normalize_coded_sp(coded_sps, coded_sps_mean, coded_sps_std)
        np.save(join(mc_dir_train, spk_name + '_' + wav_nam.replace('.wav', '.npy')), normed_coded_sp, allow_pickle=False)
    
    for wav_file in test_paths:
        wav_nam = basename(wav_file)
        f0, timeaxis, sp, ap, coded_sp = world_encode_wav(wav_file, fs=sample_rate)
        normed_coded_sp = normalize_coded_sp(coded_sps, coded_sps_mean, coded_sps_std)
        np.save(join(mc_dir_test, spk_name + '_' + wav_nam.replace('.wav', '.npy')), normed_coded_sp, allow_pickle=False)
    return 0

if __name__ == '__main__':
    sample_rate = 16000
    mc_dir_train = './mc/train'
    os.makedirs(mc_dir_train, exist_ok=True)

    mc_dir_test = './mc/test'
    os.makedirs(mc_dir_test, exist_ok=True)

    num_workers = 32 #cpu_count()
    print("number of workers: ", num_workers)
    executor = ProcessPoolExecutor(max_workers=num_workers)

    work_dir = "./wav48"
    spk_folders = os.listdir(work_dir)
    print("processing {} speaker folders".format(len(spk_folders)))
    print(spk_folders)

    # print("normalizing mel ...")
    futures = []
    for spk in spk_folders:
        spk_path = os.path.join(work_dir, spk)
        futures.append(executor.submit(partial(get_spk_world_feats, spk_path)))
    result_list = [future.result() for future in tqdm(futures)]
    print(result_list)
    sys.exit(0)

