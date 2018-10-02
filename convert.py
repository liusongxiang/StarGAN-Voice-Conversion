import argparse
from model import Generator
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import numpy as np
import os
from os.path import join, basename, dirname, split
import time
import datetime
from data_loader import to_categorical
import librosa
from utils import *
import glob
# from data_loader import TestDataset, TestDataset2

spk2acc = {'262': 'Edinburgh', #F
           '272': 'Edinburgh', #M
           '229': 'SouthEngland', #F 
           '232': 'SouthEngland', #M
           '292': 'NorthernIrishBelfast', #M 
           '293': 'NorthernIrishBelfast', #F 
           '360': 'AmericanNewJersey', #M
           '361': 'AmericanNewJersey', #F
           '248': 'India', #F
           '251': 'India'} #M

speakers = ['p262', 'p272', 'p229', 'p232', 'p292', 'p293', 'p360', 'p361', 'p226', 'p248']
spk2idx = dict(zip(speakers, range(len(speakers))))
speakers_casia = ['liuchanhg', 'wangzhe', 'zhaoquanyin', 'ZhaoZuoxiang']
speakers_comb = ['FTHU', 'FZHILING', 'MTHU', 'MWH']


class TestDataset(object):
    """Dataset for testing."""
    def __init__(self, config, src_spk='p262', src_emo='neutral', trg_spk='p272'):
        """Note: src_spk can be chosen from casia, comb and english, 
                while trg_spk should be chosen from casia and comb.
        """
        assert trg_spk in speakers, f'The trg_spk should be chosen from casia and comb, but you choose {trg_spk}.'
        # Source speaker
        self.src_spk = src_spk
        self.trg_spk = trg_spk

        if src_spk in speakers_casia:
            self.src_spk_emo = f'{src_spk}-{src_emo}'
            self.mc_files = sorted(glob.glob(join(config.test_data_dir_casia, f'{self.src_spk_emo}*.npy')))
            self.src_spk_stats = np.load(join(config.train_data_dir_casia, f'{self.src_spk_emo}_stats.npz'))
            self.src_wav_dir = f'/scratch/sxliu/data_exp/CASIA_database/wav16/{src_spk}/{src_emo}'
        elif src_spk in speakers_comb:
            self.src_spk_emo = f'{src_spk}-null'
            self.mc_files = sorted(glob.glob(join(config.test_data_dir_comb, f'{src_spk}*.npy')))
            self.src_spk_stats = np.load(join(config.train_data_dir_comb, f'{src_spk}_stats.npz'))
            self.src_wav_dir = f'/scratch/sxliu/data_exp/mandarin_full/wav16/{src_spk}'
        else:
            self.src_spk_emo = f'{src_spk}-null'
            self.mc_files = sorted(glob.glob(join(config.test_data_dir_english, f'{src_spk}*.npy')))
            self.src_spk_stats = np.load(join(config.train_data_dir_english, f'{src_spk}_stats.npz'))
            self.src_wav_dir = f'/scratch/sxliu/data_exp/VCTK-Corpus-16k/wav48/{src_spk}'

        
        self.trg_spk_stats = np.load(join(config.train_data_dir_english, f'{trg_spk}_stats.npz'))

        self.logf0s_mean_src = self.src_spk_stats['log_f0s_mean']
        self.logf0s_std_src = self.src_spk_stats['log_f0s_std']
        self.logf0s_mean_trg = self.trg_spk_stats['log_f0s_mean']
        self.logf0s_std_trg = self.trg_spk_stats['log_f0s_std']
        self.mcep_mean_src = self.src_spk_stats['coded_sps_mean']
        self.mcep_std_src = self.src_spk_stats['coded_sps_std']
        self.mcep_mean_trg = self.trg_spk_stats['coded_sps_mean']
        self.mcep_std_trg = self.trg_spk_stats['coded_sps_std']
        
        self.spk_idx = spk2idx[trg_spk]
        spk_cat = to_categorical([self.spk_idx], num_classes=len(speakers))
        self.spk_c_trg = spk_cat


    def get_batch_test_data(self, batch_size=4):
        batch_data = []
        cur_batch_size = 5 if self.src_spk in speakers_casia else batch_size 
        for i in range(cur_batch_size):
            mcfile = self.mc_files[i]
            filename = basename(mcfile).split('-')[-1]
            wavfile_path = join(self.src_wav_dir, filename.replace('npy', 'wav'))
            batch_data.append(wavfile_path)
        return batch_data 


def load_wav(wavfile, sr=16000):
    wav, _ = librosa.load(wavfile, sr=sr, mono=True)
    return wav_padding(wav, sr=sr, frame_period=5, multiple = 4)  # TODO
    # return wav

def test(config):
    os.makedirs(join(config.convert_dir, str(config.resume_iters)), exist_ok=True)
    sampling_rate, num_mcep, frame_period=16000, 36, 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    G = Generator().to(device)
    test_loader = TestDataset(config)
    # Restore model
    print(f'Loading the trained models from step {config.resume_iters}...')
    G_path = join(config.model_save_dir, f'{config.resume_iters}-G.ckpt')
    G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))

    # Read a batch of testdata
    test_wavfiles = test_loader.get_batch_test_data(batch_size=8)
    test_wavs = [load_wav(wavfile, sampling_rate) for wavfile in test_wavfiles]

    with torch.no_grad():
        for idx, wav in enumerate(test_wavs):
            print(len(wav))
            wav_name = basename(test_wavfiles[idx])
            # print(wav_name)
            f0, timeaxis, sp, ap = world_decompose(wav=wav, fs=sampling_rate, frame_period=frame_period)
            f0_converted = pitch_conversion(f0=f0, 
                mean_log_src=test_loader.logf0s_mean_src, std_log_src=test_loader.logf0s_std_src, 
                mean_log_target=test_loader.logf0s_mean_trg, std_log_target=test_loader.logf0s_std_trg)
            coded_sp = world_encode_spectral_envelop(sp=sp, fs=sampling_rate, dim=num_mcep)
            print("Before being fed into G: ", coded_sp.shape)
            coded_sp_norm = (coded_sp - test_loader.mcep_mean_src) / test_loader.mcep_std_src
            coded_sp_norm_tensor = torch.FloatTensor(coded_sp_norm.T).unsqueeze_(0).unsqueeze_(1).to(device)
            spk_conds = torch.FloatTensor(test_loader.spk_c_trg).to(device)
            # print(spk_conds.size())
            coded_sp_converted_norm = G(coded_sp_norm_tensor, spk_conds).data.cpu().numpy()
            coded_sp_converted = np.squeeze(coded_sp_converted_norm).T * test_loader.mcep_std_trg + test_loader.mcep_mean_trg
            coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
            print("After being fed into G: ", coded_sp_converted.shape)
            wav_transformed = world_speech_synthesis(f0=f0_converted, coded_sp=coded_sp_converted, 
                                                    ap=ap, fs=sampling_rate, frame_period=frame_period)
            wav_id = wav_name.split('.')[0]
            librosa.output.write_wav(join(config.convert_dir, str(config.resume_iters),
                f'{test_loader.src_spk_emo}-{wav_id}-vcto-{test_loader.trg_spk}.wav'), wav_transformed, sampling_rate)
            if [True, False][0]:
                wav_cpsyn = world_speech_synthesis(f0=f0, coded_sp=coded_sp, 
                                                ap=ap, fs=sampling_rate, frame_period=frame_period)
                librosa.output.write_wav(join(config.convert_dir, str(config.resume_iters), f'{test_loader.src_spk_emo}-cpsyn-{wav_name}'), wav_cpsyn, sampling_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--num_speakers', type=int, default=10, help='dimension of speaker labels')
    parser.add_argument('--num_emotions', type=int, default=6, help='dimension of emotions labels')
    parser.add_argument('--resume_iters', type=int, default=73000, help='resume training from this step')

    # Directories.
    parser.add_argument('--train_data_dir_casia', type=str, default='/scratch/sxliu/data_exp/CASIA_database/mc/train')
    parser.add_argument('--test_data_dir_casia', type=str, default='/scratch/sxliu/data_exp/CASIA_database/mc/test')
    parser.add_argument('--train_data_dir_comb', type=str, default='/scratch/sxliu/data_exp/mandarin/mc/train')
    parser.add_argument('--test_data_dir_comb', type=str, default='/scratch/sxliu/data_exp/mandarin/mc/test')
    parser.add_argument('--test_data_dir_english', type=str, default='/scratch/sxliu/data_exp/VCTK-Corpus-16k/mc/train')
    parser.add_argument('--train_data_dir_english', type=str, default='/scratch/sxliu/data_exp/VCTK-Corpus-16k/mc/train')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--model_save_dir', type=str, default='./models')
    parser.add_argument('--convert_dir', type=str, default='./converted')


    config = parser.parse_args()
    print(config)
    test(config)
