import torch
import numpy as np
import os
import librosa
import utils
import glob
from pathlib import Path
from os.path import join, basename, splitext
from tqdm import tqdm

from model import Generator
from data_loader import to_categorical
from utils import VCTK_PATH

# Set MODEL_PATH here. e.g. MODEL_PATH = 'models/modified_stargans/modified_23000-G.ckpt'
MODEL_PATH = None

# Below is the accent info for the used 10 speakers.
spk2acc = {'262': 'Edinburgh',  # F
           '272': 'Edinburgh',  # M
           '229': 'SouthEngland',  # F
           '232': 'SouthEngland',  # M
           '292': 'NorthernIrishBelfast',  # M
           '293': 'NorthernIrishBelfast',  # F
           '360': 'AmericanNewJersey',  # M
           '361': 'AmericanNewJersey',  # F
           '248': 'India',  # F
           '251': 'India'}  # M

speakers = ['p262', 'p272', 'p229', 'p232', 'p292', 'p293', 'p360', 'p361', 'p248', 'p251']
spk2idx = dict(zip(speakers, range(len(speakers))))


class TestDataset(object):
    """Dataset for testing."""

    def __init__(self, config):
        assert config.trg_spk in speakers, f'The trg_spk should be chosen from {speakers}, but you choose {config.trg_spk}.'
        # Source speaker
        self.src_spk = config.src_spk
        self.trg_spk = config.trg_spk

        self.mc_files = sorted(glob.glob(join(config.test_data_dir, f'{config.src_spk}*.npy')))
        self.src_spk_stats = np.load(join(config.train_data_dir, f'{config.src_spk}_stats.npz'))
        self.src_wav_dir = f'{config.wav_dir}/{config.src_spk}'

        self.trg_spk_stats = np.load(join(config.train_data_dir, f'{config.trg_spk}_stats.npz'))

        self.logf0s_mean_src = self.src_spk_stats['log_f0s_mean']
        self.logf0s_std_src = self.src_spk_stats['log_f0s_std']
        self.logf0s_mean_trg = self.trg_spk_stats['log_f0s_mean']
        self.logf0s_std_trg = self.trg_spk_stats['log_f0s_std']
        self.mcep_mean_src = self.src_spk_stats['coded_sps_mean']
        self.mcep_std_src = self.src_spk_stats['coded_sps_std']
        self.mcep_mean_trg = self.trg_spk_stats['coded_sps_mean']
        self.mcep_std_trg = self.trg_spk_stats['coded_sps_std']

        self.spk_idx = spk2idx[config.trg_spk]
        spk_cat = to_categorical([self.spk_idx], num_classes=len(speakers))
        self.spk_c_trg = spk_cat

    def get_batch_test_data(self, batch_size=4):
        batch_data = []
        for i in range(batch_size):
            mcfile = self.mc_files[i]
            filename = basename(mcfile).split('-')[-1]
            wavfile_path = join(self.src_wav_dir, filename.replace('npy', 'wav'))
            batch_data.append(wavfile_path)
        return batch_data


def get_stats(wav_files):
    f0s = []
    coded_sps = []
    for wav_file in tqdm(wav_files):
        f0, _, _, _, coded_sp = utils.world_encode_wav(wav_file, fs=utils.SAMPLING_RATE)
        f0s.append(f0)
        coded_sps.append(coded_sp)
    log_f0s_mean, log_f0s_std = utils.logf0_statistics(f0s)
    coded_sps_mean, coded_sps_std = utils.coded_sp_statistics(coded_sps)
    return {
        'log_f0s_mean': log_f0s_mean,
        'log_f0s_std': log_f0s_std,
        'coded_sps_mean': coded_sps_mean,
        'coded_sps_std': coded_sps_std
    }


def convert(src_wav_dir, trg_wav_file):
    all_src_wav_files = glob.glob(f'{src_wav_dir}/*.wav')
    # This regex for src_wav_files creates about 20 output files to get a good sample without taking too
    # much time or memory. It can be altered (including setting to a single file or all_src_wav_files)
    # to create fewer/more output files.
    src_wav_files = glob.glob(f'{src_wav_dir}/p???_0[01][0-9].wav')
    src_wavs = [utils.load_wav(src_wav_file, utils.SAMPLING_RATE) for src_wav_file in src_wav_files]
    trg_wav = utils.load_wav(trg_wav_file, utils.SAMPLING_RATE)
    trg_wav_name = splitext(basename(trg_wav_file))[0]
    converted_dir = VCTK_PATH.joinpath('converted_audio', 'trg_' + trg_wav_name)
    os.makedirs(converted_dir, exist_ok=True)

    src_stats = get_stats(all_src_wav_files)
    trg_stats = get_stats([trg_wav_file])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = get_model(device)

    _, _, trg_sp, _ = utils.world_decompose(wav=trg_wav, fs=utils.SAMPLING_RATE, frame_period=utils.FRAME_PERIOD)
    trg_coded_sp = utils.world_encode_spectral_envelop(sp=trg_sp, fs=utils.SAMPLING_RATE, dim=utils.NUM_MCEP)
    trg_coded_sp_norm = (trg_coded_sp - trg_stats['coded_sps_mean']) / trg_stats['coded_sps_std']
    assert trg_coded_sp_norm.shape[0] >= 8192
    trg_coded_sp_norm = trg_coded_sp_norm[:8192, :]
    trg_coded_sp_norm_tensor = torch.FloatTensor(trg_coded_sp_norm.T).unsqueeze_(0).unsqueeze_(1).to(device)

    trg_embed = G.trg_downsample(trg_coded_sp_norm_tensor)

    with torch.no_grad():
        for i, src_wav in enumerate(tqdm(src_wavs)):
            f0, _, sp, ap = utils.world_decompose(wav=src_wav, fs=utils.SAMPLING_RATE,
                                                  frame_period=utils.FRAME_PERIOD)
            coded_sp = utils.world_encode_spectral_envelop(sp=sp, fs=utils.SAMPLING_RATE, dim=utils.NUM_MCEP)

            f0_converted = utils.pitch_conversion(
                f0=f0,
                mean_log_src=src_stats['log_f0s_mean'],
                std_log_src=src_stats['log_f0s_std'],
                mean_log_target=trg_stats['log_f0s_mean'],
                std_log_target=trg_stats['log_f0s_std'])

            coded_sp_norm = (coded_sp - src_stats['coded_sps_mean']) / src_stats['coded_sps_std']
            coded_sp_norm_tensor = torch.FloatTensor(coded_sp_norm.T).unsqueeze_(0).unsqueeze_(1).to(device)

            # coded_sp_converted_norm = G(coded_sp_norm_tensor, trg_embed).data.cpu().numpy()
            coded_sp_converted_norm = G.forward_with_trg_embed(coded_sp_norm_tensor, trg_embed)
            coded_sp_converted_norm = coded_sp_converted_norm.data.cpu().numpy()
            coded_sp_converted = np.squeeze(coded_sp_converted_norm).T
            coded_sp_converted = coded_sp_converted * trg_stats['coded_sps_std'] + trg_stats['coded_sps_mean']
            coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
            coded_sp_converted = coded_sp_converted.astype('double')
            wav_transformed = utils.world_speech_synthesis(
                f0=f0_converted,
                coded_sp=coded_sp_converted,
                ap=ap,
                fs=utils.SAMPLING_RATE,
                frame_period=utils.FRAME_PERIOD)

            output_path = converted_dir.joinpath('src_' + os.path.basename(src_wav_files[i]))
            print(f'Saving to {output_path}')
            librosa.output.write_wav(output_path, wav_transformed, utils.SAMPLING_RATE)
            # wav_cpsyn = world_speech_synthesis(f0=f0, coded_sp=coded_sp,
            #                                 ap=ap, fs=sampling_rate, frame_period=frame_period)
            # librosa.output.write_wav(join(convert_dir, str(resume_iters), f'cpsyn-{wav_name}'), wav_cpsyn, sampling_rate)


def get_model(device):
    G = Generator().to(device)
    # test_loader = TestDataset(config)
    # Restore model
    print(f'Loading the trained model from {MODEL_PATH}...')
    G.load_state_dict(torch.load(MODEL_PATH, map_location=lambda storage, loc: storage))
    return G


if __name__ == '__main__':
    # src_dir = VCTK_PATH.joinpath('wav16', 'p262')  # Src in training data
    src_dir = VCTK_PATH.joinpath('wav16', 'p226')  # Src out of training data
    # trg_wav = VCTK_PATH.joinpath('concatted_audio', 'wav', 'p272', 'p272_concatted.wav')  # Trg in training data
    trg_wav = VCTK_PATH.joinpath('concatted_audio', 'wav', 'p226', 'p226_concatted.wav')  # Trg out of training data
    assert src_dir.exists()
    assert trg_wav.exists()
    convert(src_dir, trg_wav)
