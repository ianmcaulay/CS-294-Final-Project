from model import Generator
import torch
import numpy as np
import os
from data_loader import to_categorical
import librosa
import utils
import glob
from pathlib import Path
from os.path import join, basename

# Set MODEL_PATH here. e.g. MODEL_PATH = 'models/20190328_first_run/23000-G.ckpt'
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
    for wav_file in wav_files:
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
    src_wav_files = glob.glob(f'{src_wav_dir}/*.wav')
    src_wavs = [utils.load_wav(src_wav_file, utils.SAMPLING_RATE) for src_wav_file in src_wav_files]
    trg_wav = utils.load_wav(trg_wav_file, utils.SAMPLING_RATE)
    converted_dir = src_dir.parent.joinpath('converted_audio')
    os.makedirs(converted_dir, exist_ok=True)

    src_stats = get_stats(src_wav_files)
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
        for i, src_wav in enumerate(src_wavs):
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

            output_path = converted_dir.joinpath(os.path.basename(src_wav_files[i]))
            print(f'Saving to {output_path}')
            librosa.output.write_wav(output_path, wav_transformed, utils.SAMPLING_RATE)
            # wav_cpsyn = world_speech_synthesis(f0=f0, coded_sp=coded_sp,
            #                                 ap=ap, fs=sampling_rate, frame_period=frame_period)
            # librosa.output.write_wav(join(convert_dir, str(resume_iters), f'cpsyn-{wav_name}'), wav_cpsyn, sampling_rate)


def get_model(device):
    G = Generator().to(device)
    # test_loader = TestDataset(config)
    # Restore model
    print(f'Loading the trained models from {MODEL_PATH}...')
    G.load_state_dict(torch.load(MODEL_PATH, map_location=lambda storage, loc: storage))
    return G


if __name__ == '__main__':
    src_dir = Path('data/synth_audio/testing')
    trg_wav = Path('data/concatted_audio/wav/p262/p262_concatted.wav')
    convert(src_dir, trg_wav)
#     parser = argparse.ArgumentParser()
#
#     # Model configuration.
#     parser.add_argument('--num_speakers', type=int, default=10, help='dimension of speaker labels')
#     parser.add_argument('--num_converted_wavs', type=int, default=8, help='number of wavs to convert.')
#     parser.add_argument('--resume_iters', type=int, default=None, help='step to resume for testing.')
#     parser.add_argument('--src_spk', type=str, default='p262', help='target speaker.')
#     parser.add_argument('--trg_spk', type=str, default='p272', help='target speaker.')
#
#     # Directories.
#     parser.add_argument('--train_data_dir', type=str, default='./data/mc/train')
#     parser.add_argument('--test_data_dir', type=str, default='./data/mc/test')
#     parser.add_argument('--wav_dir', type=str, default="./data/VCTK-Corpus/wav16")
#     parser.add_argument('--log_dir', type=str, default='./logs')
#     parser.add_argument('--model_save_dir', type=str, default='./models')
#     parser.add_argument('--convert_dir', type=str, default='./converted')
#
#     config = parser.parse_args()
#
#     print(config)
#     if config.resume_iters is None:
#         raise RuntimeError("Please specify the step number for resuming.")
#     test(config)

