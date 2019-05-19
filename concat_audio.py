from pydub import AudioSegment
import glob
import os
from tqdm import tqdm

import data_loader
from preprocess import get_spk_world_feats
from utils import VCTK_PATH

# Maximum length of concatted file in seconds.
MAX_LEN = 60


def concat_wavs(wavs, out_filepath):
    out = AudioSegment.from_wav(wavs[0])
    for wav in wavs[1:]:
        out += AudioSegment.from_wav(wav)
        if len(out) / 1000 > MAX_LEN:
            break
    out.export(f'{out_filepath}.wav', format='wav')


def concat_for_all_speakers(speakers):
    for spk in tqdm(speakers):
        path = f'{VCTK_PATH}/wav16/{spk}'
        wavs = glob.glob(path + '/*.wav')
        out_dir = 'data/concatted_audio/wav/' + spk
        os.makedirs(out_dir, exist_ok=True)
        concat_wavs(wavs, out_dir + '/' + spk + '_concatted')


def create_mc(speakers):
    for spk in tqdm(speakers):
        spk_fold_path = 'data/concatted_audio/wav/' + spk
        mc_dir_train = 'data/concatted_audio/mc/'
        mc_dir_test = None  #'data/concatted_audio/mc/test'
        get_spk_world_feats(spk_fold_path, mc_dir_train, mc_dir_test, sample_rate=16000)


if __name__ == '__main__':
    print(data_loader.speakers)
    new_spk = ['p225', 'p226', 'p229', 'p232', 'p248', 'p251', 'p262', 'p272',
               'p292', 'p293', 'p300', 'p360', 'p361']

    # assert set(new_spk) & set(data_loader.speakers) == set()
    concat_for_all_speakers(new_spk)
    create_mc(new_spk)
