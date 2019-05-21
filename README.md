# StarGAN-Voice-Conversion
This is a pytorch implementation of the paper: StarGAN-VC: Non-parallel many-to-many voice conversion with star generative adversarial networks  https://arxiv.org/abs/1806.02169 .
Note that the model architecture is a little different from that of the original paper.

# Dependencies
See requirements.txt.  
* Python 3.6
* Pytorch 0.4.0
* pyworld
* tqdm
* librosa
* tensorboardX and tensorboard

# Usage
## Download Dataset

Download and unzip [VCTK](https://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html) corpus to designated directories.

```bash
mkdir ./data
wget https://datashare.is.ed.ac.uk/bitstream/handle/10283/2651/VCTK-Corpus.zip?sequence=2&isAllowed=y
unzip VCTK-Corpus.zip -d ./data
```
If the downloaded VCTK is in tar.gz, run this:

```bash
tar -xzvf VCTK-Corpus.tar.gz -C ./data
```

Set utils.VCTK_PATH to the location of the extracted VCTK directory (e.g. `"./data/VCTK-Corpus"`)

## Preprocess data

We will use Mel-cepstral coefficients(MCEPs) here.

```bash
python preprocess.py
```

## Train model

Note: you may need to early stop the training process if the training-time test samples sounds good or the you can also see the training loss curves to determine early stop or not.

```
python main.py
```

## Convert

Set convert.MODEL_PATH to the location of the generator model (e.g. `"models/23000-G.ckpt"`). Run `python convert.py` or use `convert.convert(src_dir, trg_wav)` by passing in a directory to the source audio and a wav file of the target speaker. 

