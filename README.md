# StarGAN-Voice-Conversion
This is a pytorch implementation of the paper: StarGAN-VC: Non-parallel many-to-many voice conversion with star generative adversarial networks  https://arxiv.org/abs/1806.02169 .
Note that the model architecture is a little different from that of the original paper.

# Dependencies
* Python 3.6 (or 3.5)
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

Preprocess data

We will use Mel-cepstral coefficients(MCEPs) here.

```bash
python preprocess.py --sample_rate 16000 \
                    --origin_wavpath data/VCTK-Corpus/wav48 \
                    --target_wavpath data/VCTK-Corpus/wav16 \
                    --mc_dir_train data/mc/train \
                    --mc_dir_test data/mc/test
```

Train model

Note: you may need to early stop the training process if the training-time test samples sounds good or the you can also see the training loss curves to determine early stop or not.

```
python main.py
```

Convert

For example: restore model at step 200000 and specify the source speaker and target speaker to `p262` and `p272`, respectively.

```
convert.py --resume_iters 200000 --src_spk p262 --trg_spk p272
```

## To-Do list
- [x] Post some converted samples (Please find some converted samples in the `converted_samples` folder).

## Papers that use this repo:
1. [AUTOVC: Zero-Shot Voice Style Transfer with Only Autoencoder Loss (ICML2019)](https://arxiv.org/pdf/1905.05879v2.pdf)
2. [Blow: a single-scale hyperconditioned flow for non-parallel raw-audio voice conversion (NeurIPS 2019)](https://arxiv.org/pdf/1906.00794.pdf)
3. [ADAGAN: ADAPTIVE GAN FOR MANY-TO-MANY NON-PARALLEL VOICE CONVERSION (under review for ICLR 2020)](https://openreview.net/pdf?id=HJlk-eHFwH)

