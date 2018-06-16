# This directory contains the code we used for our SeaRNN ICLR 2018 [paper](https://openreview.net/forum?id=HkUR_y-RZ)

This code is an open-source (MIT) implementation of SeaRNN. It is rather sparsely documented, so you are welcome to ask us more details using issues.


Table of Contents
=================

  * [Installation](#installation)
  * [Running](#running)
  * [Citation](#citation)

## Installation
First, set up a virtualenv to install the dependencies of the project.
The project uses Python 3 and was written with PyTorch 0.2 in mind, although it also works with PyTorch 0.3 for NMT.
You can replace the version numbers in the following commands to suit your architecture.

```bash
virtualenv -p /usr/bin/python3.5 --system-site-packages /path/to/virtualenv
pip3 install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp35-cp35m-linux_x86_64.whl
pip3 install torchvision
pip3 install numpy --upgrade
pip3 install nltk
pip3 install cython
```

Second, download the code:
```bash
git clone git@github.com:RemiLeblond/SeaRNN-open.git
cd SeaRNN-open
```

Next, compile the cython files

```bash
python setup.py build_ext --inplace
```

Finally, download the data at http://www.di.ens.fr/sierra/research/SEARNN/ and preprocess it:

```bash
scripts/prepare_iwlst14_de-en.sh
```

## Running

### Step 1: Train the model
OCR
```bash
python main_seq2seq.py --dataset ocr --dataroot /path/to/OCR --rollin learned --rollout mixed --objective target-learning --log_path /path/to/save
```

NMT (the standard MLE training)
```bash
python main_seq2seq.py --dataset nmt --dataroot /path/to/iwlst14_de-en_train_dev.train.pt --rollin gt --objective mle --log_path /path/to/save
```

Various parameters can be tuned, including the rollin and rollout policies, the objective etc.
See main_seq2seq.py for a complete description.

### Step 2: Evaluate.

```bash
python main_seq2seq.py --dataset nmt --dataroot /path/to/iwlst14_de-en_train_test.train.pt --max_iter 0 --print_iter 1 --checkpoint_file /path/to/checkpoint_file.pth
```
The arguments must be coherent with those used for training the model (such as the size of the hidden state of the RNN, whether the encoder is bidirectional or not...), otherwise the model loading will break.

To reproduce the NMT experiments of the paper, see scripts/training.sh


## Citation

```
@inproceedings{searnn2018leblond,
  author    = {Leblond, R\'emi and
               Alayrac, Jean-Baptiste and
               Osokin, Anton and
               Lacoste-Julien, Simon},
  title     = {SEARNN: Training RNNs with global-local losses},
  booktitle = {ICLR},
  year      = {2018},
}
```
