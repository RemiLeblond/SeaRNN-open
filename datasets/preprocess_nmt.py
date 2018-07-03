import argparse

import torch

from .dict import Dict

parser = argparse.ArgumentParser(description='preprocess.py')

"""
    Preprocessing options (borrowed from OpenNMT-py, https://github.com/OpenNMT/OpenNMT-py)
"""


parser.add_argument('-config',    help="Read options from this file")

parser.add_argument('-train_src', required=True, help="Path to the training source data")
parser.add_argument('-train_tgt', required=True, help="Path to the training target data")
parser.add_argument('-valid_src', required=True, help="Path to the validation source data")
parser.add_argument('-valid_tgt', required=True, help="Path to the validation target data")

parser.add_argument('-save_data', required=True, help="Output file for the prepared data")

parser.add_argument('-src_vocab_size', type=int, default=50000, help="Size of the source vocabulary")
parser.add_argument('-tgt_vocab_size', type=int, default=50000, help="Size of the target vocabulary")
parser.add_argument('-src_vocab', help="Path to an existing source vocabulary")
parser.add_argument('-tgt_vocab', help="Path to an existing target vocabulary")

parser.add_argument('-seq_length', type=int, default=50, help="Maximum sequence length")
parser.add_argument('-shuffle',    type=int, default=0, help="Shuffle data")
parser.add_argument('-sort_data',    type=int, default=0, help="Sort data by the length increase")
parser.add_argument('-seed',       type=int, default=3435, help="Random seed")

parser.add_argument('-lower', action='store_true', help='lowercase data')

parser.add_argument('-report_every', type=int, default=100000, help="Report status every this many sentences")

opt = parser.parse_args()

torch.manual_seed(opt.seed)

PAD = 0
UNK = 1
BOS = 2
EOS = 3

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'


def make_vocabulary(filename, size):
    vocab = Dict([PAD_WORD, UNK_WORD,
                  BOS_WORD, EOS_WORD], lower=opt.lower)

    with open(filename) as f:
        for sent in f.readlines():
            for word in sent.split():
                vocab.add(word)

    original_size = vocab.size()
    vocab = vocab.prune(size)
    print('Created dictionary of size %d (pruned from %d)' %
          (vocab.size(), original_size))

    return vocab


def init_vocabulary(name, data_file, vocab_file, vocab_size):
    vocab = None
    if vocab_file is not None:
        # If given, load existing word dictionary.
        print('Reading ' + name + ' vocabulary from \'' + vocab_file + '\'...')
        vocab = Dict()
        vocab.load_file(vocab_file)
        print('Loaded ' + str(vocab.size()) + ' ' + name + ' words')

    if vocab is None:
        # If a dictionary is still missing, generate it.
        print('Building ' + name + ' vocabulary...')
        gen_word_vocab = make_vocabulary(data_file, vocab_size)

        vocab = gen_word_vocab

    print()
    return vocab


def save_vocabulary(name, vocab, file):
    print('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.write_file(file)


def make_data(src_file, tgt_file, src_dicts, tgt_dicts):
    src, tgt = [], []
    sizes = []
    count, ignored = 0, 0

    print('Processing %s & %s ...' % (src_file, tgt_file))
    src_f = open(src_file)
    tgt_f = open(tgt_file)

    while True:
        src_words = src_f.readline().split()
        tgt_words = tgt_f.readline().split()

        if not src_words or not tgt_words:
            if src_words and not tgt_words or not src_words and tgt_words:
                print('WARNING: source and target do not have the same number of sentences')
            break

        if len(src_words) <= opt.seq_length and len(tgt_words) <= opt.seq_length:

            src += [src_dicts.convert_to_idx(src_words, UNK_WORD)]
            tgt += [tgt_dicts.convert_to_idx(tgt_words, UNK_WORD, BOS_WORD, EOS_WORD)]

            sizes += [len(src_words)]
        else:
            ignored += 1

        count += 1

        if count % opt.report_every == 0:
            print('... %d sentences prepared' % count)

    src_f.close()
    tgt_f.close()

    if opt.shuffle == 1:
        print('... shuffling sentences')
        perm = torch.randperm(len(src))
        src = [src[idx] for idx in perm]
        tgt = [tgt[idx] for idx in perm]
        sizes = [sizes[idx] for idx in perm]

    if opt.sort_data == 1:
        print('... sorting sentences by size')
        _, perm = torch.sort(torch.Tensor(sizes))
        src = [src[idx] for idx in perm]
        tgt = [tgt[idx] for idx in perm]

    print('Prepared %d sentences (%d ignored due to length == 0 or > %d)' %
          (len(src), ignored, opt.seq_length))

    return src, tgt


def main():
    dicts = {}
    dicts['src'] = init_vocabulary('source', opt.train_src, opt.src_vocab, opt.src_vocab_size)
    dicts['tgt'] = init_vocabulary('target', opt.train_tgt, opt.tgt_vocab, opt.tgt_vocab_size)

    print('Preparing training ...')
    train = {}
    train['src'], train['tgt'] = make_data(opt.train_src, opt.train_tgt, dicts['src'], dicts['tgt'])

    print('Preparing validation ...')
    valid = {}
    valid['src'], valid['tgt'] = make_data(opt.valid_src, opt.valid_tgt, dicts['src'], dicts['tgt'])

    if opt.src_vocab is None:
        save_vocabulary('source', dicts['src'], opt.save_data + '.src.dict')
    if opt.tgt_vocab is None:
        save_vocabulary('target', dicts['tgt'], opt.save_data + '.tgt.dict')

    print('Saving data to \'' + opt.save_data + '.train.pt\'...')
    save_data = {'dicts': dicts,
                 'train': train,
                 'valid': valid}
    torch.save(save_data, opt.save_data + '.train.pt')


if __name__ == "__main__":
    main()
