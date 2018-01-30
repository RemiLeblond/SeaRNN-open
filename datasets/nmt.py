import os
import nltk

import torch
from datasets.dataset_seq2seq import DatasetSeq2seq
from losses_utils import compute_prediction_lengths
from models import EmbeddingSeq2seq

"""
    NMT dataset helpers.
"""

# from OpenNMT-py (https://github.com/OpenNMT/OpenNMT-py).
PAD = 0
UNK = 1
BOS = 2
EOS = 3

numeric_const_pattern = "[-+]?(?:(?:\d*\.\d+)|(?:\d+\.?))(?:[Ee][+-]?\d+)?"


def _read_nmt_dataset(file_path, split='train'):
    print('Reading dataset file %s' % file_path)
    dataset = torch.load(file_path)

    print('Constructing %s set' % split)
    input_data = dataset[split]['src']
    output_data = dataset[split]['tgt']
    dicts = dataset['dicts']
    input_dict_size = dicts['src'].size()
    output_dict_size = dicts['tgt'].size()

    # chop off EOS and BOS symbols in the output
    for x in output_data:
        assert(x[0] == BOS and x[-1] == EOS)
    output_data = [x[1:-1] for x in output_data]

    print('Vocabulary size: source = %d; target = %d' % (input_dict_size, output_dict_size))
    print('Number of sentences: %d' % len(input_data))

    data_path = os.path.dirname(os.path.abspath(file_path))

    return input_data, output_data, dicts, input_dict_size, output_dict_size, data_path


class NmtDataset(DatasetSeq2seq):
    def __init__(self, file_path, batch_size, split='train', revert_input_sequence=True, input_embedding_size=128,
                 num_buckets=1, dataset_name='nmt', max_num_items=None):

        # read data
        input_data, output_data, dicts, input_dict_size, output_dict_size, data_path = _read_nmt_dataset(file_path,
                                                                                                         split)

        # init the base dataset class
        DatasetSeq2seq.__init__(self, input_data, output_data, batch_size, num_buckets, revert_input_sequence,
                                max_num_items)

        # find lengths of input/output
        self.max_output_input_ratio = max([float(y.size(0)) / x.size(0) for x, y in zip(self.input_data,
                                                                                        self.output_data)])

        self.dataset_name = dataset_name + '-' + split
        self.file_path = file_path
        self.data_path = data_path
        self.dicts = dicts

        # load dictionary sizes
        self.num_input_symbols = input_dict_size
        self.num_output_labels = output_dict_size
        input_dict_min_value, input_dict_max_value = 0, self.num_input_symbols - 1
        output_dict_min_value, output_dict_max_value = 0, self.num_output_labels - 1

        assert (all(a for a in self.input_min_value >= input_dict_min_value))
        self.input_min_value = torch.LongTensor([input_dict_min_value])

        assert (all(a for a in self.input_max_value <= input_dict_max_value))
        self.input_max_value = torch.LongTensor([input_dict_max_value])

        assert(all(a for a in self.output_min_value >= output_dict_min_value))
        self.output_min_value = torch.LongTensor([output_dict_min_value])

        assert(all(a for a in self.output_max_value <= output_dict_max_value))
        self.output_max_value = torch.LongTensor([output_dict_max_value])

        # add output special symbols
        self.output_end_of_string_token = EOS
        self.output_rare_word_token = UNK

        # add input special symbols
        self.num_input_helper_symbols = torch.LongTensor([2])  # words: empty token, rare word token
        self.input_empty_tokens = torch.LongTensor([PAD])
        self.input_rare_word_token = UNK

        # save input_embedding_size
        self.input_embedding_size = input_embedding_size

        # get output lengths per input length
        self.output_lengths = [1] * self.max_input_length  # one for EOS symbol
        for i_item in range(self.num_items):
            len_input = self.input_data[i_item].size(0)
            len_output = self.output_data[i_item].size(0) + 1  # add one to allow for EOS symbol
            self.output_lengths[len_input] = max(self.output_lengths[len_input], len_output)
        # do cumulative max
        for i_len in range(len(self.output_lengths)):
            self.output_lengths[i_len] = max(self.output_lengths[:i_len+1])

        self.evaluate_func = self.evaluate_bleu
        self.evaluate_func_name = 'BLEU'

    def get_max_output_length_per_input(self, length_inputs):
        output = []
        for len_input in length_inputs:
            if len_input < len(self.output_lengths):
                output.append(self.output_lengths[len_input])
            else:
                print('WARNING: sentence of length', len_input, 'detected')
                output.append(self.output_lengths[-1])
        return tuple(output)

    def get_embedding_layer(self):
        if self.input_embedding is None:
            self.input_embedding = EmbeddingSeq2seq(int(self.num_input_symbols), int(self.input_embedding_size),
                                                    sparse=False)
        return self.input_embedding, self.input_embedding_size

    def evaluate_bleu(self, predictions, labels, seq_lengths, item_indices):
        # interface for compatibility

        # get the write sentence lengths
        pr_lengths = compute_prediction_lengths(predictions, self.output_end_of_string_token)
        gt_lengths = compute_prediction_lengths(labels, self.output_end_of_string_token)

        # simplify types
        predictions, labels = predictions.data.cpu().squeeze(2), labels.data.cpu().squeeze(2)
        pr_lengths, gt_lengths = pr_lengths.data.cpu().view(-1), gt_lengths.data.cpu().view(-1)

        # convert into nltk format
        def decode(labels, lengths, unk_old, unk_new):
            decoded = []
            for i in range(labels.size(1)):
                sen = []
                for w in range(lengths[i]):
                    sen.append(labels[w, i] if labels[w, i] != unk_old else unk_new)
                decoded.append(sen)
            return decoded

        pr_all = decode(predictions, pr_lengths, self.output_rare_word_token, -1)
        gt_all = decode(labels, gt_lengths, self.output_rare_word_token, -2)

        # nltk wants a list of reference sentence for each entry
        gt_all = [[sen] for sen in gt_all]

        try:
            bleu_nltk = nltk.translate.bleu_score.corpus_bleu(
                gt_all, pr_all, smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method2)
            # compared against bleu nltk without smoothing: for BLEU around 0.3 difference in 1e-4
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as e:
            print("\nWARNING: Could not compute BLEU-score. Error:", str(e))
            bleu_nltk = 0

        return bleu_nltk
