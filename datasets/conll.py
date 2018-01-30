import os
import gzip
import subprocess
import re
import copy
from collections import OrderedDict
import locale

import torch

from datasets.dataset_seq2seq import DatasetSeq2seq
from models import MultiEmbedding, EmbeddingPartialTrainable, EmbeddingSeq2seq

"""
    Conll dataset helper.
"""

locale.setlocale(locale.LC_ALL, 'en_US.UTF8')

# keeping this order is important
output_dictionary = {
    'O': 0,
    'B-NP': 1,
    'I-NP': 2,
    'B-VP': 3,
    'I-VP': 4,
    'B-PP': 5,
    'I-PP': 6,
    'B-SBAR': 7,
    'I-SBAR': 8,
    'B-PRT': 9,
    'I-PRT': 10,
    'B-ADJP': 11,
    'I-ADJP': 12,
    'B-ADVP': 13,
    'I-ADVP': 14,
    'B-CONJP': 15,
    'I-CONJP': 16,
    'B-UCP': 17,
    'I-UCP': 18,
    'B-LST': 19,
    'I-LST': 20,
    'B-INTJ': 21,
    'I-INTJ': 22,
}


def read_conll_dataset(root, is_train=True, max_seq_length=None, max_lines=None):
    if is_train:
        data_file = os.path.join(root, 'train.txt.gz')
    else:
        data_file = os.path.join(root, 'test.txt.gz')

    if not os.path.isfile(data_file):
        raise(RuntimeError("Could not found CONLL-chunking dataset (file {0} in {2})".format(data_file, root)))

    # read data
    data = []
    num_entries = 3
    max_length = 0
    num_items = 0
    num_items_ignored = 0
    with gzip.open(data_file, 'rt') as f:
        seq = []
        for line in f:
            words = line[:-1].split(' ')
            if len(line) > 1:
                assert(len(words) == num_entries)
                seq.append(words)
            else:
                if len(seq) > 0:
                    if max_seq_length is None or len(seq) <= max_seq_length:
                        data.append(seq)
                        num_items += 1
                        max_length = max(max_length, len(seq))
                    else:
                        num_items_ignored += 1
                seq = []
                if max_lines is not None and num_items >= max_lines:
                    break

    print('Read CONLL-chunking-{2} dataset: read {0} items, using {1} items'.format(
        num_items + num_items_ignored, num_items, 'train' if is_train else 'test'))

    return data, max_length


def is_number(s):
    try:
        locale.atof(s)
        return True
    except ValueError:
        pass
    # put string like '3\/32' into numbers
    try:
        special_str = '\/'
        pos = s.find(special_str)
        if pos > 0:
            locale.atoi(s[:pos])
            locale.atoi(s[pos+len(special_str):])
            return True
    except ValueError:
        pass
    return False


def build_dictionary(input_data, word_pos_index, min_word_count=0, word_dict=None):
    token_count_dict = OrderedDict()
    for item in input_data:
        for tokens in item:
            word = tokens[word_pos_index]
            if word not in token_count_dict:
                token_count_dict[word] = 1
            else:
                token_count_dict[word] += 1

    num_words = 0
    token_dict = OrderedDict()
    for word, count in token_count_dict.items():
        if word_dict is not None and word.casefold() in word_dict:
            token_dict[word] = num_words
            num_words += 1
        elif count >= min_word_count and not is_number(word):
            # adding new word
            token_dict[word] = num_words
            num_words += 1

    print("Building dictionary with min_word_count={0}, found {1} tokens".format(min_word_count, num_words))
    return token_dict


def construct_dataset(data, dicts):
    input_data = []
    output_data = []
    num_tokens = len(dicts)
    dict_sizes = [len(d) for d in dicts]
    num_input_tokens = dict_sizes[:-1]
    num_output_tokens = dict_sizes[-1]
    num_numbers = 0

    for item in data:
        seq_len = len(item)
        data_th = torch.LongTensor(seq_len, num_tokens - 1)
        labels_th = torch.LongTensor(seq_len, 1)
        for i, tokens in enumerate(item):
            assert(len(tokens) == num_tokens)
            # output
            if tokens[-1] not in dicts[-1]:
                raise (RuntimeError("Unknown output token: {0}".format(tokens[-1])))
            else:
                labels_th[i] = dicts[-1][tokens[-1]]
            # input
            for i_pos in range(num_tokens-1):
                token = tokens[i_pos]
                if token in dicts[i_pos]:
                    data_th[i, i_pos] = int(dicts[i_pos][token])
                elif is_number(token):
                    data_th[i, i_pos] = dict_sizes[i_pos]
                    num_numbers += 1
                else:
                    data_th[i, i_pos] = dict_sizes[i_pos] + 1
        input_data.append(data_th)
        output_data.append(labels_th)
    print('Found %d tokens of numbers' % num_numbers)
    return input_data, output_data, num_input_tokens, num_output_tokens,


def make_lower_case(data):
    for item in data:
        for tokens in item:
            tokens[0] = tokens[0].casefold()


def read_emb_word_dict(file):
    assert (os.path.isfile(file))
    words = [line.rstrip('\n') for line in open(file, 'r')]
    emb_dict = OrderedDict()
    for i, w in enumerate(words):
        emb_dict[w] = i
    return emb_dict


class ConllDataset(DatasetSeq2seq):
    def __init__(self, data_path, batch_size, is_train=True,
                 max_lines=None, max_seq_length=None, dicts=None, min_word_count=10,
                 revert_input_sequence=True, num_buckets=1, max_num_items=None, senna_emb=''):

        # read data
        data, max_length = read_conll_dataset(data_path, is_train, max_seq_length, max_lines)
        self.raw_data = copy.deepcopy(data)
        self.senna_emb = senna_emb

        # make lower case
        make_lower_case(data)

        # build dictionaries
        if dicts is None:
            print('Building dictionaries')
            dicts = [None] * 3
            # entries: word, POS, label
            dicts[1] = build_dictionary(data, 1)
            dicts[2] = output_dictionary
            if not self.senna_emb:
                dicts[0] = build_dictionary(data, 0, min_word_count)
            else:
                print('Getting list of words from SENNA embeddings')
                assert (os.path.isdir(self.senna_emb))
                word_list_file = os.path.join(self.senna_emb, 'hash/words.lst')
                word_dict = read_emb_word_dict(word_list_file)

                # build SENNA dictionary
                # read second part of the data to build the joint dictionary
                data_extra, _ = read_conll_dataset(data_path, not is_train)
                make_lower_case(data_extra)
                dicts[0] = build_dictionary(data + data_extra, 0, max_length * len(data), word_dict)
        else:
            num_unknown = 0
            num_words = 0
            for item in data:
                num_words += len(item)
                for tokens in item:
                    if tokens[0] not in dicts[0] and not is_number(tokens[0]):
                        num_unknown += 1
            print('%d of %d words are not found in the dictionary' % (num_unknown, num_words))

        # build dataset
        input_data, output_data, num_input_tokens, num_output_tokens = construct_dataset(data, dicts)

        # init the base dataset class
        DatasetSeq2seq.__init__(self, input_data, output_data, batch_size, num_buckets, revert_input_sequence, max_num_items)

        assert(all([x.size(0) == y.size(0) for x, y in zip(self.input_data, self.output_data)]))

        self.dataset_name = 'CONLL-chunking-' + ('train' if is_train else 'test')
        self.data_path = data_path
        self.max_length = max_length
        self.revert_input_sequence = revert_input_sequence
        self.dicts = dicts

        self.num_output_helper_symbols = 1
        self.num_output_labels = num_output_tokens + self.num_output_helper_symbols # tokens + UNK + PAD token
        self.output_end_of_string_token = self.num_output_labels - 1

        self.num_input_helper_symbols = torch.LongTensor([3, 3])  # NUM + UNK + PAD token
        self.num_input_symbols = torch.LongTensor(num_input_tokens) + self.num_input_helper_symbols
        self.input_empty_tokens = self.num_input_symbols - 1
        self.input_unk_tokens = self.num_input_symbols - 2
        self.input_num_tokens = self.num_input_symbols - 3

        self.evaluate_func = self.evaluate_conll_F1
        self.evaluate_func_name = 'F1'

        self.output_inverse_dict = {}
        for k, i in self.dicts[-1].items():
            self.output_inverse_dict[i] = k

    def get_embedding_layer(self):
        if self.input_embedding is None:
            if not self.senna_emb:
                self.input_embedding_dims = (128, 44)
                embeddings = [EmbeddingSeq2seq(int(self.num_input_symbols[i]),
                                               int(self.input_embedding_dims[i]),
                                               padding_idx=int(self.input_empty_tokens[i]),
                                               sparse=False) for i in range(len(self.input_embedding_dims))]
            else:
                self.input_embedding_dims = (50, 44)
                word_embs = torch.FloatTensor(self.num_input_symbols[0], self.input_embedding_dims[0]).normal_(0, 1)
                read_mask = self.init_embeddings_senna(word_embs)

                embeddings = [EmbeddingPartialTrainable(int(self.num_input_symbols[0]),
                                                        int(self.input_embedding_dims[0]),
                                                        mask_items_to_update=read_mask != 1,
                                                        weights=word_embs),
                              EmbeddingSeq2seq(int(self.num_input_symbols[1]),
                                               int(self.input_embedding_dims[1]),
                                               padding_idx=int(self.input_empty_tokens[1]),
                                               sparse=False)]

            self.input_embedding = MultiEmbedding(embeddings)
            self.input_embedding_size = sum(self.input_embedding_dims)

        return self.input_embedding, self.input_embedding_size

    def init_embeddings_senna(self, embeddings):
        print('Initializing from SENNA embeddings from %s' % self.senna_emb)
        assert (os.path.isdir(self.senna_emb))
        word_list_file = os.path.join(self.senna_emb, 'hash/words.lst')
        emb_file = os.path.join(self.senna_emb, 'embeddings/embeddings.txt')
        words = [line.rstrip('\n') for line in open(word_list_file, 'r')]
        embs = [[float(num) for num in line.split(' ')] for line in open(emb_file, 'r')]
        assert(len(words) == len(embs))
        emb_dict = OrderedDict()
        for i, w in enumerate(words):
            emb_dict[w] = i

        num_found = 0
        read_mask = torch.ByteTensor(embeddings.size(0)).zero_()
        for w, id in self.dicts[0].items():
            w = w.casefold()
            if w in emb_dict:
                emb_id = emb_dict[w]
                emb = torch.FloatTensor(embs[emb_id])
                assert(emb.numel() == 50)
                embeddings[id, :] = emb
                read_mask[id] = 1
                num_found += 1
        print('Found {0} of {1} words in SENNA embeddings'.format(num_found, len(self.dicts[0])))
        return read_mask

    def evaluate_conll_F1(self, predictions, labels, seq_lengths, item_indices):
        score = float('nan')
        try:
            # restore the original order of the dataset
            _, order_reversed = torch.sort(item_indices, 0, descending=False)
            predictions = torch.index_select(predictions, 1, order_reversed).data.cpu()
            labels = torch.index_select(labels, 1, order_reversed).data.cpu()
            seq_lengths = torch.index_select(seq_lengths, 0, order_reversed).data.cpu()
            item_indices = torch.index_select(item_indices, 0, order_reversed).cpu()

            # make predictions
            results = []
            for i_item in range(predictions.size(1)):
                # Each line contains four symbols: the current word, its part-of-speech tag (POS),
                # the chunk tag according to the corpus and the predicted chunk tag.
                # Sentences have been separated by empty lines.
                for i_word in range(seq_lengths[i_item]):
                    line = [token for token in self.raw_data[item_indices[i_item]][i_word]]
                    pred = predictions[i_word, i_item, 0]
                    if pred in self.output_inverse_dict:
                        line.append(self.output_inverse_dict[pred])
                    else:
                        line.append('<UNK>')
                    results.append(' '.join(line))
                results.append('')

            script_name = os.path.join(self.data_path, 'conlleval.perl')
            results = '\n'.join(results)

            proc = subprocess.Popen(['perl', script_name],
                                    stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout_data = proc.communicate(input=results.encode())

            script_output = stdout_data[0].decode('ascii')

            # extract the FB1 score
            numeric_const_pattern = "[-+]?(?:(?:\d*\.\d+)|(?:\d+\.?))(?:[Ee][+-]?\d+)?"
            match = re.search('FB1:\s*(%s)' % numeric_const_pattern, script_output)
            if match:
                score = float(match.group(1))
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as e:
            print("\nWARNING: Could not compute F-score. Error:", str(e))

        return score
