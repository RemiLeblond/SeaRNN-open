import os
import numpy as np

import torch
from datasets.dataset_seq2seq import DatasetSeq2seq
from models import IdleLayer


def _read_ocr_dataset(root):
    file_data = os.path.join(root, 'letter.data')
    if not os.path.isfile(file_data):
        raise(RuntimeError("Could not found OCR dataset (file letter.data) in " + root))

    # read letters
    letter_lines = {}
    num_bits = 16 * 8
    with open(file_data) as data_file:
        for line in data_file:
            # order of entries in a line: id, letter, next_id, word_id, position, fold, p_0_0, p_0_1, ..., p_15_7
            entries = line.split()
            assert(len(entries) == 6 + num_bits)
            id = int(entries[0])
            letter_lines[id] = entries

    # find the ordering of letters
    last_letters = {}
    prev_letter = {}
    for id, entries in letter_lines.items():
        next_id = int(entries[2])
        if next_id == -1:
            last_letters[id] = True
        else:
            if next_id not in prev_letter:
                prev_letter[next_id] = id
            else:
                raise(RuntimeError("Letter {0} is after both {1} and {2}".format(id, prev_letter[next_id], next_id)))

    # merge letters into words
    words_bits = {}
    words_str = {}
    folds = {}

    used_letters = {}
    max_word_length = 0
    for last_letter in last_letters:
        cur_word = []
        cur_letter = last_letter
        while cur_letter is not None:
            used_letters[cur_letter] = True
            cur_word = [cur_letter] + cur_word
            if cur_letter in prev_letter:
                cur_letter = prev_letter[cur_letter]
                assert(cur_letter not in used_letters)  # sanity check to avoid infinite loop in case of bad data
            else:
                cur_letter = None

        # check the word and convert in to numpy
        word_length = len(cur_word)
        word_bits = np.zeros((word_length, num_bits), dtype='uint8')
        word_str = np.zeros(word_length, dtype='uint8')
        fold = None
        word_id = None
        for i_pos, id in enumerate(cur_word):
            # extract all letter data
            # order of entries in a line: id, letter, next_id, word_id, position, fold, p_0_0, p_0_1, ..., p_15_7
            entries = letter_lines[id]
            cur_id = int(entries[0])
            cur_letter = entries[1]
            cur_next_id = int(entries[2])
            cur_word_id = int(entries[3])
            cur_position = int(entries[4])
            cur_fold = int(entries[5])
            cur_bits = np.array(entries[6:], dtype='uint8')

            assert(id == cur_id)

            word_str[i_pos] = ord(cur_letter) - ord('a')

            if word_id is None:
                word_id = cur_word_id
            else:
                assert(word_id == cur_word_id)

            assert(cur_position == i_pos + 1)

            if fold is None:
                fold = cur_fold
            else:
                assert(fold == cur_fold)

            word_bits[i_pos] = cur_bits

        assert(word_id not in words_bits)
        words_bits[word_id] = word_bits
        words_str[word_id] = word_str
        folds[word_id] = fold
        max_word_length = max(max_word_length, word_length)

    assert(len(used_letters) == len(letter_lines))  # double check that all the letters are used

    return words_bits, words_str, folds, max_word_length, num_bits


class OcrDataset(DatasetSeq2seq):
    def __init__(self, root, batch_size, is_train=True, split_id=0, num_buckets=1, revert_input_sequence=True,
                 max_num_items=None):

        words_bits, words_str, folds, max_word_length, num_features = _read_ocr_dataset(root)

        # get the data from splits
        if is_train:
            self.word_ids = [word_id for word_id, fold in folds.items() if not fold == split_id]
        else:
            self.word_ids = [word_id for word_id, fold in folds.items() if fold == split_id]

        # prepare data for torch
        input_data = [torch.FloatTensor(words_bits[i].astype('float32')) for i in self.word_ids]
        output_data = [torch.LongTensor(words_str[i].astype('int64')) for i in self.word_ids]

        # init the base dataset class
        DatasetSeq2seq.__init__(self, input_data, output_data, batch_size, num_buckets, revert_input_sequence,
                                max_num_items)

        assert(all([x.size(0) == y.size(0) for x, y in zip(self.input_data, self.output_data)]))

        self.dataset_name = 'OCR-' + ('train' if is_train else 'test')
        self.root = root
        self.max_length = 16  # max_word_length
        self.num_features = num_features

        self.num_input_helper_symbols = 0
        self.input_empty_tokens = self.input_empty_tokens * 0
        self.num_input_symbols = self.num_input_symbols * 0 + 2

    def get_embedding_layer(self):
        if self.input_embedding is None:
            self.input_embedding = IdleLayer()
            self.input_embedding_size = self.num_features
        return self.input_embedding, self.input_embedding_size
