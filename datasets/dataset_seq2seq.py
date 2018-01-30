import numpy as np
import math
import copy

import torch
import torch.utils.data as data


class DatasetSeq2seq(data.Dataset):
    def __init__(self, input_data, output_data, batch_size, num_buckets=1, revert_input_sequence=False,
                 max_num_items=None):
        self.input_data = input_data
        self.output_data = output_data
        self.dataset_name = 'dataset-base'

        if max_num_items is not None:
            # crop the number of data items
            print('Cropping dataset to %d items' % max_num_items)
            self.input_data, self.output_data = self.input_data[:max_num_items], self.output_data[:max_num_items]

        assert(len(self.input_data) == len(self.output_data))

        self.num_items = len(self.input_data)
        # add two positions to always have two EOS at the end
        self.max_input_length = max([x.size(0) for x in self.input_data]) + 2
        # add two positions to always have two EOS at the end
        self.max_output_length = max([x.size(0) for x in self.output_data]) + 2

        # revert input sequences if needed
        self.revert_input_sequence = revert_input_sequence
        if self.revert_input_sequence:
            # pytorch does not have negative strides for the moment, so reversing sequences is a bit painful
            inv_idx = torch.arange(self.max_input_length - 1, -1, -1).long()
            self.input_data = [x.index_select(0, inv_idx[inv_idx.numel() - x.size(0):]).contiguous()
                               for x in self.input_data]

        self.input_min_value, self.input_max_value = self._find_min_max_values(self.input_data)
        self.output_min_value, self.output_max_value = self._find_min_max_values(self.output_data)

        self.num_output_helper_symbols = 1
        # tokens + EOS token
        self.num_output_labels = self.output_max_value - self.output_min_value + 1 + self.num_output_helper_symbols
        assert(self.num_output_labels.numel() == 1)
        self.num_output_labels = int(self.num_output_labels.view(-1)[0])
        self.output_end_of_string_token = self.num_output_labels - 1

        self.num_input_helper_symbols = torch.LongTensor([1] * self.input_min_value.numel()).type_as(
            self.input_min_value)  # empty token
        self.num_input_symbols = self.num_input_helper_symbols + self.input_max_value[0] - self.input_min_value[0] + 1
        self.input_empty_tokens = torch.LongTensor([0] * self.input_min_value.numel()).type_as(self.input_min_value)

        self.input_embedding = None
        self.input_embedding_size = None
        self.evaluate_func = None
        self.evaluate_func_name = 'Special'

        self.batch_size = batch_size
        self.num_batches = math.ceil(self.num_items / self.batch_size)
        self.num_buckets = num_buckets
        self._create_buckets()

    def get_embedding_layer(self):
        raise NotImplementedError

    def get_name(self):
        return self.dataset_name

    def _find_min_max_values(self, data):
        min_value = data[0].min(0, keepdim=True)[0]
        max_value = data[0].max(0, keepdim=True)[0]
        for x in data:
            max_value = torch.max(max_value, x.max(0, keepdim=True)[0])
            min_value = torch.min(min_value, x.min(0, keepdim=True)[0])
        return min_value, max_value

    def _create_buckets(self, indices=None):
        # truncating num_buckets to good values
        self.num_buckets = min(self.num_buckets, self.num_batches)
        self.num_buckets = max(self.num_buckets, 1)

        sequence_lengths = [output.size(0) for output in self.output_data]

        # sort in the order of decreasing length
        if not indices:
            indices = range(len(sequence_lengths))

        items = zip(indices, sequence_lengths, self.input_data, self.output_data)
        batch, lengths = zip(*sorted(zip(items, sequence_lengths), key=lambda x: -x[1]))
        indices, sequence_lengths, self.input_data, self.output_data = zip(*batch)
        self.item_indices = indices

        # collect_buckets
        num_batches_in_bucket = math.ceil(self.num_batches / self.num_buckets)
        self.buckets = []
        self.bucket_order = []
        for i_bucket in range(self.num_buckets):
            bucket_start = i_bucket * num_batches_in_bucket * self.batch_size
            bucket_end = min( (i_bucket+1) * num_batches_in_bucket * self.batch_size, self.num_items )
            bucket_input = self.input_data[bucket_start : bucket_end]
            bucket_output = self.output_data[bucket_start: bucket_end]
            bucket_indices = self.item_indices[bucket_start: bucket_end]
            self.buckets.append( [bucket_input, bucket_output, bucket_indices] )

            bucket_num_batches = math.ceil((bucket_end - bucket_start) / self.batch_size)
            self.bucket_order.extend([ (i_bucket, i_batch) for i_batch in range(bucket_num_batches) ])

    def _batchify(self, data, empty_tokens, include_lengths=False, pad_beginning=False):
        lengths = [x.size(0) for x in data]
        max_length = max(lengths)
        batch_size = len(data)

        # init torch outputs
        out = empty_tokens.unsqueeze(0).unsqueeze(1)
        out = out.repeat(batch_size, max_length + 1, 1)  # add a barrier of one empty symbol

        # copy in the data points
        for i in range(batch_size):
            data_item = data[i]
            data_length = data_item.size(0)
            if data_item.dim() == 1:
                data_item = data_item.unsqueeze(1)
            if pad_beginning:
                # pad sequence from the beginning
                out[i].narrow(0, out.size(1) - data_length, data_length).copy_(data_item)
            else:
                # pad sequence from the end
                out[i].narrow(0, 0, data_length).copy_(data_item)
        if include_lengths:
            return out, lengths
        else:
            return out

    def __getitem__(self, index):
        assert(index < self.num_batches)
        i_bucket, i_batch = self.bucket_order[index]
        batch_input = self.buckets[i_bucket][0][i_batch*self.batch_size:(i_batch+1)*self.batch_size]
        batch_output = self.buckets[i_bucket][1][i_batch * self.batch_size:(i_batch + 1) * self.batch_size]
        batch_indices = self.buckets[i_bucket][2][i_batch * self.batch_size:(i_batch + 1) * self.batch_size]
        return self._prepare_batch(batch_input, batch_output, batch_indices)

    def get_random_batch(self):
        assert (self.num_items == len(self.input_data))
        assert (self.num_items == len(self.output_data))
        assert (self.num_items == len(self.item_indices))
        selected_items = np.random.choice(self.num_items, self.batch_size, replace=False)
        batch_input = [self.input_data[i] for i in selected_items]
        batch_output = [self.output_data[i] for i in selected_items]
        batch_indices = [self.item_indices[i] for i in selected_items]
        return self._prepare_batch(batch_input, batch_output, batch_indices)

    def _prepare_batch(self, batch_input, batch_output, batch_indices):
        # construct the batches
        input_batch, input_lengths = self._batchify(batch_input, self.input_empty_tokens,
                                                    include_lengths=True, pad_beginning=False)

        if self.output_data:
            output_batch, output_lengths = self._batchify(batch_output, torch.LongTensor(
                [self.output_end_of_string_token]), include_lengths=True, pad_beginning=False)
        else:
            output_batch, output_lengths = None, None

        # within batch sorting by decreasing length for input sequences
        batch = zip(batch_indices, input_batch, input_lengths) if output_batch is None \
            else zip(batch_indices, input_batch, input_lengths, output_batch, output_lengths)
        batch = sorted(batch, key=lambda x: -x[2])
        if output_batch is None:
            batch_indices, input_batch, input_lengths = zip(*batch)
        else:
            batch_indices, input_batch, input_lengths, output_batch, output_lengths = zip(*batch)

        # concatenate batch to make it SEG_LEN x BATCH_SIZE x NUM_FEATURES
        input_batch = torch.stack(input_batch, 1).contiguous()
        output_batch = torch.stack(output_batch, 1).contiguous()
        indices = torch.LongTensor(batch_indices)

        return input_batch, input_lengths, output_batch, output_lengths, indices

    def __len__(self):
        return self.num_batches

    def get_max_output_length(self):
        return self.max_output_length

    def get_max_output_length_per_input(self, length_inputs):
        return (self.get_max_output_length(), ) * len(length_inputs)

    def shuffle(self, shuffle_buckets=True):
        self.bucket_order = [self.bucket_order[i] for i in torch.randperm(len(self.bucket_order))]

        if shuffle_buckets:
            for bucket in self.buckets:
                data = list(zip(bucket[0], bucket[1],  bucket[2]))
                bucket[0], bucket[1], bucket[2] = zip(*[data[i] for i in torch.randperm(len(data))])

    def copy_subset(self, subset_size):
        dataset_subset = copy.copy(self)  # shallow copy
        batch_size = dataset_subset.batch_size

        if subset_size is not None:
            dataset_subset.num_items = min(subset_size, dataset_subset.num_items)
        dataset_subset.num_batches = math.ceil(dataset_subset.num_items / batch_size)

        # copy data elements
        input_data_subset = []
        output_data_subset = []
        item_indices_subset = []
        for index in range(dataset_subset.num_batches):
            i_bucket, i_batch = dataset_subset.bucket_order[index]
            input_data_subset.extend(
                dataset_subset.buckets[i_bucket][0][i_batch * batch_size:(i_batch + 1) * batch_size])
            output_data_subset.extend(
                dataset_subset.buckets[i_bucket][1][i_batch * batch_size:(i_batch + 1) * batch_size])
            item_indices_subset.extend(
                dataset_subset.buckets[i_bucket][2][i_batch * batch_size:(i_batch + 1) * batch_size])
        dataset_subset.input_data = copy.deepcopy(input_data_subset)
        dataset_subset.output_data = copy.deepcopy(output_data_subset)
        dataset_subset.item_indices = copy.deepcopy(item_indices_subset)

        # recreate buckets
        dataset_subset._create_buckets(dataset_subset.item_indices)

        return dataset_subset
