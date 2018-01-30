import torch


"""
    Borrowed from OpenNMT-py (https://github.com/OpenNMT/OpenNMT-py).
"""


class Dict(object):
    def __init__(self, data=None, lower=False):
        self.idx_to_label = {}
        self.label_to_idx = {}
        self.frequencies = {}
        self.lower = lower

        # Special entries will not be pruned.
        self.special = []

        if data is not None:
            if type(data) == str:
                self.load_file(data)
            else:
                self.add_specials(data)

    def size(self):
        return len(self.idx_to_label)

    # Load entries from a file.
    def load_file(self, filename):
        for line in open(filename):
            fields = line.split()
            label = fields[0]
            idx = int(fields[1])
            self.add(label, idx)

    # Write entries to a file.
    def write_file(self, filename):
        with open(filename, 'w') as file:
            for i in range(self.size()):
                label = self.idx_to_label[i]
                file.write('%s %d\n' % (label, i))

        file.close()

    def lookup(self, key, default=None):
        key = key.lower() if self.lower else key
        try:
            return self.label_to_idx[key]
        except KeyError:
            return default

    def get_label(self, idx, default=None):
        try:
            return self.idx_to_label[idx]
        except KeyError:
            return default

    # Mark this `label` and `idx` as special (i.e. will not be pruned).
    def add_special(self, label, idx=None):
        idx = self.add(label, idx)
        self.special += [idx]

    # Mark all labels in `labels` as specials (i.e. will not be pruned).
    def add_specials(self, labels):
        for label in labels:
            self.add_special(label)

    # Add `label` in the dictionary. Use `idx` as its index if given.
    def add(self, label, idx=None):
        label = label.lower() if self.lower else label
        if idx is not None:
            self.idx_to_label[idx] = label
            self.label_to_idx[label] = idx
        else:
            if label in self.label_to_idx:
                idx = self.label_to_idx[label]
            else:
                idx = len(self.idx_to_label)
                self.idx_to_label[idx] = label
                self.label_to_idx[label] = idx

        if idx not in self.frequencies:
            self.frequencies[idx] = 1
        else:
            self.frequencies[idx] += 1

        return idx

    # Return a new dictionary with the `size` most frequent entries.
    def prune(self, size):
        if size >= self.size():
            return self

        # Only keep the `size` most frequent entries.
        freq = torch.Tensor([self.frequencies[i] for i in range(len(self.frequencies))])
        _, idx = torch.sort(freq, 0, True)

        new_dict = Dict()
        new_dict.lower = self.lower

        # Add special entries in all cases.
        for i in self.special:
            new_dict.add_special(self.idx_to_label[i])

        for i in idx[:size]:
            new_dict.add(self.idx_to_label[i])

        return new_dict

    # Convert `labels` to indices. Use `unkWord` if not found.
    # Optionally insert `bosWord` at the beginning and `eosWord` at the .
    def convert_to_idx(self, labels, unk_word, bos_word=None, eos_word=None):
        vec = []

        if bos_word is not None:
            vec += [self.lookup(bos_word)]

        unk = self.lookup(unk_word)
        vec += [self.lookup(label, default=unk) for label in labels]

        if eos_word is not None:
            vec += [self.lookup(eos_word)]

        return torch.LongTensor(vec)

    # Convert `idx` to labels. If index `stop` is reached, convert it and return.
    def convert_to_labels(self, idx, stop):
        labels = []

        for i in idx:
            labels += [self.get_label(i)]
            if i == stop:
                break

        return labels
