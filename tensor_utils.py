import torch
import torch.nn.functional as F
from torch.autograd import Variable


def lengths_to_mask(sequence_length, max_length=None, sequence_start=None):
    if max_length is None:
        max_length = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = Variable(torch.arange(0, max_length).long())
    if sequence_length.is_cuda:
        seq_range = seq_range.cuda()
    seq_range_expand = seq_range.unsqueeze(1).expand(max_length, batch_size)
    sequence_length_expand = sequence_length.unsqueeze(0).expand_as(seq_range_expand)
    mask = seq_range_expand < sequence_length_expand
    if sequence_start is not None:
        seq_start_expand = sequence_start.unsqueeze(0).expand_as(seq_range_expand)
        mask = torch.min(mask, seq_range_expand >= seq_start_expand)

    return mask


def pad_tensor(var, dim, size, value=0):
    if var.size(dim) < size:  # pad to standard size
        var_sizes = list(var.size())
        pad_size = size - var.size(dim)

        # view everything to fit the API of F.pad function
        sizes = var_sizes[:dim + 1] + [-1]
        var = var.view(sizes)
        if var.dim() == 2:
            var = var.unsqueeze(0)
        var = var.view(-1, var.size(1), var.size(2))
        var = var.unsqueeze(0)

        # padding
        var = F.pad(var, (0, 0, 0, pad_size), mode='constant', value=value)

        # view to the original dimensions
        var_sizes[dim] = size
        var = var.view(var_sizes)
    return var


def masked_scatter_with_transpose(src, src_mask, tgt, tgt_mask):
    assert(src.size() == src_mask.size())
    assert(tgt.size() == tgt_mask.size())
    labels_to_copy = src.transpose(0, 1).masked_select(src_mask.transpose(0, 1))
    tgt = tgt.transpose(0, 1)
    tgt = tgt.masked_scatter(tgt_mask.transpose(0, 1), labels_to_copy)
    return tgt.transpose(0, 1)


def count_nans_in_var(var):
    mask = torch.max(var >= 0, var < 0)
    num_nans = var.numel() - mask.data.long().sum()
    return num_nans, mask


def sync_dim_size(a, b, dim, pad_value_a, pad_value_b=None):
    pad_value_b = pad_value_a if pad_value_b is None else pad_value_b
    if a.size(dim) < b.size(dim):
        a = pad_tensor(a, dim, b.size(dim), value=pad_value_a)
    elif a.size(dim) > b.size(dim):
        b = pad_tensor(b, dim, a.size(dim), value=pad_value_b)
    return a, b


def compute_argmax_masked(scores, mask):
    scores_for_max = scores.clone()
    scores_for_max.masked_fill_(mask == 0, float("-inf"))
    _, max_pos = torch.max(scores_for_max, 1, keepdim=True)
    return max_pos


def compute_max_masked(scores, mask):
    max_pos = compute_argmax_masked(scores, mask)
    return torch.gather(scores, 1, max_pos)


def compute_sum_masked(scores, mask):
    # not good: breaking the backprod chain (to get rid of infs):
    scores.data.masked_fill_(mask.data == 0, 0.0)
    # to get zero gradient in backprop, multiply by zero:
    scores = torch.mul(scores, mask.float())  # if one of the non-masked element is inf, this produces nans
    return scores.sum(1, keepdim=True)


def compute_log_softmax_masked(scores, mask):
    # find max values
    max_vals = compute_max_masked(scores, mask)
    # exponentiate
    scores_exp = torch.exp(scores - max_vals.expand_as(scores))
    # compute sum
    norm_vals = compute_sum_masked(scores_exp, mask)
    # take log
    norm_vals = torch.log(norm_vals) + max_vals
    # subtract normalization
    scores = scores - norm_vals.expand_as(scores)
    return scores


def compute_softmax_masked(scores, mask):
    # overwriting variable in a backprop chain is not good
    # But, since d p_j / d l_i = 0 at l_i = -inf, this is probably ok
    # analogous trick for log_softmax does not have this property
    if mask is not None:
        scores.data.masked_fill_(1 - mask.data, float('-inf'))
    probs = F.softmax(scores)
    return probs


def process_bidirectional_encoder_memory(h):
    # the encoder hidden is  (layers*directions) x batch x dim
    #  we need to convert it to layers x batch x (directions*dim)
    return h.view(h.size(0) // 2, 2, h.size(1), h.size(2)) \
                   .transpose(1, 2).contiguous() \
                   .view(h.size(0) // 2, h.size(1), h.size(2) * 2)


def get_batch_sizes_from_length(lengths):
    # need to have a separate implementation for a GPU-based torch Variable; too slow otherwise
    if type(lengths) == Variable:
        max_length = lengths.data[0]
        batch_sizes = [lengths.numel()] * max_length
        if max_length == 0:
            return batch_sizes  # []

        if lengths.numel() == 1:
            # this case happens sometimes
            return batch_sizes

        diff = lengths[:-1] - lengths[1:]
        drop_ids = torch.nonzero(diff.data)
        if drop_ids.numel() == 0:
            return batch_sizes  # [lengths.numel()] * max_length

        drop_sizes = torch.index_select(diff.data, 0, drop_ids.view(-1))
        drop_ids, drop_sizes = drop_ids.cpu(), drop_sizes.cpu()
        cum_drop = drop_sizes.sum()
        for id, drop in reversed(list(zip(drop_ids, drop_sizes))):
            assert(drop >= 0)
            for i_pos in range(max_length-cum_drop, max_length-cum_drop+drop):
                batch_sizes[i_pos] = id[0]+1
            cum_drop -= drop
    else:
        assert (is_sorted_decreasing(lengths))
        max_length = lengths[0]
        num_items = len(lengths)
        batch_sizes = [None] * max_length
        if max_length == 0:
            return batch_sizes  # []
        i_step = 0
        for i_item, len_item in reversed(list(enumerate(lengths))):
            while i_step < len_item:
                batch_sizes[i_step] = i_item + 1
                i_step += 1

    return batch_sizes


def is_sorted_decreasing(l):
    return all(a >= b for a, b in zip(l[:-1], l[1:]))
