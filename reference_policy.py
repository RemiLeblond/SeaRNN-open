import fast_chunkscore
import fast_bleu_ref_rollout_with_suffix_length
import fast_bleu_ref_rollout_with_suffix_length_bleu1
import fast_gleu_ref_rollout_with_suffix_length

from losses_utils import *
from tensor_utils import *


def reference_policy_max_F1(prefixes_v, prefix_lengths_v, labels_v, eos):
    # prefixes - Variable with tensor of size  MAX_SEQ_LENGTH x BATCH_SIZE x 1
    # labels - Variable with tensor of size  MAX_SEQ_LENGTH x BATCH_SIZE x 1
    is_cuda = prefixes_v.is_cuda
    th = torch.cuda if is_cuda else torch

    prefixes_v = prefixes_v.squeeze(2)
    labels_v = labels_v.squeeze(2)

    # copy the correct prefixes, fill the rest with zeros - no chunk symbol
    last_label_v = torch.gather(prefixes_v, 0, prefix_lengths_v.unsqueeze(0) - 1)
    # After labels B-** one should put I-**
    mask_inc_v = torch.fmod(last_label_v, 2) == 1
    next_label_v = last_label_v.masked_select(mask_inc_v) + 1
    last_label_v = last_label_v.masked_scatter(mask_inc_v, next_label_v)
    # init outputs but full next label
    outputs_v = last_label_v.expand_as(prefixes_v).clone()
    # copy in the prefixes
    mask_prefixes_v = lengths_to_mask(prefix_lengths_v, max_length=labels_v.size(0))
    outputs_v = masked_scatter_with_transpose(prefixes_v, mask_prefixes_v, outputs_v, mask_prefixes_v)
    # find beginnings of all chunks
    mask_mod2_v = torch.fmod(labels_v, 2) == 1
    chunk_start_mask_v = mask_mod2_v.clone()
    chunk_start_mask_v[0] = 1  # the first symbol always starts a chunk
    extra_mask_t = (labels_v[:-1] != labels_v[1:]).data & \
                   (labels_v[1:] != 0).data & \
                   ((labels_v[:-1] + 1) != labels_v[1:]).data & \
                   (mask_mod2_v[1:] == 0).data
    extra_mask_t = extra_mask_t | (labels_v[1:] == 0).data
    if extra_mask_t.long().sum() > 0:
        extra_mask_t = torch.cat([th.ByteTensor(1, labels_v.size(1)).fill_(0), extra_mask_t], 0)
        chunk_start_mask_v.masked_fill_(Variable(extra_mask_t), 1)

    # find the earlier chunk after prefix
    chunk_start_v = Variable(th.LongTensor(labels_v.size(1)).fill_(labels_v.size(0) + 1))
    for i_step in range(labels_v.size(0) - 1, -1, -1):
        step_mask_t = chunk_start_mask_v[i_step, :].data & (prefix_lengths_v <= i_step).data
        chunk_start_v.masked_fill_(Variable(step_mask_t), i_step)

    # copy back GT after the start of the chunk
    all_length_v = Variable(th.LongTensor(prefix_lengths_v.size()).fill_(labels_v.size(0)))
    mask_suffixes_v = lengths_to_mask(all_length_v, max_length=labels_v.size(0), sequence_start=chunk_start_v)
    outputs_v = masked_scatter_with_transpose(labels_v, mask_suffixes_v, outputs_v, mask_suffixes_v)

    return outputs_v.unsqueeze(2)


def reference_policy_best_suffix(prefixes, prefix_lengths, labels, eos, loss_type = 'bleu4'):
    # prefixes - Variable with tensor of size  MAX_SEQ_LENGTH x BATCH_SIZE x 1
    # labels - Variable with tensor of size  MAX_SEQ_LENGTH x BATCH_SIZE x 1
    # bleu_type - 'bleu4' (default), 'bleu1'
    is_cuda = prefixes.is_cuda
    th = torch.cuda if is_cuda else torch

    # get the write gt lengths
    gt_lengths_th = compute_prediction_lengths(labels, eos)

    # check that vocabulary is not too big
    MAX_VOCAB_SIZE = 50000-3
    assert(prefixes.data.max() < MAX_VOCAB_SIZE)
    assert(labels.data.max() < MAX_VOCAB_SIZE)

    # equalize prefix and GT lengths
    prefixes, labels = sync_dim_size(prefixes, labels, 0, eos)

    # simplify types and move to CPU
    prefixes_np, labels_np = prefixes.data.squeeze(2).t().cpu().numpy(), labels.data.squeeze(2).t().cpu().numpy()
    prefix_lengths_np  = prefix_lengths.data.cpu().view(-1).numpy()
    gt_lengths_np = gt_lengths_th.data.cpu().view(-1).numpy()

    # run the reference policy
    n_items = prefixes_np.shape[0]
    max_length = prefixes_np.shape[1]
    prefixes_ravel = prefixes_np.ravel()
    labels_ravel = labels_np.ravel()
    if loss_type == 'bleu4':
        scores_np, suffix_length_np = fast_bleu_ref_rollout_with_suffix_length.compute_bleu(
            n_items, max_length, labels_ravel, gt_lengths_np, prefixes_ravel, prefix_lengths_np)
    elif loss_type == 'bleu1':
        scores_np, suffix_length_np = fast_bleu_ref_rollout_with_suffix_length_bleu1.compute_bleu(
            n_items, max_length, labels_ravel, gt_lengths_np, prefixes_ravel, prefix_lengths_np)
    elif loss_type == 'gleu':
        scores_np, suffix_length_np = fast_gleu_ref_rollout_with_suffix_length.compute_gleu(
            n_items, max_length, labels_ravel, gt_lengths_np, prefixes_ravel, prefix_lengths_np)
    else:
        raise (RuntimeError("Unknown type of loss used in the best suffix reference policy: %s" % loss_type))
    # construct complete labellings
    prefix_lengths_th = prefix_lengths
    suffix_lengths_th = torch.from_numpy(suffix_length_np)
    if is_cuda:
        suffix_lengths_th = suffix_lengths_th.cuda()
    suffix_lengths_th = Variable(suffix_lengths_th)

    output_lengths_th = suffix_lengths_th + prefix_lengths_th
    max_output_length = output_lengths_th.max().data[0]
    max_output_length = max(max_output_length, labels.size(0))

    prefix_th = pad_tensor(prefixes, 0, max_output_length, value=eos)
    labels_th = pad_tensor(labels, 0, max_output_length, value=eos)

    mask_prefix = lengths_to_mask(prefix_lengths_th, max_length=max_output_length)
    mask_suffix = lengths_to_mask(output_lengths_th, max_length=max_output_length, sequence_start=prefix_lengths_th)
    mask_gt_suffix = lengths_to_mask(gt_lengths_th, max_length=max_output_length,
                                     sequence_start=gt_lengths_th - suffix_lengths_th)

    outputs = Variable(th.LongTensor(max_output_length, n_items).fill_(eos))
    outputs = masked_scatter_with_transpose(prefix_th.squeeze(2), mask_prefix, outputs, mask_prefix)
    outputs = masked_scatter_with_transpose(labels_th.squeeze(2), mask_gt_suffix, outputs, mask_suffix)
    outputs = outputs.contiguous()

    # convert to torch
    scores_th = torch.from_numpy(scores_np)
    if is_cuda:
        scores_th = scores_th.cuda()
    scores_th = scores_th.contiguous()

    # convert scores to losses
    losses_th = 1.0 - scores_th
    return Variable(losses_th), outputs.unsqueeze(2)


def compute_sentence_F1_loss(predictions, labels, lengths):
    # predictions - Variable with tensor of size  MAX_SEQ_LENGTH x BATCH_SIZE x 1
    # ground_truth_labels - Variable with tensor of size  MAX_SEQ_LENGTH x BATCH_SIZE x 1
    # lengths - Variable with tensor of size BATCH_SIZE, each value is between 0 and MAX_SEQ_LENGTH-1

    is_cuda = predictions.is_cuda
    th = torch.cuda if is_cuda else torch

    # simplify types
    predictions, labels = predictions.data.squeeze(2).t().cpu().numpy(), labels.data.squeeze(2).t().cpu().numpy()
    lengths = lengths.data.cpu().view(-1).numpy()

    n_samples = labels.shape[0]
    max_len = labels.shape[1]

    # counts per sentece: chunks in labels, chunks in predictions, correct chunks
    counts = fast_chunkscore.compute_counts(n_samples, max_len, labels, predictions, lengths)
    counts = torch.from_numpy(counts.astype('float32'))
    if is_cuda:
        counts = counts.cuda()

    # precision, recall
    precision = counts[:, 2] / counts[:, 1]
    recall = counts[:, 2] / counts[:, 0]

    #FB1
    FB1 = th.FloatTensor(precision.size()).fill_(0)
    mask = (precision + recall) > 0
    masked_pr = precision.masked_select(mask)
    masked_re = recall.masked_select(mask)
    maskedFB1 = 2 * masked_pr * masked_re / (masked_pr + masked_re)
    FB1.masked_scatter_(mask, maskedFB1)

    F1_loss = 1.0 - FB1
    return Variable(F1_loss)
