import fast_bleu
import fast_bleu1
import fast_gleu
from tensor_utils import *

"""
    Helpers for loss computation.
"""


def compute_prediction_lengths(labels, eos):
    # pad labels to always have eos
    labels = pad_tensor(labels, 0, labels.size(0) + 1, eos)

    max_length = labels.size(0)
    pos_ids = torch.arange(0, max_length).long().unsqueeze(1).unsqueeze(2)
    if labels.is_cuda:
        pos_ids = pos_ids.cuda()
    pos_ids = Variable(pos_ids).expand_as(labels).contiguous()
    pos_ids.masked_fill_(labels != eos, max_length)
    _, first_eos = torch.min(pos_ids, 0, keepdim=True)
    return first_eos.view(-1)


def make_eos_final(labels, eos):
    # change all predictions after EOS symbol to EOS
    lengths = compute_prediction_lengths(labels, eos)
    mask = lengths_to_mask(sequence_length=lengths, max_length=labels.size(0))
    mask = mask == 0  # invert the mask

    # fill the mask with EOS
    labels.masked_fill_(mask.unsqueeze(2), eos)
    return labels


def compute_hamming_loss_masked(predictions, labels, lengths, normalization_constant=None):
    # predictions - Variable with tensor of size  MAX_SEQ_LENGTH x BATCH_SIZE x 1
    # labels - Variable with tensor of size  MAX_SEQ_LENGTH x BATCH_SIZE x 1
    # lengths - Variable with tensor of size BATCH_SIZE, each value is between 0 and MAX_SEQ_LENGTH-1
    errors = (predictions != labels).float().squeeze(2)
    if normalization_constant is None:
        normalization = lengths.float().unsqueeze(0).expand_as(errors)
        errors = errors / normalization
    else:
        errors = errors / normalization_constant
    hamming_loss = errors.sum(0, keepdim=False)
    return hamming_loss


def compute_sentence_loss_masked(predictions, labels, eos=None, loss_type='bleu4'):
    # predictions - Variable with tensor of size  MAX_SEQ_LENGTH x BATCH_SIZE x 1
    # labels - Variable with tensor of size  MAX_SEQ_LENGTH x BATCH_SIZE x 1
    is_cuda = predictions.is_cuda

    # get the write sentence lengths
    pr_lengths = compute_prediction_lengths(predictions, eos)
    gt_lengths = compute_prediction_lengths(labels, eos)

    # check that vocabulary is not too big
    max_vocab_size = 50000-3
    assert(predictions.data.max() < max_vocab_size)
    assert(labels.data.max() < max_vocab_size)

    # convert types and move to numpy on CPU
    predictions, labels = predictions.data.squeeze(2).t().cpu().numpy(), labels.data.squeeze(2).t().cpu().numpy()
    pr_lengths, gt_lengths = pr_lengths.data.cpu().view(-1).numpy(), gt_lengths.data.cpu().view(-1).numpy()

    n_items = predictions.shape[0]
    max_length = predictions.shape[1]
    predictions_ravel = predictions.ravel()
    labels_ravel = labels.ravel()

    if loss_type == 'bleu4':
        scores = fast_bleu.compute_bleu(n_items, max_length, labels_ravel, gt_lengths, predictions_ravel, pr_lengths)
    elif loss_type == 'bleu1':
        scores = fast_bleu1.compute_bleu(n_items, max_length, labels_ravel, gt_lengths, predictions_ravel, pr_lengths)
    elif loss_type == 'gleu':
        scores = fast_gleu.compute_gleu(n_items, max_length, labels_ravel, gt_lengths, predictions_ravel, pr_lengths)
    else:
        raise (RuntimeError("Unknown type of loss used in the best suffix reference policy: %s" % loss_type))

    # convert to torch
    scores_th = torch.from_numpy(scores)
    if is_cuda:
        scores_th = scores_th.cuda()

    # convert scores to losses
    losses_th = 1.0 - scores_th
    return Variable(losses_th)
