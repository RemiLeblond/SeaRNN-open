import numpy as np
import os
import pickle
import sys
import time
import copy

import torch
from torch.autograd import Variable

import train
from tensor_utils import pad_tensor, lengths_to_mask, sync_dim_size
from evaluation.translate import translate_batch

"""
    Evaluation methods.
"""


def evaluate(dataset, encoder, decoder, opt, max_eval_size=sys.maxsize):
    print('Evaluation on %s' % dataset.get_name(), end=': ')
    t_start = time.time()

    # init the averaged losses
    seq_loss = 0.0
    # Total number of errors divided by the total number of symbols in the ground truth outputs
    hamming_loss_norm_global = 0.0
    length_global = 0
    log_loss = 0.0
    train_loss = 0.0
    num_items = 0
    max_length_memory = 0  # about memory storage and not actual length

    encoder.eval()
    decoder.eval()

    # prepare data for final evaluation
    predictions_all = []
    labels_all = []
    gt_lengths_all = []
    item_indices_all = []

    for i_batch in range(len(dataset)):
        # get data for evaluation
        input_batch, input_lengths, labels, output_lengths_gt, item_indices = dataset[i_batch]

        # convert to the right types
        input_batch = Variable(input_batch, volatile=True)
        labels = Variable(labels, volatile=True)  # SEQ_LEN x BATCH_SIZE x 1
        batch_size = labels.size(1)

        # CPU vs GPU
        if encoder.is_cuda():
            input_batch, labels, item_indices = input_batch.cuda(), labels.cuda(), item_indices.cuda()
        th = torch.cuda if input_batch.is_cuda else torch

        # add one EOS symbol to gt to predict it
        if not opt.output_fixed_size:
            labels = pad_tensor(labels, 0, labels.size(0) + 1, decoder.end_of_string_token)
            output_lengths_gt = tuple([x + 1 for x in output_lengths_gt])

        # fix max prediction length
        if opt.output_fixed_size:
            output_lengths_pred = copy.deepcopy(output_lengths_gt)
        else:
            output_lengths_pred = (opt.max_pred_length,) * batch_size
            labels = pad_tensor(labels, 0, max(output_lengths_pred), decoder.end_of_string_token)

        # sequence_lengths
        output_lengths_gt_th = Variable(th.LongTensor(output_lengths_gt))
        output_lengths_pred_th = Variable(th.LongTensor(output_lengths_pred))

        # run the decoder
        decoder_output, _, _, decoder_attention, _ = \
            train.run_rollin(input_batch, input_lengths, output_lengths_pred, encoder, decoder,
                             output_fixed_size=opt.output_fixed_size)
        assert(decoder_output.size(0) == max(output_lengths_pred))

        # decode the predictions
        if opt.decoding == 'beam':  # beam search decoding
            hyps, scores, _ = translate_batch(encoder, decoder, input_batch, input_lengths, output_lengths_pred, opt)
            predictions = Variable(th.LongTensor(hyps).permute(2, 0, 1).contiguous())
        else:  # greedy search decoding
            predictions = decoder.decode_labels(decoder_output, opt.output_fixed_size)

        # sync dimensions
        decoder_output, labels = sync_dim_size(decoder_output, labels, 0, 0.0, decoder.end_of_string_token)
        predictions, labels = sync_dim_size(predictions, labels, 0, decoder.end_of_string_token)

        # postprocess predictions
        if opt.output_fixed_size:
            # cut off extra symbols in the case of fixed output length
            mask = lengths_to_mask(sequence_length=output_lengths_gt_th, max_length=predictions.size(0))
            mask = mask == 0  # invert the mask
            predictions.masked_fill_(mask.unsqueeze(2), decoder.end_of_string_token)

        # make EOS symbol final
        predictions = train.make_eos_final(predictions, decoder.end_of_string_token)

        # compute costs specified for training
        train_costs = train.compute_costs(predictions, labels, output_lengths_gt_th,
                                          decoder.end_of_string_token, opt)
        train_loss += train_costs.data.sum()

        # compute the log loss
        cell_mask = lengths_to_mask(sequence_length=output_lengths_pred_th, max_length=labels.size(0))
        cur_log_loss = train.compute_objective_masked(decoder_output, cell_mask, target=labels, obj_func='softmax',
                                                      obj_normalization=opt.obj_normalization)
        log_loss += cur_log_loss.data.cpu()[0]

        # get numpy copies
        predictions_np = predictions.data.cpu().numpy()
        predictions_np = np.squeeze(predictions_np, axis=(2,))
        labels_np = labels.data.cpu().numpy()
        labels_np = np.squeeze(labels_np, axis=(2,))

        # compute the losses
        labels_error = predictions_np != labels_np
        for i_item in range(batch_size):
            num_items += 1
            seq_loss += np.any(labels_error[:, i_item])
            hamming_loss_norm_global += labels_error[:, i_item].sum()
            length_global += output_lengths_gt[i_item]

        # cumulative information
        labels_all.append(labels)
        predictions_all.append(predictions)
        gt_lengths_all.append(output_lengths_gt_th)
        item_indices_all.append(item_indices)
        max_length_memory = max(max_length_memory, labels.size(0))

        if num_items > max_eval_size:
            break

    seq_loss = seq_loss / num_items
    hamming_loss_norm_global = hamming_loss_norm_global / length_global
    log_loss = log_loss / num_items
    train_loss = train_loss / num_items

    print('%d items, %d tokens' % (num_items, length_global))

    print('Hamming = %.4f, Seq = %.4f, Log loss = %.4f, Train loss (%s): %.4f' % (
        hamming_loss_norm_global, seq_loss, log_loss, opt.loss, train_loss), end='')

    # aggregate data for further evaluation
    if dataset.evaluate_func is not None:
        labels_all = torch.cat([pad_tensor(t, dim=0, size=max_length_memory, value=decoder.end_of_string_token)
                                for t in labels_all], 1)
        predictions_all = torch.cat([pad_tensor(t, dim=0, size=max_length_memory, value=decoder.end_of_string_token)
                                     for t in predictions_all], 1)
        gt_lengths_all = torch.cat(gt_lengths_all, 0)
        item_indices_all = torch.cat(item_indices_all, 0)
        global_loss = dataset.evaluate_func(predictions_all, labels_all, gt_lengths_all, item_indices_all)
        print(', %s = %.4f' % (dataset.evaluate_func_name, global_loss), end='')
    else:
        global_loss = None
    print(', Eval time: %.1fs' % (time.time() - t_start))

    return seq_loss, hamming_loss_norm_global, log_loss, train_loss, global_loss


def evaluate_and_log(encoder, decoder, train_set, val_set, full_log, t_start, i_iter, objective_avg, grad_norm_avg,
                     opt, anneal_lr=None):
    # evaluate performance: train set
    sequence_loss_train, hamming_loss_train, log_loss_train, train_loss_train, global_loss_train = evaluate(
        train_set, encoder, decoder, opt, max_eval_size=opt.eval_size)

    if global_loss_train is not None:
        full_log['dataset_specific_train'].append(global_loss_train)

    full_log['time'].append((time.time() - t_start) / 3600)
    full_log['iter'].append(i_iter + 1)
    full_log['objective'].append(objective_avg)
    full_log['hamming_error_train'].append(hamming_loss_train)
    full_log['sequence_error_train'].append(sequence_loss_train)
    full_log[opt.loss.casefold() + '_train'].append(train_loss_train)
    full_log['log_loss_train'].append(log_loss_train)
    full_log['grad_norm'].append(grad_norm_avg)

    if val_set is not None:
        # evaluate performance: validation set
        sequence_loss_val, hamming_loss_val, log_loss_val, train_loss_val, global_loss_val = evaluate(
            val_set, encoder, decoder, opt)

        if global_loss_val is not None:
            full_log['dataset_specific_val'].append(global_loss_val)

        full_log['hamming_error_val'].append(hamming_loss_val)
        full_log['sequence_error_val'].append(sequence_loss_val)
        full_log['log_loss_val'].append(log_loss_val)
        full_log[opt.loss.casefold() + '_val'].append(train_loss_val)

    # anneal learning rate
    if anneal_lr:
        lr = anneal_lr()
    else:
        lr = float('nan')
    full_log['learning_rate'].append(lr)

    # rollin reference probability (for scheduled sampling)
    full_log['rollin_ref_prob'].append(opt.rollin_ref_prob)

    # save plots
    if opt.log_path:
        try:
            if not os.path.isdir(opt.log_path):
                os.makedirs(opt.log_path)
            pickle.dump(full_log, open(os.path.join(opt.log_path, "train_log.pkl"), "wb"))
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as e:
            print("\nWARNING: could not save the log file for some reason:", str(e))
