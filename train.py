import copy
from itertools import chain

import torch.nn as nn
import torch.optim.lr_scheduler
from evaluation.logging_utils import init_logging, checkpoint_model

from evaluation.evaluate import evaluate_and_log
from optimization import setup_lr
from searnn import *
from tensor_utils import *
from utils import *


def train_one_batch(input_data, input_lengths, ground_truth_labels, output_lengths_gt, encoder, decoder, optimizer,
                    num_output_labels, opt):
    # get data stats
    batch_size = input_data.size(1)
    assert(ground_truth_labels.size(1) == batch_size)

    # GPU vs CPU
    if opt.cuda:
        input_data = input_data.cuda()
        ground_truth_labels = ground_truth_labels.cuda()
    th = torch.cuda if input_data.is_cuda else torch

    # convert types
    input_data = Variable(input_data)
    ground_truth_labels = Variable(ground_truth_labels)

    # add one EOS symbol to gt to predict it
    if not opt.output_fixed_size:
        ground_truth_labels = pad_tensor(ground_truth_labels, 0, ground_truth_labels.size(0) + 1,
                                         decoder.end_of_string_token)
        output_lengths_gt = tuple([x + 1 for x in output_lengths_gt])
    else:
        # include EOS symbol in the training loss
        output_lengths_gt = tuple([x + 1 for x in output_lengths_gt])

    # fix max prediction length
    if opt.output_fixed_size:
        output_lengths_pred = copy.deepcopy(output_lengths_gt)
    else:
        output_lengths_pred = opt.get_train_output_length(input_lengths)
        ground_truth_labels = pad_tensor(ground_truth_labels, 0, max(output_lengths_pred), decoder.end_of_string_token)

    # sequence_lengths
    input_lengths_th = Variable(th.LongTensor(input_lengths))
    output_lengths_gt_th = Variable(th.LongTensor(output_lengths_gt))
    output_lengths_pred_th = Variable(th.LongTensor(output_lengths_pred))

    # init optimization
    optimizer.zero_grad()

    # decide how many decoder steps to do
    decoder_steps_per_item = output_lengths_pred

    if opt.objective.casefold() == 'mle':
        # length of GT including one EOS symbol when opt.output_fixed_size == False
        decoder_steps_per_item = output_lengths_gt

    # decide whether to use teacher forcing or not
    rollin_mode = opt.rollin.casefold()
    rollin_reference_policy = None
    if rollin_mode == 'learned':
        use_teacher_forcing = False
    elif rollin_mode == 'gt':
        use_teacher_forcing = True
    elif rollin_mode == 'mixed':
        use_teacher_forcing = torch.bernoulli(torch.FloatTensor(batch_size).fill_(opt.rollin_ref_prob)).byte()
        if opt.cuda:
            use_teacher_forcing = use_teacher_forcing.cuda()
    elif rollin_mode == 'mixed-cells':
        use_teacher_forcing = torch.bernoulli(
            torch.FloatTensor(batch_size, max(output_lengths_pred)).fill_(opt.rollin_ref_prob)).byte()
        if opt.cuda:
            use_teacher_forcing = use_teacher_forcing.cuda()

        # set the reference policy to be used at the rollin stage
        rollin_reference_policy = lambda ground_truth_labels, rollout_data, rollin_labels:\
            apply_reference_policy(ground_truth_labels, rollout_data, rollin_labels,
                                   opt, eos=decoder.end_of_string_token)
    else:
        raise (RuntimeError("Unknown roll-in strategy %s" % rollin_mode))

    # roll-in
    encoder.train()
    decoder.train()
    decoder_output_rollin, decoder_hidden, encoder_output_states, decoder_attention, rollin_labels = \
        run_rollin(input_data, input_lengths, decoder_steps_per_item, encoder, decoder, ground_truth_labels,
                   opt.output_fixed_size, use_teacher_forcing, reference_policy=rollin_reference_policy)

    # add EOS symbol because there could be none because of the mixed-cells roll-in
    rollin_labels = pad_tensor(rollin_labels, 0, rollin_labels.size(0) + 1, decoder.end_of_string_token)
    rollin_labels = make_eos_final(rollin_labels, decoder.end_of_string_token)
    # update output lengths if rollin has become longer
    output_lengths_pred_th = torch.max(output_lengths_pred_th,
                                       compute_prediction_lengths(rollin_labels, decoder.end_of_string_token))
    # sync dimensions: to deal with a rare case when GT labels appear to be very long
    rollin_labels, ground_truth_labels = sync_dim_size(rollin_labels, ground_truth_labels, 0,
                                                       decoder.end_of_string_token)
    decoder_output_rollin, ground_truth_labels = sync_dim_size(decoder_output_rollin, ground_truth_labels, 0, 0.0,
                                                               decoder.end_of_string_token)

    if opt.objective.casefold() == 'mle':
        # compute the objective
        cell_mask = lengths_to_mask(sequence_length=output_lengths_gt_th, max_length=ground_truth_labels.size(0))
        objective = compute_objective_masked(decoder_output_rollin, cell_mask=cell_mask, target=ground_truth_labels,
                                             obj_func='softmax', obj_normalization=opt.obj_normalization,
                                             dataset_max_length=opt.max_pred_length)
    else:
        # run rollouts
        cost_tensor, target_labels, label_mask, cell_mask = \
            get_costs_by_rollouts(encoder, decoder, decoder_hidden, decoder_output_rollin, num_output_labels,
                                  output_lengths_pred_th, encoder_output_states, input_lengths_th, ground_truth_labels,
                                  use_teacher_forcing, output_lengths_gt_th, rollin_labels, opt)

        if opt.rollout == 'gt' and opt.rollin == 'gt':
            if (target_labels.masked_select(cell_mask.unsqueeze(2)) !=
                    ground_truth_labels.masked_select(cell_mask.unsqueeze(2))).long().sum().data[0] != 0:
                print('WARNING: mismatch of targets and gt for reference roll-in!')
                print(target_labels.masked_select(cell_mask.unsqueeze(2))
                      != ground_truth_labels.masked_select(cell_mask.unsqueeze(2)))
        temperature = 1
        obj_func = opt.objective.casefold()

        if obj_func == 'target-learning':
            obj_func = 'softmax'
        elif obj_func == 'target-learning-all-labels':
            obj_func = 'softmax'
            label_mask = None
        elif (obj_func == 'kl' or obj_func == 'inverse_kl' or obj_func == 'js' or obj_func == 'l2'
              or obj_func == 'loss-softmax' or obj_func == 'svm-cs'):
            temperature = opt.temperature

        objective = compute_objective_masked(decoder_output_rollin, cell_mask=cell_mask, label_mask=label_mask,
                                             costs=cost_tensor, target=target_labels, obj_func=obj_func,
                                             obj_normalization=opt.obj_normalization,
                                             dataset_max_length=opt.max_pred_length, temperature=temperature)

    # optimize
    objective.backward()

    # clip the gradient and do optimization step
    grad_norm = nn.utils.clip_grad_norm(chain(encoder.parameters(), decoder.parameters()), opt.max_grad_norm)

    # optimization step
    optimizer.step()

    assert(objective.numel() == 1)
    return objective.data[0], grad_norm


def train_seq2seq(encoder, decoder, optimizer, dataset_train, num_output_labels, opt, dataset_val=None):
    # init plotting and logging
    t_start = time.time()
    plot_objective_total, plot_grad_norm_total, num_steps_for_logging = 0.0, 0.0, 0
    full_log = init_logging(opt, train_evaluate_func=dataset_train.evaluate_func,
                            val_evaluate_func=dataset_val.evaluate_func)

    # create a subset of training set for evaluation
    dataset_train_for_eval = dataset_train.copy_subset(opt.eval_size)

    # setup the learning rate schedule
    lr_scheduler, anneal_lr_func = setup_lr(optimizer, full_log, opt)

    # evaluate the initial model
    dataset_train.shuffle()
    opt.rollin_ref_prob = apply_scheduled_sampling(0, opt)
    evaluate_and_log(encoder, decoder, dataset_train_for_eval, dataset_val, full_log, t_start, 0, float('nan'),
                     float('nan'), opt, anneal_lr_func)

    # start training
    i_epoch = 0
    i_batch = len(dataset_train)  # to start a new epoch at the first iteration, see l1005
    for i_iter in range(opt.max_iter):
        opt.iteration = i_iter
        # save models
        if opt.log_path and i_iter % opt.save_iter == 0:
            checkpoint_model(encoder, decoder, optimizer, i_iter, opt)

        # restart dataloader if needed
        data_sampling = opt.data_sampling.casefold()
        if i_batch >= len(dataset_train):
            i_epoch += 1
            i_batch = 0
            # shuffle dataset
            if data_sampling != 'fixed-order':
                dataset_train.shuffle()

        # get data for training
        if data_sampling == 'shuffle' or data_sampling == 'fixed-order':
            input_batch, input_lengths, labels, output_lengths, ids = dataset_train[i_batch]
        elif data_sampling == 'random':
            input_batch, input_lengths, labels, output_lengths, ids = dataset_train.get_random_batch()
        else:
            raise RuntimeError("Unknown data sampling strategy: " + data_sampling)

        i_batch += 1
        num_steps_for_logging += 1

        # apply scheduled sampling
        opt.rollin_ref_prob = apply_scheduled_sampling(i_iter, opt)

        # train on one batch
        objective, grad_norm = train_one_batch(input_batch, input_lengths, labels, output_lengths,
                                               encoder, decoder, optimizer, num_output_labels, opt)
        plot_objective_total += objective
        plot_grad_norm_total += grad_norm

        # print things
        if i_iter % opt.print_iter == 0:
            print('Iter %d of %d, epoch %d, time: %s, obj: %.4f, grad: %.4f' % (
                i_iter, opt.max_iter, i_epoch, time_since(t_start, (i_iter + 1) / opt.max_iter), objective, grad_norm))

        # evaluation
        if (i_iter + 1) % opt.eval_iter == 0:
            evaluate_and_log(encoder, decoder, dataset_train_for_eval, dataset_val, full_log, t_start, i_iter,
                             plot_objective_total / num_steps_for_logging, plot_grad_norm_total / num_steps_for_logging,
                             opt, (anneal_lr_func if i_iter > opt.lr_initial_patience
                                   else lambda: anneal_lr_func(anneal_now=False)))
            plot_objective_total, plot_grad_norm_total, num_steps_for_logging = 0.0, 0.0, 0

    # add the final point
    evaluate_and_log(encoder, decoder, dataset_train_for_eval, dataset_val, full_log,
                     t_start, opt.max_iter, float('nan'), float('nan'), opt)

    # save the final model
    if opt.log_path:
        checkpoint_model(encoder, decoder, optimizer, opt.max_iter, opt)
