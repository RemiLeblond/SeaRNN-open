import numpy as np

import torch.optim.lr_scheduler

from reference_policy import *
from tensor_utils import *
from utils import *


def run_rollin(input_data, input_lengths, output_lengths, encoder, decoder, ground_truth_labels=None,
               output_fixed_size=True, use_teacher_forcing=False, reference_policy=None):
    # get data stats
    batch_size = input_data.size(1)

    # CPU vs GPU
    th = torch.cuda if input_data.is_cuda else torch

    # parse ground truth if it exists
    if ground_truth_labels is not None:
        assert (ground_truth_labels.size(1) == batch_size)

    # create vars to feed encoder
    init_hidden_state = encoder.init_hidden(batch_size)

    # feed inputs into the encoder
    encoder_final_hidden_states, encoder_output_states = encoder(init_hidden_state, input_data, input_lengths)
    # need BATCH_SIZE x SEQ_LEN x DIM for attention
    encoder_output_states = encoder_output_states.transpose(0, 1).contiguous()

    # need to reshape encoder
    if encoder.bidirectional:
        encoder_final_hidden_states = process_bidirectional_encoder_memory(encoder_final_hidden_states)
        assert (decoder.project_encoder is not None and decoder.project_context is not None)
        encoder_final_hidden_states = decoder.project_encoder(encoder_final_hidden_states)
        encoder_output_states = decoder.project_context(encoder_output_states)

    # sequence_lengths (need in torch format)
    input_lengths_th = Variable(th.LongTensor(input_lengths))
    output_lengths_th = Variable(th.LongTensor(output_lengths))

    # change the order of sequences: decoder requires output sequences sorted in decreasing order of output length
    # initially data is sorted in the decreasing order of input length
    output_lengths_th_resorted, order = torch.sort(output_lengths_th, 0, descending=True)
    encoder_final_hidden_states_resorted = torch.index_select(encoder_final_hidden_states, 1, order)
    ground_truth_labels_resorted = (torch.index_select(ground_truth_labels, 1, order) if ground_truth_labels is not None
                                    else None)
    encoder_output_states_resorted = torch.index_select(encoder_output_states, 0, order)
    input_lengths_th_resorted = torch.index_select(input_lengths_th, 0, order)

    decoder_output, decoder_hidden, decoder_attention, labels_seen_by_decoder = \
        decoder(encoder_final_hidden_states_resorted, output_lengths_th_resorted,
                ground_truth_labels_resorted, output_fixed_size, use_teacher_forcing,
                encoder_outputs=encoder_output_states_resorted, input_lengths=input_lengths_th_resorted,
                reference_policy=reference_policy)

    # unsort outputs of the decoder
    _, order_reversed = torch.sort(order, 0, descending=False)
    decoder_output = torch.index_select(decoder_output, 1, order_reversed)
    decoder_hidden = torch.index_select(decoder_hidden, 2, order_reversed)
    if decoder_attention is not None:
        decoder_attention = torch.index_select(decoder_attention, 1, order_reversed)
    labels_seen_by_decoder = torch.index_select(labels_seen_by_decoder, 1, order_reversed)

    return decoder_output, decoder_hidden, encoder_output_states, decoder_attention, labels_seen_by_decoder


def compute_objective_masked(outputs, cell_mask, label_mask=None, costs=None, target=None, obj_func='softmax',
                             obj_normalization='cell-per-seq-batch', dataset_max_length=None, temperature=1):
    """
    Args:
        outputs: A Variable containing a FloatTensor of size
            (max_len, batch, num_classes) which contains the scores for each class
        cell_mask: A Variable containing a ByteTensor of size
            (max_len, batch) containing the mask of active cells
        label_mask: A Variable containing a ByteTensor of size
            (max_len, batch, num_classes) which contains the mask of all active labels
        costs: A Variable containing a FloatTensor of size
            (max_len, batch, num_classes) which contains the cost for each class
        target: A Variable containing a LongTensor of size
            (max_len, batch, 1) which contains the index of the true
            class for each corresponding step
        obj_func: the loss to use
        obj_normalization: 'none' | 'batch' | 'cell-per-seq-batch' | 'cell-global'
        dataset_max_length: biggest length in the dataset
        temperature: scaling parameter for some loss functions
    Returns:
        objective: An average objective value masked by the length
    """
    batch_size = outputs.size(1)
    num_classes = outputs.size(2)
    cell_mask_full = cell_mask.unsqueeze(2).expand_as(outputs)

    # CPU vs GPU
    th = torch.cuda if outputs.is_cuda else torch

    # flatten everything
    # outputs_flat: (num_cells, num_classes)
    outputs_flat = torch.masked_select(outputs, cell_mask_full)
    outputs_flat = outputs_flat.view(-1, outputs.size(-1))

    # target_flat: (num_cells, 1)
    if target is not None:
        target_flat = torch.masked_select(target, cell_mask.unsqueeze(2))
        target_flat = target_flat.view(-1, 1)

    # label_mask_flat: (num_cells, num_classes)
    if label_mask is not None:
        label_mask_flat = torch.masked_select(label_mask, cell_mask_full)
        label_mask_flat = label_mask_flat.view(-1, label_mask.size(-1))

    # costs_flat: (num_cells, num_classes)
    if costs is not None:
        costs_flat = torch.masked_select(costs, cell_mask_full)
        costs_flat = costs_flat.view(-1, costs.size(-1))
        if label_mask is None:
            min_costs, _ = costs_flat.min(1, keepdim=True)
            max_costs, _ = costs_flat.max(1, keepdim=True)
        else:
            min_costs = -compute_max_masked(-costs_flat, label_mask_flat)
            max_costs = compute_max_masked(costs_flat, label_mask_flat)
        costs_flat = costs_flat - min_costs.expand_as(costs_flat)

    obj_func = obj_func.casefold()
    if obj_func == 'softmax' or obj_func == 'loss-softmax':
        if obj_func == 'loss-softmax':
            outputs_flat = outputs_flat + costs_flat * temperature

        # apply label mask
        if label_mask is None:
            # log_probs_flat: (num_cells, num_classes)
            log_probs_flat = F.log_softmax(outputs_flat)
        else:
            # log_probs_flat: (num_cells, num_classes)
            log_probs_flat = compute_log_softmax_masked(outputs_flat, label_mask_flat)

        # objectives_flat: (num_cells, 1)
        objectives_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)

    elif obj_func == 'kl':
        if label_mask is None:
            log_probs_flat = F.log_softmax(outputs_flat)
        else:
            log_probs_flat = compute_log_softmax_masked(outputs_flat, label_mask_flat)

        # apply temperature
        target_distribution = compute_softmax_masked(- costs_flat * temperature, label_mask_flat)
        objectives_flat = - torch.sum(target_distribution * log_probs_flat, -1).unsqueeze(-1)

    elif obj_func == 'inverse_kl':
        if label_mask is None:
            target_distribution = F.softmax(-costs_flat * temperature)
            log_probs_flat = F.softmax(outputs_flat)
        else:
            target_distribution = compute_softmax_masked(-costs_flat * temperature, label_mask_flat) + 1e-12
            log_probs_flat = compute_softmax_masked(outputs_flat, label_mask_flat) + 1e-12

        objectives_flat = torch.sum(log_probs_flat * (torch.log(log_probs_flat) - torch.log(target_distribution)),
                                    -1).unsqueeze(-1)

    elif obj_func == 'js':
        if label_mask is None:
            target_distribution = F.softmax(-costs_flat * temperature)
            log_probs_flat = F.softmax(outputs_flat)
        else:
            target_distribution = compute_softmax_masked(-costs_flat * temperature, label_mask_flat)
            log_probs_flat = compute_softmax_masked(outputs_flat, label_mask_flat)
        mean_distribution = (target_distribution + log_probs_flat) / 2

        objectives_flat = (-1 / 2 * torch.sum(target_distribution * torch.log(mean_distribution), -1).unsqueeze(-1) +
                           1 / 2 * torch.sum(
                               log_probs_flat * (torch.log(log_probs_flat) - torch.log(mean_distribution)),
                               -1).unsqueeze(-1))
    elif obj_func == 'l2':
        centered_costs = costs_flat - max_costs.expand_as(costs_flat) / 2
        #  we're trying to learn - cost since we want the biggest one to be predicted.
        differences = outputs_flat + centered_costs * temperature
        differences.data.masked_fill_(label_mask_flat.data == 0, 0.0)
        objectives_flat = torch.sum((differences * differences), -1).unsqueeze(-1)

    elif obj_func == 'svm-cs':
        # apply label mask
        outputs_flat = outputs_flat + costs_flat * temperature
        if label_mask is None:
            # loss-augmented inference
            scores_loss_augmented, _ = torch.max(outputs_flat, 1, keepdim=True)
        else:
            scores_loss_augmented = compute_max_masked(outputs_flat, label_mask_flat)

        # ground truth scores
        scores_gt = torch.gather(outputs_flat, dim=1, index=target_flat)
        # objectives_flat: (num_cells, 1)
        objectives_flat = scores_loss_augmented - scores_gt
    else:
        raise (RuntimeError("Unknown objective function {0}".format(obj_func)))

    # normalize the objective
    if obj_normalization == 'batch':
        objectives_flat = objectives_flat / batch_size
    elif obj_normalization == 'batch-maxlen':
        assert (dataset_max_length is not None)
        objectives_flat = objectives_flat / (batch_size * dataset_max_length)
    elif obj_normalization == 'cell-per-seq-batch':
        # mask, normalization: (max_len, batch)
        normalization = cell_mask.float().sum(0).expand_as(cell_mask)
        normalization_flat = torch.masked_select(normalization, cell_mask)
        objectives_flat = objectives_flat / normalization_flat
        objectives_flat = objectives_flat / batch_size
    elif obj_normalization == 'cell-global':
        objectives_flat = objectives_flat / objectives_flat.numel()
    elif obj_normalization == 'none':
        pass
    else:
        raise (RuntimeError("Unknown normalization type {0}".format(obj_normalization)))

    # add NaN protection
    num_nans, good_item_mask = count_nans_in_var(objectives_flat)
    if num_nans > 0:
        if num_nans == objectives_flat.numel():
            raise (RuntimeError("All objectives are NaN"))
        print('WARNING: %d of %d objectives are NaN, ignoring them' % (num_nans, objectives_flat.numel()))
        objectives_flat = objectives_flat.masked_select(good_item_mask)

    # accumulate averything
    objective = torch.sum(objectives_flat)

    return objective


def create_labels_for_rollout(ground_truth_labels, output_lengths, rollin_scores, rollin_labels, num_output_labels,
                              eos_token, opt):
    batch_size = ground_truth_labels.size(1)
    rollout_mask = torch.ByteTensor(rollin_scores.size()).zero_()  # SEQ_LEN x BATCH_SIZE x NUM_LABELS

    # find lengths of rollin sequences
    rollin_lengths = compute_prediction_lengths(rollin_labels, eos_token)

    output_lengths, rollin_lengths = output_lengths.data.cpu(), rollin_lengths.data.cpu()

    # find all labels above the targeted sampling threshold if needed
    # TODO maybe only start after a few iterations, because here the init is random?
    if opt.targeted_sampling:
        rollin_probs = F.softmax(rollin_scores)
        predictions, _ = torch.max(rollin_probs, dim=2)
        candidates = torch.gt(rollin_probs, predictions.unsqueeze(-1).expand_as(rollin_probs) - opt.ts_threshold)
        total_candidates = torch.sum(candidates.float()) / batch_size

        # remove candidates if there are too many
        if total_candidates.data[0] > opt.ts_max_samples:
            random_mask = torch.FloatTensor(candidates.size()).fill_(opt.ts_max_samples / total_candidates.data[0])
            random_mask = torch.bernoulli(random_mask)
            candidates = candidates.data.cpu() * random_mask.byte()

        else:
            candidates = candidates.data.cpu()

        # Now add the ground truth things
        ground_truth_mask = torch.zeros(candidates.size()).scatter_(2, ground_truth_labels.data.cpu(),
                                                                    torch.ones(candidates.size())).byte()
        rollout_mask = torch.max(candidates, ground_truth_mask)

        # TODO find a cleaner way to do this
        for i_item in range(batch_size):
            for cell in range(ground_truth_labels.size(0)):
                if cell >= rollin_lengths[i_item]:
                    rollout_mask[cell, i_item, :] = 0

    else:
        rollout_mask = rollout_mask.view(-1)
        # find all topk labels in advance if needed
        if opt.sample_labels_policy_topk > 0 and opt.sample_labels_uniform < num_output_labels:
            _, labels_top_policy = torch.topk(rollin_scores, opt.sample_labels_policy_topk, dim=2)
            labels_top_policy = labels_top_policy.data.cpu()

        all_cells = np.arange(rollin_scores.size(0))
        for i_item in range(batch_size):
            cur_length = rollin_lengths[i_item]

            if cur_length > 0:
                num_cells_to_rollout = min(cur_length,
                                           opt.num_cells_to_rollout) if opt.num_cells_to_rollout else cur_length
                random_cells = np.random.choice(all_cells[:cur_length], size=num_cells_to_rollout, replace=False)
            else:
                random_cells = np.array([0])

            if not opt.output_fixed_size:
                # add special cells for non-fixed output length
                if (cur_length - 1) not in random_cells and cur_length > 0:
                    # add last position before EOS
                    random_cells = np.append(random_cells, cur_length - 1)
            if cur_length not in random_cells and cur_length < output_lengths[i_item]:
                # add first position after EOS if relevant
                random_cells = np.append(random_cells, cur_length)

            gt_this_item = ground_truth_labels[:, i_item, 0].data.cpu()
            # loop over all rollout positions
            for i_cell in random_cells:
                # generate labels to roll-out at the current cell
                if opt.sample_labels_uniform >= num_output_labels:
                    # simply take all the labels
                    labels_this_cell = torch.arange(0, num_output_labels).long()
                else:
                    labels_this_cell = []

                    # add ground truth
                    labels_this_cell.append(torch.LongTensor([gt_this_item[i_cell]]))

                    # always try to add EOS
                    if not opt.output_fixed_size:
                        labels_this_cell.append(torch.LongTensor([eos_token]))

                    if opt.sample_labels_uniform > 0:
                        labels_random = np.random.randint(0, high=num_output_labels, size=opt.sample_labels_uniform)
                        labels_random = torch.LongTensor(labels_random)
                        labels_this_cell.append(labels_random)

                    if opt.sample_labels_policy_topk > 0:
                        labels_this_cell.append(labels_top_policy[i_cell, i_item])

                    if opt.sample_labels_neighbors > 0:
                        start_slice = max(i_cell - opt.sample_labels_neighbors, 0)
                        end_slice = min(i_cell + opt.sample_labels_neighbors + 1, gt_this_item.size(0))
                        labels_neighbor = gt_this_item[start_slice: end_slice]
                        labels_this_cell.append(labels_neighbor)

                    labels_this_cell = torch.cat(labels_this_cell, 0)

                mask_indices = labels_this_cell + (
                    rollin_scores.size(2) * int(i_item) + rollin_scores.size(1) * rollin_scores.size(2) * int(i_cell))
                rollout_mask[mask_indices] = 1

        rollout_mask = rollout_mask.view(rollin_scores.size())  # SEQ_LEN x BATCH_SIZE x NUM_LABELS

    if rollin_scores.is_cuda:
        rollout_mask = rollout_mask.cuda()

    if rollout_mask.long().sum() == 0:
        raise RuntimeError('No (cell, action) pairs sampled for rollout')

    return rollout_mask


def encode_rollout_map(rollout_mask, output_lengths, eos_token, opt,
                       rollout_mode='gt', use_teacher_forcing=True,
                       rollout_learned_prob=0.5):
    # rollout_mask: SEQ_LEN x BATCH_SIZE x NUM_LABELS
    batch_size = rollout_mask.size(1)
    max_num_labels = rollout_mask.size(2)
    max_num_cells = rollout_mask.size(0)
    joint_num_labels = rollout_mask.long().sum()

    # CPU vs GPU
    th = torch.cuda if rollout_mask.is_cuda else torch

    batch_index = torch.arange(0, batch_size).long()
    cell_index = torch.arange(0, max_num_cells).long()
    label_index = torch.arange(0, max_num_labels).long()
    if rollout_mask.is_cuda:
        batch_index, cell_index, label_index = batch_index.cuda(), cell_index.cuda(), label_index.cuda()

    # expand sizes
    batch_index = batch_index.unsqueeze(0).unsqueeze(2).expand_as(rollout_mask)
    cell_index = cell_index.unsqueeze(1).unsqueeze(2).expand_as(rollout_mask)
    label_index = label_index.unsqueeze(0).unsqueeze(0).expand_as(rollout_mask)

    # decide which rollout to use
    rollout_mode = rollout_mode.casefold()
    if rollout_mode == 'learned' or rollout_mode == 'focused-costing':
        use_learned_rollout = torch.ones(batch_size)
    elif rollout_mode == 'gt':
        use_learned_rollout = torch.zeros(batch_size)
    elif rollout_mode == 'mixed':
        use_learned_rollout = torch.bernoulli(torch.ones(batch_size) * rollout_learned_prob)
    elif rollout_mode == 'mixed-matched':
        use_learned_rollout = torch.bernoulli(torch.ones(batch_size) * rollout_learned_prob)
        if isinstance(use_teacher_forcing, bool):
            if use_teacher_forcing:
                use_learned_rollout.fill_(False)
        else:
            use_learned_rollout.masked_fill_(use_teacher_forcing.cpu() == 0, False)
    else:
        raise (RuntimeError("Unknown roll-out strategy %s" % rollout_mode))

    if rollout_mask.is_cuda:
        use_learned_rollout = use_learned_rollout.long().cuda()

    use_learned_rollout = use_learned_rollout.unsqueeze(0).unsqueeze(2).expand_as(rollout_mask)
    output_lengths = output_lengths.data.unsqueeze(0).unsqueeze(2).expand_as(rollout_mask)
    num_steps = output_lengths - cell_index - 1

    if rollout_mode == 'focused-costing':  # first steps of learned and finish with reference
        focused_steps = math.floor(opt.fc_initial_value + opt.fc_increment * opt.iteration / opt.fc_epoch)
        learned_steps = th.LongTensor(1).fill_(focused_steps).expand_as(num_steps)
        num_steps = torch.min(num_steps, learned_steps)

    rollout_data = th.LongTensor(7, joint_num_labels).fill_(0)
    rollout_data[0, :] = torch.masked_select(batch_index, rollout_mask)
    rollout_data[1, :] = torch.masked_select(cell_index, rollout_mask)
    rollout_data[2, :] = torch.masked_select(label_index, rollout_mask)
    rollout_data[3, :] = torch.masked_select(use_learned_rollout, rollout_mask)
    rollout_data[4, :] = torch.masked_select(num_steps, rollout_mask)
    rollout_data[5, :] = torch.masked_select(output_lengths, rollout_mask)
    rollout_data[6, :] = torch.arange(0, joint_num_labels).type_as(rollout_data)  # original indices

    assert (rollout_data[4, :].min() >= 0)

    # if the first symbol is EOS do not do rollout steps
    eos_ids = torch.nonzero(rollout_data[2, :] == eos_token)
    if eos_ids.numel() > 0:
        rollout_data[4][eos_ids.squeeze(1)] = 0

    # sort in the decreasing order of num steps
    _, order = torch.sort(rollout_data[4, :], 0, descending=True)
    rollout_data = torch.index_select(rollout_data, 1, order)
    rollout_data = Variable(rollout_data.contiguous(), volatile=True)

    return rollout_data


def decode_rollout_map(rollout_mask, rollout_data, rollout_costs, target_noise_std=1e-5, eos=-1):
    # unsort the costs
    backup = rollout_costs.clone()
    rollout_costs.index_copy_(0, rollout_data[6, :], backup)

    # CPU vs GPU
    th = torch.cuda if rollout_mask.is_cuda else torch

    # init costs
    cost_tensor = th.FloatTensor(rollout_mask.size())
    cost_tensor.fill_(float('inf'))

    # collect all the costs
    cost_tensor.masked_scatter_(rollout_mask, rollout_costs.data)
    cost_tensor = Variable(cost_tensor)

    # mask of cells with costs
    cell_mask, _ = torch.max(rollout_mask, 2, keepdim=False)  # mask of active cells: SEQ_LEN x BATCH_SIZE
    cell_mask = Variable(cell_mask)

    # select targets
    target_noise = cost_tensor.data.clone().normal_(0, target_noise_std)  # add noise to target labels to break ties
    target_noise = Variable(target_noise)

    # careful computation of min to avoid items not in the mask
    losses = (cost_tensor + target_noise).view(-1, cost_tensor.size(-1))
    mask = rollout_mask.view(-1, rollout_mask.size(-1))
    target_labels = compute_argmax_masked(-losses, mask)
    target_labels = target_labels.view(rollout_mask.size(0), rollout_mask.size(1))
    # pad with EOS symbol if provided
    target_labels.masked_fill_(cell_mask.data == 0, eos)
    target_labels = target_labels.unsqueeze(2)

    return cost_tensor, target_labels, cell_mask


def compute_costs(predictions, labels, label_lengths, eos, opt):
    target_loss = opt.loss.casefold()

    if target_loss == 'hamming':
        costs = compute_hamming_loss_masked(predictions, labels, label_lengths)
    elif target_loss == 'hamming-unnorm':
        normalization_constant = 1.0
        costs = compute_hamming_loss_masked(predictions, labels, label_lengths,
                                            normalization_constant=normalization_constant)
    elif target_loss == 'bleu-smoothed':
        costs = compute_sentence_loss_masked(predictions, labels, eos=eos, loss_type='bleu4')
    elif target_loss == 'bleu1-smoothed':
        costs = compute_sentence_loss_masked(predictions, labels, eos=eos, loss_type='bleu1')
    elif target_loss == 'gleu':
        costs = compute_sentence_loss_masked(predictions, labels, eos=eos, loss_type='gleu')
    elif target_loss == 'sentence-f1':
        costs = compute_sentence_F1_loss(predictions, labels, label_lengths)
    else:
        raise (RuntimeError("Unknown target loss %s" % target_loss))

    return costs


def apply_reference_policy(ground_truth_labels, rollout_data, rollin_labels, opt, eos=None, focused_costing=False):
    index_batch = rollout_data[0]
    index_cell = rollout_data[1]
    first_label_to_feed_all = rollout_data[2]
    use_learned_rollout_all = rollout_data[3]
    max_length = rollin_labels.size(0)

    # CPU vs GPU
    th = torch.cuda if rollout_data.is_cuda else torch
    # In case we're doing focused costing, this is the last cell that was predicted in the partial learned rollout
    if focused_costing:
        use_learned_rollout_all = Variable(th.LongTensor(1).fill_(0).expand_as(rollout_data[3]))
        index_cell = index_cell + rollout_data[4] + 1

    # copy predictions from ground truth
    # reference_labels contains the ground truth
    reference_labels = torch.index_select(ground_truth_labels, 1, index_batch).clone()
    reference_labels = reference_labels[:max_length]  # in case GT labels are longer than necessary

    # preparations
    rollin_labels_unrolled = rollin_labels if focused_costing else torch.index_select(rollin_labels, 1, index_batch)
    mask_known_labels = lengths_to_mask(index_cell + 1, max_length=max_length)

    # copy in current labels: has to be before rollin labels as it gets overwritten later
    # reference_labels contains the arbitrary deviations from cell 0 to the actual index cell, and then GT tokens
    first_label_expanded = first_label_to_feed_all.unsqueeze(0).unsqueeze(2).expand_as(rollin_labels_unrolled)
    labels_to_copy = first_label_expanded.masked_select(mask_known_labels.unsqueeze(2))
    if not focused_costing:
        reference_labels.masked_scatter_(mask_known_labels.unsqueeze(2), labels_to_copy)

    # copy in roll-in labels. Overwrites the previous filling, where we put the arbitrary decision everywhere.
    # reference_labels contains the roll-in labels, the arbitrary deviations and then the ground truth labels
    mask_rollin = lengths_to_mask(index_cell, max_length=max_length)
    labels_to_copy = rollin_labels_unrolled.masked_select(mask_rollin.unsqueeze(2))
    if labels_to_copy.numel() > 0:
        reference_labels.masked_scatter_(mask_rollin.unsqueeze(2), labels_to_copy)

    # apply fancier reference policy
    if hasattr(opt, 'reference_policy'):
        reference_policy = opt.reference_policy.casefold()

        # select positions where reference policy has to be applied
        indices_for_reference = Variable(torch.nonzero(1 - use_learned_rollout_all.data))
        indices_for_reference = indices_for_reference.view(-1)
        if rollout_data.is_cuda:
            indices_for_reference = indices_for_reference.cuda()

        if indices_for_reference.numel() > 0:
            # get prefixes for the ref policy
            prefixes = torch.index_select(reference_labels, 1, indices_for_reference)
            prefix_lengths = torch.index_select(index_cell + 1, 0, indices_for_reference)

            # get gt labels for the ref policy
            gt_for_policy = torch.index_select(ground_truth_labels, 1, index_batch)
            gt_for_policy = torch.index_select(gt_for_policy, 1, indices_for_reference)

            # apply the ref policy
            flag_update_reference_labels = False
            if reference_policy == 'bleu-best-suffix':
                reference_costs, labels_for_costs = reference_policy_best_suffix(prefixes, prefix_lengths,
                                                                                 gt_for_policy, eos, loss_type='bleu4')
                labels_for_costs = make_eos_final(labels_for_costs, eos)
                flag_update_reference_labels = True
            elif reference_policy == 'bleu1-best-suffix':
                reference_costs, labels_for_costs = reference_policy_best_suffix(prefixes, prefix_lengths,
                                                                                 gt_for_policy, eos, loss_type='bleu1')
                labels_for_costs = make_eos_final(labels_for_costs, eos)
                flag_update_reference_labels = True
            elif reference_policy == 'gleu-best-suffix':
                reference_costs, labels_for_costs = reference_policy_best_suffix(prefixes, prefix_lengths,
                                                                                 gt_for_policy, eos, loss_type='gleu')
                labels_for_costs = make_eos_final(labels_for_costs, eos)
                flag_update_reference_labels = True
            elif reference_policy == 'maximize-f1':
                labels_for_costs = reference_policy_max_F1(prefixes, prefix_lengths, gt_for_policy, eos)
                flag_update_reference_labels = True
            elif reference_policy == 'copy-gt':
                pass
            else:
                raise (RuntimeError("Unknown reference policy %s" % reference_policy))

            # copy results of the reference policy into the joint array
            if flag_update_reference_labels:
                if labels_for_costs.size(0) > reference_labels.size(0):
                    reference_labels = pad_tensor(reference_labels, 0, labels_for_costs.size(0),
                                                  value=eos if eos is not None else 0)
                reference_labels.index_copy_(1, indices_for_reference, labels_for_costs)

    return reference_labels


def get_costs_by_rollouts(encoder, decoder, decoder_hidden, decoder_output_rollin, num_output_labels,
                          output_lengths_pred_th, encoder_output_states, input_lengths_th, ground_truth_labels,
                          use_teacher_forcing, output_lengths_gt_th, rollin_labels, opt):
    # sync dimensions
    rollin_labels, ground_truth_labels = sync_dim_size(rollin_labels, ground_truth_labels, 0,
                                                       decoder.end_of_string_token)
    # combine rollin with the output length (the last scores are filled with garbage)
    mask_rollin = lengths_to_mask(sequence_length=output_lengths_pred_th, max_length=rollin_labels.size(0))
    rollin_labels.masked_fill_(mask_rollin.unsqueeze(2) == 0, decoder.end_of_string_token)

    # set which labels to rollout (can depend on the results of rollin)
    rollout_mask = create_labels_for_rollout(ground_truth_labels, output_lengths_pred_th, decoder_output_rollin,
                                             rollin_labels, num_output_labels, decoder.end_of_string_token, opt)
    rollout_data = encode_rollout_map(rollout_mask, output_lengths_pred_th, decoder.end_of_string_token, opt,
                                      rollout_mode=opt.rollout, rollout_learned_prob=1.0 - opt.rollout_ref_prob,
                                      use_teacher_forcing=use_teacher_forcing)

    # apply reference policy
    reference_labels = apply_reference_policy(ground_truth_labels, rollout_data, rollin_labels,
                                              opt, eos=decoder.end_of_string_token)

    # do rollout
    encoder.eval()
    decoder.eval()
    predictions = decoder.rollout_one_batch(rollout_data, decoder_hidden, reference_labels,
                                            encoder_output_states=encoder_output_states, input_lengths=input_lengths_th,
                                            output_fixed_size=opt.output_fixed_size,
                                            rollout_batch_size=opt.rollout_batch_size)

    if opt.rollout == 'focused-costing':
        predictions = apply_reference_policy(ground_truth_labels, rollout_data, predictions, opt,
                                             eos=decoder.end_of_string_token, focused_costing=True)

    # change all predictions after EOS symbol to EOS
    predictions = make_eos_final(predictions, decoder.end_of_string_token)

    # collect data and sync lengths
    # collect the ground-truth labels
    ground_truth_labels_all = torch.index_select(ground_truth_labels, 1, rollout_data[0])
    # retrieve sequence length from the batch level info
    rollout_output_lengths_gt = torch.index_select(output_lengths_gt_th, 0, rollout_data[0])
    predictions, ground_truth_labels_all = sync_dim_size(predictions, ground_truth_labels_all, 0,
                                                        decoder.end_of_string_token)

    # compute the losses
    rollout_costs = compute_costs(predictions, ground_truth_labels_all, rollout_output_lengths_gt,
                                  decoder.end_of_string_token, opt)

    # decode costs
    cost_tensor, target_labels, cell_mask = \
        decode_rollout_map(rollout_mask, rollout_data, rollout_costs, target_noise_std=1e-5,
                           eos=decoder.end_of_string_token)
    label_mask = Variable(rollout_mask)

    return cost_tensor, target_labels, label_mask, cell_mask


def apply_scheduled_sampling(i_iter, opt):
    scheduled_sampling = opt.scheduled_sampling.casefold()

    if scheduled_sampling == 'none':
        rollin_ref_prob = opt.rollin_ref_prob
    elif scheduled_sampling == 'sigmoid':
        rollin_ref_prob = 1 / (1 + math.exp(12.0 * i_iter / (opt.max_iter - 1) - 6))
    elif scheduled_sampling == 'exponential-samy':
        if i_iter >= 10**5:
            rollin_ref_prob = 0.9
        else:
            rollin_ref_prob = math.exp(math.log(0.9) * i_iter / 10**5)
    else:
        raise RuntimeError("Invalid value of scheduled_sampling: " + scheduled_sampling)

    return rollin_ref_prob
