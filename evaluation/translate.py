from evaluation.beam import Beam
from tensor_utils import *


def translate_batch(encoder, decoder, input_data, input_lengths, gt_lengths, opt):
    beam_size = opt.beam_size
    batch_size = input_data.size(1)
    # number of things to return, TODO make it a user setting
    n_best = 1
    max_output_size = max(gt_lengths)
    step_batch_sizes = get_batch_sizes_from_length(gt_lengths)

    # (1) Run the encoder on the src.
    init_hidden_state = encoder.init_hidden(batch_size)
    enc_states, context = encoder(init_hidden_state, input_data, input_lengths)
    dec_states = decoder.init_decoder_state(context, enc_states)

    th = torch.cuda if input_data.is_cuda else torch
    input_length_mask = lengths_to_mask(Variable(th.LongTensor(input_lengths)), max_length=context.size(0)).t()

    #  (1b) Initialize for the decoder.
    def var(a):
        return Variable(a, volatile=True)

    def rvar(a):
        return var(a.repeat(1, beam_size, 1))

    # Repeat everything beam_size times.
    context = rvar(context.data)
    dec_states.repeat_beam_size_times(beam_size)
    scorer = None
    beam = [Beam(beam_size, decoder.start_of_string_token, decoder.end_of_string_token, n_best=n_best, cuda=opt.cuda,
                 vocab=None, global_scorer=scorer) for _ in range(batch_size)]

    # (2) run the decoder to generate sentences, using beam search.
    def unbottle(m):
        return m.view(beam_size, batch_size, -1)

    for i in range(max_output_size):
        if all((b.done() for b in beam)):
            break
        # Construct batch x beam_size nxt words.
        # Get all the pending current beam words and arrange for forward.
        inp = var(torch.stack([b.get_current_state() for b in beam]).t().contiguous().view(1, -1))

        # Temporary kludge solution to handle changed dim expectation in the decoder
        inp = inp.unsqueeze(2)

        input_length_mask_this_step = input_length_mask[:step_batch_sizes[i], :]
        size_0, size_1 = input_length_mask_this_step.size()
        if size_0 != context.size(1):
            assert (context.size(1) == size_0 * beam_size)
            input_length_mask_this_step = input_length_mask_this_step.repeat(beam_size, 1)

        # Run one step.
        dec_out, dec_states, attn = decoder.original_forward(inp, context, dec_states, input_length_mask_this_step)
        dec_out = dec_out.squeeze(0)
        # dec_out: beam x rnn_size

        # (b) Compute a vector of batch*beam word scores.
        out = F.log_softmax(decoder.out(dec_out.unsqueeze(0)).squeeze(0) * opt.beam_scaling).data
        out = unbottle(out)
        # beam x tgt_vocab

        # (c) Advance each beam.
        for j, b in enumerate(beam):
            b.advance(out[:, j], unbottle(attn["std"]).data[:, j], opt.output_fixed_size)
            dec_states.beam_update(j, b.get_current_origin(), beam_size)

    # (3) Package everything up.
    all_hyps, all_scores, all_attn = [], [], []
    for (elem, b) in enumerate(beam):
        n_best = n_best
        scores, ks = b.sort_finished(minimum=n_best)
        hyps, attn = [], []
        for i, (times, k) in enumerate(ks[:n_best]):
            hyp, att = b.get_hyp(times, k)
            if len(hyp) < max_output_size:
                padding = [decoder.end_of_string_token] * (max_output_size - len(hyp))
                hyp.extend(padding)

            hyp[gt_lengths[elem]:max_output_size] = [decoder.end_of_string_token] * (max_output_size - gt_lengths[elem])
            hyps.append(hyp)
            attn.append(att)
        all_hyps.append(hyps)
        all_scores.append(scores)
        all_attn.append(attn)

    return all_hyps, all_scores, all_attn
