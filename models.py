import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

from utils import aeq
from tensor_utils import *
from gru_cells import StackedGRU

"""
    Definition of the RNN models and the associated attention mechanism.
    Inspired by openNMT-py (https://github.com/OpenNMT/OpenNMT-py).
"""


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False, input_embedding=None, dropout=0):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.dropout_prob = dropout
        if input_embedding is not None:
            self.input_embedding = WrapForRNN(input_embedding)
        else:
            self.input_embedding = None
        self.rnn = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=False,
                          bidirectional=bidirectional, dropout=self.dropout_prob)

        # add dropout on embeddings
        self.dropout_emb = nn.Dropout(self.dropout_prob)

    def forward(self, hidden, input, lengths=None):
        if self.input_embedding is not None:
            input = self.input_embedding(input)

        # add dropout on embeddings
        input = self.dropout_emb(input)

        if lengths is not None:
            input = pack(input, lengths)

        output, hidden_t = self.rnn(input, hidden)

        if lengths is not None:
            output, lengths = unpack(output)

        return hidden_t, output

    def init_hidden(self, batch_size):
        num_chains = self.num_layers
        if self.bidirectional:
            num_chains *= 2
        th = torch.cuda if self.is_cuda() else torch
        return Variable(th.FloatTensor(num_chains, batch_size, self.hidden_size).fill_(0))

    def is_cuda(self):
        return next(self.parameters()).is_cuda


class GlobalAttention(nn.Module):
    """
    Attention as implemented in OpenNMT-py:
    https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/modules/GlobalAttention.py
    """
    def __init__(self, dim, context_dim=None, attn_type='matrix'):
        super(GlobalAttention, self).__init__()
        context_dim = dim if context_dim is None else context_dim
        self.attn_type = attn_type
        if self.attn_type == 'matrix':
            self.linear_in = WrapForRNN(LinearSeq2seq(dim, context_dim, bias=False))
        elif self.attn_type == 'sum-tanh':
            self.linear_in_context = WrapForRNN(LinearSeq2seq(context_dim, dim, bias=False))
            self.linear_in = WrapForRNN(LinearSeq2seq(dim, dim, bias=False))
            self.linear_v = WrapForRNN(LinearSeq2seq(dim, 1, bias=False))
        else:
            raise (RuntimeError("Unknown attention type %s" % self.attn_type))

        self.linear_out = WrapForRNN(LinearSeq2seq(dim + context_dim, dim, bias=False))
        self.mask = None

    def forward(self, input, context, mask=None):
        """
        input: 1 x BATCH_SIZE x DIM
        context: BATCH_SIZE x SEQ_LEN x CONT_DIM
        mask:  BATCH_SIZE x SEQ_LEN
        """

        if self.attn_type == 'matrix':
            query = self.linear_in(input)  # 1 x BATCH_SIZE x CONT_DIM
            # transform dimensions for search
            query_search = query.squeeze(0).unsqueeze(2)  # BATCH_SIZE x CONT_DIM x 1
            # Get attention
            attn = torch.bmm(context, query_search).squeeze(2)  # BATCH_SIZE x SEQ_LEN

        elif self.attn_type == 'sum-tanh':
            # apply transformations
            query = self.linear_in(input)
            query_search = query.squeeze(0).unsqueeze(1)  # BATCH_SIZE x 1 x DIM
            context_transformed = self.linear_in_context(context)  # BATCH_SIZE x SEQ_LEN x DIM

            # combine
            combined = context_transformed + query_search.expand_as(context_transformed)
            combined = F.tanh(combined)   # BATCH_SIZE x SEQ_LEN x DIM

            attn = self.linear_v(combined)  # BATCH_SIZE x SEQ_LEN x 1
            attn = attn.squeeze(2)  # BATCH_SIZE x SEQ_LEN
        else:
            raise (RuntimeError("Unknown attention type %s" % self.attn_type))

        # use the mask not to consider attention to non-existing elements
        if mask is not None:
            soft_mask = compute_softmax_masked(attn, mask)
        else:
            soft_mask = F.softmax(attn)

        # multiply attention by context
        soft_mask = soft_mask.view(attn.size(0), 1, soft_mask.size(1))  # BATCH_SIZE x 1 x SEQ_LEN
        context_with_attention = torch.bmm(soft_mask, context).squeeze(1)  # BATCH_SIZE x CONT_DIM

        # combine context with original input: 1 x BATCH_SIZE x (CONT_DIM + DIM)
        context_combined = torch.cat([context_with_attention, input.squeeze(0)], 1).unsqueeze(0)

        out = self.linear_out(context_combined)
        if self.attn_type == 'matrix':
            out = F.tanh(out)
        return out, soft_mask


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, dictionary_size, end_of_string_token, num_layers=1, emb_size=None,
                 use_attention=False, encoder_state_size=None, bidirectional_encoder=False, dropout=0, input_feed=False,
                 attn_type='matrix', start_of_string_token=None):
        super(DecoderRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.end_of_string_token = end_of_string_token
        self.dictionary_size = dictionary_size
        self.emb_size = emb_size if emb_size is not None else hidden_size
        self.input_feed = input_feed
        self.dropout_prob = dropout
        self.bidirectional_encoder = bidirectional_encoder
        self.encoder_state_size = encoder_state_size * 2 if self.bidirectional_encoder else encoder_state_size

        input_size = self.emb_size
        if self.input_feed:
            input_size += self.hidden_size

        # create embeddings for output symbols
        if start_of_string_token is not None:
            self.start_of_string_token = start_of_string_token
            self.embedding = WrapForRNN(EmbeddingSeq2seq(dictionary_size, self.emb_size))
        else:
            self.start_of_string_token = dictionary_size
            self.embedding = WrapForRNN(EmbeddingSeq2seq(dictionary_size + 1, self.emb_size))

        self.rnn = StackedGRU(input_size, hidden_size, num_layers=num_layers, dropout=self.dropout_prob)
        self.out = WrapForRNN(LinearSeq2seq(hidden_size, dictionary_size))

        self.use_attention = use_attention
        if self.use_attention:
            self.attention = GlobalAttention(hidden_size, self.encoder_state_size,
                                             attn_type=attn_type)
        else:
            self.attention = None

        # add dropout on embeddings
        self.dropout_emb = nn.Dropout(self.dropout_prob)
        self.dropout_output = nn.Dropout(self.dropout_prob)

        if self.encoder_state_size is not None:
            # layer to project inout to hidden size (for bidirectional encoders)
            if self.encoder_state_size != hidden_size:
                self.project_encoder = nn.Sequential(
                    WrapForRNN(LinearSeq2seq(self.encoder_state_size, hidden_size)),
                    nn.Tanh())
            else:
                self.project_encoder = IdleLayer()

            self.project_context = IdleLayer()
        else:
            self.project_encoder = None
            self.project_context = None

    def forward_given_inputs(self, hidden, input, encoder_context=None, mask=None, prev_output=None):
        # push data through RNN
        embedding = self.embedding(input)
        embedding = self.dropout_emb(embedding)

        # concatenate for input feed
        if self.input_feed:
            assert(prev_output is not None)
            embedding = torch.cat([embedding, prev_output], 2)

        # push data through RNN
        output_rnn, hidden_t = self.rnn(embedding, hidden)

        # attention
        if self.use_attention:
            output_rnn, attn = self.attention(output_rnn, encoder_context, mask)
        else:
            attn = None

        # project output to the correct size
        output_rnn = self.dropout_output(output_rnn)
        output_projected = self.out(output_rnn.contiguous())

        # attn is for visualizing attention masks
        # output_rnn is to apply the input feed
        return hidden_t, output_projected, attn, output_rnn

    def decode_labels(self, decoder_output, output_fixed_size):
        # get the max score to construct the next input
        if output_fixed_size:
            # do not do inplace operation on the original decoder output (this breaks backprop)
            decoder_output = decoder_output.clone()
            # block the EOS symbol
            decoder_output[:, :, self.end_of_string_token] = float('-inf')
        max_scores, labels = torch.max(decoder_output, dim=2, keepdim=True)
        return labels

    def forward(self, hidden, output_lengths, ground_truth_labels=None, output_fixed_size=True,
                use_teacher_forcing=False, first_label_to_feed=None, encoder_outputs=None, input_lengths=None,
                reference_policy=None):

        # get the full batch_size
        batch_size = hidden.size(1)
        assert(hidden.size(2) == self.hidden_size)

        # CPU vs GPU
        th = torch.cuda if hidden.is_cuda else torch

        # get batch sizes for each element of the sequence
        step_batch_sizes = get_batch_sizes_from_length(output_lengths)
        max_length = output_lengths.data[0] if type(output_lengths) == Variable else output_lengths[0]
        assert(max_length > 0)

        # initial input to the decoder
        if first_label_to_feed is None:
            decoder_input_init = [[self.start_of_string_token] for _ in range(batch_size)]
            decoder_input_init = th.LongTensor(decoder_input_init).unsqueeze(0)
            decoder_input_init = Variable(decoder_input_init)
        else:
            decoder_input_init = first_label_to_feed

        # check the teacher forcing status
        if isinstance(use_teacher_forcing, bool):
            use_teacher_forcing = th.ByteTensor(batch_size).fill_(use_teacher_forcing)
        flag_teacher_forcing = use_teacher_forcing.long().sum() > 0
        use_teacher_forcing = Variable(use_teacher_forcing)

        # prepare to run the reference policy
        if reference_policy is not None:
            # construct data structure similar to rollout data
            # rollout_data - torch Variable to do rollouts:
            #   rollout_data[0] - index in the batch
            #   rollout_data[1] - cell_index
            #   rollout_data[2] - label to feed
            #   rollout_data[3] - type of rollout to do: 0 - reference, 1 - learned
            #   rollout_data[4] - required number of steps
            #   rollout_data[5] - length of the original sequence
            #   rollout_data[6] - indices in the original order (needed to 'unsort' later)
            reference_policy_data = torch.LongTensor(7, batch_size).fill_(0)
            reference_policy_data[0, :] = torch.arange(0, batch_size)
            reference_policy_data[1, :].fill_(-1)  # start with feeding the go symbol
            reference_policy_data[2, :].fill_(self.start_of_string_token)  # start with feeding the go symbol
            reference_policy_data[3, :].fill_(0)
            reference_policy_data[4, :].fill_(-1)  # unused here
            reference_policy_data[5, :].fill_(-1)  # unused here
            reference_policy_data[6, :] = torch.arange(0, batch_size)
            if hidden.is_cuda:
                reference_policy_data = reference_policy_data.cuda()
            reference_policy_data = Variable(reference_policy_data)

        # create array of labels to feed into the reference policy
        labels_seen_by_decoder = th.LongTensor(max_length, batch_size, 1).fill_(self.end_of_string_token)
        labels_seen_by_decoder = Variable(labels_seen_by_decoder)

        # run the forward pass
        # init the output variables
        decoder_output = [None] * max_length
        decoder_hidden = [None] * (max_length + 1)
        decoder_attention = [None] * max_length
        decoder_hidden[0] = hidden

        # create vars to feed decoder
        decoder_input = decoder_input_init
        output_for_input_feed = Variable(th.FloatTensor(1, batch_size, self.hidden_size).fill_(0.0))

        # input length_mask
        if input_lengths is not None:
            # BATCH_SIZE x SEQ_LEN
            input_length_mask = lengths_to_mask(input_lengths, max_length=encoder_outputs.size(1)).t()
        else:
            input_length_mask = None

        for i_step in range(max_length):
            hidden_this_step = decoder_hidden[i_step][:, :step_batch_sizes[i_step], :].contiguous()
            input_this_step = decoder_input[:, :step_batch_sizes[i_step], :].contiguous()
            context_this_step = encoder_outputs[:step_batch_sizes[i_step], :, :] \
                if encoder_outputs is not None else None
            input_length_mask_this_step = input_length_mask[:step_batch_sizes[i_step], :] \
                if input_length_mask is not None else None
            output_for_input_feed = output_for_input_feed[:, :step_batch_sizes[i_step], :]

            decoder_hidden[i_step + 1], \
            decoder_output[i_step], \
            decoder_attention[i_step], \
            output_for_input_feed = \
                self.forward_given_inputs(hidden_this_step, input_this_step, encoder_context=context_this_step,
                                          mask=input_length_mask_this_step, prev_output=output_for_input_feed)

            # pad the outputs to the standard size
            if step_batch_sizes[i_step] < batch_size:
                decoder_hidden[i_step + 1] = pad_tensor(decoder_hidden[i_step + 1], 1, batch_size, 0)
                decoder_output[i_step] = pad_tensor(decoder_output[i_step], 1, batch_size, 0)

            # get the max score to construct the next input
            new_feed = self.decode_labels(decoder_output[i_step], output_fixed_size)

            decoder_input = Variable(new_feed.data.clone())
            if flag_teacher_forcing:
                if reference_policy is None:
                    reference_this_step = ground_truth_labels[i_step]
                else:
                    reference_results = reference_policy(ground_truth_labels, reference_policy_data,
                                                         labels_seen_by_decoder)
                    reference_this_step = reference_results[i_step, :]

                if use_teacher_forcing.dim() == 1:
                    cur_tf_flags = use_teacher_forcing
                else:
                    cur_tf_flags = use_teacher_forcing[:,i_step]
                if cur_tf_flags.data.long().sum() > 0:
                    selected_this_step = reference_this_step.squeeze(1).masked_select(cur_tf_flags)
                    decoder_input.masked_scatter_(cur_tf_flags.unsqueeze(0).unsqueeze(2), selected_this_step)

            if reference_policy is not None:
                # update data structured for further calls of the reference policy
                reference_policy_data[1, :].data.fill_(i_step)  # the index of the current input symbol
                reference_policy_data[2, :] = decoder_input[0, :, 0]

            # save all the decoder inputs for usage outside of this function
            labels_seen_by_decoder[i_step, :, 0] = decoder_input[0, :, 0]

            # pad attention if needed
            if decoder_attention[i_step] is not None:
                decoder_attention[i_step] = pad_tensor(decoder_attention[i_step], 0, batch_size)

        # cat all the outputs
        decoder_output = torch.cat(decoder_output, 0)
        decoder_output = decoder_output.contiguous()

        # cat the memory cells
        decoder_hidden = torch.stack(decoder_hidden[1:], 1)
        decoder_hidden = decoder_hidden.contiguous()

        if decoder_attention[0] is not None:
            decoder_attention = torch.stack(decoder_attention, 0)
        else:
            decoder_attention = None

        return decoder_output, decoder_hidden, decoder_attention, labels_seen_by_decoder

    def rollout_one_batch(self, rollout_data, decoder_hidden, reference_labels, encoder_output_states, input_lengths,
                          output_fixed_size=True, rollout_batch_size=512):

        # rollout_data - torch Variable to do rollouts:
        #   rollout_data[0] - index in the batch
        #   rollout_data[1] - cell_index
        #   rollout_data[2] - label to feed
        #   rollout_data[3] - type of rollout to do: 0 - reference, 1 - learned
        #   rollout_data[4] - required number of steps
        #   rollout_data[5] - length of the original sequence
        #   rollout_data[6] - indices in the original order (needed to 'unsort' later)

        batch_size = decoder_hidden.size(2)
        num_rollouts = rollout_data.size(1)

        i_rollout = 0
        predictions_per_big_batch = []
        decoder_hidden_flat = decoder_hidden.view(decoder_hidden.size(0), -1, decoder_hidden.size(3))

        # CPU vs GPU
        th = torch.cuda if rollout_data.is_cuda else torch

        while i_rollout < num_rollouts:
            # construct a giant batch for all rollouts at the same time
            start_rollout_batch = i_rollout
            end_rollout_batch = min(i_rollout + rollout_batch_size, num_rollouts)
            i_rollout += rollout_batch_size

            index_batch = rollout_data[0, start_rollout_batch : end_rollout_batch]
            index_cell = rollout_data[1, start_rollout_batch : end_rollout_batch]
            first_label_to_feed_all = rollout_data[2, start_rollout_batch : end_rollout_batch]
            use_learned_rollout_all = rollout_data[3, start_rollout_batch : end_rollout_batch]
            rollout_num_steps = rollout_data[4, start_rollout_batch : end_rollout_batch]
            max_length = reference_labels.size(0)

            # initial states for rollout
            init_states_all = torch.index_select(decoder_hidden_flat, 1, index_cell * batch_size + index_batch)

            # labels and predictions
            num_learned_rollouts = use_learned_rollout_all.long().sum().data[0]
            predictions_all = reference_labels[:, start_rollout_batch : end_rollout_batch, :].clone()

            # construct a mask to fill-in roll-outs
            mask_rollout = lengths_to_mask(index_cell + 1 + rollout_num_steps, max_length=max_length,
                                           sequence_start=index_cell + 1)

            # for attention
            encoder_output_states_all = torch.index_select(encoder_output_states, 0, index_batch)
            input_lengths_all = torch.index_select(input_lengths, 0, index_batch)

            # do roll-outs if needed
            if num_learned_rollouts > 0:
                indices_for_rollout = Variable(torch.nonzero(use_learned_rollout_all.data))
                indices_for_rollout = indices_for_rollout.view(-1)
                if init_states_all.is_cuda:
                    indices_for_rollout = indices_for_rollout.cuda()

                # select elements for learned roll-out
                init_states_selected = torch.index_select(init_states_all, 1, indices_for_rollout)
                first_label_to_feed_selected = torch.index_select(first_label_to_feed_all, 0, indices_for_rollout)
                first_label_to_feed_selected = first_label_to_feed_selected.unsqueeze(0).unsqueeze(2)
                predictions_selected = torch.index_select(predictions_all, 1, indices_for_rollout)
                rollout_num_steps_selected = torch.index_select(rollout_num_steps, 0, indices_for_rollout)
                encoder_output_states_selected = torch.index_select(encoder_output_states_all, 0, indices_for_rollout)
                input_lengths_selected = torch.index_select(input_lengths_all, 0, indices_for_rollout)
                mask_rollout_selected = torch.index_select(mask_rollout, 1, indices_for_rollout)

                # initial input to the decoder
                use_teacher_forcing = False
                cur_max_length = rollout_num_steps_selected.max()
                if cur_max_length.data[0] > 0:
                    decoder_output, _, _, _ = \
                        self.forward(init_states_selected, rollout_num_steps_selected, None, output_fixed_size,
                                     use_teacher_forcing, first_label_to_feed_selected,
                                     encoder_outputs=encoder_output_states_selected,
                                     input_lengths=input_lengths_selected)

                    # get target labels from the scores
                    rollout_labels_selected = self.decode_labels(decoder_output, output_fixed_size)
                    assert(rollout_labels_selected.dim() == 3)

                    # update predictions with the roll-out results
                    rollout_labels_selected = pad_tensor(rollout_labels_selected, 0, max_length,
                                                         self.end_of_string_token)
                    mask_labels_to_copy = lengths_to_mask(rollout_num_steps_selected, max_length=max_length)
                    # need to transpose everything to copy labels in the correct order
                    predictions_selected = \
                        masked_scatter_with_transpose(rollout_labels_selected, mask_labels_to_copy.unsqueeze(2),
                                                      predictions_selected.clone(), mask_rollout_selected.unsqueeze(2))

                    # merge the results of roll-out with the gt labels
                    predictions_all.index_copy_(1, indices_for_rollout,  predictions_selected)

            predictions_per_big_batch.append(predictions_all.clone())

        predictions_all = torch.cat(predictions_per_big_batch, 1).contiguous()

        return predictions_all

    def original_forward(self, input, context, state, mask):
        """
        Only used for beam search. Forward through the decoder.
        Args:
            input (LongTensor): a sequence of input tokens tensors
                                of size (len x batch x nfeats).
            context (FloatTensor): output(tensor sequence) from the encoder
                        RNN of size (src_len x batch x hidden_size).
            state (FloatTensor): hidden state from the encoder RNN for
                                 initializing the decoder.
        Returns:
            outputs (FloatTensor): a Tensor sequence of output from the decoder
                                   of shape (len x batch x hidden_size).
            state (FloatTensor): final hidden state from the decoder.
            attns (dict of (str, FloatTensor)): a dictionary of different
                                type of attention Tensor from the decoder
                                of shape (src_len x batch).
        """
        # Args Check
        assert isinstance(state, RNNDecoderState)
        input_len, input_batch, _ = input.size()
        contxt_len, contxt_batch, _ = context.size()
        aeq(input_batch, contxt_batch)
        # END Args Check

        # Run the forward pass of the RNN.
        hidden, outputs, attns, coverage = self._run_forward_pass(input, context, state, mask)

        # Update the state with the result.
        final_output = outputs[-1]
        state.update_state(hidden, final_output.unsqueeze(0), coverage.unsqueeze(0) if coverage is not None else None)

        # Concatenates sequence of tensors along a new dimension.
        outputs = torch.stack(outputs)
        for k in attns:
            attns[k] = torch.stack(attns[k])

        return outputs, state, attns

    def _run_forward_pass(self, input, context, state, mask):
        """
        Only used for beam search.
        Only compatible with runs with attention. Todo: implementation without attention.
        Private helper for running the specific RNN forward pass.
        Must be overriden by all subclasses.
        Args:
            input (LongTensor): a sequence of input tokens tensors
                                of size (len x batch x nfeats).
            context (FloatTensor): output(tensor sequence) from the encoder
                        RNN of size (src_len x batch x hidden_size).
            state (FloatTensor): hidden state from the encoder RNN for
                                 initializing the decoder.
        Returns:
            hidden (Variable): final hidden state from the decoder.
            outputs ([FloatTensor]): an array of output of every time
                                     step from the decoder.
            attns (dict of (str, [FloatTensor]): a dictionary of different
                            type of attention Tensor array of every time
                            step from the decoder.
            coverage (FloatTensor, optional): coverage from the decoder.
        """
        # Initialize local and return variables.
        attns = {"std": []}
        coverage = None
        emb = self.embedding(input)

        if emb.size(2) != state.hidden[0].size(2):
            state.hidden = [self.project_encoder(state.hidden[0])]

        # Run the forward pass of the RNN.
        if isinstance(self.rnn, StackedGRU):
            rnn_output, hidden = self.rnn(emb, state.hidden[0])
        else:
            rnn_output, hidden = self.rnn(emb, state.hidden)
        # Result Check
        input_len, input_batch, _ = input.size()
        output_len, output_batch, _ = rnn_output.size()
        aeq(input_len, output_len)
        aeq(input_batch, output_batch)
        # END Result Check

        # Calculate the attention.
        attn_outputs, attn_scores = self.attention(
            rnn_output.contiguous(),  # (output_len, batch, d)
            context.transpose(0, 1).contiguous(),  # (contxt_len, batch, d)
            mask
        )
        attns["std"] = attn_scores
        outputs = attn_outputs  # (input_len, batch, d)

        # Return result.
        return hidden, outputs, attns, coverage

    def _fix_enc_hidden(self, h):
        """
        The encoder hidden is  (layers*directions) x batch x dim.
        We need to convert it to layers x batch x (directions*dim).
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def init_decoder_state(self, context, enc_hidden):
        if isinstance(enc_hidden, tuple):  # GRU
            return RNNDecoderState(context, self.hidden_size,
                                   tuple([self._fix_enc_hidden(enc_hidden[i]) for i in range(len(enc_hidden))]))
        else:  # LSTM
            return RNNDecoderState(context, self.hidden_size,
                                   self._fix_enc_hidden(enc_hidden))

    def is_cuda(self):
        return next(self.parameters()).is_cuda


class DecoderState(object):
    """
    Adapted from openNMT (https://github.com/OpenNMT/OpenNMT-py).
    DecoderState is a base class for models, used during translation
    for storing translation states.
    """
    def detach(self):
        """
        Detaches all Variables from the graph
        that created it, making it a leaf.
        """
        for h in self._all:
            if h is not None:
                h.detach_()

    def beam_update(self, idx, positions, beam_size):
        """ Update when beam advances. """
        for e in self._all:
            a, br, d = e.size()
            sent_states = e.view(a, beam_size, br // beam_size, d)[:, :, idx]
            sent_states.data.copy_(sent_states.data.index_select(1, positions))


class RNNDecoderState(DecoderState):
    def __init__(self, context, hidden_size, rnn_state):
        """
        Args:
            context (FloatTensor): output from the encoder of size
                                   len x batch x rnn_size.
            hidden_size (int): the size of hidden layer of the decoder.
            rnn_state (Variable): final hidden state from the encoder.
                transformed to shape: layers x batch x (directions*dim).
            input_feed (FloatTensor): output from last layer of the decoder.
            coverage (FloatTensor): coverage output from the decoder.
        """
        if not isinstance(rnn_state, tuple):
            self.hidden = (rnn_state,)
        else:
            self.hidden = rnn_state
        self.coverage = None

        # Init the input feed.
        batch_size = context.size(1)
        h_size = (batch_size, hidden_size)
        self.input_feed = Variable(context.data.new(*h_size).zero_(), requires_grad=False).unsqueeze(0)

    @property
    def _all(self):
        return self.hidden + (self.input_feed,)

    def update_state(self, rnn_state, input_feed, coverage):
        if not isinstance(rnn_state, tuple):
            self.hidden = (rnn_state,)
        else:
            self.hidden = rnn_state
        self.input_feed = input_feed
        self.coverage = coverage

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        vars = [Variable(e.data.repeat(1, beam_size, 1), volatile=True)
                for e in self._all]
        self.hidden = tuple(vars[:-1])
        self.input_feed = vars[-1]


class WrapForRNN(nn.Module):
    """
    Simple reshape, since RNN tensors have one more dimension than usual (the length of the sequence).
    """
    def __init__(self, layer):
        super(WrapForRNN, self).__init__()
        self.layer = layer

    def forward(self, input):
        assert(input.dim() == 3)
        seq_len = input.size(0)
        batch_size = input.size(1)
        feature_dim = input.size(2)

        # reshape from SEQ_LEN x BATCH_SIZE x FEATURE_DIM to (SEQ_LEN * BATCH_SIZE) x FEATURE_DIM
        input = input.view(seq_len * batch_size, feature_dim)

        # run the usual linear layer
        output = self.layer(input)

        # reshape back: from (SEQ_LEN * BATCH_SIZE) x EMD_DIM to  SEQ_LEN x BATCH_SIZE x EBM_DIM
        output = output.view(seq_len, batch_size, -1)

        return output


class IdleLayer(nn.Module):
    def __init__(self):
        super(IdleLayer, self).__init__()

    def forward(self, input):
        return input


class EmbeddingSeq2seq(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 max_norm=None, norm_type=2, scale_grad_by_freq=False,
                 sparse=False):
        super(EmbeddingSeq2seq, self).__init__(num_embeddings, embedding_dim, padding_idx=padding_idx,
                                               max_norm=max_norm, norm_type=norm_type,
                                               scale_grad_by_freq=scale_grad_by_freq, sparse=sparse)

    def reset_parameters(self):
        # use kaiming instead of xavier so as not to depend on the dictionary size
        nn.init.kaiming_uniform(self.weight, mode='fan_in')


class LinearSeq2seq(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, bias_constant=0.0):
        self.has_bias = bias
        self.bias_constant = bias_constant
        super(LinearSeq2seq, self).__init__(in_features=in_features, out_features=out_features, bias=bias)

    def reset_parameters(self):
        nn.init.xavier_uniform(self.weight)
        if self.has_bias and self.bias_constant is not None:
            nn.init.constant(self.bias, self.bias_constant)


class EmbeddingPartialTrainable(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, mask_items_to_update, weights=None):
        super(EmbeddingPartialTrainable, self).__init__()

        self.mask_items_to_update = mask_items_to_update.byte().view(-1)
        self.num_frozen = (self.mask_items_to_update.long() != 1).sum()
        self.num_learnable = self.mask_items_to_update.long().sum()
        assert (self.mask_items_to_update.numel() == num_embeddings)

        self.emb_learnable = EmbeddingSeq2seq(self.num_learnable, embedding_dim, sparse=False)
        self.emb_frozen = EmbeddingSeq2seq(self.num_frozen, embedding_dim, sparse=False)
        self.emb_frozen.weight.requires_grad = False
        if weights is not None:
            ids_l = self.mask_items_to_update.nonzero().view(-1)
            self.emb_learnable.weight.data.copy_(weights.index_select(0, ids_l))
            ids_f = (self.mask_items_to_update != 1).nonzero().view(-1)
            self.emb_frozen.weight.data.copy_(weights.index_select(0, ids_f))

        self.learnable_lookup = torch.LongTensor(num_embeddings).fill_(-1)
        self.learnable_lookup.masked_scatter_(self.mask_items_to_update, torch.arange(0, self.num_learnable).long())

        self.frozen_lookup = torch.LongTensor(num_embeddings).fill_(-1)
        self.frozen_lookup.masked_scatter_(self.mask_items_to_update != 1, torch.arange(0, self.num_frozen).long())

        self.mask_items_to_update = Parameter(self.mask_items_to_update)
        self.mask_items_to_update.requires_grad = False
        self.learnable_lookup = Parameter(self.learnable_lookup)
        self.learnable_lookup.requires_grad = False
        self.frozen_lookup = Parameter(self.frozen_lookup)
        self.frozen_lookup.requires_grad = False

    def forward(self, indices):
        # get indices
        item_mask = self.mask_items_to_update[indices]
        ids_l = Variable(item_mask.data.nonzero().view(-1))
        ids_f = Variable((item_mask != 1).data.nonzero().view(-1))
        assert(ids_l.numel() > 0 or ids_f.numel() > 0)

        # learnable embeddings
        if ids_l.numel() > 0:
            ind_l = indices[ids_l]
            emb = self.emb_learnable(self.learnable_lookup[ind_l])
            order = ids_l

        # frozen embeddings
        if ids_f.numel() > 0:
            ind_f = indices[ids_f]
            emb_f = self.emb_frozen(self.frozen_lookup[ind_f])
            if ids_l.numel() > 0:
                # merge the two embeddings
                emb = torch.cat([emb, emb_f], 0)
                order = torch.cat([order, ids_f], 0)
            else:
                emb = emb_f
                order = ids_f

        # unsort
        _, order_reversed = torch.sort(order, 0, descending=False)
        emb_right = torch.index_select(emb, 0, order_reversed)
        return emb_right


class MultiEmbedding(nn.Module):
    def __init__(self, embeddings):
        super(MultiEmbedding, self).__init__()
        self.num_embeddings = len(embeddings)
        self.embeddings = nn.ModuleList()
        for i_emb in range(self.num_embeddings):
            self.embeddings.append(embeddings[i_emb])

    def forward(self, input):
        assert(input.dim() == 2)
        assert(input.size(1) == self.num_embeddings)
        embeddings = [None] * self.num_embeddings
        for i_emb in range(self.num_embeddings):
            embeddings[i_emb] = self.embeddings[i_emb](input[:, i_emb])
        output = torch.cat(embeddings, 1)
        return output

