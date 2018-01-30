import argparse
import random
from itertools import chain

import numpy as np
import torch.backends.cudnn

import optimization
import train
from datasets.conll import ConllDataset
from datasets.nmt import NmtDataset
from datasets.ocr import OcrDataset
from evaluation.logging_utils import restore_from_checkpoint
from models import EncoderRNN, DecoderRNN

"""
    Entry point.
"""

parser = argparse.ArgumentParser()

# dataset
parser.add_argument('--dataset', required=True, help='ocr | conll | nmt')
parser.add_argument('--dataroot', required=True, help='Path to dataset')
parser.add_argument('--split', type=str, default='valid', help='Split (only used for NMT)')
parser.add_argument('--split_id', type=int, default=0, help='Split ID (only used for OCR)')
parser.add_argument('--revert_input_sequence', action='store_true', help='Revert input sequence')
parser.add_argument('--max_train_items', type=int, default=None, help='Training set is cropped to this number of items')
parser.add_argument('--num_buckets', type=int, default=1,
                    help='Number of buckets used to group the input data, default 1 (no buckets)')

# CONLL specific settings
parser.add_argument('--min_word_count', type=int, default=10,
                    help='Minimum number of word entries required to be in the dictionary')
parser.add_argument('--max_seq_length', type=int, default=None, help='Max length of sequences for training')
parser.add_argument('--lower_case', action='store_true', help='Put all the words to lower case')
parser.add_argument('--senna_emb', type=str, default='', help='Path to Senna embedding')

# model
parser.add_argument('--memory_size', type=int, default=128, help='RNN memory size, default=128')
parser.add_argument('--memory_size_encoder', type=int, default=None,
                    help='Memory size of RNN cells in the encoder, default - same as the decoder if not BRNN otherwise'
                         ' twice smaller')
parser.add_argument('--rnn_depth', type=int, default=1, help='Depth of RNN layers')
parser.add_argument('--bidirectional', action='store_true', help='Use bidirectional encoder')
parser.add_argument('--attention', action='store_true', help='Use attention mechanism')
parser.add_argument('--attn_type', type=str, default='matrix', help='Type of attention: matrix | sum-tanh')
parser.add_argument('--input_feed', type=int, default=0,
                    help='Feed the context vector at each time step as additional input (via concatenation with the '
                         'word embeddings) to the decoder.')
parser.add_argument('--decoder_emb_size', type=int, default=None,
                    help='Size of the decoder embedding, default - the same as the decoder hidden units')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout between RNN layers (default: 0 - no dropout)')
parser.add_argument('--target_noise_std', type=float, default=1e-5,
                    help='When selecting a target label, we add noise to the costs to break ties randomly')
parser.add_argument('--encoder_file', type=str, default='', help='File to load the encoder model')
parser.add_argument('--decoder_file', type=str, default='', help='File to load the decoder model')
parser.add_argument('--checkpoint_file', type=str, default='', help='File to load encoder, decoder, optimizer')
parser.add_argument('--decoding', type=str, default='greedy', help='Decoding algorithm: greedy|beam')
parser.add_argument('--beam_size', type=int, default=1, help='Size of the beam if using beam search for decoding')
parser.add_argument('--beam_scaling', type=float, default=1.0, help='scaling factor for output distributions')

# training approach
parser.add_argument('--loss', type=str, default='hamming',
                    help='hamming | hamming-unnorm | bleu-smoothed | bleu1-smoothed | gleu | sentence-F1')
parser.add_argument('--rollin', type=str, default='gt', help='gt | learned | mixed | mixed-cells')
parser.add_argument('--rollout', type=str, default='gt', help='gt | learned | mixed | mixed-matched | focused-costing')
parser.add_argument('--reference_policy', type=str, default='copy-gt',
                    help='copy-gt | bleu-best-suffix | bleu1-best-suffix | gleu-best-suffix | maximize-F1')
parser.add_argument('--num_cells_to_rollout', type=int, default=100, help='Number of cells to do rollout')
parser.add_argument('--objective', type=str, default='mle',
                    help='mle | target-learning | target-learning-all-labels | loss-softmax | kl | inverse_kl | js | l2 | svm-cs')
parser.add_argument('--temperature', type=int, default=1, help='Temperature used for KL, LLCAS and other losses')
parser.add_argument('--obj_normalization', type=str, default='cell-per-seq-batch',
                    help="How to normalize training loss: 'none' | 'batch' | 'cell-global' | 'cell-per-seq-batch' | "
                         "'batch-maxlen'")
parser.add_argument('--rollin_ref_prob', type=float, default=0.5,
                    help='Probability to pick reference rollin in mixed strategies')
parser.add_argument('--rollout_ref_prob', type=float, default=0.5,
                    help='Probability to pick reference rollout in mixed strategies')
parser.add_argument('--data_sampling', type=str, default='shuffle', help='shuffle | random | fixed-order')
parser.add_argument('--scheduled_sampling', type=str, default='none', help='none | sigmoid')
parser.add_argument('--fc_initial_value', type=int, default=0, help='Amount of learned steps in FC rollouts')
parser.add_argument('--fc_increment', type=int, default=0,
                    help='Increment to the amount of learned steps in focused costing rollouts')
parser.add_argument('--fc_epoch', type=int, default=1e8,
                    help='Number of steps before increase of focused costing learned steps in rollouts')

# labels sampling
parser.add_argument('--sample_labels_uniform', type=int, default=30,
                    help='Number of tokens to sample uniformly, default=30 (all labels for OCR and CONLL)')
parser.add_argument('--sample_labels_policy_topk', type=int, default=5,
                    help='Number of best tokens according to the current policy')
parser.add_argument('--sample_labels_neighbors', type=int, default=2,
                    help='Number of neighboring labels (on each side) in ground truth (skipped words in nmt)')
parser.add_argument('--targeted_sampling', action='store_true', help='Whether or not to do targeted sampling')
parser.add_argument('--ts_threshold', type=float, default=0.0, help='Threshold to pick actions in targeted sampling')
parser.add_argument('--ts_max_samples', type=int, default=0,
                    help='Maximum amount of samples to pick in targeted sampling')

# optimization
parser.add_argument('--optim_method', default='adam',
                    help='Optimization method: sgd | adagrad | adadelta | adam | adamax | asgd | rmsprop | rprop')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size, default=64')
parser.add_argument('--learning_rate', type=float, default=None,
                    help='learning rate, default: sgd - 0.5, adagrad - 0.01, adadelta - 1, adam - 0.001, '
                         'adamax - 0.002, asgd - 0.01, rmsprop - 0.01, rprop - 0.01')
parser.add_argument('--max_iter', type=int, default=10000, help='Number of iterations to train, default=10000')
parser.add_argument('--max_grad_norm', type=float, default=5, help='Maximum gradient norm. Renormalize if necessary.')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay (L2 regularization), default=0.0')
parser.add_argument('--param_init', type=float, default=0.0,
                    help='Parameters are initialized over uniform distribution with support (-param_init, param_init)')
parser.add_argument('--change_learning_rate', action='store_true', help='Change the learning rate when warm starting')

# parameters of the schedule to anneal learning rate
parser.add_argument('--anneal_learning_rate', action='store_true',
                    help='Anneal the learning rate with torch.optim.lr_scheduler.ReduceLROnPlateau strategy')
parser.add_argument('--lr_quantity_to_monitor', default='log_loss_val', type=str,
                    help='Monitor this quantity when deciding to anneal learning rate')
parser.add_argument('--lr_quantity_mode', default='min', type=str,
                    help='"min" | "max" depending on whether the quantity of interest is supposed to go up or down')
parser.add_argument('--lr_quantity_epsilon', default=1e-3, type=float,
                    help='Quantity improvement has to be at least this much to be significant '
                         '(this threshold is relative to the characteritic value)')
parser.add_argument('--lr_reduce_factor', default=0.5, type=float,
                    help='Multiply learning rate by this factor when decreasing, default: 0.5')
parser.add_argument('--lr_min_value', default=1e-5, type=float, help='The minimal value of learning rate')
parser.add_argument('--lr_patience', default=1000, type=int,
                    help='Wait for this number of steps before annealing the learning rate after previous lr decrease')
parser.add_argument('--lr_initial_patience', default=0, type=int,
                    help='Wait for this number of steps before annealing the learning rate initially '
                         '(e.g. for warm starting)')
parser.add_argument('--lr_cooldown', default=5000, type=int,
                    help='Number of calls to wait before resuming normal operation after lr has been reduced.')
parser.add_argument('--lr_quantity_smoothness', default=0, type=int,
                    help='When deciding to reduce LR use sliding window averages of this width.')

# logging
parser.add_argument('--log_path', type=str, default='', help='Where to store results and models (default: do not save)')
parser.add_argument('--print_iter', type=int, default=10, help='Print after this number of steps')
parser.add_argument('--eval_iter', type=int, default=200, help='Evaluate after this number of steps')
parser.add_argument('--save_iter', type=int, default=1000, help='Save models at these iterations')
parser.add_argument('--eval_size', type=int, default=10000, help='Max number of items for intermediate evaluations')

# misc
parser.add_argument('--cuda', type=int, default=1, help='GPU vs CPU')
parser.add_argument('--free_random_seed', action='store_true', help='Fix random seed or not.')
parser.add_argument('--random_seed', type=int, default=42, help='Random seed, default=42')
parser.add_argument('--rollout_batch_size', type=int, default=512,
                    help='Size of the batch to use for rollout computations, default 512 is safe for 12G GPUs')

opt = parser.parse_args()

# set this to use faster convolutions
opt.cuda = torch.cuda.is_available() and opt.cuda == 1
if opt.cuda:
    torch.backends.cudnn.benchmark = True

# print all the options
print(opt)

# random seed
if not opt.free_random_seed:
    random.seed(opt.random_seed)
    np.random.seed(opt.random_seed)
    torch.manual_seed(opt.random_seed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.random_seed)

# load the dataset
if opt.dataset == 'ocr':
    train_set = OcrDataset(opt.dataroot, opt.batch_size, is_train=True, split_id=opt.split_id,
                           num_buckets=opt.num_buckets, revert_input_sequence=opt.revert_input_sequence,
                           max_num_items=opt.max_train_items)
    val_set = OcrDataset(opt.dataroot, opt.batch_size, is_train=False, split_id=opt.split_id,
                         revert_input_sequence=opt.revert_input_sequence)
    opt.output_fixed_size = True
elif opt.dataset == 'conll':
    train_set = ConllDataset(opt.dataroot, opt.batch_size, is_train=True,
                             min_word_count=opt.min_word_count, max_seq_length=opt.max_seq_length,
                             num_buckets=opt.num_buckets, revert_input_sequence=opt.revert_input_sequence,
                             max_num_items=opt.max_train_items, senna_emb=opt.senna_emb)
    val_set = ConllDataset(opt.dataroot, opt.batch_size, is_train=False, dicts=train_set.dicts,
                           revert_input_sequence=opt.revert_input_sequence)
    opt.output_fixed_size = True
elif opt.dataset == 'nmt':
    train_set = NmtDataset(opt.dataroot, opt.batch_size, split='train',
                           revert_input_sequence=opt.revert_input_sequence, input_embedding_size=opt.memory_size,
                           num_buckets=opt.num_buckets, max_num_items=opt.max_train_items)
    val_set = NmtDataset(opt.dataroot, opt.batch_size, split='valid',
                         revert_input_sequence=opt.revert_input_sequence)
    opt.output_fixed_size = False
    opt.output_unknown_word_token = train_set.output_rare_word_token
else:
    raise (RuntimeError("Unknown dataset"))

opt.max_pred_length = train_set.get_max_output_length()
opt.get_train_output_length = train_set.get_max_output_length_per_input


# function to init params
def init_params(model):
    for p in model.parameters():
        p.data.uniform_(-opt.param_init, opt.param_init)

# encoder
input_embedding, input_embedding_size = train_set.get_embedding_layer()
if opt.memory_size_encoder is None:
    encoder_state_size = (opt.memory_size // 2) if opt.bidirectional else opt.memory_size
else:
    encoder_state_size = opt.memory_size_encoder
encoder = EncoderRNN(input_embedding_size, encoder_state_size, num_layers=opt.rnn_depth,
                     input_embedding=input_embedding, bidirectional=opt.bidirectional, dropout=opt.dropout)
print(encoder)
if opt.param_init:
    init_params(encoder)

# decoder
decoder = DecoderRNN(opt.memory_size, train_set.num_output_labels, train_set.output_end_of_string_token,
                     num_layers=opt.rnn_depth, emb_size=opt.decoder_emb_size, use_attention=opt.attention,
                     encoder_state_size=encoder_state_size, bidirectional_encoder=opt.bidirectional,
                     dropout=opt.dropout, input_feed=opt.input_feed, attn_type=opt.attn_type)
print(decoder)
if opt.param_init:
    init_params(decoder)


# optimizer
def good_params(model):
    return filter(lambda p: p.requires_grad, model.parameters())

parameters = chain(good_params(encoder), good_params(decoder))
optimizer = optimization.create_optimizer(parameters, opt)

# check number of parameters
num_params_encoder = sum([p.nelement() for p in encoder.parameters()])
num_params_decoder = sum([p.nelement() for p in decoder.parameters()])
print('Number of parameters: encoder - %d; decoder - %d' % (num_params_encoder, num_params_decoder))

# restore from checkpoint
encoder, decoder, optimizer = restore_from_checkpoint(encoder, decoder, optimizer, opt)

# move models to GPU
if opt.cuda:
    encoder.cuda()
    decoder.cuda()

# start training
train.train_seq2seq(encoder, decoder, optimizer, train_set,
                    train_set.num_output_labels, opt, dataset_val=val_set)
