import os
import torch

"""
    Logging helpers
"""


def init_logging(opt, train_evaluate_func=None, val_evaluate_func=None):
    full_log = {}
    full_log['time'] = []
    full_log['iter'] = []
    full_log['grad_norm'] = []
    full_log['learning_rate'] = []
    full_log['objective'] = []
    full_log['hamming_error_train'] = []
    full_log['hamming_error_val'] = []
    full_log['sequence_error_train'] = []
    full_log['sequence_error_val'] = []
    full_log['log_loss_train'] = []
    full_log['log_loss_val'] = []
    full_log['rollin_ref_prob'] = []
    full_log[opt.loss.casefold()+'_train'] = []
    full_log[opt.loss.casefold()+'_val'] = []
    if train_evaluate_func:
        full_log['dataset_specific_train'] = []
    if val_evaluate_func:
        full_log['dataset_specific_val'] = []

    return full_log


def checkpoint_model(encoder, decoder, optimizer, i_iter, opt):
    try:
        if not os.path.isdir(opt.log_path):
            os.makedirs(opt.log_path)
        encoder.cpu()
        decoder.cpu()
        checkpoint = {}
        checkpoint['encoder'] = encoder.state_dict()
        checkpoint['decoder'] = decoder.state_dict()
        checkpoint['optimizer'] = optimizer.state_dict()
        checkpoint_file = '{0}/checkpoint_iter_{1}.pth'.format(opt.log_path, i_iter)
        print('Saving checkpoint', checkpoint_file)
        torch.save(checkpoint, checkpoint_file)
        if opt.cuda:
            encoder.cuda()
            decoder.cuda()
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:
        print("\nWARNING: could not save the checkpoint model for some reason:", str(e))


def restore_from_checkpoint(encoder, decoder, optimizer, opt):
    # read the checkpoint file
    if opt.checkpoint_file:
        print('Reading checkpoint file', opt.checkpoint_file)
        checkpoint = torch.load(opt.checkpoint_file)
    else:
        checkpoint = None

    if checkpoint and 'encoder' in checkpoint:
        encoder.load_state_dict(checkpoint['encoder'])
        print('Loaded encoder from checkpoint')
    elif opt.encoder_file:
        encoder.load_state_dict(torch.load(opt.encoder_file))
        print('Loaded encoder from file', opt.encoder_file)

    if checkpoint and 'decoder' in checkpoint:
        decoder.load_state_dict(checkpoint['decoder'])
        print('Loaded decoder from checkpoint')
    elif opt.decoder_file:
        decoder.load_state_dict(torch.load(opt.decoder_file))
        print('Loaded decoder from file', opt.decoder_file)

    if checkpoint and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('Loaded optimizer from checkpoint')
        if opt.change_learning_rate:
            if opt.learning_rate is None:
                print('WARNING: could not change learning rate. New one was not specified.')
            else:
                for p in optimizer.param_groups:
                    if 'lr' in p:
                        print('Changing learning rate from %f to %f' % (p['lr'], opt.learning_rate) )
                        p['lr'] = opt.learning_rate

    return encoder, decoder, optimizer
