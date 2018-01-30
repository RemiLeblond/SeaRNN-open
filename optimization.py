from torch import optim

from utils import ceildiv


def create_optimizer(parameters, opt):
    lr = opt.learning_rate
    # default learning rates:
    # sgd - 0.5, adagrad - 0.01, adadelta - 1, adam - 0.001, adamax - 0.002, asgd - 0.01, rmsprop - 0.01, rprop - 0.01
    optim_method = opt.optim_method.casefold()
    if optim_method == 'sgd':
        optimizer = optim.SGD(parameters, lr=lr if lr else 0.5, weight_decay=opt.weight_decay)
    elif optim_method == 'adagrad':
        optimizer = optim.Adagrad(parameters, lr=lr if lr else 0.01, weight_decay=opt.weight_decay)
    elif optim_method == 'adadelta':
        optimizer = optim.Adadelta(parameters, lr=lr if lr else 1, weight_decay=opt.weight_decay)
    elif optim_method == 'adam':
        optimizer = optim.Adam(parameters, lr=lr if lr else 0.001, weight_decay=opt.weight_decay)
    elif optim_method == 'adamax':
        optimizer = optim.Adamax(parameters, lr=lr if lr else 0.002, weight_decay=opt.weight_decay)
    elif optim_method == 'asgd':
        optimizer = optim.ASGD(parameters, lr=lr if lr else 0.01, t0=5000, weight_decay=opt.weight_decay)
    elif optim_method == 'rmsprop':
        optimizer = optim.RMSprop(parameters, lr=lr if lr else 0.01, weight_decay=opt.weight_decay)
    elif optim_method == 'rprop':
        optimizer = optim.Rprop(parameters, lr=lr if lr else 0.01)
    else:
        raise RuntimeError("Invalid optim method: " + opt.optim_method)
    return optimizer


def get_learning_rate(optimizer):
    for p in optimizer.param_groups:
        if 'lr' in p:
            return p['lr']


def setup_lr(optimizer, full_log, opt):
    # annealing learning rate
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, verbose=True, factor=opt.lr_reduce_factor, min_lr=opt.lr_min_value,
        threshold=opt.lr_quantity_epsilon, threshold_mode='rel', mode=opt.lr_quantity_mode,
        patience=ceildiv(opt.lr_patience, opt.eval_iter), cooldown=ceildiv(opt.lr_cooldown, opt.eval_iter))

    # create a function and a closure
    averaging_buffer_max_length = ceildiv(opt.lr_quantity_smoothness, opt.eval_iter)
    if averaging_buffer_max_length <= 1:
        averaging_buffer_max_length = 1
    averaging_buffer = []

    def anneal_lr_func(anneal_now=True):
        value_to_monitor = full_log[opt.lr_quantity_to_monitor][-1]
        averaging_buffer.append(value_to_monitor)
        if len(averaging_buffer) > averaging_buffer_max_length:
            averaging_buffer.pop(0)
        averaged_value = sum(averaging_buffer) / float(len(averaging_buffer))
        counter = len(full_log[opt.lr_quantity_to_monitor])
        if opt.anneal_learning_rate and anneal_now:
            lr_scheduler.step(averaged_value, counter)
        return get_learning_rate(optimizer)

    return lr_scheduler, anneal_lr_func
