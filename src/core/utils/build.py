# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import math
import logging
import random
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from src.nets.utils import param_groups_layer_decay, param_groups_weight_decay


def get_logger(name, save_path=None, level='INFO'):
    """
    create logger function
    """
    logger = logging.getLogger(name)
    logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s', level=getattr(logging, level))

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        log_format = logging.Formatter('[%(asctime)s %(levelname)s] %(message)s')
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)

    return logger


def get_optimizer(net, 
                  optim_name='SGD', 
                  lr=0.1, 
                  momentum=0.9, 
                  weight_decay=0, 
                  layer_decay=1.0,
                  nesterov=True, 
                  bn_wd_skip=True):
    '''
    return optimizer (name) in torch.optim.

    Args:
        net: model witth parameters to be optimized
        optim_name: optimizer name, SGD|AdamW
        lr: learning rate
        momentum: momentum parameter for SGD
        weight_decay: weight decay in optimizer
        layer_decay: layer-wise decay learning rate for model, requires the model have group_matcher function
        nesterov: SGD parameter
        bn_wd_skip: if bn_wd_skip, the optimizer does not apply weight decay regularization on parameters in batch normalization.
    '''
    assert layer_decay <= 1.0

    no_decay = {}
    if hasattr(net, 'no_weight_decay') and bn_wd_skip:
        no_decay = net.no_weight_decay()
    
    if layer_decay != 1.0:
        per_param_args = param_groups_layer_decay(net, lr, weight_decay, no_weight_decay_list=no_decay, layer_decay=layer_decay)
    else:
        per_param_args = param_groups_weight_decay(net, weight_decay, no_weight_decay_list=no_decay)
        
    if optim_name == 'SGD':
        optimizer = torch.optim.SGD(per_param_args, lr=lr, momentum=momentum, weight_decay=weight_decay,
                                    nesterov=nesterov)
    elif optim_name == 'AdamW':
        optimizer = torch.optim.AdamW(per_param_args, lr=lr, weight_decay=weight_decay)

    return optimizer


def get_cosine_schedule_with_warmup(optimizer,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    num_warmup_steps=0,
                                    last_epoch=-1):
    '''
    Get cosine scheduler (LambdaLR).
    if warmup is needed, set num_warmup_steps (int) > 0.
    '''
    from torch.optim.lr_scheduler import LambdaLR
    def _lr_lambda(current_step):
        '''
        _lr_lambda returns a multiplicative factor given an interger parameter epochs.
        Decaying criteria: last_epoch
        '''

        if current_step < num_warmup_steps:
            _lr = float(current_step) / float(max(1, num_warmup_steps))
        else:
            num_cos_steps = float(current_step - num_warmup_steps)
            num_cos_steps = num_cos_steps / float(max(1, num_training_steps - num_warmup_steps))
            _lr = max(0.0, math.cos(math.pi * num_cycles * num_cos_steps))
        return _lr

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def get_port():
    """
    find a free port to used for distributed learning
    """
    pscmd = "netstat -ntl |grep -v Active| grep -v Proto|awk '{print $4}'|awk -F: '{print $NF}'"
    procs = os.popen(pscmd).read()
    procarr = procs.split("\n")
    tt= random.randint(15000, 30000)
    if tt not in procarr:
        return tt
    else:
        return get_port()
