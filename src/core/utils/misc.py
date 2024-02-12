# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import argparse
import torch
import torch.nn as nn
from ruamel.yaml import YAML
from torch.utils.tensorboard import SummaryWriter


def over_write_args_from_dict(args, dict):
    """
    overwrite arguments acocrding to config file
    """
    for k in dict:
        setattr(args, k, dict[k])


def over_write_args_from_file(args, yml):
    """
    overwrite arguments according to config file
    """
    if yml == '':
        return
    with open(yml, 'r', encoding='utf-8') as f:
        yaml = YAML(typ='safe')
        dic = yaml.load(f.read())
        for k in dic:
            setattr(args, k, dic[k])


def setattr_cls_from_kwargs(cls, kwargs):
    # if default values are in the cls,
    # overlap the value by kwargs
    for key in kwargs.keys():
        if hasattr(cls, key):
            print(f"{key} in {cls} is overlapped by kwargs: {getattr(cls, key)} -> {kwargs[key]}")
        setattr(cls, key, kwargs[key])


def send_model_cuda(args, model, clip_batch=True):
    if not torch.cuda.is_available():
        raise Exception('ONLY GPU TRAINING IS SUPPORTED')
    elif args.distributed:
        ngpus_per_node = torch.cuda.device_count()  # number of gpus of each node

        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)

            '''
            batch_size: batch_size per node -> batch_size per gpu
            workers: workers per node -> workers per gpu
            '''
            if clip_batch:
                args.batch_size = int(args.batch_size / ngpus_per_node)
            model.cuda(args.gpu)
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, broadcast_buffers=False,
                                                                     find_unused_parameters=True,
                                                                     device_ids=[args.gpu])
        else:
            # if arg.gpu is None, DDP will divide and allocate batch_size
            # to all available GPUs if device_ids are not set.
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model,  broadcast_buffers=False, 
                                                                      find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()
    return model


def count_parameters(model):
    # count trainable parameters
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class TBLog:
    """
    Construct tensorboard writer (self.writer).
    The tensorboard is saved at os.path.join(tb_dir, file_name).
    """

    def __init__(self, tb_dir, file_name, use_tensorboard=False):
        self.tb_dir = tb_dir
        self.use_tensorboard = use_tensorboard
        if self.use_tensorboard:
            self.writer = SummaryWriter(os.path.join(self.tb_dir, file_name))
            

    def update(self, log_dict, it, suffix=None, mode="train"):
        """
        Args
            log_dict: contains scalar values for updating tensorboard
            it: contains information of iteration (int).
            suffix: If not None, the update key has the suffix.
        """
        if suffix is None:
            suffix = ''
        if self.use_tensorboard:
            for key, value in log_dict.items():
                self.writer.add_scalar(suffix + key, value, it)


class Argument(object):
    """
    Algrithm specific argument
    """
    def __init__(self, name, type, default, help='', *args, **kwargs):
        """
        Model specific arguments should be added via this class.
        """
        self.name = name
        self.type = type
        self.default = default
        self.help = help
        self.args = args
        self.kwargs = kwargs


def str2bool(v):
    """
    str to bool
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class EMA:
    """
    EMA model
    Implementation from https://fyubang.com/2019/06/01/ema/
    """

    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def load(self, ema_model):
        for name, param in ema_model.named_parameters():
            self.shadow[name] = param.data.clone()

    def register(self):
        for name, param in self.model.named_parameters():
            self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
            self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            self.backup[name] = param.data
            param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            param.data = self.backup[name]
        self.backup = {}