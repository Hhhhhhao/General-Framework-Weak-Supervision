

import os
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np

from src.core.algorithmbase import AlgorithmBase
from src.core.utils import Argument, str2bool, get_optimizer, send_model_cuda
from src.core.criterions import CELoss
from src.core.hooks import CheckpointHook, TimerHook, LoggingHook, DistSamplerSeedHook, ParamUpdateHook, EvaluationHook, EMAHook
from src.datasets import get_sym_noisy_labels, get_cifar10_asym_noisy_labels, get_cifar100_asym_noisy_labels, get_data, get_dataloader, ImgBaseDataset, ImgTwoViewBaseDataset



class NoiseMatrixLayer(torch.nn.Module):
    def __init__(self, num_classes, scale=1.0):
        super().__init__()
        self.num_classes = num_classes
        
        self.noise_layer = nn.Linear(self.num_classes, self.num_classes, bias=False)
        # initialization
        self.noise_layer.weight.data.copy_(torch.eye(self.num_classes))
        
        init_noise_matrix = torch.eye(self.num_classes) 
        self.noise_layer.weight.data.copy_(init_noise_matrix)
        
        self.eye = torch.eye(self.num_classes).cuda()
        self.scale = scale
    
    def forward(self, x):
        noise_matrix = self.noise_layer(self.eye)
        # noise_matrix = noise_matrix ** 2
        noise_matrix = F.normalize(noise_matrix, dim=0)
        noise_matrix = F.normalize(noise_matrix, dim=1)
        return noise_matrix * self.scale


class NoiseParamUpdateHook(ParamUpdateHook):
    def before_train_step(self, algorithm):
        if hasattr(algorithm, 'start_run'):
            torch.cuda.synchronize()
            algorithm.start_run.record()

    # call after each train_step to update parameters
    def after_train_step(self, algorithm):
        
        loss = algorithm.out_dict['loss']
        # algorithm.optimizer.zero_grad()
        # update parameters
        if algorithm.use_amp:
            raise NotImplementedError
        else:
            loss.backward()
            if (algorithm.clip_grad > 0):
                torch.nn.utils.clip_grad_norm_(algorithm.model.parameters(), algorithm.clip_grad)
            algorithm.optimizer.step()
            algorithm.optimizer_noise.step()

        algorithm.model.zero_grad()
        algorithm.noise_model.zero_grad()
        
        if algorithm.scheduler is not None:
            algorithm.scheduler.step()

        if hasattr(algorithm, 'end_run'):
            algorithm.end_run.record()
            torch.cuda.synchronize()
            algorithm.log_dict['train/run_time'] = algorithm.start_run.elapsed_time(algorithm.end_run) / 1000.



class ImpreciseNoisyLabelLearning(AlgorithmBase):
    def __init__(self, args, tb_log=None, logger=None, **kwargs):
        self.init(args)
        super().__init__(args, tb_log, logger, **kwargs)
        
        self.num_train_iter = self.epochs * len(self.loader_dict['train'])
        self.num_eval_iter = len(self.loader_dict['train'])
        self.ce_loss = CELoss()
    
    def init(self, args):
        # extra arguments 
        self.average_entropy_loss = args.average_entropy_loss
        self.noise_ratio = args.noise_ratio
        self.noise_type =args.noise_type
        self.noise_matrix_scale = args.noise_matrix_scale
        
        self.noise_model = NoiseMatrixLayer(num_classes=args.num_classes, scale=self.noise_matrix_scale)
        self.noise_model = send_model_cuda(args, self.noise_model)
        self.optimizer_noise = torch.optim.SGD(self.noise_model.parameters(), lr=args.lr, weight_decay=0, momentum=0)


    def set_hooks(self):
        # parameter update hook is called inside each train_step
        self.register_hook(NoiseParamUpdateHook(), None, "HIGHEST")
        if self.ema_model is not None:
            self.register_hook(EMAHook(), None, "HIGH")
        self.register_hook(EvaluationHook(), None, "HIGH")
        self.register_hook(CheckpointHook(), None, "HIGH")
        self.register_hook(DistSamplerSeedHook(), None, "NORMAL")
        self.register_hook(TimerHook(), None, "LOW")
        self.register_hook(LoggingHook(), None, "LOWEST")

    def set_optimizer(self):
        """
        set optimizer for algorithm
        """
        self.print_fn("Create optimizer and scheduler")
        optimizer = get_optimizer(self.model, self.args.optim, self.args.lr, self.args.momentum, self.args.weight_decay, self.args.layer_decay, nesterov=False, bn_wd_skip=False)
        if self.args.dataset == 'webdataset':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[50 * len(self.loader_dict['train'])]) 
        elif self.args.dataset == 'clothing1m':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[7 * len(self.loader_dict['train'])]) 
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(self.epochs * len(self.loader_dict['train'])), eta_min=2e-4)
        return optimizer, scheduler

    def set_dataset(self):
        # get initial data
        train_data, train_targets, test_data, test_targets, extra_data = get_data(self.args.data_dir, self.args.dataset)
        
        if self.noise_type == 'sym':
            assert self.args.dataset in ['cifar10', 'cifar100']
            noise_idx, train_data, train_noisy_targets = get_sym_noisy_labels(train_data, train_targets, self.num_classes, self.noise_ratio)
        elif self.noise_type == 'asym':
            if self.args.dataset == 'cifar10':
                noise_idx, train_data, train_noisy_targets = get_cifar10_asym_noisy_labels(train_data, train_targets, self.num_classes, self.noise_ratio)
            elif self.args.dataset == 'cifar100':
                noise_idx, train_data, train_noisy_targets = get_cifar100_asym_noisy_labels(train_data, train_targets, self.num_classes, self.noise_ratio)
            else:
                raise NotImplementedError
        elif self.noise_type == 'ins':
            if self.args.dataset == 'cifar10n':
                noise_file = torch.load(os.path.join(self.args.data_dir, 'cifar10n', 'CIFAR-10_human.pt'))
                assert self.noise_ratio in ['clean_label', 'worse_label', 'aggre_label', 'random_label1', 'random_label2', 'random_label3']
                train_noisy_targets = noise_file[self.noise_ratio]
            elif self.args.dataset == 'cifar100n':
                noise_file = torch.load(os.path.join(self.args.data_dir, 'cifar100n', 'CIFAR-100_human.pt'))
                assert self.noise_ratio in ['clean_label', 'noisy_label']
                train_noisy_targets = noise_file[self.noise_ratio]
            else:
                # noisy labels is directly loaded in train_targets
                train_noisy_targets = train_targets
        else:
            raise NotImplementedError
        
        if self.args.dataset in ['cifar10', 'cifar100', 'cifar10n', 'cifar100n']:
            resize = 'resize_crop_pad'
            if self.args.dataset == 'cifar10' or self.args.dataset == 'cifar10n':
                # autoaug = 'randaug'
                autoaug = 'autoaug_cifar'
            else:
                autoaug = 'autoaug_cifar'
            test_resize = 'resize'
        elif self.args.dataset == 'webvision':
            resize = 'resize_rpc'
            autoaug = 'autoaug'
            test_resize = 'resize'
        elif self.args.dataset == 'clothing1m':
            resize = 'resize_crop'
            autoaug = 'autoaug'
            test_resize = 'resize_crop'
        else:
            resize = 'rpc'
            autoaug = 'autoaug'
            test_resize = 'resize_crop'
            
        if not self.strong_aug:
            autoaug = None
        
        # make dataset
        train_dataset = ImgTwoViewBaseDataset(self.args.dataset, train_data, train_noisy_targets, 
                                              num_classes=self.num_classes, is_train=True,
                                              img_size=self.args.img_size, crop_ratio=self.args.crop_ratio,
                                              autoaug=autoaug, resize=resize,
                                              return_target=True, return_keys=['x_w', 'x_s', 'y'])
        test_dataset = ImgBaseDataset(self.args.dataset, test_data, test_targets, 
                                      num_classes=self.num_classes, is_train=False,
                                      img_size=self.args.img_size, crop_ratio=self.args.crop_ratio, resize=test_resize,
                                      return_keys=['x', 'y'])

        self.print_fn("Create datasets!")
        return {'train': train_dataset, 'eval': test_dataset}

    def set_data_loader(self):
        loader_dict = {}

        loader_dict['train'] = get_dataloader(self.dataset_dict['train'], 
                                              num_epochs=self.epochs, 
                                              batch_size=self.args.batch_size, 
                                              shuffle=True, 
                                              num_workers=self.args.num_workers, 
                                              pin_memory=True, 
                                              drop_last=True,
                                              distributed=self.args.distributed)
        loader_dict['eval'] = get_dataloader(self.dataset_dict['eval'], 
                                             num_epochs=self.epochs, 
                                             batch_size=self.args.eval_batch_size, 
                                             shuffle=False, 
                                             num_workers=self.args.num_workers, 
                                             pin_memory=True, 
                                             drop_last=False)
        self.print_fn("Create data loaders!")
        return loader_dict


    def train(self):
        """
        train function
        """
        self.model.train()
        self.call_hook("before_run")

        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            
            # prevent the training iterations exceed args.num_train_iter
            if self.it >= self.num_train_iter:
                break
            
            self.call_hook("before_train_epoch")
            

            for data in self.loader_dict['train']:
                # prevent the training iterations exceed args.num_train_iter
                if self.it >= self.num_train_iter:
                    break
                
                self.call_hook("before_train_step")
                self.out_dict, self.log_dict = self.train_step(**self.process_batch(**data))
                self.call_hook("after_train_step")
                self.it += 1
            
            self.call_hook("after_train_epoch")

        self.call_hook("after_run")

    def train_step(self, x_w, x_s, y):    
            
        inputs = torch.cat((x_w, x_s))
        true_outputs = self.model(inputs)
        logits_x_w, logits_x_s = true_outputs.chunk(2)
        noise_matrix = self.noise_model(logits_x_w)
        # noise_matrix *= 2
        
        # convert logits_w to probs
        probs_x_w = logits_x_w.softmax(dim=-1).detach()
             
        # convert logits_s to probs
        probs_x_s = logits_x_s.softmax(dim=-1)
        
        # compute forward-backward on graph x_w
        with torch.no_grad():
            # model p(y_hat | y, x) p(y|x)
            noise_matrix_col = noise_matrix.softmax(dim=-1)[:, y].detach().transpose(0, 1)
            em_y = probs_x_w * noise_matrix_col
            em_y = em_y / em_y.sum(dim=1, keepdim=True)

        # compute forward_backward on graph x_s
        em_probs_x_s = probs_x_s * noise_matrix_col
        em_probs_x_s = em_probs_x_s / em_probs_x_s.sum(dim=1, keepdim=True)
        
        # compute observed noisy labels
        noise_matrix_row = noise_matrix.softmax(dim=0)
        noisy_probs_x_w = torch.matmul(logits_x_w.softmax(dim=-1), noise_matrix_row)
        noisy_probs_x_w = noisy_probs_x_w / noisy_probs_x_w.sum(dim=-1, keepdims=True)

        # compute noisy loss 
        noise_loss = torch.mean(-torch.sum(F.one_hot(y, self.num_classes) * torch.log(noisy_probs_x_w), dim = -1))
        
        # compute em loss
        em_loss =  torch.mean(-torch.sum(em_y * torch.log(em_probs_x_s), dim=-1), dim=-1)
        
        # compute consistency loss
        con_loss = self.ce_loss(logits_x_s, probs_x_w, reduction='mean')
        
        # total loss
        loss = noise_loss + em_loss + con_loss
        
        # computer average entropy loss
        if self.average_entropy_loss:
            avg_prediction = torch.mean(logits_x_w.softmax(dim=-1), dim=0)
            prior_distr = 1.0/self.num_classes * torch.ones_like(avg_prediction)
            avg_prediction = torch.clamp(avg_prediction, min = 1e-6, max = 1.0)
            balance_kl =  torch.mean(-(prior_distr * torch.log(avg_prediction)).sum(dim=0))
            entropy_loss = 0.1 * balance_kl
            loss += entropy_loss

        out_dict = self.process_out_dict(loss=loss)
        log_dict = self.process_log_dict(loss=loss.item(), 
                                         noise_loss=noise_loss.item(),
                                         em_loss=em_loss.item(),
                                         con_loss=con_loss.item())
        return out_dict, log_dict

    def get_save_dict(self):
        save_dict = super().get_save_dict()
        # additional saving arguments
        save_dict['noise_model'] = self.noise_model.state_dict()
        save_dict['optimizer_noise'] = self.optimizer_noise.state_dict()
        return save_dict


    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.noise_model.load_state_dict(checkpoint['noise_model'])
        self.optimizer_noise.load_state_dict(checkpoint['optimizer_noise'])
        self.print_fn("additional parameter loaded")
        return checkpoint

    @staticmethod
    def get_argument():
        return [
            Argument('--average_entropy_loss', str2bool, True, 'use entropy loss'),
            Argument('--noise_ratio', float, 0.1, 'noise ratio for noisy label learning'),
            Argument('--noise_type', str, 'sym', 'noise type (sym, asym, ins) noisy label learning'),
            Argument('--noise_matrix_scale', float, 1.0, 'scale for noise matrix'),
        ]