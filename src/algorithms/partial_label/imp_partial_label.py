

import torch 

from src.core.algorithmbase import AlgorithmBase
from src.core.utils import Argument, get_optimizer, str2bool
from src.core.criterions import CELoss
from src.datasets import get_partial_labels, get_data, get_dataloader, ImgBaseDataset, ImgThreeViewBaseDataset


class ImprecisePartialLabelLearning(AlgorithmBase):
    def __init__(self, args, tb_log=None, logger=None, **kwargs):

        self.init(args)
        super().__init__(args, tb_log, logger, **kwargs)
        
        # set number train iterations
        self.num_train_iter = self.epochs * len(self.loader_dict['train'])
        self.num_eval_iter = len(self.loader_dict['train'])
        self.ce_loss = CELoss()


    def init(self, args):
        # extra arguments 
        self.average_entropy_loss = args.average_entropy_loss
        self.ema_p = 0.999
        self.partial_ratio = args.partial_ratio


    def set_optimizer(self):
        """
        set optimizer for algorithm
        """
        # defautl is for semi-supervised learning
        self.print_fn("Create optimizer and scheduler")
        optimizer = get_optimizer(self.model, self.args.optim, self.args.lr, self.args.momentum, self.args.weight_decay, self.args.layer_decay, nesterov=False, bn_wd_skip=False)
        
        if self.args.net == 'wrn_34_10':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[100 * len(self.loader_dict['train']), 150 * len(self.loader_dict['train'])]) 
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(self.epochs * len(self.loader_dict['train'])), eta_min=1e-4)
        return optimizer, scheduler

    def set_dataset(self):
        # get initial data
        train_data, train_targets, test_data, test_targets, extra_data = get_data(self.args.data_dir, self.args.dataset)
        
        # make partial labels
        train_data, train_partial_targets = get_partial_labels(train_data, train_targets, self.num_classes, self.args.partial_ratio)
        
        if self.args.dataset in ['mnist', 'fmnist']:
            resize = 'resize'
            test_resize = 'resize'
            autoaug = 'randaug' 
        elif self.args.dataset in ['cifar10', 'svhn', 'cifar100']:
            resize = 'resize_crop_pad'
            test_resize = 'resize'
            autoaug = 'randaug'
        elif self.args.dataset in ['stl10']:
            resize = 'resize_crop'
            autoaug = 'randaug'
        elif self.args.dataset in ['imagenet1k', 'imagenet100']:
            resize = 'resize_rpc'
            autoaug = 'autoaug'
        test_resize = 'resize'
        
        if not self.strong_aug:
            autoaug = None
        
        # make dataset
        train_dataset = ImgThreeViewBaseDataset(self.args.dataset, train_data, train_partial_targets, 
                                                num_classes=self.num_classes, is_train=True,
                                                img_size=self.args.img_size, crop_ratio=self.args.crop_ratio,
                                                autoaug=autoaug, resize=resize,
                                                return_target=True, return_keys=['x_w', 'x_s', 'x_s_', 'y'])
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


    def train_step(self, x_w, x_s, x_s_, y):        
        inputs = torch.cat((x_w, x_s, x_s_), dim=0)
        outputs = self.model(inputs)
        logits_x_w, logits_x_s, logits_x_s_ = outputs.chunk(3)
        
        # convert logots_w to probs
        probs_x_w = logits_x_w.softmax(dim=-1)
        
        # compute forward-backward on graph
        with torch.no_grad():
            # pseudo_y_ulb, avg_paths = self.nfa_builder.compute(torch.log(probs_x_w.detach()), y)
            pseudo_y_ulb = y * probs_x_w.detach()
            pseudo_y_ulb = pseudo_y_ulb / pseudo_y_ulb.sum(dim=-1, keepdim=True)

        sup_loss = -torch.mean(torch.sum(torch.log(1.0000001 - logits_x_w.softmax(dim=-1)) * (1 - y), dim=1), dim=0)
        
        # compute unsupervised loss 
        unsup_loss = self.ce_loss(torch.cat([logits_x_w, logits_x_s, logits_x_s_], dim=0), torch.cat([pseudo_y_ulb, pseudo_y_ulb, pseudo_y_ulb], dim=0), reduction='mean')
        
        # total_loss = sup_loss + min(self.epoch / 100, 1) * unsup_loss
        total_loss = sup_loss + unsup_loss
        
        # computer average entropy loss
        if self.average_entropy_loss:
            avg_prediction = torch.mean(probs_x_w, dim=0)
            prior_distr = 1.0/self.num_classes * torch.ones_like(avg_prediction)
            avg_prediction = torch.clamp(avg_prediction, min = 1e-6, max = 1.0)
            balance_kl =  torch.mean(-(prior_distr * torch.log(avg_prediction)).sum(dim=0))
            entropy_loss = 0.1 * balance_kl
            total_loss += entropy_loss

        out_dict = self.process_out_dict(loss=total_loss)
        log_dict = self.process_log_dict(loss=total_loss.item(), sup_loss=sup_loss.item(), unsup_loss=unsup_loss.item())
        return out_dict, log_dict


    @staticmethod
    def get_argument():
        return [
            Argument('--average_entropy_loss', str2bool, False, 'use entropy loss'),
            Argument('--partial_ratio', float, 0.1, 'ambiguity level (q) in partial label learning'),
        ]