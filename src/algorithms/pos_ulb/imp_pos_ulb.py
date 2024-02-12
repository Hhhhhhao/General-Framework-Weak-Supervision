

import torch 
import numpy as np

from src.core.algorithmbase import AlgorithmBase
from src.core.utils import Argument, str2bool, get_optimizer
from src.core.criterions import CELoss, BCELoss
from src.nets import get_model
from src.core.nfa import create_proportion_graph


from src.datasets import get_pos_ulb_labels, get_data, get_dataloader, ImgBaseDataset, ImgTwoViewBaseDataset


class ImprecisePositiveUnlabeledLearning(AlgorithmBase):
    def __init__(self, args, tb_log=None, logger=None, **kwargs):
        self.init(args)
        super().__init__(args, tb_log, logger, **kwargs)
        
        # set number train iterations
        self.num_eval_iter = self.num_train_iter // self.epochs
        
        self.ce_loss = CELoss()
        self.bce_loss = BCELoss()
        

    def init(self, args):        
        self.target_classes = args.target_classes
        if isinstance(self.target_classes, int):
            self.target_classes = [self.target_classes]
        self.target_classes = sorted(self.target_classes)
        self.neg_classes = args.neg_classes
        if self.neg_classes is None or self.neg_classes == 'None':
            self.neg_classes = [i for i in range(args.num_classes) if i not in self.target_classes]
        
        self.output_classes = 2
        self.class_map = {}
        for i in range(args.num_classes):
            if i in self.target_classes:
                self.class_map[i] = 1 
            elif i in self.neg_classes:
                self.class_map[i] = 0
        self.num_pos_data = args.num_pos_data
        self.num_ulb_data = args.num_ulb_data
        self.include_lb_to_ulb = args.include_lb_to_ulb
        self.uratio = args.uratio


    def set_model(self):
        """
        initialize model
        """
        model = get_model(model_name=self.args.net, num_classes=self.output_classes, pretrained=self.args.use_pretrain)
        return model

    def set_ema_model(self):
        """
        initialize ema model from model
        """
        ema_model = get_model(model_name=self.args.net, num_classes=self.output_classes,)
        ema_model.load_state_dict(self.model.state_dict())
        return ema_model

    def set_optimizer(self):
        """
        set optimizer for algorithm
        """
        # defautl is for semi-supervised learning
        self.print_fn("Create optimizer and scheduler")
        optimizer = get_optimizer(self.model, self.args.optim, self.args.lr, self.args.momentum, self.args.weight_decay, self.args.layer_decay, nesterov=False, bn_wd_skip=False)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_train_iter, eta_min=1e-6)
        return optimizer, scheduler

    def set_dataset(self):
        # get initial data
        train_data, train_targets, test_data, test_targets, extra_data = get_data(self.args.data_dir, self.args.dataset)
        
        # make partial labels
        train_pos_idx, train_pos_data, train_pos_targets, \
        train_ulb_idx, train_ulb_data, train_ulb_targets, \
        class_prior = get_pos_ulb_labels(train_data, train_targets, self.num_classes, 
                                         target_classes=self.target_classes, 
                                         num_pos_data=self.num_pos_data,
                                         num_ulb_data=self.num_ulb_data,
                                         include_lb_to_ulb=self.include_lb_to_ulb)
        
        # TODO: check this
        if self.args.dataset == 'stl10':
            if train_ulb_data is not None:
                train_ulb_data = np.concatenate((train_ulb_data, extra_data))
                train_ulb_targets = np.concatenate((train_ulb_targets, np.zeros((len(extra_data),), dtype=train_targets.dtype)))
            else:
                train_ulb_data = extra_data
                train_ulb_targets = np.zeros((len(extra_data),), dtype=train_targets.dtype)
        
        self.class_prior = class_prior
        self.num_ulb_data = len(train_ulb_data)
        
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
            resize = 'rpc'
            autoaug = 'autoaug'
        test_resize = 'resize'
        
        if not self.strong_aug:
            autoaug = None
        
        # make dataset
        train_lb_dataset = ImgBaseDataset(self.args.dataset, train_pos_data, train_pos_targets, 
                                          num_classes=self.num_classes, class_map=self.class_map, is_train=True,
                                          img_size=self.args.img_size, crop_ratio=self.args.crop_ratio,
                                          autoaug=None, resize=resize,
                                          return_keys=['x_lb', 'y_lb'])
        train_ulb_dataset = ImgTwoViewBaseDataset(self.args.dataset, train_ulb_data, train_ulb_targets, 
                                       num_classes=self.num_classes, class_map=self.class_map,
                                       is_train=True,
                                       img_size=self.args.img_size, crop_ratio=self.args.crop_ratio,
                                       autoaug=autoaug, resize=resize,
                                       # autoaug=None, resize=resize,
                                       return_target=False, return_keys=['x_ulb_w', 'x_ulb_s'])
        test_dataset = ImgBaseDataset(self.args.dataset, test_data, test_targets, 
                                      num_classes=self.num_classes, class_map=self.class_map,
                                      is_train=False,
                                      # aggregation='proportion',
                                      img_size=self.args.img_size, crop_ratio=self.args.crop_ratio, resize=test_resize,
                                      return_keys=['x', 'y'])

        self.print_fn("Create datasets!")
        return {'train_lb': train_lb_dataset, 'train_ulb': train_ulb_dataset, 'eval': test_dataset}

    def set_data_loader(self):
        loader_dict = {}

        loader_dict['train_lb'] = get_dataloader(self.dataset_dict['train_lb'], 
                                                 num_epochs=self.epochs,
                                                 num_train_iter=self.num_train_iter, 
                                                 batch_size=self.args.batch_size, 
                                                 shuffle=True, 
                                                 num_workers=self.args.num_workers, 
                                                 pin_memory=True, 
                                                 drop_last=True,
                                                 distributed=self.args.distributed)
        loader_dict['train_ulb'] = get_dataloader(self.dataset_dict['train_ulb'], 
                                                  num_epochs=self.epochs,
                                                  num_train_iter=self.num_train_iter, 
                                                  batch_size=int(self.args.uratio * self.args.batch_size), 
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
        # default is for semi-sup setting
        return super().train()
    

    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s):
        
        num_lb = y_lb.shape[0]
        num_ulb = x_ulb_w.shape[0]
        
        # forward model
        inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
        outputs = self.model(inputs)
        logits_x_lb = outputs[:num_lb]
        logits_x_ulb_w, logits_x_ulb_s = outputs[num_lb:].chunk(2)
        
        # calculate labeled loss
        lb_sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
        
        # forward-backward algorithm
        with torch.no_grad():
            probs_x_ulb_w = logits_x_ulb_w.softmax(dim=1)
            pseudo_probs_x, count_probs_x = create_proportion_graph(torch.log(probs_x_ulb_w), int(self.class_prior * num_ulb))
        
        # calculate unsup loss
        unsup_loss = self.ce_loss(logits_x_ulb_s, pseudo_probs_x, reduction='mean')
        total_loss = lb_sup_loss + unsup_loss 
        
        
        out_dict = self.process_out_dict(loss=total_loss)
        log_dict = self.process_log_dict(loss=total_loss.item(),  lb_sup_loss=lb_sup_loss.item(), unsup_loss=unsup_loss.item(), 
                                         )
        return out_dict, log_dict



    @staticmethod
    def get_argument():
        return [
            Argument('--target_classes', int, 9, 'positive class for positive unlabeled learning', nargs='+'),
            Argument('--neg_classes', int, None, 'negative class for positive unlabeled learning', nargs='+'),
            Argument('--num_pos_data', int, 1000, 'number of labeled positive samples'),
            Argument('--num_ulb_data', int, 1000, 'number of unlabeled samples'),
            Argument('--uratio', int, 2, 'ratio of unlabeled batch size to labeled batch size'),
            Argument('--include_lb_to_ulb', str2bool, True, 'flag of adding labeled data into unlabeled data'),
            # Argument('--std_bag_len', int, 1, 'std for bag length'),
            # Argument('--num_bags_train', int, 500, 'number of training bags'),
            # Argument('--num_bags_test', int, 20, 'number of testing bags'),
            # Argument('--balanced_bags', str2bool, False, 'bags are balanced'),
            # Argument('--multi_label', str2bool, False, 'bags has multi label'),
            
        ]