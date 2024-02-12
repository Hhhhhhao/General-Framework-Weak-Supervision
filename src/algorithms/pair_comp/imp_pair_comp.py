

import torch 
import numpy as np

from src.core.algorithmbase import AlgorithmBase
from src.core.utils import Argument, str2bool, get_optimizer
from src.core.criterions import CELoss, BCELoss
from src.nets import get_model
from src.core.nfa import create_pair_comp_graph


from src.datasets import get_pairwise_comp_labels, get_data, get_dataloader, bag_collate_fn, ImgBaseDataset, ImageTwoViewBagDataset, ImgTwoViewBaseDataset


class ImprecisePairComparisonLearning(AlgorithmBase):
    def __init__(self, args, tb_log=None, logger=None, **kwargs):
        self.init(args)
        super().__init__(args, tb_log, logger, **kwargs)
        
        # set number train iterations
        self.num_train_iter = self.epochs * len(self.loader_dict['train'])
        self.num_eval_iter = len(self.loader_dict['train'])
        
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
        self.num_pair_data = args.num_pair_data
        self.class_prior = args.class_prior


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
        ema_model = get_model(model_name=self.args.net, num_classes=self.output_classes)
        ema_model.load_state_dict(self.model.state_dict())
        return ema_model

    def set_optimizer(self):
        """
        set optimizer for algorithm
        """
        # defautl is for semi-supervised learning
        self.print_fn("Create optimizer and scheduler")
        optimizer = get_optimizer(self.model, self.args.optim, self.args.lr, self.args.momentum, self.args.weight_decay, self.args.layer_decay, nesterov=False, bn_wd_skip=False)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(self.epochs * len(self.loader_dict['train'])), eta_min=1e-6)
        return optimizer, scheduler

    def set_dataset(self):
        # get initial data
        train_data, train_targets, test_data, test_targets, extra_data = get_data(self.args.data_dir, self.args.dataset)
        
        # make pair data
        train_pair_samples, train_pair_targets = get_pairwise_comp_labels(train_data, train_targets, self.num_pair_data, 
                                                              target_classes=self.target_classes, 
                                                              neg_classes=self.neg_classes,
                                                              class_map=self.class_map,
                                                              num_pair_data=self.num_pair_data,
                                                              class_prior=self.class_prior)
        
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
        train_dataset = ImageTwoViewBagDataset(self.args.dataset, train_pair_samples, train_pair_targets, 
                                               num_classes=self.num_classes, class_map=self.class_map, is_train=True,
                                               img_size=self.args.img_size, crop_ratio=self.args.crop_ratio,
                                               autoaug=autoaug, resize=resize,
                                               aggregation='pair_comp',
                                               return_target=False,
                                               return_keys=['x_bag_w', 'x_bag_s'])
        test_dataset = ImgBaseDataset(self.args.dataset, test_data, test_targets, 
                                      num_classes=self.num_classes, class_map=self.class_map,
                                      is_train=False,
                                      # aggregation='proportion',
                                      img_size=self.args.img_size, crop_ratio=self.args.crop_ratio, resize=test_resize,
                                      return_keys=['x', 'y'])

        self.print_fn("Create datasets!")
        return {'train': train_dataset,  'eval': test_dataset}

    def set_data_loader(self):
        loader_dict = {}

        loader_dict['train'] = get_dataloader(self.dataset_dict['train'], 
                                                 num_epochs=self.epochs,
                                                 batch_size=self.args.batch_size, 
                                                 shuffle=True, 
                                                 num_workers=self.args.num_workers, 
                                                 pin_memory=True, 
                                                 drop_last=True,
                                                 distributed=self.args.distributed,
                                                 collate_fn=bag_collate_fn)
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
    

    def train_step(self, x_bag_w, x_bag_s):
        
        bag_batch_size = x_bag_w.shape[0]
        
        x_ins_w = x_bag_w.view(-1, *x_bag_w.shape[2:])
        x_ins_s = x_bag_s.view(-1, *x_bag_s.shape[2:])
        
        # forward model
        inputs = torch.cat((x_ins_w, x_ins_s))
        outputs = self.model(inputs)
        logits_x_ins_w, logits_x_ins_s = outputs.chunk(2)
        logits_x_bag_w = logits_x_ins_w.view(bag_batch_size, 2, -1)
        
        # forward-backward algorithm on pairwise comparison bag
        with torch.no_grad():
            probs_x_bag_w = logits_x_bag_w.softmax(dim=1)
            pseudo_probs_x_bag, count_probs_x_bag = create_pair_comp_graph(torch.log(probs_x_bag_w))
        
        # calculate labeled loss
        unsup_loss = self.ce_loss(logits_x_ins_s, pseudo_probs_x_bag.reshape(-1, 2), reduction='mean')
    
        total_loss = unsup_loss
        
        out_dict = self.process_out_dict(loss=total_loss)
        log_dict = self.process_log_dict(loss=total_loss.item(),  
                                         unsup_loss=unsup_loss.item())
        return out_dict, log_dict

    @staticmethod
    def get_argument():
        return [
            Argument('--target_classes', int, 9, 'positive class for positive unlabeled learning', nargs='+'),
            Argument('--neg_classes', int, None, 'negative class for positive unlabeled learning', nargs='+'),
            Argument('--num_pair_data', int, 10000, 'number of data pairs'),
            Argument('--class_prior', float, 0.5, 'positive data class prior'),
            
        ]