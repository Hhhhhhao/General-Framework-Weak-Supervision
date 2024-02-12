

import torch 

from src.core.algorithmbase import AlgorithmBase
from src.core.utils import Argument, str2bool, get_optimizer
from src.core.criterions import CELoss, BCELoss
from src.nets import get_model
from src.core.nfa import create_proportion_graph


from src.datasets import get_ulb_ulb_labels, get_data, get_dataloader, ImgBaseDataset, ImgTwoViewBaseDataset


class ImpreciseUnlabeledUnlabeledLearning(AlgorithmBase):
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
            
        self.num_ulb1_data = args.num_ulb1_data
        self.num_ulb2_data = args.num_ulb2_data
        self.cls_prior_ulb1 = args.cls_prior_ulb1
        self.cls_prior_ulb2 = args.cls_prior_ulb2


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
        
        train_ulb1_idx, train_ulb1_data, train_ulb1_targets, \
        train_ulb2_idx, train_ulb2_data, train_ulb2_targets, \
        cls_prior_ulb1, cls_prior_ulb2 = get_ulb_ulb_labels(train_data, train_targets, self.num_classes,
                                                            target_classes=self.target_classes,
                                                            neg_classes=self.neg_classes,
                                                            num_ulb1_data=self.num_ulb1_data,
                                                            num_ulb2_data=self.num_ulb2_data,
                                                            cls_prior_ulb1=self.cls_prior_ulb1,
                                                            cls_prior_ulb2=self.cls_prior_ulb2)
        self.ulb_prior = len(train_ulb1_data) / (len(train_ulb1_data) + len(train_ulb2_data))
        
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
        train_ulb1_dataset = ImgTwoViewBaseDataset(self.args.dataset, train_ulb1_data, train_ulb1_targets,
                                       num_classes=self.num_classes, class_map=self.class_map,
                                       is_train=True,
                                       img_size=self.args.img_size, crop_ratio=self.args.crop_ratio,
                                       autoaug=autoaug, resize=resize,
                                       # autoaug=None, resize=resize,
                                       return_target=False, return_keys=['x_ulb1_w', 'x_ulb1_s'])
        
        train_ulb2_dataset = ImgTwoViewBaseDataset(self.args.dataset, train_ulb2_data, train_ulb2_targets, 
                                       num_classes=self.num_classes, class_map=self.class_map,
                                       is_train=True,
                                       img_size=self.args.img_size, crop_ratio=self.args.crop_ratio,
                                       autoaug=autoaug, resize=resize,
                                       # autoaug=None, resize=resize,
                                       return_target=False, return_keys=['x_ulb2_w', 'x_ulb2_s'])
        
        test_dataset = ImgBaseDataset(self.args.dataset, test_data, test_targets, 
                                      num_classes=self.num_classes, class_map=self.class_map,
                                      is_train=False,
                                      # aggregation='proportion',
                                      img_size=self.args.img_size, crop_ratio=self.args.crop_ratio, resize=test_resize,
                                      return_keys=['x', 'y'])

        self.print_fn("Create datasets!")
        return {'train_ulb1': train_ulb1_dataset, 'train_ulb2': train_ulb2_dataset, 'eval': test_dataset}

    def set_data_loader(self):
        loader_dict = {}

        loader_dict['train_ulb1'] = get_dataloader(self.dataset_dict['train_ulb1'], 
                                                 num_epochs=self.epochs,
                                                 num_train_iter=self.num_train_iter, 
                                                 batch_size=self.args.batch_size, 
                                                 shuffle=True, 
                                                 num_workers=self.args.num_workers, 
                                                 pin_memory=True, 
                                                 drop_last=True,
                                                 distributed=self.args.distributed)
        loader_dict['train_ulb2'] = get_dataloader(self.dataset_dict['train_ulb2'], 
                                                  num_epochs=self.epochs,
                                                  num_train_iter=self.num_train_iter, 
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
            

            for data_ulb1, data_ulb2 in zip(self.loader_dict['train_ulb1'],
                                            self.loader_dict['train_ulb2']):
                
                # prevent the training iterations exceed args.num_train_iter
                if self.it >= self.num_train_iter:
                    break
                
                self.call_hook("before_train_step")
                self.out_dict, self.log_dict = self.train_step(**self.process_batch(**data_ulb1, **data_ulb2))
                self.call_hook("after_train_step")
                self.it += 1
            
            self.call_hook("after_train_epoch")

        self.call_hook("after_run")
    

    def train_step(self, x_ulb1_w, x_ulb1_s, x_ulb2_w, x_ulb2_s):
        
        batch_size = x_ulb1_w.size(0)
        
        inputs = torch.cat((x_ulb1_w, x_ulb1_s, x_ulb2_w, x_ulb2_s))
        outputs = self.model(inputs)
        logits_x_ulb1_w, logits_x_ulb1_s, logits_x_ulb2_w, logits_x_ulb2_s = outputs.chunk(4)
        
        # forward-backward algorithm
        probs_x_ulb1_w = logits_x_ulb1_w.softmax(dim=1)
        pseudo_probs_x_ulb1, count_probs_x_ulb1 = create_proportion_graph(torch.log(probs_x_ulb1_w), int(self.cls_prior_ulb1 * batch_size))
        probs_x_ulb2_w = logits_x_ulb2_w.softmax(dim=1)
        pseudo_probs_x_ulb2, count_probs_x_ulb2 = create_proportion_graph(torch.log(probs_x_ulb2_w), int(self.cls_prior_ulb2 * batch_size))
        
        # compute loss
        unsup_ulb1_loss = self.ce_loss(logits_x_ulb1_s, pseudo_probs_x_ulb1, reduction='mean')
        unsup_ulb2_loss = self.ce_loss(logits_x_ulb2_s, pseudo_probs_x_ulb2, reduction='mean')
        
        # total loss
        total_loss = unsup_ulb1_loss + unsup_ulb2_loss
        
        out_dict = self.process_out_dict(loss=total_loss)
        log_dict = self.process_log_dict(loss=total_loss.item(),  
                                         unsup_ulb1_loss=unsup_ulb1_loss.item(), unsup_ulb2_loss=unsup_ulb2_loss.item())
        return out_dict, log_dict



    @staticmethod
    def get_argument():
        return [
            Argument('--target_classes', int, 9, 'positive class for positive unlabeled learning', nargs='+'),
            Argument('--neg_classes', int, None, 'negative class for positive unlabeled learning', nargs='+'),
            Argument('--num_ulb1_data', int, 1000, 'number of samples in unlabeled set 1'),
            Argument('--num_ulb2_data', int, 1000, 'number of samples in unlabeled set 2'),
            Argument('--cls_prior_ulb1', float, 0.5, 'class prior in unlabeled set 1'),
            Argument('--cls_prior_ulb2', float, 0.5, 'class prior in unlabeled set 2'),
        ]