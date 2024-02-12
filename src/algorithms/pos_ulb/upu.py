

import torch 

from src.algorithms.pos_ulb.imp_pos_ulb import ImprecisePositiveUnlabeledLearning
from src.datasets import ImgBaseDataset


class UnbiasedPositiveUnLabeledLearning(ImprecisePositiveUnlabeledLearning):
    def __init__(self, args, tb_log=None, logger=None, **kwargs):
        super().__init__(args, tb_log, logger, **kwargs)
    
    
    def set_dataset(self): 
        dataset_dict = super().set_dataset()
        if self.args.dataset in ['mnist', 'fmnist']:
            resize = 'resize'
            test_resize = 'resize'
        elif self.args.dataset in ['cifar10', 'svhn', 'cifar100']:
            resize = 'resize_crop_pad'
            test_resize = 'resize'
        elif self.args.dataset in ['stl10']:
            resize = 'resize_crop'
        elif self.args.dataset in ['imagenet1k', 'imagenet100']:
            resize = 'rpc'
        train_ulb_dataset = dataset_dict['train_ulb']
        train_ulb_dataset =  ImgBaseDataset(self.args.dataset, train_ulb_dataset.data, train_ulb_dataset.targets, 
                                          num_classes=self.num_classes, class_map=self.class_map, is_train=True,
                                          img_size=self.args.img_size, crop_ratio=self.args.crop_ratio,
                                          autoaug=None, resize=resize,
                                          return_target=False,
                                          return_keys=['x_ulb'])
        dataset_dict['train_ulb'] = train_ulb_dataset
        return dataset_dict


    def train_step(self, x_lb, y_lb, x_ulb):

        num_lb = y_lb.shape[0]
        
        # forward model
        inputs = torch.cat((x_lb, x_ulb))
        outputs = self.model(inputs)
        logits_x_lb = outputs[:num_lb]
        logits_x_ulb = outputs[num_lb:]
        
        # define targets
        targets_lb_sub = torch.zeros_like(y_lb, dtype=torch.long)
        targets_ulb_sub = torch.zeros((x_ulb.shape[0], ), device=x_lb.device, dtype=torch.long)
        
        # compute loss
        loss_pos = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
        loss_pos_neg = self.ce_loss(logits_x_lb, targets_lb_sub, reduction='mean') 
        loss_ulb = self.ce_loss(logits_x_ulb, targets_ulb_sub, reduction='mean')
        
        # total loss
        total_loss = self.class_prior * (loss_pos - loss_pos_neg)+ loss_ulb
        
        out_dict = self.process_out_dict(loss=total_loss)
        log_dict = self.process_log_dict(loss=total_loss.item(),  loss_pos=loss_pos.item(), loss_pos_neg=loss_pos_neg.item(), loss_ulb=loss_ulb.item())
        return out_dict, log_dict