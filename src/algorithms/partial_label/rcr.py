

import torch 
from copy import deepcopy

from src.datasets import get_partial_labels, get_data, ImgBaseDataset, ImgThreeViewBaseDataset

from .imp_partial_label import ImprecisePartialLabelLearning


class RCRPartialLabelLearning(ImprecisePartialLabelLearning):
    def __init__(self, args, tb_log=None, logger=None, **kwargs):
        self.init(args)
        super().__init__(args, tb_log, logger, **kwargs)

    def set_dataset(self):
        # get initial data
        train_data, train_targets, test_data, test_targets, extra_data = get_data(self.args.data_dir, self.args.dataset)
        
        # make partial labels
        train_data, train_partial_targets = get_partial_labels(train_data, train_targets, self.num_classes, self.args.partial_ratio)
        
        self.confidence = torch.from_numpy(deepcopy(train_partial_targets))
        self.confidence = self.confidence / self.confidence.sum(dim=-1, keepdim=True)
        
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
                                                return_target=True, return_idx=True, return_keys=['x_idx', 'x_w', 'x_s', 'x_s_', 'y'])
        test_dataset = ImgBaseDataset(self.args.dataset, test_data, test_targets, 
                                      num_classes=self.num_classes, is_train=False,
                                      img_size=self.args.img_size, crop_ratio=self.args.crop_ratio, resize=test_resize,
                                      return_keys=['x', 'y'])

        self.print_fn("Create datasets!")
        return {'train': train_dataset, 'eval': test_dataset}


    def train_step(self, x_idx, x_w, x_s, x_s_, y):    
            
        inputs = torch.cat((x_w, x_s, x_s_), dim=0)
        outputs = self.model(inputs)
        logits_x_w, logits_x_s, logits_x_s_ = outputs.chunk(3)
        
        # convert logots_w to probs
        probs_x_w = logits_x_w.softmax(dim=-1)

        sup_loss = -torch.mean(torch.sum(torch.log(1.0000001 - logits_x_w.softmax(dim=-1)) * (1 - y), dim=1), dim=0)
        
        # compute unsupervised loss 
        pseudo_y_ulb = self.confidence[x_idx.cpu()].to(x_w.device)
        unsup_loss = self.ce_loss(torch.cat([logits_x_w, logits_x_s, logits_x_s_], dim=0), torch.cat([pseudo_y_ulb, pseudo_y_ulb, pseudo_y_ulb], dim=0), reduction='mean')
        
        # update confidence
        with torch.no_grad():
            probs_x_w = logits_x_w.detach().softmax(dim=-1)
            probs_x_s = logits_x_s.detach().softmax(dim=-1)
            probs_x_s_ = logits_x_s_.detach().softmax(dim=-1)
            revised_y = y.clone()
            
            revised_y = revised_y * torch.pow(probs_x_w, 1.0 / 3.0)  * torch.pow(probs_x_s, 1.0 / 3.0) * torch.pow(probs_x_s_, 1.0 / 3.0)
            revised_y = revised_y / revised_y.sum(dim=-1, keepdim=True)
            self.confidence[x_idx] = revised_y.cpu()
        
        total_loss = sup_loss + min(self.epoch / 100, 1) * unsup_loss
        out_dict = self.process_out_dict(loss=total_loss)
        log_dict = self.process_log_dict(loss=total_loss.item(), sup_loss=sup_loss.item(), unsup_loss=unsup_loss.item())
        return out_dict, log_dict