

import torch 
import torch.nn.functional as F
from copy import deepcopy

from src.core.utils import Argument
from src.datasets import get_partial_labels, get_data, ImgBaseDataset

from .imp_partial_label import ImprecisePartialLabelLearning


def partial_loss(output1, target):
    output = F.softmax(output1, dim=1)
    l = target * torch.log(output)
    loss = (-torch.sum(l)) / l.size(0)

    revisedY = target.clone()
    revisedY[revisedY > 0]  = 1
    revisedY = revisedY * output
    revisedY = revisedY / revisedY.sum(dim=1, keepdims=True)

    new_target = revisedY


    return loss, new_target


class ProdenPartialLabelLearning(ImprecisePartialLabelLearning):
    def __init__(self, args, tb_log=None, logger=None, **kwargs):
        self.init(args)
        super().__init__(args, tb_log, logger, **kwargs)

    def set_dataset(self):
        # get initial data
        train_data, train_targets, test_data, test_targets, extra_data = get_data(self.args.data_dir, self.args.dataset)
        
        # make partial labels
        train_data, train_partial_targets = get_partial_labels(train_data, train_targets, self.num_classes, self.args.partial_ratio)
        
        # initialize confidence
        self.confidence = torch.from_numpy(deepcopy(train_partial_targets))
        # self.confidence = self.confidence / self.confidence.sum(dim=-1, keepdim=True)
        
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
        train_dataset = ImgBaseDataset(self.args.dataset, train_data, train_partial_targets, 
                                                num_classes=self.num_classes, is_train=True,
                                                img_size=self.args.img_size, crop_ratio=self.args.crop_ratio,
                                                autoaug=autoaug, resize=resize,
                                                return_target=True, return_idx=True, return_keys=['x_idx', 'x', 'y'])
        test_dataset = ImgBaseDataset(self.args.dataset, test_data, test_targets, 
                                      num_classes=self.num_classes, is_train=False,
                                      img_size=self.args.img_size, crop_ratio=self.args.crop_ratio, resize=test_resize,
                                      return_keys=['x', 'y'])

        self.print_fn("Create datasets!")
        return {'train': train_dataset, 'eval': test_dataset}


    def train_step(self, x_idx, x, y):    
            
        logits = self.model(x)
        
        loss, updated_y = partial_loss(logits, self.confidence[x_idx.cpu()].to(x.device))
        
        with torch.no_grad():
            self.confidence[x_idx.cpu()] = updated_y.detach().cpu()

        out_dict = self.process_out_dict(loss=loss)
        log_dict = self.process_log_dict(loss=loss.item())
        return out_dict, log_dict