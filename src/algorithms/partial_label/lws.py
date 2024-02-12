

import torch 
import torch.nn.functional as F
from copy import deepcopy

from src.core.utils import Argument
from src.datasets import get_partial_labels, get_data, ImgBaseDataset

from .imp_partial_label import ImprecisePartialLabelLearning



def lwc_loss(outputs, partialY, confidence, index, lw_weight, lw_weight0, epoch_ratio):
    device = outputs.device

    onezero = torch.zeros(outputs.shape[0], outputs.shape[1])
    onezero[partialY > 0] = 1
    counter_onezero = 1 - onezero
    onezero = onezero.to(device)
    counter_onezero = counter_onezero.to(device)

    sm_outputs = F.softmax(outputs, dim=1)

    sig_loss1 = - torch.log(sm_outputs + 1e-8)
    l1 = confidence[index, :].to(outputs.device) * onezero * sig_loss1
    average_loss1 = torch.sum(l1) / l1.size(0)

    sig_loss2 = - torch.log(1 - sm_outputs + 1e-8)
    l2 = confidence[index, :].to(outputs.device) * counter_onezero * sig_loss2
    average_loss2 = torch.sum(l2) / l2.size(0)

    average_loss = lw_weight0 * average_loss1 + lw_weight * average_loss2
    return average_loss, average_loss1, lw_weight * average_loss2


class LWSPartialLabelLearning(ImprecisePartialLabelLearning):
    def __init__(self, args, tb_log=None, logger=None, **kwargs):
        self.init(args)
        super().__init__(args, tb_log, logger, **kwargs)

    def set_dataset(self):
        # get initial data
        train_data, train_targets, test_data, test_targets, extra_data = get_data(self.args.data_dir, self.args.dataset)
        
        # make partial labels
        train_data, train_partial_targets = get_partial_labels(train_data, train_targets, self.num_classes, self.args.partial_ratio)
        
        # TODO: initialize confidence
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
        
        loss, _, _ = lwc_loss(logits, y.float(), self.confidence, x_idx.cpu(), self.args.lw, self.args.lw0, None)

        out_dict = self.process_out_dict(loss=loss)
        log_dict = self.process_log_dict(loss=loss.item())
        return out_dict, log_dict

    @staticmethod
    def get_argument():
        argument_list = ImprecisePartialLabelLearning.get_argument()
        argument_list.extend([
            Argument('-lw', default=0, type=float, help='lw sigmoid loss weight',),
            Argument('-lw0', default=1, type=float, help='lw of first term',),
        ])
        return argument_list