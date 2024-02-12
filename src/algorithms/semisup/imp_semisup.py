

import torch 
import numpy as np

from src.core.algorithmbase import AlgorithmBase
from src.core.utils import Argument, str2bool
from src.core.criterions import CELoss
from src.datasets import get_semisup_labels, get_data, get_dataloader, ImgBaseDataset, ImgTwoViewBaseDataset


class ImpreciseSemiSupervisedLearning(AlgorithmBase):
    def __init__(self, args, tb_log=None, logger=None, **kwargs):
        self.init(args)
        super().__init__(args, tb_log, logger, **kwargs)
        self.ce_loss = CELoss()
    
    def init(self, args):
        # extra arguments
        self.average_entropy_loss = args.average_entropy_loss
        self.num_labels = args.num_labels
    
        # initialize distribution alignment 
        self.ema_p = 0.999
        self.p_hat = torch.ones((args.num_classes, )) / args.num_classes
        self.p_hat = self.p_hat.cuda()
    
    def set_dataset(self):
        # get initial data
        train_data, train_targets, test_data, test_targets, extra_data = get_data(self.args.data_dir, self.args.dataset)

        # make labeled and unlabeled data
        lb_index, lb_train_data, lb_train_targets, ulb_index, ulb_train_data, ulb_train_targets \
            = get_semisup_labels(train_data, train_targets, self.num_classes, self.args.num_labels, self.args.include_lb_to_ulb)
        self.print_fn("labeled data: {}, unlabeled data {}".format(len(lb_index), len(ulb_index)))
        
        # determine the resize methods
        if self.args.dataset in ['cifar10', 'cifar100']:
            resize = 'resize_crop_pad'
        elif self.args.dataset in ['stl10', 'svhn']:
            resize = 'resize_crop'
        test_resize = 'resize'
        
        # make dataset
        train_lb_dataset = ImgBaseDataset(self.args.dataset, lb_train_data, lb_train_targets, 
                                          num_classes=self.num_classes, is_train=True,
                                          img_size=self.args.img_size, crop_ratio=self.args.crop_ratio,
                                          autoaug=None, resize=resize,
                                          return_keys=['x_lb', 'y_lb'])
        if extra_data is not None:
            ulb_train_data = np.concatenate([ulb_train_data, extra_data], axis=0)
            if ulb_train_targets.ndim == 1:
                ulb_train_targets = np.concatenate([ulb_train_targets, -1 * np.ones((extra_data.shape[0],))], axis=0)
            else:
                ulb_train_targets = np.concatenate([ulb_train_targets, -1 * np.ones((extra_data.shape[0], ulb_train_targets.shape[1]))], axis=0)
        train_ulb_dataset = ImgTwoViewBaseDataset(self.args.dataset, ulb_train_data, ulb_train_targets,
                                                  num_classes=self.num_classes, is_train=True,
                                                  img_size=self.args.img_size, crop_ratio=self.args.crop_ratio,
                                                  autoaug='randaug', resize=resize,
                                                  return_target=False, return_keys=['x_ulb_w', 'x_ulb_s'])
        test_dataset = ImgBaseDataset(self.args.dataset, test_data, test_targets, 
                                      num_classes=self.num_classes, is_train=False,
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
        inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
        outputs = self.model(inputs)
        logits_x_lb = outputs[:num_lb]
        logits_x_ulb_w, logits_x_ulb_s = outputs[num_lb:].chunk(2)

        # compute supervised loss 
        sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
           
        # compute forward-backward on graph
        with torch.no_grad():
            probs_x_ulb_w = logits_x_ulb_w.detach().softmax(dim=-1)

            # distribution alignment
            self.p_hat = self.ema_p * self.p_hat + (1 - self.ema_p) * probs_x_ulb_w.mean(dim=0)
            probs_x_ulb_w = probs_x_ulb_w / self.p_hat
            probs_x_ulb_w = (probs_x_ulb_w / probs_x_ulb_w.sum(dim=-1, keepdim=True))
            
            values, indices = torch.topk(probs_x_ulb_w, k=3, dim=1)
            # Create a zero tensor of the same shape
            pseudo_y_ulb = torch.zeros_like(probs_x_ulb_w)
            # Place the top k values in the result tensor
            pseudo_y_ulb.scatter_(1, indices, values)
            pseudo_y_ulb = pseudo_y_ulb / pseudo_y_ulb.sum(dim=1, keepdim=True)

        # compute unsupervised loss 
        unsup_loss = self.ce_loss(logits_x_ulb_s, pseudo_y_ulb, reduction='mean')
        
        # total loss
        total_loss = sup_loss + unsup_loss 
        
        # computer average entropy loss
        if self.average_entropy_loss:
            avg_prediction = torch.mean(logits_x_ulb_w.softmax(dim=-1), dim=0)
            prior_distr = 1.0/self.num_classes * torch.ones_like(avg_prediction)
            avg_prediction = torch.clamp(avg_prediction, min = 1e-6, max = 1.0)
            balance_kl =  torch.mean(-(prior_distr * torch.log(avg_prediction)).sum(dim=0))
            entropy_loss = 0.1 * balance_kl
            total_loss += entropy_loss


        out_dict = self.process_out_dict(loss=total_loss)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         unsup_loss=unsup_loss.item(), 
                                         loss=total_loss.item())
        
        return out_dict, log_dict

    def get_save_dict(self):
        save_dict = super().get_save_dict()
        # additional saving arguments
        save_dict['p_hat'] = self.p_hat.cpu()
        return save_dict


    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.p_hat = checkpoint['p_hat'].cuda(self.args.gpu)
        self.print_fn("additional parameter loaded")
        return checkpoint


    @staticmethod
    def get_argument():
        return [
            Argument('--average_entropy_loss', str2bool, True, 'use entropy loss'),
            Argument('--num_labels', int, 40, 'number of labels used in semi-supervised learning'),
            Argument('--uratio', int, 7, 'ratio of unlabeled batch size to labeled batch size'),
            Argument('--include_lb_to_ulb', str2bool, True, 'flag of adding labeled data into unlabeled data'),
        ]