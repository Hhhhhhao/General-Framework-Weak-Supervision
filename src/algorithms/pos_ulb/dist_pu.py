
import math
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.core.utils import Argument, get_optimizer, str2bool
from src.core.hooks import Hook
from src.algorithms.pos_ulb.upu import UnbiasedPositiveUnLabeledLearning


class WarmupEndHookl(Hook):
    
    def before_run(self, algorithm):
        # set warmup optimizer, scheduler
        algorithm.optimizer = get_optimizer(algorithm.model, algorithm.args.optim, algorithm.args.warmup_lr, algorithm.args.momentum, algorithm.args.warmup_weight_decay, algorithm.args.layer_decay, nesterov=False, bn_wd_skip=False)
        algorithm.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(algorithm.optimizer , T_max=int(algorithm.iter_per_epoch *  algorithm.warmup_epochs), eta_min=1e-6)
        
    def after_train_epoch(self, algorithm):
        if algorithm.epoch >= algorithm.warmup_epochs:
            algorithm.optimizer = get_optimizer(algorithm.model, algorithm.args.optim, algorithm.args.lr, algorithm.args.momentum, algorithm.args.weight_decay, algorithm.args.layer_decay, nesterov=False, bn_wd_skip=False)
            algorithm.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(algorithm.optimizer , T_max=int(algorithm.iter_per_epoch *  (algorithm.epochs - algorithm.warmup_epochs)), eta_min=1e-6)
            print('warmup end!')


# adapted from https://github.com/valerystrizh/pytorch-histogram-loss
class LabelDistributionLoss(nn.Module):
    def __init__(self, prior, device, num_bins=1, proxy='polar', dist='L1'):
        super(LabelDistributionLoss, self).__init__()
        self.prior = prior
        self.frac_prior = 1.0 / (2*prior)

        self.step = 1 / num_bins # bin width. predicted scores in [0, 1]. 
        self.device = device
        self.t = torch.arange(0, 1+self.step, self.step).view(1, -1).requires_grad_(False) # [0, 1+bin width)
        self.t_size = num_bins + 1

        self.dist = None
        if dist == 'L1':
            self.dist = F.l1_loss
        else:
            raise NotImplementedError("The distance: {} is not defined!".format(dist))

        # proxy
        proxy_p, proxy_n = None, None
        if proxy == 'polar':
            proxy_p = np.zeros(self.t_size, dtype=float)
            proxy_n = np.zeros_like(proxy_p)
            proxy_p[-1] = 1
            proxy_n[0] = 1
        else:
            raise NotImplementedError("The proxy: {} is not defined!".format(proxy))
        
        proxy_mix = prior*proxy_p + (1-prior)*proxy_n
        print('#### Label Distribution Loss ####')
        print('ground truth P:')
        print(proxy_p)
        print('ground truth U:')
        print(proxy_mix)

        # to torch tensor
        self.proxy_p = torch.from_numpy(proxy_p).requires_grad_(False).float()
        self.proxy_mix = torch.from_numpy(proxy_mix).requires_grad_(False).float()

        # to device
        self.t = self.t.to(self.device)
        self.proxy_p = self.proxy_p.to(self.device)
        self.proxy_mix = self.proxy_mix.to(self.device)

    def histogram(self, scores):
        scores_rep = scores.repeat(1, self.t_size)
        
        hist = torch.abs(scores_rep - self.t)

        inds = (hist>self.step)
        hist = self.step-hist # switch values
        hist[inds] = 0

        return hist.sum(dim=0)/(len(scores)*self.step)

    def forward(self, scores_p, scores_n):
        # scores=torch.sigmoid(outputs)
        # labels=labels.view(-1,1)

        s_p = scores_p.view(-1,1)
        s_u = scores_n.view(-1,1)

        l_p = 0
        l_u = 0
        if s_p.numel() > 0:
            hist_p = self.histogram(s_p)
            l_p = self.dist(hist_p, self.proxy_p, reduction='mean')
        if s_u.numel() > 0:
            hist_u = self.histogram(s_u)
            l_u = self.dist(hist_u, self.proxy_mix, reduction='mean')

        return l_p + self.frac_prior * l_u


@torch.no_grad()
def mixup_two_target(x, y, alpha=1.0, is_bias=False):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    if is_bias:
        lam = max(lam, 1 - lam)

    index = torch.randperm(x.size(0)).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


@torch.no_grad()
def mixup_one_target(x, y, alpha=1.0, is_bias=False):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    if is_bias:
        lam = max(lam, 1 - lam)

    index = torch.randperm(x.size(0)).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y, lam


class LabelDistPositiveUnlabeledLearning(UnbiasedPositiveUnLabeledLearning):
    def __init__(self, args, tb_log=None, logger=None, **kwargs):
        super().__init__(args, tb_log, logger, **kwargs)
        self.label_dist_loss = LabelDistributionLoss(prior=self.class_prior, device=torch.device(f'cuda:{self.gpu}'))
        self.iter_per_epoch = self.num_train_iter // self.epochs
    
    def init(self, args):
        super().init(args)
        self.mixup_alpha = args.mixup_alpha
        self.loss_weight_ent = args.loss_weight_ent
        self.loss_weight_mixup_ent = args.loss_weight_mixup_ent
        self.loss_weight_mixup = args.loss_weight_mixup
        self.warmup_epochs = args.warmup_epoch
        self.warmup_lr = args.warmup_lr
        self.warmup_weight_decay = args.warmup_weight_decay
    

    def train_step(self, x_lb, y_lb, x_ulb):

        num_lb = y_lb.shape[0]
        
        # forward model
        inputs = torch.cat((x_lb, x_ulb))
        outputs = self.model(inputs)
        logits_x_lb = outputs[:num_lb]
        logits_x_ulb = outputs[num_lb:]
        
        # get probs
        probs_x_lb = logits_x_lb.softmax(dim=1)
        probs_x_ulb = logits_x_ulb.softmax(dim=1)
        
        # calculate label dist loss
        label_dist_loss = self.label_dist_loss(probs_x_lb[:, 1], probs_x_ulb[:, 0])
        total_loss = label_dist_loss
        
        # calculate entropy minimization loss
        if self.epoch >= self.warmup_epochs:
            entropy_loss = - (probs_x_ulb * torch.log(probs_x_ulb)).sum(dim=1).mean()

            # get pseudo labels for mixup
            with torch.no_grad():
                pseudo_labels = probs_x_ulb.argmax(dim=1)
                x, y = torch.cat((x_lb, x_ulb), dim=0), torch.cat((y_lb, pseudo_labels), dim=0)
                mixed_x, mixed_y, lam = mixup_one_target(x, y, self.mixup_alpha)
            
            mixed_logits = self.model(mixed_x)
            mixup_loss = self.ce_loss(mixed_logits, mixed_y.to(torch.long), reduction='mean')
            
            # calculate mixup entropy loss
            mixed_probs = mixed_logits.softmax(dim=1)
            mixup_entropy_loss = - (mixed_probs * torch.log(mixed_probs)).sum(dim=1).mean()
            
            loss_weight_ent = (1-math.cos((float(self.epoch - self.warmup_epochs)/ (self.epochs - self.warmup_epochs) ) * (math.pi/2))) * self.loss_weight_ent
            total_loss +=  self.loss_weight_mixup * mixup_loss + loss_weight_ent * entropy_loss + self.loss_weight_mixup_ent * mixup_entropy_loss
            
            log_dict = self.process_log_dict(loss=total_loss.item(),  entropy_loss=entropy_loss.item(), mixup_loss=mixup_loss.item(), mixup_entropy_loss=mixup_entropy_loss.item())
        else:
            log_dict = self.process_log_dict(loss=total_loss.item(), )
        
        out_dict = self.process_out_dict(loss=total_loss)
        return out_dict, log_dict

    @staticmethod
    def get_argument():
        return [
            Argument('--target_classes', int, 9, 'positive class for positive unlabeled learning', nargs='+'),
            Argument('--neg_classes', int, None, 'negative class for positive unlabeled learning', nargs='+'),
            Argument('--num_pos_data', int, 1000, 'number of labeled positive samples'),
            Argument('--num_ulb_data', int, 1000, 'number of unlabeled samples'),
            Argument('--uratio', int, 1, 'ratio of unlabeled batch size to labeled batch size'),
            Argument('--include_lb_to_ulb', str2bool, True, 'flag of adding labeled data into unlabeled data'),
            Argument('--mixup_alpha', float, 6.0, 'mixup alpha'),
            Argument('--loss_weight_ent', float, 0.004, 'entropy loss weight'),
            Argument('--loss_weight_mixup', float, 5.0, 'mixup loss weight'),
            Argument('--loss_weight_mixup_ent', float, 0.04, 'mixup entropy loss weight'),
            Argument('--warmup_epoch', int, 60, 'warmup epoch'),
            Argument('--warmup_lr', float, 5e-4, 'warmup lr'),
            Argument('--warmup_weight_decay', float, 5e-3, 'warmup weight decay')
        ]