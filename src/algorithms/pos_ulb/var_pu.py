
import math
import torch 
import numpy as np

from src.core.utils import Argument, get_optimizer, str2bool
from src.algorithms.pos_ulb.upu import UnbiasedPositiveUnLabeledLearning


class VariationalPositiveUnlabeledLearning(UnbiasedPositiveUnLabeledLearning):
    def __init__(self, args, tb_log=None, logger=None, **kwargs):
        super().__init__(args, tb_log, logger, **kwargs)

    
    def init(self, args):
        super().init(args)
        self.mixup_alpha = args.mixup_alpha
        self.loss_weight_mixup = args.loss_weight_mixup


    def train_step(self, x_lb, y_lb, x_ulb):

        num_lb = y_lb.shape[0]
        
        # forward model
        inputs = torch.cat((x_lb, x_ulb))
        outputs = self.model(inputs)
        logits_x_lb = outputs[:num_lb]
        logits_x_ulb = outputs[num_lb:]
        
        # get probability
        log_probs_x_lb = logits_x_lb.log_softmax(dim=1)[:, 1]
        log_probs_x_ulb = logits_x_ulb.log_softmax(dim=1)[:, 1]
        
        # compute varitional loss
        var_loss = torch.logsumexp(log_probs_x_ulb, dim=0).mean() - math.log(len(log_probs_x_ulb)) - torch.mean(log_probs_x_lb)
        
        # mixup regularization
        with torch.no_grad():
            target_x_ulb = logits_x_ulb.softmax(dim=1)[:, 1]
            target_x_lb = y_lb
            rand_perm = torch.randperm(x_lb.size(0))
            x_lb_perm, target_x_lb_perm = x_lb[rand_perm], target_x_lb[rand_perm]
            m = torch.distributions.beta.Beta(self.mixup_alpha, self.mixup_alpha)
            lam = m.sample()
            mixed_x = lam * x_ulb + (1 - lam) * x_lb_perm
            mixed_y = lam * target_x_ulb + (1 - lam) * target_x_lb_perm
        mixed_logits = self.model(mixed_x)
        reg_mix_loss = ((torch.log(mixed_y) - mixed_logits[:, 1]) ** 2).mean()
        
        # calculate total loss
        total_loss = var_loss + self.loss_weight_mixup * reg_mix_loss
        
        out_dict = self.process_out_dict(loss=total_loss)
        log_dict = self.process_log_dict(loss=total_loss.item(),  var_loss=var_loss.item(), reg_mix_loss=reg_mix_loss.item())
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
            Argument('--mixup_alpha', float, 0.3, 'mixup alpha'),
            Argument('--loss_weight_mixup', float, 0.03, 'mixup loss weight'),
        ]