

import contextlib
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import unpad_sequence, pad_sequence

from src.core.utils import Argument
from src.algorithms.proportion_label.imp_proportion_label import ImpreciseProportionLabelLearning


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


class VATLoss(nn.Module):
    def __init__(self, xi=10.0, eps=1.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x, x_):
        with torch.no_grad():
            pred = F.softmax(model(x), dim=1)

        # prepare random unit tensor
        # d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = torch.randn_like(x)
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                pred_hat = model(x_ + self.xi * d)
                logp_hat = F.log_softmax(pred_hat, dim=1)
                adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()

            # calc LDS
            r_adv = d * self.eps
            pred_hat = model(x_ + r_adv)
            logp_hat = F.log_softmax(pred_hat, dim=1)
            lds = F.kl_div(logp_hat, pred, reduction='batchmean')

        return lds

def cross_entropy_loss(input, target, eps=1e-8):
    input = torch.clamp(input, eps, 1 - eps)
    loss = -target * torch.log(input)
    return loss

class ProportionLoss(nn.Module):
    def __init__(self, metric, alpha, eps=1e-8):
        super(ProportionLoss, self).__init__()
        self.metric = metric
        self.eps = eps
        self.alpha = alpha
        # self.num_classes = num_classes
        

    def forward(self, input, target):
        assert input.shape == target.shape
        
        # if input.shape[-1] < self.num_classes:
        #     input = input[1:]
        #     target = target[1:]

        if self.metric == "ce":
            loss = cross_entropy_loss(input, target, eps=self.eps)
        elif self.metric == "l1":
            loss = F.l1_loss(input, target, reduction="none")
        elif self.metric == "mse":
            loss = F.mse_loss(input, target, reduction="none")
        else:
            raise NameError("metric {} is not supported".format(self.metric))

        loss = torch.sum(loss, dim=-1).mean()
        return self.alpha * loss



class LLPVATProportionLabelLearning(ImpreciseProportionLabelLearning):
    def __init__(self, args, tb_log=None, logger=None, **kwargs):
        super().__init__(args, tb_log, logger, **kwargs)

    def init(self, args):
        super().init(args)
        self.vat_loss = VATLoss(args.vat_xi, args.vat_eps, args.vat_ip)
        self.prop_loss = ProportionLoss(args.prop_metric, 1.0)
        self.con_loss_weight = args.loss_weight_cons


    def train_step(self, x_bag_w, x_bag_s, x_bag_len, y_bag):      
        batch_size = x_bag_w.shape[0]
        x_bag_len = x_bag_len.cpu()
        
        # get unpadded data
        unpad_x_bag_w = unpad_sequence(x_bag_w, x_bag_len, batch_first=True)
        unpad_x_bag_s = unpad_sequence(x_bag_s, x_bag_len, batch_first=True)
        
        # get output
        inputs = torch.cat(unpad_x_bag_w + unpad_x_bag_s, dim=0)
        outputs = self.model(inputs)
        logits_x_w, logits_x_s = outputs.chunk(2)

        # get softmax
        probs_x_w = logits_x_w.softmax(dim=-1)
        probs_x_s = logits_x_s.softmax(dim=-1)
        
        # split to list
        probs_x_w_list = probs_x_w.split(x_bag_len.tolist())
        probs_x_s_list = probs_x_s.split(x_bag_len.tolist())
        
        
        # compute consistency loss
        consistency_loss = self.vat_loss(self.model, torch.cat(unpad_x_bag_w, dim=0), torch.cat(unpad_x_bag_s, dim=0))
        
        # compute proportion loss
        prop_loss = 0.0
        for probs, targets, length in zip(probs_x_s_list, y_bag, x_bag_len):
            avg_probs = torch.mean(probs, dim=0)
            targets = targets / length
            loss = self.prop_loss(avg_probs, targets)
            prop_loss += loss
        prop_loss /= batch_size
        
        # total loss
        total_loss = self.con_loss_weight * consistency_loss + prop_loss
        
        out_dict = self.process_out_dict(loss=total_loss)
        log_dict = self.process_log_dict(loss=total_loss.item(), con_loss=consistency_loss.item(), prop_loss=prop_loss.item())
        return out_dict, log_dict
        

    @staticmethod
    def get_arguments():
        argument_list = ImpreciseProportionLabelLearning.get_arguments()
        argument_list.extend([
            Argument('--vat_xi', float, 10.0, 'hyperparameter of VAT (default: 10.0)'),
            Argument('--vat_eps', float, 1.0, 'hyperparameter of VAT (default: 1.0)'),
            Argument('--vat_ip', int, 1, 'iteration times of computing adv noise (default: 1)'),
            Argument('--prop_metric', str, 'mse', 'metric function for proportion loss'),
            Argument('--loss_weight_cons', float, 1.0, 'loss weight for consistency loss'),
            
        ])