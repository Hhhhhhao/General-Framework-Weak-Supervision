

import torch 
import torch.nn.functional as F

from src.algorithms.sim_dsim_ulb import ImpreciseSimilarDisimilarUnlabeledLearning



def double_hinge_loss(logits, t):
    hinge_1 = F.relu(0.5 - 0.5 * t * logits)
    hinge_2, _ = torch.max(torch.cat([- t * logits.unsqueeze(1), hinge_1.unsqueeze(1)], dim=1), dim=1)
    return hinge_2.mean()


def sd_loss(logits, t, class_prior):
    loss_weight_0 = class_prior / (class_prior - (1 - class_prior))
    loss_weight_1 = (1 - class_prior) / (class_prior - (1 - class_prior))
    double_hinge_loss_0 = double_hinge_loss(logits, t)
    double_hinge_loss_1 = double_hinge_loss(logits, -t)
    loss = loss_weight_0 * double_hinge_loss_0 - loss_weight_1 * double_hinge_loss_1
    return loss 


def s_loss(logits, class_prior):
    loss_weight_0 = 1 / (class_prior - (1 - class_prior))
    loss_weight_1 = 1 / (class_prior - (1 - class_prior))
    double_hinge_loss_0 = double_hinge_loss(logits, 1.0)
    double_hinge_loss_1 = double_hinge_loss(logits, -1.0)
    loss = loss_weight_0 * double_hinge_loss_0 - loss_weight_1 * double_hinge_loss_1
    return loss 

def ds_loss(logits, class_prior):
    loss_weight_0 = 1 / (class_prior - (1 - class_prior))
    loss_weight_1 = 1 / (class_prior - (1 - class_prior))
    double_hinge_loss_0 = double_hinge_loss(logits, 1.0)
    double_hinge_loss_1 = double_hinge_loss(logits, -1.0)
    loss = loss_weight_0 * double_hinge_loss_0 - loss_weight_1 * double_hinge_loss_1
    return -loss 


def ulb_loss(logits, t, class_prior):
    pass
    


class RsikSDPairSimilarDisimilarUnlabeledLearning(ImpreciseSimilarDisimilarUnlabeledLearning):
    def __init__(self, args, tb_log=None, logger=None, **kwargs):
        super().__init__(args, tb_log, logger, **kwargs)
        
        self.sim_weight = self.class_prior ** 2 + (1 - self.class_prior) ** 2
        self.dsim_weight = 2 * self.class_prior * (1 - self.class_prior)
        
    def train_step(self, x_bag_lb_w, y_bag_lb, x_ulb_w):
        
        bag_lb_batch_size = x_bag_lb_w.shape[0]
        
        # get inputs
        x_ins_lb_w = x_bag_lb_w.view(-1, *x_bag_lb_w.shape[2:])
        inputs = torch.cat([x_ins_lb_w, x_ulb_w], dim=0)
        
        # forward model
        logits = self.model(inputs)
        logits_x_ins_lb_w = logits[:bag_lb_batch_size * 2]
        logits_x_ulb_w = logits[bag_lb_batch_size * 2:]
        logits_x_bag_w = logits_x_ins_lb_w.view(bag_lb_batch_size, 2, -1)
        
        # compute sd loss
        if self.lb_data == 'sim_dsim':
            sim_mask = (y_bag_lb == 1)
            dsim_mask = (y_bag_lb == 0)
            sim_loss = self.sim_weight * 0.5 * (sd_loss(logits_x_bag_w[sim_mask, 0, 1], 1.0, self.class_prior) + sd_loss(logits_x_bag_w[sim_mask, 1, 1], 1.0, self.class_prior))
            dsim_loss = self.dsim_weight * 0.5 * (sd_loss(logits_x_bag_w[dsim_mask, 0, 1], -1.0, self.class_prior) + sd_loss(logits_x_bag_w[dsim_mask, 1, 1], -1.0, self.class_prior))
            lb_loss = sim_loss + dsim_loss
            ulb_loss = sd_loss(logits_x_ulb_w[:, 1], -1.0, self.class_prior) + sd_loss(logits_x_ulb_w[:, 1], 1.0, self.class_prior)
        elif self.lb_data == 'sim':
            lb_loss = self.sim_weight * 0.5 * (s_loss(logits_x_bag_w[:, 0, 1], self.class_prior) + s_loss(logits_x_bag_w[:, 1, 1], self.class_prior))
            ulb_loss = sd_loss(logits_x_ulb_w[:, 1], -1.0, self.class_prior)
        elif self.lb_data == 'dsim':
            lb_loss = self.dsim_weight * 0.5 * (ds_loss(logits_x_bag_w[:, 0, 1], self.class_prior) + ds_loss(logits_x_bag_w[:, 1, 1], self.class_prior))
            ulb_loss = sd_loss(logits_x_ulb_w[:, 1], 1.0, self.class_prior)
        else:
            raise NotImplementedError

        # compute total loss
        total_loss = lb_loss + ulb_loss

        out_dict = self.process_out_dict(loss=total_loss)
        log_dict = self.process_log_dict(loss=total_loss.item(), lb_loss=lb_loss.item(), ulb_loss=ulb_loss.item())
        return out_dict, log_dict