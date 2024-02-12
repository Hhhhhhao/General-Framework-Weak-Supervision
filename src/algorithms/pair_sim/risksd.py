

import torch 
import torch.nn.functional as F

from src.algorithms.pair_sim import ImprecisePairSimilarityLearning



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
    


class RsikSDPairSimilarityLearning(ImprecisePairSimilarityLearning):
    def __init__(self, args, tb_log=None, logger=None, **kwargs):
        super().__init__(args, tb_log, logger, **kwargs)
        
        self.sim_weight = self.class_prior ** 2 + (1 - self.class_prior) ** 2
        self.dsim_weight = 2 * self.class_prior * (1 - self.class_prior)
        
    def train_step(self, x_bag_w, y_bag):
        
        bag_batch_size = x_bag_w.shape[0]
        
        x_ins_w = x_bag_w.view(-1, *x_bag_w.shape[2:])
        
        # forward model
        logits_x_ins_w = self.model(x_ins_w)
        logits_x_bag_w = logits_x_ins_w.view(bag_batch_size, 2, -1)
        
        # compute loss
        sim_mask = (y_bag == 1)
        dsim_mask = (y_bag == 0)

        sim_loss = self.sim_weight * 0.5 * (sd_loss(logits_x_bag_w[sim_mask, 0, 1], 1.0, self.class_prior) + sd_loss(logits_x_bag_w[sim_mask, 1, 1], 1.0, self.class_prior))
        dsim_loss = self.dsim_weight * 0.5 * (sd_loss(logits_x_bag_w[dsim_mask, 0, 1], -1.0, self.class_prior) + sd_loss(logits_x_bag_w[dsim_mask, 1, 1], -1.0, self.class_prior))

        # total loss
        loss = sim_loss + dsim_loss

        out_dict = self.process_out_dict(loss=loss)
        log_dict = self.process_log_dict(loss=loss.item())
        return out_dict, log_dict