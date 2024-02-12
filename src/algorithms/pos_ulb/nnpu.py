

import torch 

from src.algorithms.pos_ulb.upu import UnbiasedPositiveUnLabeledLearning
from src.datasets import ImgBaseDataset


class NonNegativePositiveUnlabeledLearning(UnbiasedPositiveUnLabeledLearning):
    def __init__(self, args, tb_log=None, logger=None, **kwargs):
        super().__init__(args, tb_log, logger, **kwargs)


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
        if torch.gt((loss_ulb - self.class_prior * loss_pos_neg ), 0):
            total_loss = self.class_prior * (loss_pos - loss_pos_neg) + loss_ulb
        else: 
            total_loss = self.class_prior  * loss_pos_neg - loss_ulb
        
        out_dict = self.process_out_dict(loss=total_loss)
        log_dict = self.process_log_dict(loss=total_loss.item(),  loss_pos=loss_pos.item(), loss_pos_neg=loss_pos_neg.item(), loss_ulb=loss_ulb.item())
        return out_dict, log_dict