

import torch 

from src.algorithms.ulb_ulb.imp_ulb_ulb import ImpreciseUnlabeledUnlabeledLearning


class UULearnUnlabeledUnlabeledLearning(ImpreciseUnlabeledUnlabeledLearning):
    def __init__(self, args, tb_log=None, logger=None, **kwargs):
        super().__init__(args, tb_log, logger, **kwargs)

    def train_step(self, x_ulb1_w, x_ulb2_w):
        inputs = torch.cat((x_ulb1_w, x_ulb2_w))
        outputs = self.model(inputs)
        logits_x_ulb1, logits_x_ulb2 = outputs.chunk(2)
        
        # get probs
        probs_x_ulb1 = logits_x_ulb1.softmax(dim=1)
        probs_x_ulb2 = logits_x_ulb2.softmax(dim=1)

        # compute loss
        weight1 = self.ulb_prior * (1. - self.cls_prior_ulb2) / (self.cls_prior_ulb1 - self.cls_prior_ulb2)
        weight2 = self.cls_prior_ulb2 * (1. - self.ulb_prior) / (self.cls_prior_ulb1 - self.cls_prior_ulb2)
        weight3 = self.ulb_prior * (1. - self.cls_prior_ulb1) / (self.cls_prior_ulb1 - self.cls_prior_ulb2)
        weight4 = self.cls_prior_ulb1 * (1. - self.ulb_prior) / (self.cls_prior_ulb1 - self.cls_prior_ulb2)
        pos_loss = torch.mean((weight1 * probs_x_ulb1[:, 0] - weight2 * probs_x_ulb1[:, 1]))
        neg_loss = torch.mean((-weight3 * probs_x_ulb2[:, 0] + weight4 * probs_x_ulb2[:, 1]))
        total_loss = pos_loss + neg_loss
        
        
        out_dict = self.process_out_dict(loss=total_loss)
        log_dict = self.process_log_dict(loss=total_loss.item(),  
                                         pos_loss=pos_loss.item(), neg_loss=neg_loss.item())
        return out_dict, log_dict