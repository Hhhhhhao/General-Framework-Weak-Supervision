

import torch 
import torch.nn.functional as F

from src.algorithms.pair_sim import ImprecisePairSimilarityLearning


def uum_pair_sim_loss(probs_w, probs_s, y):    
    k = probs_w.shape[-1]
    p_z1_s = probs_s[:, 0] # activation(t1)
    p_z2_s = probs_s[:, 1] # activation(t2)
    
    lossz1 = torch.log(p_z1_s + 1e-32)
    lossz2 = torch.log(p_z2_s + 1e-32)
    
    p_z1 = probs_w[:, 0].detach() # activation(t1)
    p_z2 = probs_w[:, 1].detach()
    
    p_y1 = (p_z1 * p_z2).sum(dim=1)
    p_y0 = 1. - p_y1
    weight1 = (p_z1 * p_z2)/(p_y1.unsqueeze(-1)+1e-32)
    weight01 = (p_z1 * (1-p_z2))/(p_y0.unsqueeze(-1)+1e-32)
    weight02 = (p_z2 * (1-p_z1))/(p_y0.unsqueeze(-1)+1e-32)
    
    loss11 = (weight1 * lossz1).sum(dim=-1)
    loss12 = (weight1 * lossz2).sum(dim=-1)
    loss01 = (weight01 * lossz1).sum(dim=-1)
    loss02 = (weight02 * lossz2).sum(dim=-1)
    
    loss1 = torch.stack((loss01,loss11),dim=-1)
    loss2 = torch.stack((loss02,loss12),dim=-1)
    
    loss1 = F.nll_loss(loss1,y)
    loss2 = F.nll_loss(loss2,y)
    
    return (loss1+loss2)/2


class UUMPairSimilarityLearning(ImprecisePairSimilarityLearning):
    def __init__(self, args, tb_log=None, logger=None, **kwargs):
        super().__init__(args, tb_log, logger, **kwargs)

    def train_step(self, x_bag_w, x_bag_s, y_bag):
        
        bag_batch_size = x_bag_w.shape[0]
        
        x_ins_w = x_bag_w.view(-1, *x_bag_w.shape[2:])
        x_ins_s = x_bag_s.view(-1, *x_bag_s.shape[2:])
        
        # forward model
        inputs = torch.cat((x_ins_w, x_ins_s))
        outputs = self.model(inputs)
        logits_x_ins_w, logits_x_ins_s = outputs.chunk(2)
        logits_x_bag_w = logits_x_ins_w.view(bag_batch_size, 2, -1)
        logits_x_bag_s = logits_x_ins_s.view(bag_batch_size, 2, -1)
        
        # get probs
        probs_x_bag_w = F.softmax(logits_x_bag_w, dim=-1).detach()
        probs_x_bag_s = F.softmax(logits_x_bag_s, dim=-1)
        
        # compute loss
        loss = uum_pair_sim_loss(probs_x_bag_w, probs_x_bag_s, y_bag.to(torch.long))

        out_dict = self.process_out_dict(loss=loss)
        log_dict = self.process_log_dict(loss=loss.item())
        return out_dict, log_dict