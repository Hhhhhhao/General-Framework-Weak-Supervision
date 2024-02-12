

import torch 
import torch.nn.functional as F
from torch.nn.utils.rnn import unpad_sequence, pad_sequence

from src.algorithms.multi_ins_label.imp_multi_ins import ImpreciseMultipleInstanceLearning


def uum_multi_ins_loss(probs_w, probs_s, targets):

    p_z0 = probs_s[:, 0].unsqueeze(0)
    p_z1 = probs_s[:, 1].unsqueeze(0)
    lossz0 = torch.log(p_z0 + 1e-32)
    lossz1 = torch.log(p_z1 + 1e-32)
    
    with torch.no_grad():
        probs_w = probs_w.detach()
        p_z0 = probs_w[:, 0].unsqueeze(0)
        p_z1 = probs_w[:, 1].unsqueeze(0)
    
        log_p_y0 = torch.sum(torch.log(p_z0), dim=1, keepdim=True)
        p_y0 = torch.exp(log_p_y0)
        p_y1 = 1. - p_y0
        
        weight1 = p_z1
        weight0 = (1 - p_y0 /(p_z0 + 1e-32)) * p_z0

    lossc0 = weight0 * lossz0 / (p_y1 + 1e-32)
    lossc1 = weight1 * lossz1 / (p_y1 + 1e-32)
    
    loss1 = lossc0.sum(dim=-1, keepdim=True) + lossc1.sum(dim=-1, keepdim=True)
    loss0 = lossz0.sum(dim=-1, keepdim=True) 
    log_pred = torch.cat((loss0,loss1),dim=-1)
    
    return F.nll_loss(log_pred, targets)    


def uum_multi_ins_loss_batch(probs_w, probs_s, lengths, targets):
    lengths = lengths.to(probs_s.device)
    
    batch_size, max_seq_length, _ = probs_s.shape

    # create a mask based on lengths
    mask = torch.arange(max_seq_length, device=lengths.device)[None, :] < lengths[:, None]
    
    # compute loss for z0 and z1
    p_z0 = probs_s[:, :, 0]
    p_z1 = probs_s[:, :, 1]
    lossz0 = torch.log(p_z0 + 1e-32) * mask
    lossz1 = torch.log(p_z1 + 1e-32) * mask

    with torch.no_grad():
        probs_w = probs_w.detach()
        p_z0_w = probs_w[:, :, 0]
        p_z1_w = probs_w[:, :, 1]

        log_p_y0 = torch.sum(torch.log(p_z0_w + 1e-32) * mask, dim=1, keepdim=True)
        p_y0 = torch.exp(log_p_y0)
        p_y1 = 1. - p_y0

        weight1 = p_z1_w
        weight0 = (1 - p_y0 /(p_z0_w + 1e-32)) * p_z0_w

    lossc0 = weight0 * lossz0 / (p_y1 + 1e-32)
    lossc1 = weight1 * lossz1 / (p_y1 + 1e-32)

    # Apply mask and sum over sequence length
    loss1 = lossc0.sum(dim=1, keepdim=True) + lossc1.sum(dim=1, keepdim=True)
    loss0 = lossz0.sum(dim=1, keepdim=True)
    
    log_pred = torch.cat((loss0, loss1), dim=-1)

    # Reshape targets to match the log_pred shape
    targets = targets.view(-1).to(torch.long)

    return F.nll_loss(log_pred, targets)


class UUMMultipleInstanceLearning(ImpreciseMultipleInstanceLearning):
    def __init__(self, args, tb_log=None, logger=None, **kwargs):
        super().__init__(args, tb_log, logger, **kwargs)

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
        
        loss = 0.0
        for target_class in self.target_classes:
            cls_idx = self.class_map[target_class]
            binary_y_bag = y_bag[:, cls_idx]
            
            # construct binary_probs_x_w
            neg_probs_x_w = torch.cat([probs_x_w[:, idx].unsqueeze(1) for idx in range(probs_x_w.shape[1]) if idx != cls_idx], dim=1)
            neg_probs_x_w = torch.sum(neg_probs_x_w, dim=1, keepdim=True)
            pos_probs_x_w =  probs_x_w[:, cls_idx].unsqueeze(1)
            binary_probs_x_w =  torch.cat([neg_probs_x_w, pos_probs_x_w], dim=1)
            
            # construct binary_probs_x_s
            neg_probs_x_s = torch.cat([probs_x_s[:, idx].unsqueeze(1) for idx in range(probs_x_s.shape[1]) if idx != cls_idx], dim=1)
            neg_probs_x_s = torch.sum(neg_probs_x_s, dim=1, keepdim=True)
            pos_probs_x_s =  probs_x_s[:, cls_idx].unsqueeze(1)
            binary_probs_x_s =  torch.cat([neg_probs_x_s, pos_probs_x_s], dim=1)
            
            batch_loss = 0.0
            for i in range(batch_size):
                batch_loss += uum_multi_ins_loss(probs_w=binary_probs_x_w[i].unsqueeze(0), probs_s=binary_probs_x_s[i].unsqueeze(0), targets=binary_y_bag[i].unsqueeze(0).to(torch.long))
            batch_loss /= batch_size
            loss += batch_loss
        loss /= len(self.target_classes)
        
        out_dict = self.process_out_dict(loss=loss)
        log_dict = self.process_log_dict(loss=loss.item())
        return out_dict, log_dict