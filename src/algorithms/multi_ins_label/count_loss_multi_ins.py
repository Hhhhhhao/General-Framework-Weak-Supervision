

import torch 
from torch.nn.utils.rnn import unpad_sequence, pad_sequence

from src.algorithms.multi_ins_label.imp_multi_ins import ImpreciseMultipleInstanceLearning


def dp_count_multi_ins(log_probs, lengths):
    lengths = lengths.to(log_probs.device)
    batch_size, max_seq_length, _ = log_probs.shape
    
    # dynamic programming initialization
    # where the column is the number of positive instances and row is the idx of samples
    log_alpha = torch.full((batch_size, max_seq_length + 1, max_seq_length + 1), -float("Inf"),  device=log_probs.device)
    log_alpha[:, 0, 0] = 0
    
    for i in range(1, max_seq_length + 1):
        for j in range(max_seq_length + 1):
            valid_mask = (i <= lengths)
            
            alpha_plus_zero = log_alpha[:, i - 1, j] + log_probs[:, i - 1, 0]
            alpha_plus_one = log_alpha[:, i - 1, j - 1] + log_probs[:, i - 1, 1]

            # Mask to check if alpha_plus_zero and alpha_plus_one are -inf
            alpha_plus_zero[alpha_plus_zero == -float("Inf")] = -1e10
            alpha_plus_one[alpha_plus_one == -float("Inf")] = -1e10
            
            log_alpha[:, i, j] = torch.where(
                valid_mask,
                torch.logsumexp(torch.stack([alpha_plus_zero, alpha_plus_one], dim=-1), dim=-1),
                log_alpha[:, i - 1, j]
            )
    
    # print("log_alpha", log_alpha)
    
    m = torch.logsumexp(log_alpha[torch.arange(batch_size), lengths, :], dim=1)
    alpha = torch.exp(log_alpha[torch.arange(batch_size), lengths, :] - m.unsqueeze(-1))
    # print("alpha", alpha)
    
    sup_preds = []
    for i in range(batch_size):
        single_alpha = alpha[i, :lengths[i] + 1]
        normalized_alpha = single_alpha / (single_alpha.sum() + 1e-12)
        count_prob = torch.cat([normalized_alpha[:1], normalized_alpha[1:].sum().unsqueeze(0)])
        sup_preds.append(count_prob)
    sup_preds = torch.stack(sup_preds, dim=0)
    sup_preds = torch.clamp(sup_preds, min=1e-12, max=1. - 1e-12)
    # print("sup_preds", sup_preds)
    
    return sup_preds
    


class CountLossMultipleInstanceLearning(ImpreciseMultipleInstanceLearning):
    def __init__(self, args, tb_log=None, logger=None, **kwargs):
        super().__init__(args, tb_log, logger, **kwargs)

    def train_step(self, x_bag_w, x_bag_len, y_bag):      
        x_bag_len = x_bag_len.cpu()
        
        # get unpadded data
        unpad_x_bag = unpad_sequence(x_bag_w, x_bag_len, batch_first=True)
        
        inputs = torch.cat(unpad_x_bag, dim=0)
        logits = self.model(inputs)
        
        # get softmax
        probs = logits.softmax(dim=-1)
        
        # handle multiple classes
        loss = 0.0
        for target_class in self.target_classes:
            cls_idx = self.class_map[target_class]
            binary_y_bag = y_bag[:, cls_idx]
            
            # construct binary  probs
            neg_probs_x = torch.cat([probs[:, idx].unsqueeze(1) for idx in range(probs.shape[1]) if idx != cls_idx], dim=1)
            neg_probs_x = torch.sum(neg_probs_x, dim=1, keepdim=True)
            pos_probs_x =  probs[:, cls_idx].unsqueeze(1)
            binary_probs_x =  torch.cat([neg_probs_x, pos_probs_x], dim=1)
            
            # pad probs_x_w
            pad_binary_probs_x = pad_sequence(binary_probs_x.split(x_bag_len.tolist()), batch_first=True)
            # dynamic programming
            sup_preds = dp_count_multi_ins(torch.log(pad_binary_probs_x), x_bag_len)
        
            loss += self.bce_loss(sup_preds[:, 1].unsqueeze(1), binary_y_bag.unsqueeze(1).to(torch.float), reduction='mean')
        
        loss /= len(self.target_classes)

        out_dict = self.process_out_dict(loss=loss)
        log_dict = self.process_log_dict(loss=loss.item())

        return out_dict, log_dict