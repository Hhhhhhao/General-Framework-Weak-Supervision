

import torch 

from src.algorithms.pos_ulb.upu import UnbiasedPositiveUnLabeledLearning



def dp_count_proportion(log_probs, targets):    
    batch_size, _ = log_probs.shape
    
    # dynamic programming initialization
    # where the column is the number of positive instances and row is the idx of samples
    log_alpha = torch.full((batch_size + 1, batch_size + 1), -float("Inf"),  device=log_probs.device)
    log_alpha[0, 0] = 0
    
    for i in range(1, batch_size + 1):
        for j in range(batch_size + 1):
            
            alpha_plus_zero = log_alpha[i - 1, j] + log_probs[i - 1, 0]
            alpha_plus_one = log_alpha[i - 1, j - 1] + log_probs[i - 1, 1]

            # Mask to check if alpha_plus_zero and alpha_plus_one are -inf
            alpha_plus_zero[alpha_plus_zero == -float("Inf")] = -1e10
            alpha_plus_one[alpha_plus_one == -float("Inf")] = -1e10
            
            log_alpha[i, j] = torch.logsumexp(torch.stack([alpha_plus_zero, alpha_plus_one], dim=-1), dim=-1)
    
    m = torch.logsumexp(log_alpha[-1, :], dim=0)
    alpha = torch.exp(log_alpha[-1, :] - m)
    alpha = alpha / (alpha.sum() + 1e-12)
    count_prob = alpha[targets].view(1, -1)
    return count_prob


class CountLossPositiveUnlabeledLearning(UnbiasedPositiveUnLabeledLearning):
    def __init__(self, args, tb_log=None, logger=None, **kwargs):
        super().__init__(args, tb_log, logger, **kwargs)


    def train_step(self, x_lb, y_lb, x_ulb):

        num_lb = y_lb.shape[0]
        num_ulb = x_ulb.shape[0]
        
        # forward model
        inputs = torch.cat((x_lb, x_ulb))
        outputs = self.model(inputs)
        logits_x_lb = outputs[:num_lb]
        logits_x_ulb = outputs[num_lb:]
        
        # calculate labeled loss
        sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
        
        # get probs for unlabeled 
        probs_x_ulb = logits_x_ulb.softmax(dim=1)
        # get count probability
        count_prob = dp_count_proportion(torch.log(probs_x_ulb), int(num_ulb * self.class_prior))
        # get unsup loss
        unsup_loss = - torch.log(count_prob).mean()
        
        # total loss
        total_loss = sup_loss + unsup_loss
        
        
        out_dict = self.process_out_dict(loss=total_loss)
        log_dict = self.process_log_dict(loss=total_loss.item(),  sup_loss=sup_loss.item(), unsup_loss=unsup_loss.item())
        return out_dict, log_dict