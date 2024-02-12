import numpy as np
import torch
import torch.nn.functional as F


def check_for_nan(tensor, label):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {label}!")


def log_sum_exp(*args):
    return torch.logsumexp(torch.stack(args), dim=0)


def create_mask(lengths, max_len):
    return torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]


def create_multi_ins_graph(log_probs, bag_lengths, targets):
    bag_lengths = bag_lengths.to(log_probs.device)
    
    # get size
    batch_size, max_seq_length, _ = log_probs.shape
    k = 4
    
    # create mask from lengths
    bag_masks = create_mask(bag_lengths, max_seq_length)
    
    # forward initialization    
    log_alpha = torch.full((batch_size, k, max_seq_length), -1e12, device=log_probs.device)
    log_alpha[:, 0, 0], log_alpha[:, 1, 0] = log_probs[:, 0, 0], log_probs[:, 0, 1]
    
    # forward process
    for i in range(1, max_seq_length):
        valid_mask = bag_masks[:, i]
        
        log_alpha[:, 0, i] = torch.where(valid_mask, log_alpha[:, 0, i - 1] + log_probs[:, i, 0], -1e12)
        log_alpha[:, 1, i] = torch.where(valid_mask, log_alpha[:, 0, i - 1] + log_probs[:, i, 1], -1e12)
        
        extended_mask = (i >= 2) & valid_mask
        log_alpha[:, 2, i] = torch.where(
            extended_mask,
            log_sum_exp(log_alpha[:, 1, i - 1], log_alpha[:, 2, i - 1], log_alpha[:, 3, i - 1]) + log_probs[:, i, 0],
            torch.where(valid_mask, log_alpha[:, 1, i - 1] + log_probs[:, i, 0], -1e12)
        )
        log_alpha[:, 3, i] = torch.where(
            extended_mask,
            log_sum_exp(log_alpha[:, 1, i - 1], log_alpha[:, 2, i - 1], log_alpha[:, 3, i - 1]) + log_probs[:, i, 1],
            torch.where(valid_mask, log_alpha[:, 1, i - 1] + log_probs[:, i, 1], -1e12)
        )

    # forward probability
    batch_idxs = torch.arange(batch_size, device=log_probs.device)
    last_valid_idxs = bag_lengths - 1
    m = torch.logsumexp(log_alpha[batch_idxs, :, last_valid_idxs], dim=1, keepdims=True)
    alpha = torch.exp(log_alpha[batch_idxs, :, last_valid_idxs] - m)
    sup_preds = torch.sum(alpha[:, 1:], dim=1)
    sup_preds = torch.clamp(sup_preds, min=1e-12, max=1. - 1e-12)
    
    with torch.no_grad():
        
        log_alpha_ = log_alpha.clone()
        log_alpha_[0, 0, last_valid_idxs] = -1e12
        
        log_beta = torch.full((batch_size, k, max_seq_length), -1e12, device=log_probs.device)
        log_beta[batch_idxs, 1, last_valid_idxs] = log_probs[torch.arange(batch_size), last_valid_idxs, 1]
        log_beta[batch_idxs, 2, last_valid_idxs] = log_probs[torch.arange(batch_size), last_valid_idxs, 0]
        log_beta[batch_idxs, 3, last_valid_idxs] = log_probs[torch.arange(batch_size), last_valid_idxs, 1]
        
        for i in range(max_seq_length - 2, -1, -1):
            valid_mask = bag_masks[:, i]
            extended_mask = (i < bag_lengths - 1) & valid_mask
            
            log_beta[:, 0, i] = torch.where(valid_mask, log_sum_exp(log_beta[:, 0, i + 1], log_beta[:, 1, i + 1]) + log_probs[:, i, 0], -1e12)
            log_beta[:, 1, i] = torch.where(
                extended_mask, 
                log_sum_exp(log_beta[:, 2, i + 1], log_beta[:, 3, i + 1]) + log_probs[:, i, 1], 
                torch.where(valid_mask, log_probs[:, i, 1], -1e12)
            )
            
            if i > 0:
                log_beta[:, 2, i] = torch.where(
                    extended_mask, 
                    log_sum_exp(log_beta[:, 2, i + 1], log_beta[:, 3, i + 1]) + log_probs[:, i, 0], 
                    torch.where(valid_mask, log_probs[:, i, 0], -1e12))
                log_beta[:, 3, i] = torch.where(
                    extended_mask, 
                    log_sum_exp(log_beta[:, 2, i + 1], log_beta[:, 3, i + 1]) + log_probs[:, i, 0],
                    torch.where(valid_mask, log_probs[:, i, 1], -1e12))
            
        repeated_log_probs = torch.cat([log_probs] * 2, dim=2).transpose(1, 2)
        log_beta -= repeated_log_probs
        
        
        gamma = log_alpha_ + log_beta
        m = torch.logsumexp(gamma, dim=1, keepdim=True)
        gamma = torch.exp(gamma - m)
        
        # compute default em_targets
        em_targets_default = torch.tensor([1, 0], device=log_probs.device).repeat(batch_size, max_seq_length, 1)  # Shape: [batch_size, max_seq_length, 2]
        
        # comput em_targets from gamma
        em_targets_from_gamma =  gamma[:, :2, :].transpose(1, 2) + gamma[:, 2:, :].transpose(1, 2)

        # final em_targets
        em_targets = torch.where(targets.view(-1, 1, 1).to(dtype=torch.bool), em_targets_from_gamma, em_targets_default)
        
        em_targets = em_targets / em_targets.sum(dim=-1, keepdim=True)
    
    return em_targets, sup_preds


def create_proportion_graph(log_probs, targets):
    b, _ = log_probs.shape
    count = int(targets)
    k = 2 * count + 1
    
    # forward initialization
    log_alpha = torch.full((k, b), -1e12,  device=log_probs.device)
    log_alpha[0, 0] = log_probs[0, 0]
    if count > 0:
        log_alpha[1, 0] = log_probs[0, 1]
    
    # forward process
    for i in range(1, b):
        
        log_alpha[0, i] = log_alpha[0, i - 1] + log_probs[i, 0]
        if count > 0:
            log_alpha[1, i] = log_alpha[0, i - 1] + log_probs[i, 1]
         
        for j in range(2, k):
            
            if i < count - (k - j) // 2:
                continue
            
            if j % 2 == 0:
                log_alpha[j, i] = log_sum_exp(log_alpha[j, i - 1], log_alpha[j - 1, i - 1]) + log_probs[i, 0]
            else:
                log_alpha[j, i] = log_sum_exp(log_alpha[j - 1, i - 1], log_alpha[j - 2, i - 1]) + log_probs[i, 1]

    
    # forward probability
    m = torch.logsumexp(log_alpha[:, -1], dim=0)
    alpha = torch.exp(log_alpha[:, -1] - m)
    if count == 0:
        sup_preds = alpha[-1]
    else:
        sup_preds = torch.sum(alpha[-2:])
    sup_preds = torch.clamp(sup_preds, min=1e-12, max=1. - 1e-12)
    
    log_alpha_ = torch.clone(log_alpha)
    if count == b:
        log_alpha_[0, 0] = -1e12
    for i in range(b):
        if i >= b - count:
            log_alpha_[0, i] = -1e12
        if i >= b - count + 1 and count > 0:
            log_alpha_[1, i] = -1e12
            
        for j in range(2, k):
            
            if i >= b - count + (j + 1) // 2:
                log_alpha_[j, i] = -1e12
    
    with torch.no_grad():
        # backward initialization
        log_beta = torch.full((k, b), -1e12,  device=log_probs.device)
        if count < b:
            log_beta[-1, -1] = log_probs[-1, 0]
        if count > 0:
            log_beta[-2, -1] = log_probs[-1, 1]
            
        # backward process
        for i in range(b - 2, -1, -1):
            
            if i >= count:
                log_beta[-1, i] = log_beta[-1, i + 1] + log_probs[i, 0]
            if i >= count - 1 and count > 0:
                log_beta[-2, i] = log_beta[-1, i + 1] + log_probs[i, 0]
            
            for j in range(k - 2):
                if i < count - (k - j) // 2:
                    continue
                
                if j % 2 == 0:
                    log_beta[j, i] = log_sum_exp(log_beta[j, i + 1], log_beta[j + 1, i + 1]) + log_probs[i, 0]
                else:
                    log_beta[j, i] = log_sum_exp(log_beta[j + 1, i + 1], log_beta[j + 2, i + 1]) + log_probs[i, 1]
        

        log_beta -= torch.cat([log_probs] * count + [log_probs[:, 0].unsqueeze(1)], dim=1).transpose(0, 1)
        
        # em targets
        gamma = log_alpha_ + log_beta
        m = torch.logsumexp(gamma, dim=0)
        gamma = torch.exp(gamma - m)
        
        gamma_split = torch.split(gamma, 2)
        em_targets = torch.zeros_like(log_probs)
        for g in gamma_split[:-1]:
            em_targets += g.transpose(0, 1)
        em_targets[:, 0] += gamma_split[-1].transpose(0, 1).squeeze()
        em_targets = em_targets / em_targets.sum(dim=1, keepdim=True)
        
    return em_targets, sup_preds


def create_pair_sim_dsim_graph(log_probs, targets):
    b, l, k = log_probs.shape
    assert l == 2
    k = 4
        
    # forward initialization
    log_alpha = torch.full((b, k, l), float('-inf'),   device=log_probs.device)
    
    # forward process
    log_alpha[:, 0, 0] = log_probs[:, 0, 0]
    log_alpha[:, 1, 0] = log_probs[:, 0, 1]
    log_alpha[:, 2, 0] = log_probs[:, 0, 0]
    log_alpha[:, 3, 0] = log_probs[:, 0, 1]
    
    
    log_alpha[:, :2, 1] = log_alpha[:, :2, 0] + log_probs[:, 1, :] 
    log_alpha[:, 2, 1] = log_alpha[:, 3, 0] + log_probs[:, 1, 0]
    log_alpha[:, 3, 1] = log_alpha[:, 2, 0] + log_probs[:, 1, 1]
    
    m = torch.logsumexp(log_alpha[:, :, -1], dim=1, keepdims=True)
    alpha = torch.exp(log_alpha[:, :, -1] - m)
    sim_probs = alpha[:, :2].sum(dim=-1) 
    sup_preds = sim_probs
    sup_preds = torch.clamp(sup_preds, min=1e-12, max=1. - 1e-12)
    
    
    with torch.no_grad():        
        # log_alpha
        # forward initialization
        log_alpha_ = torch.full((b, k, l), float('-inf'),   device=log_probs.device)
    
        # forward process
        mask = targets == 1
        log_alpha_[mask, 0, 0] = log_probs[mask, 0, 0] 
        log_alpha_[mask, 1, 0] = log_probs[mask, 0, 1] 
        log_alpha_[mask, :2, 1] = log_alpha_[mask, :2, 0] + log_probs[mask, 1, :]
    
        mask = targets == 0
        log_alpha_[mask, 2, 0] = log_probs[mask, 0, 0] 
        log_alpha_[mask, 3, 0] = log_probs[mask, 0, 1]
        log_alpha_[mask, 2, 1] = log_alpha_[mask, 3, 0] + log_probs[mask, 1, 0]
        log_alpha_[mask, 3, 1] = log_alpha_[mask, 2, 0] + log_probs[mask, 1, 1]
        
        
        # backward initialization
        log_beta = torch.full((b, k, l), float('-inf'),   device=log_probs.device)
        
        # backward process
        mask = targets == 1
        log_beta[mask, 0, 1] = log_probs[mask, 1, 0]
        log_beta[mask, 1, 1] = log_probs[mask, 1, 1]
        log_beta[mask, :2, 0] = log_beta[mask, :2, 1] + log_probs[mask, 0, :]
        
        mask = targets == 0
        log_beta[mask, 2, 1] = log_probs[mask, 1, 0]
        log_beta[mask, 3, 1] = log_probs[mask, 1, 1]
        log_beta[mask, 2, 0] = log_beta[mask, 3, 1] + log_probs[mask, 0, 0]
        log_beta[mask, 3, 0] = log_beta[mask, 2, 1] + log_probs[mask, 0, 1]

        repeated_log_probs = torch.cat([log_probs] * 2, dim=2).transpose(1, 2)
        log_beta -= repeated_log_probs
        
        # em_targets
        gamma = log_alpha_ + log_beta
        gamma = torch.exp(gamma)
        em_targets_from_gamma =  gamma[:, :2, :].transpose(1, 2)  + gamma[:, 2:, :].transpose(1, 2) 
        em_targets = em_targets_from_gamma / em_targets_from_gamma.sum(dim=-1, keepdim=True)
        
    return em_targets, sup_preds


def create_pair_sim_dsim_ulb_graph(log_probs, targets, class_prior):
    b, l, k = log_probs.shape
    assert l == 2
    k = 4
        
    # forward initialization
    log_alpha = torch.full((b, k, l), float('-inf'),   device=log_probs.device)
    
    # forward process
    log_alpha[:, 0, 0] = log_probs[:, 0, 0]
    log_alpha[:, 1, 0] = log_probs[:, 0, 1]
    log_alpha[:, 2, 0] = log_probs[:, 0, 0]
    log_alpha[:, 3, 0] = log_probs[:, 0, 1]
    
    
    log_alpha[:, :2, 1] = log_alpha[:, :2, 0] + log_probs[:, 1, :] 
    log_alpha[:, 2, 1] = log_alpha[:, 3, 0] + log_probs[:, 1, 0]
    log_alpha[:, 3, 1] = log_alpha[:, 2, 0] + log_probs[:, 1, 1]
    
    m = torch.logsumexp(log_alpha[:, :, -1], dim=1, keepdims=True)
    alpha = torch.exp(log_alpha[:, :, -1] - m)
    sup_preds = alpha[:, :2].sum(dim=-1) 
    sup_preds = torch.clamp(sup_preds, min=1e-12, max=1. - 1e-12)
    
    
    with torch.no_grad():        
        
        sim_weights = torch.log(torch.FloatTensor(np.array([class_prior ** 2, (1 - class_prior) ** 2])).to(log_probs.device)).view(1, 2)
        dsim_weights = torch.log(torch.FloatTensor(np.array([class_prior * (1 - class_prior)])).to(log_probs.device)).view(1)
        
        # log_alpha
        # forward initialization
        log_alpha_ = torch.full((b, k, l), float('-inf'),   device=log_probs.device)
    
        # forward process
        mask = targets == 1
        log_alpha_[mask, 0, 0] = log_probs[mask, 0, 0] 
        log_alpha_[mask, 1, 0] = log_probs[mask, 0, 1] 
        log_alpha_[mask, :2, 1] = log_alpha_[mask, :2, 0] + log_probs[mask, 1, :] + sim_weights
    
        mask = targets == 0
        log_alpha_[mask, 2, 0] = log_probs[mask, 0, 0] 
        log_alpha_[mask, 3, 0] = log_probs[mask, 0, 1]
        log_alpha_[mask, 2, 1] = log_alpha_[mask, 3, 0] + log_probs[mask, 1, 0] + dsim_weights
        log_alpha_[mask, 3, 1] = log_alpha_[mask, 2, 0] + log_probs[mask, 1, 1] + dsim_weights
        
        
        # backward initialization
        log_beta = torch.full((b, k, l), float('-inf'),   device=log_probs.device)
        
        # backward process
        mask = targets == 1
        log_beta[mask, 0, 1] = log_probs[mask, 1, 0]
        log_beta[mask, 1, 1] = log_probs[mask, 1, 1]
        log_beta[mask, :2, 0] = log_beta[mask, :2, 1] + log_probs[mask, 0, :] + sim_weights
        
        mask = targets == 0
        log_beta[mask, 2, 1] = log_probs[mask, 1, 0]
        log_beta[mask, 3, 1] = log_probs[mask, 1, 1]
        log_beta[mask, 2, 0] = log_beta[mask, 3, 1] + log_probs[mask, 0, 0] + dsim_weights
        log_beta[mask, 3, 0] = log_beta[mask, 2, 1] + log_probs[mask, 0, 1] + dsim_weights

        repeated_log_probs = torch.cat([log_probs] * 2, dim=2).transpose(1, 2)
        log_beta -= repeated_log_probs
        
        # em_targets
        gamma = log_alpha_ + log_beta
        gamma = torch.exp(gamma)
        em_targets_from_gamma =  gamma[:, :2, :].transpose(1, 2)  + gamma[:, 2:, :].transpose(1, 2) 
        
        em_targets = em_targets_from_gamma / em_targets_from_gamma.sum(dim=-1, keepdim=True)
        
    return em_targets, sup_preds



def create_pair_comp_graph(log_probs):
    b, l, _ = log_probs.shape
    assert l == 2
    k = 4

    # forward initialization
    log_alpha = torch.full((b, k, l), float('-inf'),   device=log_probs.device)
    
    # forward process
    log_alpha[:, 0, 0] = log_probs[:, 0, 0]
    log_alpha[:, 1, 0] = log_probs[:, 0, 1]
    # log_alpha[:, 2, 0] = log_probs[:, 0, 0]
    log_alpha[:, 3, 0] = log_probs[:, 0, 1]
    
    log_alpha[:, :2, 1] = log_alpha[:, :2, 0] + log_probs[:, 1, :] # + log_sim_score
    log_alpha[:, 2, 1] = log_alpha[:, 3, 0] + log_probs[:, 1, 0] # + log_dsim_score
    
    # forward probability
    m = torch.logsumexp(log_alpha[:, :, -1], dim=1, keepdims=True)
    alpha = torch.exp(log_alpha[:, :, -1] - m)
    sup_preds = alpha[:, :3].sum(dim=1)

    
    
    with torch.no_grad():
        # backward initialization
        log_beta = torch.full((b, k, l), float('-inf'),  device=log_probs.device)
        
        # backward process
        log_beta[:, 0, 1] = log_probs[:, 1, 0]
        log_beta[:, 1, 1] = log_probs[:, 1, 1]
        log_beta[:, 2, 1] = log_probs[:, 1, 0]
        
        log_beta[:, :2, 0] = log_beta[:, :2, 1] + log_probs[:, 0, :] 
        log_beta[:, 3, 0] = log_beta[:, 2, 1] + log_probs[:, 0, 1] 

        repeated_log_probs = torch.cat([log_probs] * 2, dim=2).transpose(1, 2)
        log_beta -= repeated_log_probs
        
        # em_targets
        gamma = log_alpha + log_beta
        m = torch.logsumexp(gamma, dim=1, keepdim=True)
        gamma = torch.exp(gamma - m)
        em_targets_from_gamma =  gamma[:, :2, :].transpose(1, 2)  + gamma[:, 2:, :].transpose(1, 2) 
        
        em_targets = em_targets_from_gamma / em_targets_from_gamma.sum(dim=-1, keepdim=True)
        
    return em_targets , sup_preds



def create_pos_conf_graph(log_probs, targets):
    b, k = log_probs.shape 
    
    # em_targets
    with torch.no_grad():
        log_targets = torch.zeros_like(log_probs)
        log_targets[:, 0] = torch.log(1 - targets)
        log_targets[:, 1] = torch.log(targets)
        
        log_em_targets = log_targets + log_probs
        em_targets = torch.exp(log_em_targets)
        em_targets = em_targets / em_targets.sum(dim=1, keepdim=True)
    
    return em_targets, None




def create_sim_conf_graph(log_probs, targets):
    b, l, k = log_probs.shape
    assert l == 2
    k = 4
    
    log_sim_score = torch.log(targets).view(-1, 1)
    log_dsim_score = torch.log(1 - targets)
    
    # forward initialization
    log_alpha = torch.full((b, k, l), -1e12,  device=log_probs.device)
    
    # forward process
    log_alpha[:, 0, 0] = log_probs[:, 0, 0]
    log_alpha[:, 1, 0] = log_probs[:, 0, 1]
    log_alpha[:, 2, 0] = log_probs[:, 0, 0]
    log_alpha[:, 3, 0] = log_probs[:, 0, 1]
    
    log_alpha[:, :2, 1] = log_alpha[:, :2, 0] + log_probs[:, 1, :]
    log_alpha[:, 2, 1] = log_alpha[:, 3, 0] + log_probs[:, 1, 0]
    log_alpha[:, 3, 1] = log_alpha[:, 2, 0] + log_probs[:, 1, 1]
    # print("log_alpha\n", log_alpha)
    
    # forward probability
    alpha = torch.exp(log_alpha[:, :, -1])
    probs = alpha / alpha.sum(dim=1, keepdim=True)
    sup_preds = probs[:, :2].sum(dim=-1, keepdim=True)
    sup_preds = torch.clamp(sup_preds, min=1e-12, max=1. - 1e-12)
    
    with torch.no_grad():

        log_alpha_ = log_alpha.clone()
        log_alpha_[:, :2, 1] = log_alpha_[:, :2, 0] + log_probs[:, 1, :] + log_sim_score
        log_alpha_[:, 2, 1] = log_alpha_[:, 3, 0] + log_probs[:, 1, 0] + log_dsim_score
        log_alpha_[:, 3, 1] = log_alpha_[:, 2, 0] + log_probs[:, 1, 1] + log_dsim_score

        # backward initialization
        log_beta = torch.full((b, k, l), -1e12,  device=log_probs.device)
        
        # backward process
        log_beta[:, 0, 1] = log_probs[:, 1, 0]
        log_beta[:, 1, 1] = log_probs[:, 1, 1]
        log_beta[:, 2, 1] = log_probs[:, 1, 0]
        log_beta[:, 3, 1] = log_probs[:, 1, 1]
        
        log_beta[:, :2, 0] = log_beta[:, :2, 1] + log_probs[:, 0, :] + log_sim_score 
        log_beta[:, 2, 0] = log_beta[:, 3, 1] + log_probs[:, 0, 0] + log_dsim_score 
        log_beta[:, 3, 0] = log_beta[:, 2, 1] + log_probs[:, 0, 1] + log_dsim_score

        repeated_log_probs = torch.cat([log_probs] * 2, dim=2).transpose(1, 2)
        log_beta -= repeated_log_probs
        
        # em_targets
        gamma = log_alpha + log_beta
        m = torch.logsumexp(gamma, dim=1, keepdim=True)
        gamma = torch.exp(gamma - m).transpose(1, 2)
        em_targets = gamma[:, :, :2]  + gamma[:, :, 2:] 
        em_targets = em_targets / em_targets.sum(dim=-1, keepdim=True)
    
    return em_targets, sup_preds
    
    
    

def create_conf_diff_graph(log_probs, targets, class_prior):
    b, l, k = log_probs.shape
    assert l == 2
    k = 4
    
    target_positive = targets >= 0
    abs_targets = torch.abs(targets)
    log_dsim_score = torch.log(abs_targets)
    log_sim_score = torch.log(1 - abs_targets).view(-1, 1)

    
    # forward initialization
    log_alpha = torch.full((b, k, l), -1e12,  device=log_probs.device)
    
    # forward process
    log_alpha[:, 0, 0] = log_probs[:, 0, 0]
    log_alpha[:, 1, 0] = log_probs[:, 0, 1]
    log_alpha[:, 2, 0] = torch.where(target_positive,  -1e12, log_probs[:, 0, 0])
    log_alpha[:, 3, 0] = torch.where(target_positive, log_probs[:, 0, 1], -1e12)
    
    
    log_alpha[:, :2, 1] = log_alpha[:, :2, 0] + log_probs[:, 1, :] # + log_sim_score
    log_alpha[:, 2, 1] = log_alpha[:, 3, 0] + log_probs[:, 1, 0]  # + log_dsim_score
    log_alpha[:, 3, 1] = log_alpha[:, 2, 0] + log_probs[:, 1, 1] # + log_dsim_score
    
    
    # forward probability
    alpha = torch.exp(log_alpha[:, :, -1])
    probs = alpha / alpha.sum(dim=1, keepdim=True)
    sup_preds = torch.clamp(probs, min=1e-12, max=1. - 1e-12)
    
    with torch.no_grad():
        
        log_alpha_ = log_alpha.clone()
        log_alpha_[:, :2, 1] = log_alpha_[:, :2, 0] + log_probs[:, 1, :] + log_sim_score
        log_alpha_[:, 2, 1] = torch.where(target_positive, log_alpha_[:, 3, 0] + log_probs[:, 1, 0] + log_dsim_score, -1e12)
        log_alpha_[:, 3, 1] = torch.where(target_positive,  -1e12, log_alpha_[:, 2, 0] + log_probs[:, 1, 1] + log_dsim_score)

        # backward initialization
        log_beta = torch.full((b, k, l), -1e12,  device=log_probs.device)
        
        # backward process
        log_beta[:, 0, 1] = log_probs[:, 1, 0]
        log_beta[:, 1, 1] = log_probs[:, 1, 1]
        log_beta[:, 2, 1] = torch.where(target_positive, log_probs[:, 1, 0], -1e12)
        log_beta[:, 3, 1] = torch.where(target_positive, -1e12, log_probs[:, 1, 1])
        
        log_beta[:, :2, 0] = log_beta[:, :2, 1] + log_probs[:, 0, :] + log_sim_score
        log_beta[:, 2, 0] = torch.where(target_positive, -1e12, log_beta[:, 3, 1] + log_probs[:, 0, 0] + log_dsim_score)
        log_beta[:, 3, 0] = torch.where(target_positive, log_beta[:, 2, 1] + log_probs[:, 0, 1] + log_dsim_score, -1e12)

        repeated_log_probs = torch.cat([log_probs] * 2, dim=2).transpose(1, 2)
        log_beta -= repeated_log_probs
        
        # em_targets
        gamma = log_alpha_ + log_beta
        gamma = torch.exp(gamma).transpose(1, 2)
        

        em_targets = gamma[:, :, :2] + gamma[:, :, 2:] 
        em_targets = em_targets / em_targets.sum(dim=-1, keepdim=True)

    return em_targets, sup_preds