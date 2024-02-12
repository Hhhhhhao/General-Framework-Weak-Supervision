import torch
import torch.nn.functional as F

from .imp_pair_comp import ImprecisePairComparisonLearning

from src.core.utils import Argument



class RankPruningPairComparisonLearning(ImprecisePairComparisonLearning):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_step(self, x_bag_w, x_bag_s):

        bag_batch_size = x_bag_w.shape[0]
        
        # forward pass
        x_ins_w = x_bag_w.view(-1, *x_bag_w.shape[2:])
        x_ins_s = x_bag_s.view(-1, *x_bag_s.shape[2:])
        
        # forward model
        inputs = torch.cat((x_ins_w, x_ins_s))
        outputs = self.model(inputs)
        logits_x_ins_w, logits_x_ins_s = outputs.chunk(2)
        logits_x_ins_w = logits_x_ins_w.detach()
        logits_x_bag_w = logits_x_ins_w.view(bag_batch_size, 2, -1)
        logits_x_bag_s = logits_x_ins_s.view(bag_batch_size, 2, -1)
        
        # compute loss
        logits_x_w = logits_x_bag_w[:, 0]
        logits_x_w_ = logits_x_bag_w[:, 1]
        pos_mask = logits_x_w.argmax(dim=1) == 1 
        neg_mask = logits_x_w_.argmax(dim=1) == 0
        logits_x_s = logits_x_bag_s[:, 0]
        logits_x_s_ = logits_x_bag_s[:, 1]
        if pos_mask.sum() == 0:
            loss_1 = torch.tensor([0.0]).to(logits_x_s.device)
        else:
            loss_1 = self.ce_loss(logits_x_s[pos_mask], torch.ones(pos_mask.sum()).long().to(logits_x_s.device), reduction='mean')
        if neg_mask.sum() == 0:
            loss_0 = torch.tensor([0.0]).to(logits_x_s.device)
        else:
            loss_0 = self.ce_loss(logits_x_s_[neg_mask], torch.zeros(neg_mask.sum()).long().to(logits_x_s.device), reduction='mean')
        pho_1 = self.class_prior / (self.class_prior + 1)
        pho_0 = (1 - self.class_prior) / (1 + 1 - self.class_prior)
        loss = pho_1 * loss_1 + pho_0 * loss_0
        
        out_dict = self.process_out_dict(loss=loss)
        log_dict = self.process_log_dict(loss=loss.item())
        return out_dict, log_dict