import torch

from .imp_pair_comp import ImprecisePairComparisonLearning



class PCompUnbiasedPairComparisonLearning(ImprecisePairComparisonLearning):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def train_step(self, x_bag_w):

        bag_batch_size = x_bag_w.shape[0]
        
        # forward pass
        inputs = x_bag_w.view(-1, *x_bag_w.shape[2:])
        logits_x_ins = self.model(inputs)
        logits_x_bag = logits_x_ins.view(bag_batch_size, 2, -1)
        logits_x = logits_x_bag[:, 0]
        logits_x_ = logits_x_bag[:, 1]
        
        # comput loss
        labels_one = torch.ones(bag_batch_size).long().to(logits_x.device)
        labels_zero = torch.zeros(bag_batch_size).long().to(logits_x.device)
        loss = self.ce_loss(logits_x, labels_one, reduction='none').sum(dim=-1, keepdim=True) + \
               self.ce_loss(logits_x_, labels_zero, reduction='none').sum(dim=-1, keepdim=True) + \
               -self.class_prior * self.ce_loss(logits_x, labels_zero, reduction='none').sum(dim=-1, keepdim=True) + \
               -(1 - self.class_prior) * self.ce_loss(logits_x_, labels_one, reduction='none').sum(dim=-1, keepdim=True)
        loss = loss.mean()
        
        out_dict = self.process_out_dict(loss=loss)
        log_dict = self.process_log_dict(loss=loss.item())
        return out_dict, log_dict