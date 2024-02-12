import torch
import torch.nn.functional as F

from .imp_pair_comp import ImprecisePairComparisonLearning

from src.core.utils import Argument



class PCompCorrectedPairComparisonLearning(ImprecisePairComparisonLearning):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init(self, args):
        super().init(args)
        if args.activation == 'relu':
            self.corr_g = lambda x: F.relu(x)
        elif args.activation == 'abs':
            self.corr_g = lambda x: torch.abs(x)

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
        
        loss_0 = self.corr_g((self.ce_loss(logits_x_, labels_zero, reduction='none').sum(dim=-1, keepdim=True) - self.class_prior * self.ce_loss(logits_x, labels_zero, reduction='none').sum(dim=-1, keepdim=True)).mean())
        loss_1 = self.corr_g((self.ce_loss(logits_x, labels_one, reduction='none').sum(dim=-1, keepdim=True) - (1 - self.class_prior) * self.ce_loss(logits_x_, labels_one, reduction='none').sum(dim=-1, keepdim=True)).mean())
        loss = loss_0 + loss_1
        
        out_dict = self.process_out_dict(loss=loss)
        log_dict = self.process_log_dict(loss=loss.item())
        return out_dict, log_dict
    
    @staticmethod
    def get_argument():
        argument_list = ImprecisePairComparisonLearning.get_argument()
        argument_list.append(Argument('--activation', default='relu', type=str, help='correction activation'))
        return argument_list