import torch
import torch.nn.functional as F

from .imp_sim_conf import ImpreciseSimilarConfidenceLearning

from src.core.utils import Argument

class SconfCorrectedPairComparisonLearning(ImpreciseSimilarConfidenceLearning):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init(self, args):
        super().init(args)
        
        if 'nn' in args.activation:
            self.activation = 'nn'
            args.activation = args.activation.replace('nn_', '')
        else:
            self.activation = args.activation
        print('activation: ', args.activation)
        
        if args.activation == 'relu':
            self.corr_g = lambda x: F.relu(x)
        elif args.activation == 'abs':
            self.corr_g = lambda x: torch.abs(x)

    def train_step(self, x_bag_w, y_bag_sim):

        bag_batch_size = x_bag_w.shape[0]
        
        # forward pass
        inputs = x_bag_w.view(-1, *x_bag_w.shape[2:])
        logits_x_ins = self.model(inputs)
        logits_x_bag = logits_x_ins.view(bag_batch_size, 2, -1)
        logits_x = logits_x_bag[:, 0, 1]
        logits_x_ = logits_x_bag[:, 1, 1]
        probs_x = torch.sigmoid(logits_x)
        probs_x_ = torch.sigmoid(logits_x_)
        
        # comput loss
        weight = (2 * self.class_prior - 1)
        loss_1 = (y_bag_sim - (1 - self.class_prior)) * (-torch.log(probs_x + 1e-12) + -torch.log(probs_x_ + 1e-12))
        loss_0 = (self.class_prior - y_bag_sim) * (-torch.log(1 - probs_x + 1e-12) +  -torch.log(1 - probs_x_ + 1e-12))
        loss_1 = loss_1.mean() / weight
        loss_0 = loss_0.mean() / weight
        
        if self.activation != 'nn':
            loss_1 = loss_1 / 2 + self.corr_g(loss_1)
            loss_0 = loss_0 / 2 + self.corr_g(loss_0)
        else:
            loss_1 = self.corr_g(loss_1) 
            loss_0 = self.corr_g(loss_0)
        
        loss = 0.5 * (loss_0 + loss_1)
        
        out_dict = self.process_out_dict(loss=loss)
        log_dict = self.process_log_dict(loss=loss.item())
        return out_dict, log_dict

    @staticmethod
    def get_argument():
        argument_list = ImpreciseSimilarConfidenceLearning.get_argument()
        argument_list.append(Argument('--activation', default='relu', type=str, help='correction activation'))
        return argument_list