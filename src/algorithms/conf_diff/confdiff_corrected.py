import torch
import torch.nn as nn
import torch.nn.functional as F

from .confdiff_unbiased import ConfDiffUnbiasedConfidenceDifferenceLearning
from src.core.utils import Argument



def logistic_loss(pred):
     negative_logistic = nn.LogSigmoid()
     logistic = -1. * negative_logistic(pred)
     return logistic



class ConfDiffCorrectedConfidenceDifferenceLearning(ConfDiffUnbiasedConfidenceDifferenceLearning):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init(self, args):
        super().init(args)
        if args.activation == 'relu':
            self.corr_g = lambda x: F.relu(x)
        elif args.activation == 'abs':
            self.corr_g = lambda x: torch.abs(x)

    def train_step(self, x_bag_w, y_bag_conf_diff):

        bag_batch_size = x_bag_w.shape[0]
        
        # forward pass
        inputs = x_bag_w.view(-1, *x_bag_w.shape[2:])
        logits_x_ins = self.model(inputs)
        logits_x_bag = logits_x_ins.view(bag_batch_size, 2, -1)
        logits_x = logits_x_bag[:, 0, 1]
        logits_x_ = logits_x_bag[:, 1, 1]
        
        # comput loss
        labels_one = torch.ones(bag_batch_size).long().to(logits_x.device)
        labels_zero = torch.zeros(bag_batch_size).long().to(logits_x.device)
        weight1 = (self.class_prior - y_bag_conf_diff)  
        weight2 = (1 - self.class_prior - y_bag_conf_diff) 
        
        loss_0 = (self.class_prior - y_bag_conf_diff) * logistic_loss(logits_x)
        loss_1 = (1 - self.class_prior + y_bag_conf_diff) * logistic_loss(-logits_x)
        loss_2 = (self.class_prior + y_bag_conf_diff)  * logistic_loss(logits_x_)
        loss_3 = (1 - self.class_prior - y_bag_conf_diff)  * logistic_loss(-logits_x_)

        loss = 0.5 * (self.corr_g(loss_0.mean()) + self.corr_g(loss_1.mean()) + self.corr_g(loss_2.mean()) + self.corr_g(loss_3.mean()))
        
        out_dict = self.process_out_dict(loss=loss)
        log_dict = self.process_log_dict(loss=loss.item())
        return out_dict, log_dict



    @staticmethod
    def get_argument():
        argument_list = ConfDiffUnbiasedConfidenceDifferenceLearning.get_argument()
        argument_list.append(Argument('--activation', default='relu', type=str, help='correction activation'))
        return argument_list