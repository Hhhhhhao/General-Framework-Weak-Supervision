import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import linear_sum_assignment as hungarian

from .imp_conf_diff import ImpreciseConfidenceDifferenceLearning



def logistic_loss(pred):
     negative_logistic = nn.LogSigmoid()
     logistic = -1. * negative_logistic(pred)
     return logistic


class ConfDiffUnbiasedConfidenceDifferenceLearning(ImpreciseConfidenceDifferenceLearning):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.loss_fn = lambda x: - torch.log(torch.sigmoid(x))

    def train_step(self, x_bag_w, y_bag_conf_diff):

        bag_batch_size = x_bag_w.shape[0]
        
        # forward pass
        inputs = x_bag_w.view(-1, *x_bag_w.shape[2:])
        logits_x_ins = self.model(inputs)
        logits_x_bag = logits_x_ins.view(bag_batch_size, 2, -1)
        logits_x = logits_x_bag[:, 0, 1]
        logits_x_ = logits_x_bag[:, 1, 1]
        
        # comput loss        
        loss_0 =  (self.class_prior - y_bag_conf_diff)   * logistic_loss(logits_x) +  (1 - self.class_prior + y_bag_conf_diff)   * logistic_loss(-logits_x)
        loss_0 = loss_0.mean()
        loss_1 =  (self.class_prior + y_bag_conf_diff)   * logistic_loss(logits_x_) + (1 - self.class_prior - y_bag_conf_diff)   * logistic_loss(-logits_x_)
        loss_1 = loss_1.mean()
        loss = 0.5 * (loss_0 + loss_1)
        
        out_dict = self.process_out_dict(loss=loss)
        log_dict = self.process_log_dict(loss=loss.item())
        return out_dict, log_dict