

import torch 
import torch.nn.functional as F
from copy import deepcopy



from .lws import LWSPartialLabelLearning

def cc_loss(outputs, partialY):
    sm_outputs = F.softmax(outputs, dim=1)
    final_outputs = sm_outputs * partialY
    average_loss = -torch.log(final_outputs.sum(dim=1)).mean()
    return average_loss


class CCPartialLabelLearning(LWSPartialLabelLearning):
    def __init__(self, args, tb_log=None, logger=None, **kwargs):
        self.init(args)
        super().__init__(args, tb_log, logger, **kwargs)

    def train_step(self, x, y):    
            
        logits = self.model(x)
        
        loss = cc_loss(logits, y.float())

        out_dict = self.process_out_dict(loss=loss)
        log_dict = self.process_log_dict(loss=loss.item())
        return out_dict, log_dict