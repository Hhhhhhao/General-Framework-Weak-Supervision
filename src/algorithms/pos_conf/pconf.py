import torch 


from .imp_pos_conf import ImprecisePositiveConfidenceLearning



class PconfPositiveConfidenceLearning(ImprecisePositiveConfidenceLearning):
    def __init__(self, args, tb_log=None, logger=None, **kwargs):
        super().__init__(args, tb_log, logger, **kwargs)
    
    
    def train_step(self, x_w, y_conf):
        
        logits = self.model(x_w)
        probs = logits.softmax(dim=-1)
        
        # calculate loss
        loss = - ( torch.log(probs[:, 1]) + (1 - y_conf) / y_conf * torch.log(probs[:, 0]) ).mean() 
        
        out_dict = self.process_out_dict(loss=loss)
        log_dict = self.process_log_dict(loss=loss.item())
        return out_dict, log_dict