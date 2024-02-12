

import torch 
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src.core.utils import Argument, str2bool
from src.core.criterions import BCELoss

from src.algorithms.multi_ins_label.imp_multi_ins import ImpreciseMultipleInstanceLearning



class AttnMultipleInstanceLearning(ImpreciseMultipleInstanceLearning):
    def __init__(self, args, tb_log=None, logger=None, **kwargs):
        assert args.net in ['attn_lenet5']
        super().__init__(args, tb_log, logger, **kwargs)

    def train_step(self, x_bag_w, x_bag_s, y_bag, y_ins):      
        
        # batch size is 1 for different bag lengh
        x_bag_w = x_bag_w.squeeze(0)
        x_bag_s = x_bag_s.squeeze(0)
        
        outputs = self.model(x_bag_w)
        
        # convert logots_w to probs
        probs = outputs.softmax(dim=-1)
        
        # comput loss
        loss = self.ce_loss(probs, y_bag.to(torch.long))
        # sup_loss = -1. * (y_bag * torch.log(pseudo_y_bag) + (1. - y_bag) * torch.log(1. - pseudo_y_bag))

        out_dict = self.process_out_dict(loss=loss)
        log_dict = self.process_log_dict(loss=loss.item())
        return out_dict, log_dict

    def evaluate(self, eval_dest='eval', return_logits=False):
        """
        evaluation function
        """
        self.model.eval()
        if self.ema is not None:
            self.ema.apply_shadow()

        eval_loader = self.loader_dict[eval_dest]
        total_loss = 0.0
        total_num = 0.0
        y_bag_true = []
        y_bag_pred = []
        # y_probs = []
        y_logits = []
        with torch.no_grad():
            for data in eval_loader:
                bag_x = data['x_bag']
                bag_y = data['y_bag']
                ins_y = data['y_ins']
                
                if isinstance(bag_x, dict):
                    x = {k: v.cuda(self.gpu) for k, v in x.items()}
                else:
                    bag_x  = bag_x.cuda(self.gpu)
                bag_y = bag_y.cuda(self.gpu)

                num_batch = bag_y.shape[0]
                total_num += num_batch

                logits = self.model(bag_x.squeeze(0))
                
                # compute instance label
                bag_label = logits.argmax(dim=-1)
                # bag_label = F.one_hot(bag_label, num_classes=2).float()
                # ins_labels = pred_labels == self.target_class
                bag_y = bag_y.argmax(dim=-1)
            
                # loss = F.cross_entropy(logits, y, reduction='mean', ignore_index=-1)
                y_bag_true.append(bag_y.cpu().numpy())
                y_bag_pred.append(bag_label.cpu().numpy())
                
                # total_loss += loss.item() * num_batch
                
        y_bag_true = np.concatenate(y_bag_true, axis=0)
        y_bag_pred = np.concatenate(y_bag_pred, axis=0)
        
        # y_logits = np.concatenate(y_logits)
        bag_top1 = accuracy_score(y_bag_true, y_bag_pred)
        # balanced_top1 = balanced_accuracy_score(y_true, y_pred)
        # precision = precision_score(y_true, y_pred, average='macro')
        # recall = recall_score(y_true, y_pred, average='macro')
        bag_F1 = f1_score(y_bag_true, y_bag_pred, average='macro')


        cf_mat = confusion_matrix(y_bag_true, y_bag_pred, normalize='true')
        self.print_fn('confusion matrix:\n' + np.array_str(cf_mat))
        if self.ema is not None:
            self.ema.restore()
        self.model.train()

        eval_dict = {eval_dest+'/top-1-acc': bag_top1, eval_dest+'/F1': bag_F1}
        if return_logits:
            eval_dict[eval_dest+'/logits'] = y_logits
        return eval_dict


class GatedAttnMultipleInstanceLearning(AttnMultipleInstanceLearning):
    def __init__(self, args, tb_log=None, logger=None, **kwargs):
        assert args.net in ['gated_attn_lenet5']
        super().__init__(args, tb_log, logger, **kwargs)