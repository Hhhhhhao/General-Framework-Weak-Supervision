import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment as hungarian

from .imp_sim_conf import ImpreciseSimilarConfidenceLearning



class SconfUnbiasedPairComparisonLearning(ImpreciseSimilarConfidenceLearning):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


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
        loss_1 = loss_1.mean() / weight
        loss_0 = (self.class_prior - y_bag_sim) * (-torch.log(1 - probs_x + 1e-12) +  -torch.log(1 - probs_x_ + 1e-12))
        loss_0 = loss_0.mean() / weight
        
        loss = 0.5 * (loss_0 + loss_1)
        
        out_dict = self.process_out_dict(loss=loss)
        log_dict = self.process_log_dict(loss=loss.item())
        return out_dict, log_dict


    def evaluate(self, eval_dest='eval', **kwargs):
        """
        evaluation function
        """
        self.model.eval()
        if self.ema is not None:
            self.ema.apply_shadow()

        eval_loader = self.loader_dict[eval_dest]
        
        total_loss = 0.0
        total_num = 0.0
        y_true = []
        y_pred = []
        confusion_matrix = torch.zeros((self.output_classes, self.output_classes), dtype=torch.long)
        # y_probs = []
        # y_logits = []
        with torch.no_grad():
            for data in eval_loader:
                x = data['x']
                y = data['y']
                
                if isinstance(x, dict):
                    x = {k: v.cuda(self.gpu) for k, v in x.items()}
                else:
                    x = x.cuda(self.gpu)
                y = y.cuda(self.gpu)

                num_batch = y.shape[0]
                total_num += num_batch

                logits = self.model(x)
                
                # loss = F.binary_cross_entropy(logits[:, 1], y.to(torch.float), reduction='mean')
                loss = -F.logsigmoid(logits[:, 1]).mean()
                # loss = F.cross_entropy(logits, y, reduction='mean', ignore_index=-1)
                total_loss += loss.item() * num_batch
                
                preds =  (logits[:, 1] >= 0).float()
                y = y.cpu()
                
                indices = (y * confusion_matrix.stride(0) + preds.squeeze_().type_as(y)).type_as(confusion_matrix)
                ones = torch.ones(1).type_as(confusion_matrix).expand(indices.size(0))
                conf_flat = confusion_matrix.view(-1)
                conf_flat.index_add_(0, indices, ones)
                
                y_true.extend(y.cpu().tolist())
                y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
                
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # optimal 
        mat = -confusion_matrix.cpu().numpy() #hungaian finds the minimum cost
        r,assign = hungarian(mat)
        conf = confusion_matrix[:,assign]
        
        TP = conf.diag().sum().item()
        total = conf.sum().item()
        acc = TP/total
        
        self.print_fn('confusion matrix:\n' + np.array_str(conf.cpu().numpy()))
        if self.ema is not None:
            self.ema.restore()
        self.model.train()

        eval_dict = {eval_dest+'/loss': total_loss / total_num, eval_dest+'/top-1-acc': acc}
        return eval_dict