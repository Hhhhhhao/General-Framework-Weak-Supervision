

import torch 
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment as hungarian

from src.core.algorithmbase import AlgorithmBase
from src.core.utils import Argument, str2bool, get_optimizer
from src.core.criterions import CELoss, BCELoss
from src.nets import get_model
from src.core.nfa import create_conf_diff_graph


from src.datasets import get_conf_diff_labels, get_data, get_dataloader, ImgBaseDataset, ImageTwoViewBagDataset


class ImpreciseConfidenceDifferenceLearning(AlgorithmBase):
    def __init__(self, args, tb_log=None, logger=None, **kwargs):
        self.init(args)
        super().__init__(args, tb_log, logger, **kwargs)

        # set number train iterations
        self.num_train_iter = self.epochs * len(self.loader_dict['train'])
        self.num_eval_iter = len(self.loader_dict['train'])        
        
        self.ce_loss = CELoss()
        self.bce_loss = BCELoss()
        

    def init(self, args):        
        self.target_classes = args.target_classes
        if isinstance(self.target_classes, int):
            self.target_classes = [self.target_classes]
        self.target_classes = sorted(self.target_classes)
        self.neg_classes = args.neg_classes
        if self.neg_classes is None or self.neg_classes == 'None':
            self.neg_classes = [i for i in range(args.num_classes) if i not in self.target_classes]
        
        self.output_classes = 2
        self.class_map = {}
        for i in range(args.num_classes):
            if i in self.target_classes:
                self.class_map[i] = 1 
            elif i in self.neg_classes:
                self.class_map[i] = 0
        
        self.conf_model_name = args.conf_model_name
        self.num_pair_data = args.num_pair_data
        self.class_prior = args.class_prior


    def set_model(self):
        """
        initialize model
        """
        model = get_model(model_name=self.args.net, num_classes=self.output_classes, pretrained=self.args.use_pretrain)
        return model

    def set_ema_model(self):
        """
        initialize ema model from model
        """
        ema_model = get_model(model_name=self.args.net, num_classes=self.output_classes,)
        ema_model.load_state_dict(self.model.state_dict())
        return ema_model

    def set_optimizer(self):
        """
        set optimizer for algorithm
        """
        # defautl is for semi-supervised learning
        self.print_fn("Create optimizer and scheduler")
        optimizer = get_optimizer(self.model, self.args.optim, self.args.lr, self.args.momentum, self.args.weight_decay, self.args.layer_decay, nesterov=False, bn_wd_skip=False)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(self.epochs * len(self.loader_dict['train'])), eta_min=1e-6)
        return optimizer, scheduler

    def set_dataset(self):
        # get initial data
        train_data, train_targets, test_data, test_targets, extra_data = get_data(self.args.data_dir, self.args.dataset)
        
        # make partial labels
        train_pos_data, train_conf_targets = get_conf_diff_labels(train_data, train_targets, self.num_classes,
                                                                 self.target_classes, self.neg_classes,
                                                                 num_pair_data=self.num_pair_data,
                                                                 class_prior=self.class_prior,
                                                                 data_dir=self.args.data_dir,
                                                                 dataset_name=self.args.dataset,
                                                                 conf_model_name=self.conf_model_name,
                                                                 zero_shot='clip' in self.conf_model_name)
        
        
        if self.args.dataset in ['mnist', 'fmnist']:
            resize = 'resize'
            test_resize = 'resize'
            autoaug = 'randaug' 
        elif self.args.dataset in ['cifar10', 'svhn', 'cifar100']:
            resize = 'resize_crop_pad'
            test_resize = 'resize'
            autoaug = 'randaug'
        elif self.args.dataset in ['stl10']:
            resize = 'resize_crop'
            autoaug = 'randaug'
        elif self.args.dataset in ['imagenet1k', 'imagenet100']:
            resize = 'rpc'
            autoaug = 'autoaug'
        test_resize = 'resize'
        
        if not self.strong_aug:
            autoaug = None
        
        # make dataset
        train_dataset = ImageTwoViewBagDataset(self.args.dataset, train_pos_data, train_conf_targets,
                                               num_classes=self.num_classes, class_map=None, is_train=True,
                                               img_size=self.args.img_size, crop_ratio=self.args.crop_ratio,
                                               autoaug=autoaug, resize=resize,
                                               aggregation='conf_diff',
                                               return_target=True,
                                               return_keys=['x_bag_w', 'x_bag_s', 'y_bag_conf_diff', 'y_ins'])
        test_dataset = ImgBaseDataset(self.args.dataset, test_data, test_targets, 
                                      num_classes=self.num_classes, class_map=self.class_map,
                                      is_train=False,
                                      # aggregation='proportion',
                                      img_size=self.args.img_size, crop_ratio=self.args.crop_ratio, resize=test_resize,
                                      return_keys=['x', 'y'])

        self.print_fn("Create datasets!")
        return {'train': train_dataset, 'eval': test_dataset}

    def set_data_loader(self):
        loader_dict = {}

        loader_dict['train'] = get_dataloader(self.dataset_dict['train'], 
                                                 num_epochs=self.epochs,
                                                 batch_size=self.args.batch_size, 
                                                 shuffle=True, 
                                                 num_workers=self.args.num_workers, 
                                                 pin_memory=True, 
                                                 drop_last=True,
                                                 distributed=self.args.distributed)
        loader_dict['eval'] = get_dataloader(self.dataset_dict['eval'], 
                                             num_epochs=self.epochs, 
                                             batch_size=self.args.eval_batch_size, 
                                             shuffle=False, 
                                             num_workers=self.args.num_workers, 
                                             pin_memory=True, 
                                             drop_last=False)
        self.print_fn("Create data loaders!")
        return loader_dict

    def train(self):
        """
        train function
        """
        self.model.train()
        self.call_hook("before_run")

        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            
            # prevent the training iterations exceed args.num_train_iter
            if self.it >= self.num_train_iter:
                break
            
            self.call_hook("before_train_epoch")
            

            for data in self.loader_dict['train']:
                # prevent the training iterations exceed args.num_train_iter
                if self.it >= self.num_train_iter:
                    break
                
                self.call_hook("before_train_step")
                self.out_dict, self.log_dict = self.train_step(**self.process_batch(**data))
                self.call_hook("after_train_step")
                self.it += 1
            
            self.call_hook("after_train_epoch")

        self.call_hook("after_run")
    

    def train_step(self, x_bag_w, x_bag_s, y_bag_conf_diff):
        
        bag_batch_size = x_bag_w.shape[0]
        
        x_ins_w = x_bag_w.view(-1, *x_bag_w.shape[2:])
        x_ins_s = x_bag_s.view(-1, *x_bag_s.shape[2:])
        
        # forward model
        inputs = torch.cat((x_ins_w, x_ins_s))
        outputs = self.model(inputs)
        logits_x_ins_w, logits_x_ins_s = outputs.chunk(2)
        logits_x_bag_w = logits_x_ins_w.view(bag_batch_size, 2, -1)
        logits_x_bag_s = logits_x_ins_s.view(bag_batch_size, 2, -1)
        
        # forward-backward
        probs_x_bag_w = logits_x_bag_w.softmax(dim=-1)
        pseudo_probs_x_bag, count_probs_x_bag = create_conf_diff_graph(torch.log(probs_x_bag_w), y_bag_conf_diff, self.class_prior)
        
        # calculate loss
        y_bag_conf_diff = torch.abs(y_bag_conf_diff.unsqueeze(1))
        target_positive = y_bag_conf_diff > 0
        sup_targets = torch.where(target_positive, 
                                  torch.cat([(1 - y_bag_conf_diff) // 2, (1 - y_bag_conf_diff) // 2, y_bag_conf_diff, torch.zeros_like(y_bag_conf_diff)], dim=1),
                                  torch.cat([(1 - y_bag_conf_diff) // 2, (1 - y_bag_conf_diff) // 2, torch.zeros_like(y_bag_conf_diff), y_bag_conf_diff], dim=1))
        sup_loss = self.bce_loss(count_probs_x_bag.squeeze(), sup_targets, reduction='mean')
        unsup_loss = self.ce_loss(logits_x_ins_s, pseudo_probs_x_bag.reshape(-1, 2), reduction='mean')
        
        # total loss
        total_loss = sup_loss + unsup_loss
        
        out_dict = self.process_out_dict(loss=total_loss)
        log_dict = self.process_log_dict(loss=total_loss.item(),
                                         sup_loss=sup_loss.item(),
                                         unsup_loss=unsup_loss.item(), 
                                         )
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
                
                                
                loss = F.cross_entropy(logits, y, reduction='mean', ignore_index=-1)
                total_loss += loss.item() * num_batch
                
                preds = logits.argmax(dim=-1).cpu()
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
        
        # Assuming confusion_matrix is your confusion matrix tensor
        true_positives = torch.diag(confusion_matrix)
        false_negatives = confusion_matrix.sum(dim=1) - true_positives
        false_positives = confusion_matrix.sum(dim=0) - true_positives

        # Compute recall for each class
        recall_per_class = true_positives / (true_positives + false_negatives)
        precision_per_class = true_positives / (true_positives + false_positives)

        # Handle division by zero if needed
        precision_per_class[torch.isnan(precision_per_class)] = 0
        recall_per_class[torch.isnan(recall_per_class)] = 0
        
        precision = precision_per_class.mean()
        # Compute balanced accuracy
        balanced_accuracy = recall_per_class.mean()
        
        f1_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class)
        f1_per_class[torch.isnan(f1_per_class)] = 0
        
        self.print_fn('confusion matrix:\n' + np.array_str(conf.cpu().numpy()))
        if self.ema is not None:
            self.ema.restore()
        self.model.train()

        eval_dict = {eval_dest+'/loss': total_loss / total_num, eval_dest+'/top-1-acc': acc, eval_dest+'/balanced-acc': balanced_accuracy.item(), eval_dest+'/recall': balanced_accuracy.item(), eval_dest+'/precision': precision.item(), eval_dest+'/f1': f1_per_class.mean().item()}
        return eval_dict


    @staticmethod
    def get_argument():
        return [
            Argument('--target_classes', int, 9, 'positive class for positive unlabeled learning', nargs='+'),
            Argument('--neg_classes', int, None, 'negative class for positive unlabeled learning', nargs='+'),
            Argument('--num_pair_data', int, 10000, 'number of samples'),
            Argument('--class_prior', float, 0.5, 'positive samples class prior'),
            Argument('--conf_model_name', str, 'openclip_RN50_openai', 'clip model name for positive confidence learning'),
            # Argument('--std_bag_len', int, 1, 'std for bag length'),
            # Argument('--num_bags_train', int, 500, 'number of training bags'),
            # Argument('--num_bags_test', int, 20, 'number of testing bags'),
            # Argument('--balanced_bags', str2bool, False, 'bags are balanced'),
            # Argument('--multi_label', str2bool, False, 'bags has multi label'),
            
        ]