

import torch 
import torch.nn.functional as F
from torch.nn.utils.rnn import unpad_sequence, pad_sequence
import numpy as np
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src.core.algorithmbase import AlgorithmBase
from src.core.utils import Argument, str2bool, get_optimizer
from src.core.criterions import CELoss, BCELoss
from src.nets import get_model
from src.core.nfa import create_proportion_graph


from src.datasets import get_proportion_bags_labels, get_data, get_dataloader, bag_collate_fn, ImgBagDataset, ImageTwoViewBagDataset, ImgBaseDataset


class ImpreciseProportionLabelLearning(AlgorithmBase):
    def __init__(self, args, tb_log=None, logger=None, **kwargs):

        self.init(args)
        super().__init__(args, tb_log, logger, **kwargs)
        
        # set number train iterations
        self.num_train_iter = self.epochs * len(self.loader_dict['train'])
        self.num_eval_iter = len(self.loader_dict['train'])
        
        self.ce_loss = CELoss()
        self.bce_loss = BCELoss()
        
        
        self.p_model =  torch.ones((self.output_classes, ))/ self.output_classes
        self.p_model =  self.p_model.to(self.gpu)
        self.m = 0.999


    def init(self, args):        
        self.target_classes = args.target_classes
        if isinstance(self.target_classes, int):
            self.target_classes = [self.target_classes]
        self.target_classes = sorted(self.target_classes)
        # create class maps
        if len(self.target_classes) == args.num_classes:
            self.class_map = {i: i for i in range(args.num_classes)}
            self.output_classes = args.num_classes
        else:
            self.class_map = {i: 0 for i in range(args.num_classes) if i not in self.target_classes}
            for i, target_class in enumerate(self.target_classes):
                self.class_map[target_class] = i + 1
            self.output_classes = len(self.target_classes) + 1
        self.cls_idx_list = np.unique(list(self.class_map.values()))
        if self.output_classes < len(self.target_classes):
            self.cls_idx_list = self.cls_idx_list[1:]
        self.mean_bag_len = args.mean_bag_len
        self.std_bag_len = args.std_bag_len 
        self.num_bags_train = args.num_bags_train

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
        ema_model = get_model(model_name=self.args.net, num_classes=self.output_classes)
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
        train_bag_data, train_bag_targets = get_proportion_bags_labels(train_data, train_targets, self.num_classes, 
                                                                      target_classes=self.target_classes, 
                                                                      class_map=self.class_map,
                                                                      mean_bag_len=self.mean_bag_len, std_bag_len=self.std_bag_len, num_bags=self.num_bags_train, 
                                                                      )
        
        # TODO: check autoaug
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
            test_resize = 'resize'
        elif self.args.dataset in ['imagenet1k', 'imagenet100']:
            resize = 'resize_rpc'
            autoaug = 'autoaug'
        test_resize = 'resize'
        
        if not self.strong_aug:
            autoaug = None
        
        # make dataset
        train_dataset = ImageTwoViewBagDataset(self.args.dataset, train_bag_data, train_bag_targets, 
                                       num_classes=self.num_classes, class_map=self.class_map, target_classes=self.target_classes,
                                       is_train=True,
                                       img_size=self.args.img_size, crop_ratio=self.args.crop_ratio,
                                       autoaug=autoaug, resize=resize,
                                       # autoaug=None, resize=resize,
                                       aggregation='proportion',
                                       return_target=True, return_keys=['x_bag_w', 'x_bag_s', 'y_bag', 'y_ins'])
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
                                              drop_last=False,
                                              distributed=self.args.distributed,
                                              collate_fn=bag_collate_fn)
        loader_dict['eval'] = get_dataloader(self.dataset_dict['eval'], 
                                             num_epochs=self.epochs, 
                                             batch_size=self.args.batch_size, 
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

    def train_step(self, x_bag_w, x_bag_s, x_bag_len, y_bag):      

        x_bag_len = x_bag_len.cpu()
        
        # get unpadded data
        unpad_x_bag_w = unpad_sequence(x_bag_w, x_bag_len, batch_first=True)
        unpad_x_bag_s = unpad_sequence(x_bag_s, x_bag_len, batch_first=True)
        
        # get output
        inputs = torch.cat(unpad_x_bag_w + unpad_x_bag_s, dim=0)
        outputs = self.model(inputs)
        logits_x_w, logits_x_s = outputs.chunk(2)
        
        # get softmax
        probs_x_w = logits_x_w.softmax(dim=-1)
        probs_x_s = logits_x_s.softmax(dim=-1)

        with torch.no_grad():
            probs = probs_x_w.detach()
            if self.p_model == None:
                self.p_model = torch.mean(probs, dim=0)
            else:
                self.p_model = self.p_model * self.m + torch.mean(probs, dim=0) * (1 - self.m)
        
        # handle multiple classes
        unsup_loss = 0.0
        sup_loss = 0.0
        for cls_idx in self.cls_idx_list:
            binary_y_bag = y_bag[:, cls_idx].unsqueeze(1)
            
            # construct binary  probs
            neg_probs_x_w = torch.cat([probs_x_w[:, idx].unsqueeze(1) for idx in range(probs_x_w.shape[1]) if idx != cls_idx], dim=1)
            neg_probs_x_w = torch.sum(neg_probs_x_w, dim=1, keepdim=True)
            pos_probs_x_w =  probs_x_w[:, cls_idx].unsqueeze(1)
            binary_probs_x_w =  torch.cat([neg_probs_x_w, pos_probs_x_w], dim=1)
            # split to batch
            binary_probs_x_w_list = binary_probs_x_w.split(x_bag_len.tolist())
            
            # forward-backward
            binary_pseudo_y_ins_list = []
            binary_probs_y_bag_list = []
            for batch_idx, probs in enumerate(binary_probs_x_w_list):
                pseudo_y_ins, probs_y_bag = create_proportion_graph(torch.log(probs), binary_y_bag[batch_idx])
                binary_pseudo_y_ins_list.append(pseudo_y_ins)
                binary_probs_y_bag_list.append(probs_y_bag.view(1, -1))
            binary_pseudo_y_ins_list = torch.cat(binary_pseudo_y_ins_list, dim=0)
            binary_probs_y_bag_list = torch.cat(binary_probs_y_bag_list, dim=0)
            
            neg_probs_x_s = torch.cat([probs_x_s[:, idx].unsqueeze(1) for idx in range(probs_x_s.shape[1]) if idx != cls_idx], dim=1)
            neg_probs_x_s = torch.sum(neg_probs_x_s, dim=1, keepdim=True)
            pos_probs_x_s =  probs_x_s[:, cls_idx].unsqueeze(1)
            binary_probs_x_s =  torch.cat([neg_probs_x_s, pos_probs_x_s], dim=1)
            
            unsup_loss += torch.sum(-binary_pseudo_y_ins_list.detach() * torch.log(binary_probs_x_s), dim=1).mean()
            
            sup_loss += - torch.log(binary_probs_y_bag_list).mean()
            
        unsup_loss = unsup_loss / len(self.class_map)
        sup_loss = sup_loss / len(self.class_map)
        total_loss = unsup_loss + sup_loss

        out_dict = self.process_out_dict(loss=total_loss)
        log_dict = self.process_log_dict(loss=total_loss.item(),  unsup_loss=unsup_loss.item(), sup_loss=sup_loss.item())
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
        y_true = []
        y_pred = []
        # y_probs = []
        y_logits = []
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
                y_true.extend(y.cpu().tolist())
                y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
                y_logits.append(logits.cpu().numpy())
                total_loss += loss.item() * num_batch
                
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_logits = np.concatenate(y_logits)
        top1 = balanced_accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        F1 = f1_score(y_true, y_pred, average='macro')

        cf_mat = confusion_matrix(y_true, y_pred, normalize='true')
        self.print_fn('confusion matrix:\n' + np.array_str(cf_mat))
        if self.ema is not None:
            self.ema.restore()
        self.model.train()

        eval_dict = {eval_dest+'/loss': total_loss / total_num, eval_dest+'/top-1-acc': top1, 
                     eval_dest+'/precision': precision, eval_dest+'/recall': recall, eval_dest+'/F1': F1}
        if return_logits:
            eval_dict[eval_dest+'/logits'] = y_logits
        return eval_dict

    @staticmethod
    def get_argument():
        return [
            Argument('--target_classes', int, 9, 'target classes for learning with proportion', nargs='+'),
            Argument('--mean_bag_len', int, 10, 'mean for bag length'),
            Argument('--std_bag_len', int, 1, 'std for bag length'),
            Argument('--num_bags_train', int, 500, 'number of training bags'),
            # Argument('--num_bags_test', int, 20, 'number of testing bags'),
            # Argument('--balanced_bags', str2bool, False, 'bags are balanced'),
            # Argument('--multi_label', str2bool, False, 'bags has multi label'),
            
        ]