

import torch 

from src.algorithms.pos_ulb.upu import UnbiasedPositiveUnLabeledLearning
from src.datasets import ImgBaseDataset, get_dataloader
from src.core.hooks import Hook


class KeepSampleHook(Hook):
    @torch.no_grad()
    def before_train_epoch(self, algorithm):
        algorithm.model.eval()
        
        all_probs = []
        all_idxs = []
        for data_batch in algorithm.loader_dict['eval_ulb']:
            x_idx = data_batch['idx_ulb']
            x = data_batch['x_ulb']

            logits = algorithm.model(x)
            probs = logits.softmax(dim=-1)
            all_probs.append(probs.detach())
            all_idxs.append(x_idx.detach())
        algorithm.model.train()
        
        # TODO: bug here
        all_probs = torch.cat(all_probs, dim=0)[:, 0]
        all_idxs = torch.cat(all_idxs, dim=0)
        sorted_idx = torch.argsort(all_probs, descending=True)
        keep_samples = torch.ones_like(all_probs)
        keep_samples[sorted_idx[algorithm.num_ulb_data - int(algorithm.class_prior * algorithm.num_ulb_data):]] = 0
        
        algorithm.keep_samples = keep_samples



class CVIRPositiveUnlabeled(UnbiasedPositiveUnLabeledLearning):
    def __init__(self, args, tb_log=None, logger=None, **kwargs):
        super().__init__(args, tb_log, logger, **kwargs)

    def set_dataset(self): 
        if self.args.dataset in ['mnist', 'fmnist']:
            resize = 'resize'
            test_resize = 'resize'
        elif self.args.dataset in ['cifar10', 'svhn', 'cifar100']:
            resize = 'resize_crop_pad'
            test_resize = 'resize'
        elif self.args.dataset in ['stl10']:
            resize = 'resize_crop'
        elif self.args.dataset in ['imagenet1k', 'imagenet100']:
            resize = 'rpc'
        dataset_dict = super().set_dataset()
        train_ulb_dataset = dataset_dict['train_ulb']
        train_ulb_dataset =  ImgBaseDataset(self.args.dataset, train_ulb_dataset.data, train_ulb_dataset.targets, 
                                          num_classes=self.num_classes, class_map=self.class_map, is_train=True,
                                          img_size=self.args.img_size, crop_ratio=self.args.crop_ratio,
                                          autoaug=None, resize=resize,
                                          return_idx=True,
                                          return_target=False,
                                          return_keys=['idx_ulb', 'x_ulb'])
        dataset_dict['train_ulb'] = train_ulb_dataset
        return dataset_dict

    def set_data_loader(self):
        loader_dict = super().set_data_loader()
        loader_dict['eval_ulb'] = get_dataloader(self.dataset_dict['train_ulb'], 
                                             num_epochs=self.epochs, 
                                             batch_size=self.args.eval_batch_size, 
                                             shuffle=False, 
                                             num_workers=self.args.num_workers, 
                                             pin_memory=True, 
                                             drop_last=False)
        return loader_dict
    

    def set_hooks(self):
        self.register_hook(KeepSampleHook())
        super().set_hooks()

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
            

            for data_lb, data_ulb in zip(self.loader_dict['train_lb'],
                                         self.loader_dict['train_ulb']):
                # prevent the training iterations exceed args.num_train_iter
                if self.it >= self.num_train_iter:
                    break
                
                self.call_hook("before_train_step")
                self.out_dict, self.log_dict = self.train_step(**self.process_batch(**data_lb, **data_ulb))
                
                if len(self.out_dict) == 0:
                    continue
                
                self.call_hook("after_train_step")
                self.it += 1
            
            self.call_hook("after_train_epoch")

        self.call_hook("after_run")

    def train_step(self, x_lb, y_lb, idx_ulb, x_ulb):

        num_lb = y_lb.shape[0]
        keep_ulb = torch.nonzero(self.keep_samples[idx_ulb] == 1).squeeze()
        
        if len(keep_ulb) == 0:
            return {}, {}
        else:
            x_ulb = x_ulb[keep_ulb]
            
            
            # forward model
            inputs = torch.cat((x_lb, x_ulb))
            outputs = self.model(inputs)
            logits_x_lb = outputs[:num_lb]
            logits_x_ulb = outputs[num_lb:]
            
            
            # compute loss
            loss_pos = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
            loss_neg = self.ce_loss(logits_x_ulb, torch.zeros((x_ulb.shape[0], ), device=x_lb.device, dtype=torch.long), reduction='mean')

            # total loss
            total_loss = 0.5 * (loss_pos + loss_neg)
            
            out_dict = self.process_out_dict(loss=total_loss)
            log_dict = self.process_log_dict(loss=total_loss.item(),  loss_pos=loss_pos.item(), loss_neg=loss_neg.item())
            return out_dict, log_dict