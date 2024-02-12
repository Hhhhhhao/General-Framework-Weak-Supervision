import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.core.algorithmbase import AlgorithmBase
from src.core.utils import Argument, str2bool, get_optimizer, send_model_cuda
from src.core.criterions import CELoss
from src.core.hooks import CheckpointHook, TimerHook, LoggingHook, DistSamplerSeedHook, ParamUpdateHook, EvaluationHook, EMAHook
from src.datasets import get_semisup_labels, get_partial_labels, get_partial_noisy_labels, get_data, get_dataloader, ImgBaseDataset, ImgTwoViewBaseDataset


class NoiseMatrixLayer(torch.nn.Module):
    def __init__(self, num_classes, scale=1.0):
        super().__init__()
        self.num_classes = num_classes
        
        self.noise_layer = nn.Linear(self.num_classes, self.num_classes, bias=False)
        # initialization
        self.noise_layer.weight.data.copy_(torch.eye(self.num_classes))
        
        init_noise_matrix = torch.eye(self.num_classes) 
        self.noise_layer.weight.data.copy_(init_noise_matrix)
        
        self.eye = torch.eye(self.num_classes).cuda()
        self.scale = scale
    
    def forward(self, x):
        noise_matrix = self.noise_layer(self.eye)
        # noise_matrix = noise_matrix ** 2
        noise_matrix = F.normalize(noise_matrix, dim=0)
        noise_matrix = F.normalize(noise_matrix, dim=1)
        return noise_matrix * self.scale

class NoiseParamUpdateHook(ParamUpdateHook):
    def before_train_step(self, algorithm):
        if hasattr(algorithm, 'start_run'):
            torch.cuda.synchronize()
            algorithm.start_run.record()

    # call after each train_step to update parameters
    def after_train_step(self, algorithm):
        
        loss = algorithm.out_dict['loss']
        # algorithm.optimizer.zero_grad()
        # update parameters
        if algorithm.use_amp:
            raise NotImplementedError
        else:
            loss.backward()
            if (algorithm.clip_grad > 0):
                torch.nn.utils.clip_grad_norm_(algorithm.model.parameters(), algorithm.clip_grad)
            algorithm.optimizer.step()
            algorithm.optimizer_noise.step()

        algorithm.model.zero_grad()
        algorithm.noise_model.zero_grad()
        
        if algorithm.scheduler is not None:
            algorithm.scheduler.step()

        if hasattr(algorithm, 'end_run'):
            algorithm.end_run.record()
            torch.cuda.synchronize()
            algorithm.log_dict['train/run_time'] = algorithm.start_run.elapsed_time(algorithm.end_run) / 1000.



class ImpPartialNoisyUnlabeledLearning(AlgorithmBase):
    def __init__(self, args, tb_log=None, logger=None, **kwargs):
        self.init(args)
        super().__init__(args, tb_log, logger, **kwargs)
        
        self.ce_loss = CELoss()
    
    def init(self, args):
        # extra arguments
        self.average_entropy_loss = args.average_entropy_loss
        self.num_labels = args.num_labels
        self.partial_ratio = args.partial_ratio
        self.noise_ratio = args.noise_ratio 
        self.noise_matrix_scale = args.noise_matrix_scale
    
        # initialize distribution alignment 
        self.ema_p = 0.999
        self.p_hat = torch.ones((args.num_classes, )) / args.num_classes
        self.p_hat = self.p_hat.cuda()
    
        # initialize noise matrix layer
        self.noise_model = NoiseMatrixLayer(num_classes=args.num_classes, scale=self.noise_matrix_scale)
        self.noise_model = send_model_cuda(args, self.noise_model)
        self.optimizer_noise = torch.optim.SGD(self.noise_model.parameters(), lr=args.lr, weight_decay=0, momentum=0)    

    def set_hooks(self):
        # parameter update hook is called inside each train_step
        self.register_hook(NoiseParamUpdateHook(), None, "HIGHEST")
        if self.ema_model is not None:
            self.register_hook(EMAHook(), None, "HIGH")
        self.register_hook(EvaluationHook(), None, "HIGH")
        self.register_hook(CheckpointHook(), None, "HIGH")
        self.register_hook(DistSamplerSeedHook(), None, "NORMAL")
        self.register_hook(TimerHook(), None, "LOW")
        self.register_hook(LoggingHook(), None, "LOWEST")

    def set_optimizer(self):
        """
        set optimizer for algorithm
        """
        self.print_fn("Create optimizer and scheduler")
        optimizer = get_optimizer(self.model, self.args.optim, self.args.lr, self.args.momentum, self.args.weight_decay, self.args.layer_decay, nesterov=False, bn_wd_skip=False)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_train_iter, eta_min=1e-4)
        return optimizer, scheduler


    def set_dataset(self):
        # get initial data
        train_data, train_targets, test_data, test_targets, extra_data = get_data(self.args.data_dir, self.args.dataset)

        # make labeled and unlabeled data
        lb_index, lb_train_data, lb_train_targets, ulb_index, ulb_train_data, ulb_train_targets \
            = get_semisup_labels(train_data, train_targets, self.num_classes, self.args.num_labels, self.args.include_lb_to_ulb)
        self.print_fn("labeled data: {}, unlabeled data {}".format(len(lb_index), len(ulb_index)))
        
        # make partial labels on labeled data
        lb_train_data, lb_train_partial_targets = get_partial_labels(lb_train_data, lb_train_targets, self.num_classes, self.args.partial_ratio)
        
        # make noise labels on partial labels
        if self.noise_ratio > 0:
            lb_train_partial_noisy_targets = get_partial_noisy_labels(lb_train_targets, lb_train_partial_targets, self.args.noise_ratio)
        else:
            lb_train_partial_noisy_targets = lb_train_partial_targets
        
        # TODO: change here
        # determine the resize methods
        if self.args.dataset in ['cifar10', 'cifar100']:
            resize = 'resize_crop_pad'
        elif self.args.dataset in ['stl10', 'svhn']:
            resize = 'resize_crop'
        test_resize = 'resize'
        
        # make dataset
        train_lb_dataset = ImgTwoViewBaseDataset(self.args.dataset, lb_train_data, lb_train_partial_noisy_targets, 
                                                 num_classes=self.num_classes, is_train=True,
                                                 img_size=self.args.img_size, crop_ratio=self.args.crop_ratio,
                                                 autoaug='randaug', resize=resize, return_target=True,
                                                 return_keys=['x_lb_w', 'x_lb_s', 'y_lb'])
        if extra_data is not None:
            ulb_train_data = np.concatenate([ulb_train_data, extra_data], axis=0)
            if ulb_train_targets.ndim == 1:
                ulb_train_targets = np.concatenate([ulb_train_targets, -1 * np.ones((extra_data.shape[0],))], axis=0)
            else:
                ulb_train_targets = np.concatenate([ulb_train_targets, -1 * np.ones((extra_data.shape[0], ulb_train_targets.shape[1]))], axis=0)
        train_ulb_dataset = ImgTwoViewBaseDataset(self.args.dataset, ulb_train_data, ulb_train_targets,
                                                  num_classes=self.num_classes, is_train=True,
                                                  img_size=self.args.img_size, crop_ratio=self.args.crop_ratio,
                                                  autoaug='randaug', resize=resize,
                                                  return_target=False, return_keys=['x_ulb_w', 'x_ulb_s'])
        test_dataset = ImgBaseDataset(self.args.dataset, test_data, test_targets, 
                                      num_classes=self.num_classes, is_train=False,
                                      img_size=self.args.img_size, crop_ratio=self.args.crop_ratio, resize=test_resize,
                                      return_keys=['x', 'y'])

        self.print_fn("Create datasets!")
        return {'train_lb': train_lb_dataset, 'train_ulb': train_ulb_dataset, 'eval': test_dataset}


    def set_data_loader(self):
        loader_dict = {}

        loader_dict['train_lb'] = get_dataloader(self.dataset_dict['train_lb'], 
                                                 num_epochs=self.epochs, 
                                                 num_train_iter=self.num_train_iter, 
                                                 batch_size=self.args.batch_size, 
                                                 shuffle=True, 
                                                 num_workers=self.args.num_workers, 
                                                 pin_memory=True, 
                                                 drop_last=True,
                                                 distributed=self.args.distributed)
        loader_dict['train_ulb'] = get_dataloader(self.dataset_dict['train_ulb'], 
                                                  num_epochs=self.epochs, 
                                                  num_train_iter=self.num_train_iter, 
                                                  batch_size=int(self.args.uratio * self.args.batch_size), 
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


    def train_step(self, x_lb_w, x_lb_s, y_lb, x_ulb_w, x_ulb_s):

        num_lb = y_lb.shape[0]

        inputs = torch.cat((x_lb_w, x_lb_s, x_ulb_w, x_ulb_s))
        outputs = self.model(inputs)
        logits_x_lb_w, logits_x_lb_s = outputs[:num_lb * 2].chunk(2)
        logits_x_ulb_w, logits_x_ulb_s = outputs[2 * num_lb:].chunk(2)
        noise_matrix = self.noise_model(logits_x_lb_w)
        
        # compute observed noisy partial labels
        noise_matrix_row = noise_matrix.softmax(dim=0)
        noisy_probs_x_lb_w = torch.matmul(logits_x_lb_w.softmax(dim=-1), noise_matrix_row)
        noisy_probs_x_lb_w = noisy_probs_x_lb_w / noisy_probs_x_lb_w.sum(dim=-1, keepdims=True)
        
        # compute sup partial noisy loss
        # sup_partial_noise_loss = torch.mean(-torch.sum(y_lb * torch.log(noisy_probs_x_lb_w), dim=1) / torch.sum(y_lb, dim=1))
        sup_partial_noise_loss = torch.mean(-torch.sum(torch.log(1.0000001 - noisy_probs_x_lb_w) * (1 - y_lb), dim=1))
        
        with torch.no_grad():
            
            noise_matrix_col =  noise_matrix.softmax(dim=-1).detach()
            noise_matrix_col_all = []
            for i in range(num_lb):
                y_lb_tmp = torch.nonzero(y_lb[i], as_tuple=True)[0]
                tmp = noise_matrix_col[:, y_lb_tmp]
                tmp_prod = tmp.prod(dim=1, keepdims=True)
                # tmp = torch.log(noise_matrix_col[:, y_lb_tmp])
                # tmp_prod = torch.exp(tmp.sum(dim=1, keepdims=True))
                noise_matrix_col_all.append(tmp_prod)
            noise_matrix_col_all = torch.cat(noise_matrix_col_all, dim=1).transpose(0, 1)

            em_probs_x_lb_w_noise = logits_x_lb_w.softmax(dim=-1) * noise_matrix_col_all            
            em_probs_x_lb_w_noise = em_probs_x_lb_w_noise / (em_probs_x_lb_w_noise.sum(dim=-1, keepdims=True) + 1e-12)
            # em_probs_x_lb_w_noise = em_probs_x_lb_w_noise / em_probs_x_lb_w_noise.sum(dim=-1, keepdims=True)
           
            # unsupervised (distribution alignment)
            probs_x_ulb_w = logits_x_ulb_w.softmax(dim=-1)
            self.p_hat = self.ema_p * self.p_hat + (1 - self.ema_p) * probs_x_ulb_w.mean(dim=0)
            probs_x_ulb_w = probs_x_ulb_w / self.p_hat
            em_probs_x_ulb_w = probs_x_ulb_w / probs_x_ulb_w.sum(dim=-1, keepdim=True)
            pseudo_y_ulb = em_probs_x_ulb_w
        
        # compute noisy partial em loss 
        em_partial_noise_loss = torch.mean(-torch.sum(em_probs_x_lb_w_noise * torch.log_softmax(logits_x_lb_s, dim=-1), dim=-1), dim=-1)
        
        # compute consistency loss
        unsup_loss = self.ce_loss(logits_x_ulb_s, pseudo_y_ulb, reduction='mean')
        
        total_loss = em_partial_noise_loss + sup_partial_noise_loss + unsup_loss # + em_noise_loss
        
        # computer average entropy loss
        if self.average_entropy_loss:
            avg_prediction = torch.mean(logits_x_lb_w.softmax(dim=-1), dim=0)
            prior_distr = 1.0/self.num_classes * torch.ones_like(avg_prediction)
            avg_prediction = torch.clamp(avg_prediction, min = 1e-6, max = 1.0)
            balance_kl =  torch.mean(-(prior_distr * torch.log(avg_prediction)).sum(dim=0))
            entropy_loss = 0.1 * balance_kl
            total_loss += entropy_loss


        out_dict = self.process_out_dict(loss=total_loss)
        log_dict = self.process_log_dict(sup_partial_noise_loss=sup_partial_noise_loss.item(), 
                                         em_partial_noise_loss=em_partial_noise_loss.item(), 
                                         unsup_loss=unsup_loss.item(),
                                         loss=total_loss.item())
        
        return out_dict, log_dict

    def get_save_dict(self):
        save_dict = super().get_save_dict()
        # additional saving arguments
        save_dict['p_hat'] = self.p_hat.cpu()
        save_dict['noise_model'] = self.noise_model.state_dict()
        save_dict['optimizer_noise'] = self.optimizer_noise.state_dict()
        return save_dict


    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.p_hat = checkpoint['p_hat'].cuda(self.args.gpu)
        self.noise_model.load_state_dict(checkpoint['noise_model'])
        self.optimizer_noise.load_state_dict(checkpoint['optimizer_noise'])
        self.print_fn("additional parameter loaded")
        return checkpoint


    @staticmethod
    def get_argument():
        return [
            Argument('--average_entropy_loss', str2bool, True, 'use entropy loss'),
            # semisupervised arguments
            Argument('--num_labels', int, 40, 'number of labels used in semi-supervised learning'),
            Argument('--uratio', int, 7, 'ratio of unlabeled batch size to labeled batch size'),
            Argument('--include_lb_to_ulb', str2bool, True, 'flag of adding labeled data into unlabeled data'),
            # partial label arguments
            Argument('--partial_ratio', float, 0.1, 'ambiguity level (q) in partial label learning'),
            # noisy arguments
            Argument('--noise_ratio', float, 0.1, 'noise ratio for noisy label learning'),
            Argument('--noise_matrix_scale', float, 1.0, 'scale for noise matrix'),
        ]