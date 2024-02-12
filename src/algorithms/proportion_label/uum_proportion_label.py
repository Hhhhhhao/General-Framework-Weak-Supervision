

import torch 
from torch.nn.utils.rnn import unpad_sequence, pad_sequence
from itertools import permutations


# from src.core.hooks import Hook
from src.algorithms.proportion_label.imp_proportion_label import ImpreciseProportionLabelLearning
from src.datasets import get_proportion_bags_labels, get_data, ImgBagDataset, ImgBaseDataset, ImageTwoViewBagDataset



class LLPperm:
    def __init__(self, aggregate_labels, output_classes):
        print("generating permutations")
        self.perm = {}
        for idx, labels in enumerate(aggregate_labels):
            self.perm[idx] = torch.tensor(list(permutations(torch.from_numpy(labels))))
        self.output_classes = output_classes
        
    def generate_matrix(self, aggregate_labels):
        perm_matrix_list = []
        for label in aggregate_labels:
            perm_matrix_list.append(self.generate_matrix_single(label))
        return perm_matrix_list
    
    def generate_matrix_single(self, aggregate_label):
        perm_matrix = torch.zeros(aggregate_label.shape + (self.output_classes,)).type(torch.LongTensor)
        src = torch.ones(aggregate_label.shape + (1,)).type(torch.LongTensor)
        perm_matrix.scatter_(-1, aggregate_label.unsqueeze(-1), src)
        return perm_matrix
    
    def get_perm(self,index):
        aggregate_labels_list = []
        for i in index:
            aggregate_labels_list.append(self.perm[i.item()].unsqueeze(0))
        perm_matrix_list = self.generate_matrix(aggregate_labels_list)
        return aggregate_labels_list, perm_matrix_list



def uum_proportion_loss(prob_w, prob_s, agg_label, perm):
    p_z = prob_s # torch.cat(ts, dim=1)
    loss_z = torch.log(p_z + 1e-32)

    p_z = prob_w # torch.cat(weight, dim=1)
    
    p_y = perm * p_z.unsqueeze(dim=1)

    p_y = p_y.sum(-1).prod(-1).sum(-1)
    loss_weight = torch.ones_like(loss_z)
    for i in range(prob_w.size(1)):
        for j in range(p_z.size(-1)):
            temp = perm.clone()
            temp[agg_label[:,:,i]!=j] = 0
            temp = temp * p_z.unsqueeze(1)

            temp = temp.sum(-1).prod(-1).sum(-1)
            loss_weight[:,i,j] = temp
    loss_y = loss_weight * loss_z / (p_y.unsqueeze(-1).unsqueeze(-1) + 1e-32)
    return -loss_y.view(-1).mean()


class UUMProportionLabelLearning(ImpreciseProportionLabelLearning):
    def __init__(self, args, tb_log=None, logger=None, **kwargs):
        super().__init__(args, tb_log, logger, **kwargs)
    
    def set_dataset(self):
        # get initial data
        train_data, train_targets, test_data, test_targets, extra_data = get_data(self.args.data_dir, self.args.dataset)
        
        # make partial labels
        train_bag_data, train_bag_targets = get_proportion_bags_labels(train_data, train_targets, self.num_classes, 
                                                                      target_classes=self.target_classes, 
                                                                      class_map=self.class_map,
                                                                      mean_bag_len=self.mean_bag_len, std_bag_len=self.std_bag_len, num_bags=self.num_bags_train, 
                                                                      )
        self.llp_perm = LLPperm(train_bag_targets, self.output_classes)
        
        
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
            resize = 'rpc'
            autoaug = 'autoaug'
        test_resize = 'resize'
        
        # make dataset
        train_dataset = ImageTwoViewBagDataset(self.args.dataset, train_bag_data, train_bag_targets, 
                                       num_classes=self.num_classes, class_map=self.class_map, target_classes=self.target_classes,
                                       is_train=True,
                                       img_size=self.args.img_size, crop_ratio=self.args.crop_ratio,
                                       # autoaug=None, resize=resize,
                                       autoaug=autoaug, resize=resize,
                                       # autoaug=None, resize=resize,
                                       aggregation='proportion',
                                       return_target=True, 
                                       return_idx=True,
                                       return_keys=['x_idx', 'x_bag_w', 'x_bag_s', 'y_bag', 'y_ins'])
        test_dataset = ImgBaseDataset(self.args.dataset, test_data, test_targets, 
                                      num_classes=self.num_classes, class_map=self.class_map,
                                      is_train=False,
                                      img_size=self.args.img_size, crop_ratio=self.args.crop_ratio, resize=test_resize,
                                      return_keys=['x', 'y'])

        self.print_fn("Create datasets!")
        return {'train': train_dataset, 'eval': test_dataset}


    def train_step(self, x_idx, x_bag_w, x_bag_s, x_bag_len, y_bag):      
        batch_size = x_bag_w.shape[0]
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
        
        probs_x_w_list = probs_x_w.split(x_bag_len.tolist())
        probs_x_s_list = probs_x_s.split(x_bag_len.tolist())
        label_list, perm_matrix_list = self.llp_perm.get_perm(x_idx)
        loss = 0.0
        for batch_idx in range(batch_size):
            prob_w = probs_x_w_list[batch_idx].unsqueeze(0)
            prob_s = probs_x_s_list[batch_idx].unsqueeze(0)
            loss += uum_proportion_loss(prob_w, prob_s, label_list[batch_idx].to(prob_w.device), perm_matrix_list[batch_idx].to(prob_w.device))
        loss /= batch_size

        out_dict = self.process_out_dict(loss=loss)
        log_dict = self.process_log_dict(loss=loss.item())
        return out_dict, log_dict