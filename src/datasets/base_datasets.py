import copy
import torch
import numpy as np 
from PIL import Image
import torchvision
from torchvision import transforms
from torchvision import datasets as vision_datasets
from torch.utils.data import Dataset 
from torch.nn.utils.rnn import pad_sequence

from .rand_aug import RandAugment


norm_mean_std_dict = {
    'cifar100': [(0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)],
    'clothing1m': [(0.6959, 0.6537, 0.6371),(0.3113, 0.3192, 0.3214)],
    'mnist': [(0.1307,), (0.3081, )],
    'fmnist': [(0.1307,), (0.3081, )],
}


def get_img_transform(img_size=32, 
                      crop_ratio=0.875, 
                      is_train=True,
                      resize='rpn',
                      autoaug='randaug',
                      rand_erase=True,
                      hflip=True,
                      norm_mean_std=[(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]):

    if not is_train:
        transform_list = []
        if resize == 'resize_crop':
            transform_list.extend([
                transforms.Resize((int(img_size / crop_ratio), int(img_size / crop_ratio))),
                transforms.CenterCrop((img_size, img_size)),
            ])
        else:
            transform_list.append(transforms.Resize((img_size, img_size)))
            
        transfrom = transforms.Compose([
            *transform_list,
            transforms.ToTensor(),
            transforms.Normalize(*norm_mean_std),
        ])
        return transfrom
    
    transform_list = []
    if resize == 'rpc':
        transform_list.append(transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)))
    elif resize == 'resize_rpc':
        transform_list.append(transforms.Resize((int(img_size / crop_ratio), int(img_size / crop_ratio))))
        transform_list.append(transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)))
    elif resize == 'resize_crop':
        transform_list.append(transforms.Resize((int(img_size / crop_ratio), int(img_size / crop_ratio))))
        transform_list.append(transforms.RandomCrop((img_size, img_size)))
    elif resize == 'resize_crop_pad':
        transform_list.append(transforms.Resize((img_size, img_size)))
        transform_list.append(transforms.RandomCrop((img_size, img_size), padding=int(img_size * (1 - crop_ratio)), padding_mode='reflect'))
    
    if hflip:
        transform_list.append(transforms.RandomHorizontalFlip())
    
    if autoaug == 'randaug':
        transform_list.append(RandAugment(3, 5))
        rand_erase = False
    elif autoaug == 'autoaug_cifar':
        transform_list.append(transforms.AutoAugment(transforms. AutoAugmentPolicy.CIFAR10))  
    elif autoaug == 'autoaug':
        transform_list.append(transforms.AutoAugment()) 
    elif autoaug is None:
        rand_erase = False
    else:
        raise NotImplementedError
    
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(*norm_mean_std),
    ])
    

    if rand_erase and autoaug != 'randaug' and autoaug is not None:
        # transform_list.append(CutoutDefault(scale=cutout))
        transform_list.append(transforms.RandomErasing())
    
    print(transform_list)
    
    transform = transforms.Compose(transform_list)
    return transform


class ImgBaseDataset(Dataset):
    def __init__(self, data_name, data, targets, is_train=True, num_classes=10, class_map=None,
                 img_size=32, crop_ratio=0.875, autoaug='randaug', resize='rpc',
                 return_target=True, return_idx=False, return_keys=['x_lb', 'y_lb']):
        super(ImgBaseDataset, self).__init__()

        self.data_name = data_name
        self.data = data 
        self.targets = targets
        self.num_classes = num_classes
        self.return_target = return_target
        self.return_keys = return_keys
        self.return_idx = return_idx
        self.class_map = class_map
    
        if self.class_map is not None and len(self.class_map) != num_classes:
            print("select data from %s" % str(self.class_map))
            selected_data = []
            selected_targets = []
            for idx in range(len(targets)):
                if targets[idx] not in self.class_map:
                    continue
                selected_data.append(data[idx])
                selected_targets.append(targets[idx])
            self.data = selected_data
            self.targets = selected_targets
        
        self.transform = get_img_transform(img_size, crop_ratio, is_train=is_train, resize=resize, autoaug=autoaug, norm_mean_std=norm_mean_std_dict.get(data_name, [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]))
    
    def __getitem__(self, index):
        data, target = self.data[index], self.targets[index]
        if isinstance(data, str):
            data = Image.open(data).convert('RGB')   
        else:
            data = Image.fromarray(data)
        
        data_aug = self.transform(data)
        
        if self.class_map is not None:
            target = self.class_map[target]
        
        if self.return_idx:
            return_items = [index, data_aug]
        else:
            return_items = [data_aug]
            
        if self.return_target:
            return_items.append(target)
            
        return_dict = {k:v for k,v in zip(self.return_keys, return_items)}
        return return_dict

    def __len__(self):
        return len(self.data)


class ImgBagDataset(ImgBaseDataset):
    def __init__(self, data_name, data, targets, is_train=True, num_classes=10, class_map=None, target_classes=[9],
                 img_size=32, crop_ratio=0.875, autoaug='randaug', resize='rpc',
                 return_target=True, 
                 return_idx=False,
                 return_keys=['x_bag', 'y_bag', 'y_ins'], 
                 aggregation='multi_ins'):
        super().__init__(data_name, data, targets, is_train, num_classes, class_map, img_size, crop_ratio, autoaug, resize, return_target, return_idx, return_keys)
        self.aggregation = aggregation
        self.transform = get_img_transform(img_size, crop_ratio, is_train=is_train, resize=resize, autoaug=autoaug, norm_mean_std=norm_mean_std_dict.get(data_name, [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]))
        self.target_classes = target_classes
        
    def __getitem__(self, index):
        bag_data, bag_ins_target = self.data[index], self.targets[index]
        
        bag_data_aug = []
        for data in bag_data:
            if isinstance(data, str):
                data = Image.open(data).convert('RGB')   
            else:
                data = Image.fromarray(data)
            data_aug = self.transform(data)
            bag_data_aug.append(data_aug.unsqueeze(0))
        bag_data_aug = torch.cat(bag_data_aug, dim=0)
        
        if self.aggregation == 'multi_ins':
            
            bag_target = np.zeros((min(len(self.target_classes) + 1, self.num_classes), ), dtype=np.float32)
            for target_class in self.target_classes:
                bag_target[self.class_map[target_class]] = np.max(bag_ins_target == self.class_map[target_class])
            if np.sum(bag_target) == 0 and len(self.target_classes) < self.num_classes:
                bag_target[0] = 1
                
        elif self.aggregation == 'proportion':
            
            bag_target = np.zeros((min(len(self.target_classes) + 1, self.num_classes), ), dtype=np.float32)
            for target_class in self.target_classes:
                bag_target[self.class_map[target_class]] = np.sum((bag_ins_target == self.class_map[target_class]).astype(np.float32))
            # bag_target[0] = 1 - np.sum(bag_target[1:])
            if len(self.target_classes) < self.num_classes:
                bag_target[0] = len(bag_ins_target) - np.sum(bag_target[1:])
        
        elif self.aggregation == 'sim_dsim_ulb':
            
            bag_target = (bag_ins_target[0] == bag_ins_target[1]).astype(np.int64)
        
        elif self.aggregation == 'pair_comp':
            
            if bag_ins_target[0] == 1 and bag_ins_target[1] == 1:
                bag_target = 0
            elif bag_ins_target[0] == 1 and bag_ins_target[1] == 0:
                bag_target = 1
            elif bag_ins_target[0] == 0 and bag_ins_target[1] == 0:
                bag_target = 2
        
        elif self.aggregation == 'sim_conf' or self.aggregation == 'conf_diff':
            
            bag_target = bag_ins_target
        
        if self.return_idx:
            return_items = [index, bag_data_aug]
        else:
            return_items = [bag_data_aug]
        
        if self.return_target:
            return_items.extend([bag_target, bag_ins_target])
            
        return_dict = {k:v for k,v in zip(self.return_keys, return_items)}
        return return_dict



class ImgTwoViewBaseDataset(ImgBaseDataset):
    def __init__(self, data_name, data, targets, is_train=True, num_classes=10, class_map=None,
                 img_size=32, crop_ratio=0.875, autoaug='randaug', resize='rpc',
                 return_target=True, 
                 return_idx=False,
                 return_keys=['x_ulb_w', 'x_ulb_s', 'y_ulb']):
        super().__init__(data_name, data, targets, is_train, num_classes, class_map, img_size, crop_ratio, None, resize, return_target, return_idx, return_keys)
        self.strong_transform = get_img_transform(img_size, crop_ratio, is_train=is_train, resize=resize, autoaug=autoaug, norm_mean_std=norm_mean_std_dict.get(data_name, [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]))
    
    def __getitem__(self, index):
        data, target = self.data[index], self.targets[index]
        if isinstance(data, str):
            data = Image.open(data).convert('RGB')   
        else:
            data = Image.fromarray(data)
        data_aug_w = self.transform(data)
        data_aug_s = self.strong_transform(data)
        if self.class_map is not None:
            target = self.class_map[target]
        
        if self.return_idx:
            return_items = [index, data_aug_w, data_aug_s]
        else:
            return_items = [data_aug_w, data_aug_s]
            
        if self.return_target:
            return_items.append(target)
            
        return_dict = {k:v for k,v in zip(self.return_keys, return_items)}
        return return_dict

class ImgThreeViewBaseDataset(ImgTwoViewBaseDataset):
    def __init__(self, data_name, data, targets, is_train=True, num_classes=10, class_map=None,
                 img_size=32, crop_ratio=0.875, autoaug='randaug', resize='rpc',
                 return_target=True, 
                 return_idx=False,
                 return_keys=['x_ulb_w', 'x_ulb_s', 'x_ulb_s_', 'y_ulb']):
        super().__init__(data_name, data, targets, is_train, num_classes, class_map, img_size, crop_ratio, None, resize, return_target, return_idx, return_keys)
        self.strong_transform = get_img_transform(img_size, crop_ratio, is_train=is_train, resize=resize, autoaug=autoaug, norm_mean_std=norm_mean_std_dict.get(data_name, [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]))
    
    def __getitem__(self, index):
        data, target = self.data[index], self.targets[index]
        if isinstance(data, str):
            data = Image.open(data).convert('RGB')   
        else:
            data = Image.fromarray(data)
        data_aug_w = self.transform(data)
        data_aug_s = self.strong_transform(data)
        data_aug_s_ = self.strong_transform(data)
        if self.class_map is not None:
            target = self.class_map[target]
        
        if self.return_idx:
            return_items = [index, data_aug_w, data_aug_s, data_aug_s_]
        else:
            return_items = [data_aug_w, data_aug_s, data_aug_s_]
            
        if self.return_target:
            return_items.append(target)
            
        return_dict = {k:v for k,v in zip(self.return_keys, return_items)}
        return return_dict

    

class ImageTwoViewBagDataset(ImgTwoViewBaseDataset):
    def __init__(self, data_name, data, targets, is_train=True, num_classes=10, class_map=None, target_classes=[9],
                 img_size=32, crop_ratio=0.875, autoaug='randaug', resize='rpc',
                 return_target=False, 
                 return_idx=False,
                 return_keys=['x_bag_w', 'x_bag_s', 'y_bag', 'y_ins'], 
                 aggregation='multi_ins'):
        super().__init__(data_name, data, targets, is_train, num_classes, class_map, img_size, crop_ratio, None, resize, return_target, return_idx, return_keys)
        # self.strong_transform = get_img_transform(img_size, crop_ratio, is_train=is_train, resize=resize, autoaug=autoaug, norm_mean_std=norm_mean_std_dict.get(data_name, [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]))
        self.aggregation = aggregation
        self.transform = get_img_transform(img_size, crop_ratio, is_train=is_train, resize=resize, autoaug=autoaug, norm_mean_std=norm_mean_std_dict.get(data_name, [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]))
        self.target_classes = target_classes
        
        
    def __getitem__(self, index):
        # data (L, 1, 28, 28)
        # targets (L, )
        bag_data, bag_ins_target = self.data[index], self.targets[index]
        
        bag_data_aug = []
        bag_data_aug_s = []
        for data in bag_data:
            if isinstance(data, str):
                data = Image.open(data).convert('RGB')   
            else:
                data = Image.fromarray(data)
            data_aug = self.transform(data)
            data_aug_s = self.strong_transform(data)
            bag_data_aug.append(data_aug.unsqueeze(0))
            bag_data_aug_s.append(data_aug_s.unsqueeze(0))
        bag_data_aug = torch.cat(bag_data_aug, dim=0)
        bag_data_aug_s = torch.cat(bag_data_aug_s, dim=0)
        
        if self.aggregation == 'multi_ins':
            
            bag_target = np.zeros((min(len(self.target_classes) + 1, self.num_classes), ), dtype=np.float32)
            for target_class in self.target_classes:
                bag_target[self.class_map[target_class]] = np.max(bag_ins_target == self.class_map[target_class])
            if np.sum(bag_target) == 0 and len(self.target_classes) < self.num_classes:
                bag_target[0] = 1
                
        elif self.aggregation == 'proportion':
            
            bag_target = np.zeros((min(len(self.target_classes) + 1, self.num_classes), ), dtype=np.float32)
            for target_class in self.target_classes:
                bag_target[self.class_map[target_class]] = np.sum((bag_ins_target == self.class_map[target_class]).astype(np.float32))
            # bag_target[0] = 1 - np.sum(bag_target[1:])
            if len(self.target_classes) < self.num_classes:
                bag_target[0] = len(bag_ins_target) - np.sum(bag_target[1:])

        elif self.aggregation == 'sim_dsim_ulb':
            
            bag_target = (bag_ins_target[0] == bag_ins_target[1]).astype(np.int64)
        
        elif self.aggregation == 'pair_comp':
            
            if bag_ins_target[0] == 1 and bag_ins_target[1] == 1:
                bag_target = 0
            elif bag_ins_target[0] == 1 and bag_ins_target[1] == 0:
                bag_target = 1
            elif bag_ins_target[0] == 0 and bag_ins_target[1] == 0:
                bag_target = 2

        elif self.aggregation == 'sim_conf' or self.aggregation == 'conf_diff':
            
            bag_target = bag_ins_target

        if self.return_idx:
            return_items = [index, bag_data_aug, bag_data_aug_s]
        else:
            return_items = [bag_data_aug, bag_data_aug_s]

        if self.return_target:
            return_items.extend([bag_target, bag_ins_target])
        
        return_dict = {k:v for k,v in zip(self.return_keys, return_items)}
        return return_dict


def bag_collate_fn(data_list):
    
    keys = list(data_list[0].keys())
    
    # fetch data
    batch_data = {k:[] for k in keys}
    for data_dict in data_list:
        for key in keys:
            batch_data[key].append(data_dict[key])
    
    # pad x sequences
    padded_batch_data = {k:[] for k in keys}
    for key in keys:
        if key in ['x_bag', 'x_bag_w', 'x_bag_s']:
            padded_batch_data[key] = pad_sequence(batch_data[key], batch_first=True)
            lenghts = torch.LongTensor(np.array([t.shape[0] for t in batch_data[key]]))
            padded_batch_data['x_bag_len'] = lenghts
        elif key == 'y_ins':
            y_ins = batch_data[key]
            y_ins = [torch.from_numpy(y).long() for y in y_ins]
            padded_batch_data['y_ins'] = pad_sequence(y_ins, batch_first=True)
        else:
            padded_batch_data[key] = torch.from_numpy(np.stack(batch_data[key], axis=0))

    return padded_batch_data
    
        