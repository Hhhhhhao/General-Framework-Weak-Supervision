import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image 

from src.nets import get_model
from src.datasets.base_datasets import get_img_transform, norm_mean_std_dict
from src.datasets.base_data import get_data



class CustomDataset(Dataset):
    def __init__(self, samples, targets, transform=None):
        super().__init__()
        self.samples = samples
        self.targets = targets
        self.transform = transform
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        if isinstance(sample, str):
            sample = Image.open(sample).convert('RGB')   
        else:
            sample = Image.fromarray(sample)
        target = self.targets[index]
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample, target   


def load_data_loader(samples, targets, dataset_name, is_train=True):
    
    if dataset_name in ['mnist', 'fmnist', 'cifar10', 'cifar100', 'svhn']:
        if dataset_name in ['mnist', 'fmnist']:
            img_size = 28
            crop_ratio = 1.0
        else:
            img_size = 32
            crop_ratio = 0.875
        batch_size = 128
        resize = 'resize_crop'
        autoaug = None 
        if dataset_name == 'cifar100':
            num_classes = 100
        else:
            num_classes = 10
    elif dataset_name == 'stl10':
        img_size = 96
        crop_ratio = 0.875
        batch_size = 64
        resize = 'rpc'
        autoaug = 'randaug'
        num_classes = 10
    else:
        img_size = 224
        crop_ratio = 0.875
        batch_size = 32
        resize = 'rpc'
        autoaug = 'randaug'
        num_classes = 100
        
    transform = get_img_transform(img_size=img_size, crop_ratio=crop_ratio, is_train=True, resize=resize, autoaug=autoaug, norm_mean_std=norm_mean_std_dict.get(dataset_name, [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]))
    dataset = CustomDataset(samples, targets, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True if is_train else False, pin_memory=True, drop_last=True if is_train else False)
    return loader




def train_model(model, optimizer, scheduler, train_loader, val_loader, epochs=10):
    
    loss_fn = torch.nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        
        for i, (samples, targets) in enumerate(train_loader):
            samples, targets = samples.cuda(), targets.cuda()
            logits = model(samples)
            loss = loss_fn(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if i % 100 == 0:
                print('epoch: {}, iter: {}, loss: {}'.format(epoch, i, loss.item()))
        
        preds_list = []
        targets_list = []
        with torch.no_grad():
            for i, (samples, targets) in enumerate(val_loader):
                samples, targets = samples.cuda(), targets.cuda()
                logits = model(samples)
                preds = torch.argmax(logits, dim=1)
                preds_list.append(preds.cpu())
                targets_list.append(targets.cpu())
        preds_list = torch.cat(preds_list, dim=0)
        targets_list = torch.cat(targets_list, dim=0)
        correct = (preds_list == targets_list).sum()
        acc = correct / len(targets_list)
        print('epoch: {}, val acc: {}'.format(epoch, acc))



def get_confidence(model, loader):
    model.eval()
    
    confidence_list = []
    with torch.no_grad():
        for samples, targets in loader:
            
            logits = model(samples.cuda())
            probs = torch.softmax(logits, dim=1)
            
            confidence_list.append(probs.cpu().numpy())
    
    confidence_list = np.concatenate(confidence_list, axis=0)
    return confidence_list
            



def train_eval(model_name, dataset_name, test_samples, test_targets):
    
    train_samples, train_targets, val_samples, val_targets, _ = get_data(data_dir='./data', data_name=dataset_name)
    train_loader = load_data_loader(train_samples, train_targets, dataset_name, is_train=True)
    val_loader = load_data_loader(val_samples, val_targets, dataset_name, is_train=False)
    test_loader = load_data_loader(test_samples, test_targets, dataset_name, is_train=False)
    model = get_model(model_name, num_classes=len(np.unique(train_targets)))
    model = model.to('cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
    
    if dataset_name in ['mnist', 'fmnist']:
        epochs = 5
    elif dataset_name in ['cifar10', 'stl10']:
        epochs = 25
    elif dataset_name in ['svhn']:
        epochs = 10
    else:
        epochs = 20
    
    # train
    train_model(model, optimizer, scheduler, train_loader, val_loader, epochs)
    
    
    sample_confidence = get_confidence(model, test_loader)
    
    return sample_confidence
    