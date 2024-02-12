import os
import argparse
from tqdm import tqdm
import math

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image

import clip
import torchvision.transforms as transforms
from open_clip import create_model_and_transforms
from timm.models import create_model
import src.datasets.utils.templates as templates


class TimmWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        if hasattr(model, 'fc'):
            self.num_features = model.fc.in_features
            self.fc = self.model.fc
        elif hasattr(model, 'head'):
            self.num_features = model.head.in_features
            self.fc = self.model.head
        else:
            self.num_features = model.classifier.in_features
            self.fc = self.model.classifier
    
    def encode_image(self, inputs):
        features = self.model.forward_features(inputs)
        features = self.model.forward_head(features, pre_logits=True)
        return features


def timm_load(model):
    model = create_model(model, pretrained=True)
    model_cfg = model.pretrained_cfg
    input_size = model_cfg['input_size'][-1]
    image_size = int(math.ceil(input_size / model_cfg['crop_pct']))
    mean = model_cfg['mean']
    std = model_cfg['std']
    interpolation = model_cfg['interpolation']
    if interpolation == 'bicubic':
        inter_mode = transforms.InterpolationMode.BICUBIC
    elif interpolation == 'bilinear':
        inter_mode = transforms.InterpolationMode.BILINEAR
    else:
        raise NotImplementedError
    
    # warp model as feature extractor
    model = TimmWrapper(model)

    
    scale = (0.8, 1.0)

    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(input_size, interpolation=inter_mode, scale=scale), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(mean), std=list(std))])
    transform_val = transforms.Compose([
            transforms.Resize(image_size, interpolation=inter_mode),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(mean), std=list(std))])
    return model, transform_train, transform_val


class SampleDataset(Dataset):
    def __init__(self, samples, transforms):
        super().__init__()
        self.samples = samples 
        self.transforms = transforms
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        sample = Image.fromarray(sample)
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample


class ImageEncoder(torch.nn.Module):
    def __init__(self, model_source, model, device='cuda', keep_lang=False):
        super().__init__()

        if model_source == 'clip':
            self.model, self.train_preprocess, self.val_preprocess = clip.load(
                model, device, jit=False)
        elif model_source == 'openclip':
            model_name = model.split('_')[0]
            pretrained = '_'.join(model.split('_')[1:])
            self.model, self.train_preprocess, self.val_preprocess = create_model_and_transforms(
                model_name,
                pretrained,
                device='cuda'
            )
        elif model_source == 'timm':
            self.model, self.train_preprocess, self.val_preprocess = timm_load(model)
        else:
            raise NotImplementedError

        if not keep_lang and hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images):
        assert self.model is not None
        return self.model.encode_image(images)



class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)


class ImageClassifier(torch.nn.Module):
    def __init__(self, image_encoder, classification_head, process_images=True, return_mid_feats=False):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_head = classification_head
        self.process_images = process_images
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess
        self.return_mid_feats = return_mid_feats

    def forward(self, inputs):
        if self.process_images:
            inputs = self.image_encoder(inputs)
        
        outputs = self.classification_head(inputs)
        
        return outputs

    def forward_encoder(self, inputs):
        if self.process_images:
            # TODO: make this part better
            if self.return_mid_feats:
                inputs = self.image_encoder(inputs, retrun_mid_feats=True)
            else:
                inputs = self.image_encoder(inputs)
        return inputs

    def forward_cls_head(self, inputs):
        return self.classification_head(inputs)



def get_zeroshot_classifier(template, dataset_classnames, clip_model, device='cuda'):
    template = getattr(templates, template)
    logit_scale = clip_model.logit_scale
    clip_model.eval()
    clip_model.to(device)

    print('Getting zeroshot weights.')
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(dataset_classnames):
            texts = []
            for t in template:
                texts.append(t(classname))
            texts = clip.tokenize(texts).to(device) # tokenize
            embeddings = clip_model.encode_text(texts) # embed with text encoder
            embeddings /= embeddings.norm(dim=-1, keepdim=True)

            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm()

            zeroshot_weights.append(embeddings)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device)
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)

        zeroshot_weights *= logit_scale.exp()
        
        zeroshot_weights = zeroshot_weights.squeeze().float()
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)

    classification_head = ClassificationHead(normalize=True, weights=zeroshot_weights)

    return classification_head


@torch.no_grad()
def zeroshot_eval(model_source, model, template, data_classnames, samples, targets):
    image_encoder = ImageEncoder(model_source, model, device='cuda', keep_lang=True)
    classification_head = get_zeroshot_classifier(template, data_classnames, image_encoder.model)
    delattr(image_encoder.model, 'transformer')
    classifier = ImageClassifier(image_encoder, classification_head, process_images=True)
    classifier = classifier.to('cuda')
    
    # get dataloader
    dataset = SampleDataset(samples, classifier.val_preprocess)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, drop_last=False, num_workers=4)
    
    # get embeddings and confidence
    feat_list = []
    probs_list = []
    preds_list = []
    for batch_data in tqdm(dataloader, total=len(dataloader)):
        
        batch_data = batch_data.to('cuda')
        feats = classifier.forward_encoder(batch_data)
        logits = classifier.forward_cls_head(feats)
        probs = logits.softmax(dim=-1)
        
        feat_list.append(feats.cpu())
        probs_list.append(probs.cpu())
        preds_list.append(probs.argmax(dim=-1).cpu())
    
    feat_list = torch.cat(feat_list, dim=0).numpy()
    probs_list = torch.cat(probs_list, dim=0).numpy()
    preds_list = torch.cat(preds_list, dim=0).numpy()
    acc = (preds_list == targets).mean()
    print(f"Model {model_source}_{model} with template {template} has zero-shot accuracy {acc}")
    
    return probs_list  