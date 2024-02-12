import os
import math
import numpy as np
import random
from itertools import combinations, chain

from src.datasets.utils.zero_shot import zeroshot_eval
from src.datasets.utils.custom import train_eval
from src.datasets.utils.metadata import classnames



def get_semisup_labels(samples, targets, num_classes=10, num_labels=400, include_lb_to_ulb=True):
    samples, targets = np.array(samples), np.array(targets)

    lb_samples_per_class = [int(num_labels / num_classes)] * num_classes

    lb_idx = []
    ulb_idx = []
    
    for c in range(num_classes):
        idx = np.where(targets == c)[0]
        np.random.shuffle(idx)
        lb_idx.extend(idx[:lb_samples_per_class[c]])
        ulb_idx.extend(idx[lb_samples_per_class[c]:])
    
    if include_lb_to_ulb and len(lb_idx) < len(samples):
        ulb_idx = np.concatenate([lb_idx, ulb_idx], axis=0)
    else:
        ulb_idx = lb_idx

    lb_samples, lb_targets = samples[lb_idx], targets[lb_idx]
    ulb_samples, ulb_targets = samples[ulb_idx], targets[ulb_idx]
    return lb_idx, lb_samples, lb_targets, ulb_idx, ulb_samples, ulb_targets


def get_partial_labels(samples, targets, num_classes=10, partial_ratio=0.5):
    samples, targets = np.array(samples), np.array(targets)
    num_samples = targets.shape[0]

    partial_targets = np.zeros((num_samples, num_classes))
    partial_targets[np.arange(num_samples), targets] = 1.0

    transition_matrix =  np.eye(num_classes)
    transition_matrix[np.where(~np.eye(transition_matrix.shape[0],dtype=bool))] = partial_ratio

    random_n = np.random.uniform(0, 1, size=(num_samples, num_classes))

    for j in range(num_samples):  # for each instance
        partial_targets[j, :] = (random_n[j, :] < transition_matrix[targets[j], :]) * 1
    
    # labels are one-hot
    return samples, partial_targets


def get_partial_noisy_labels(targets, partial_targets,  noise_ratio=0.5):
    noise_partial_targets = []
    
    for idx in range(len(targets)):
        target = targets[idx]
        partial_target = partial_targets[idx]
        noise_flag = (random.uniform(0, 1) <= noise_ratio) 
        if noise_flag:
            candidate_idx = np.nonzero(1 - partial_target)[0]
            if len(candidate_idx) == 0:
                noise_partial_targets.append(partial_target)
                continue
            noise_partial_target = np.copy(partial_target)
            noise_idx = np.random.choice(candidate_idx, 1)
            noise_partial_target[noise_idx] = 1
            noise_partial_target[target] = 0
            noise_partial_targets.append(noise_partial_target)
        else:
            noise_partial_targets.append(partial_target)
    
    noise_partial_targets = np.array(noise_partial_targets)
    return noise_partial_targets

def get_sym_noisy_labels(samples, targets, num_classes=10, noise_ratio=0.5):
    samples, targets = np.array(samples), np.array(targets)

    noise_idx = []
    noisy_targets = np.copy(targets)
    indices = np.random.permutation(len(samples))
    for i, idx in enumerate(indices):
        if i < noise_ratio * len(samples):
            noise_idx.append(idx)
            noisy_targets[idx] = np.random.randint(num_classes, dtype=np.int32)

    return noise_idx, samples, noisy_targets


def get_cifar10_asym_noisy_labels(samples, targets, num_classes=10, noise_ratio=0.5):
    samples, targets = np.array(samples), np.array(targets)

    noise_idx = []
    noisy_targets = np.copy(targets)
    for i in range(num_classes):
        indices = np.where(targets == i)[0]
        np.random.shuffle(indices)
        for j, idx in enumerate(indices):
            if j < noise_ratio * len(indices):
                noise_idx.append(idx)
                # truck -> automobile
                if i == 9:
                    noisy_targets[idx] = 1
                # bird -> airplane
                elif i == 2:
                    noisy_targets[idx] = 0
                # cat -> dog
                elif i == 3:
                    noisy_targets[idx] = 5
                # dog -> cat
                elif i == 5:
                    noisy_targets[idx] = 3
                # deer -> horse
                elif i == 4:
                    noisy_targets[idx] = 7
    return noise_idx, samples, noisy_targets


def get_cifar100_asym_noisy_labels(samples, targets, num_classes=100, noise_ratio=0.5):
    p = np.eye(num_classes)
    num_superclasses = 20
    num_subclasses = 5
    samples, targets = np.array(samples), np.array(targets)


    def build_for_cifar100(size, noise):
        p = np.eye(size)
        cls1, cls2 = np.random.choice(range(size), size=2, replace=False)
        p[cls1, cls2] = noise
        p[cls2, cls1] = noise
        p[cls1, cls1] = 1.0 - noise
        p[cls2, cls2] = 1.0 - noise
        return p

    if noise_ratio > 0.0:
        for i in np.arange(num_superclasses):
            init, end = i * num_subclasses, (i+1) * num_subclasses
            p[init:end, init:end] = build_for_cifar100(num_subclasses, noise_ratio) 

        noise_idx = []
        noisy_targets = np.copy(targets)
        for idx in np.arange(noisy_targets.shape[0]):
            y = targets[idx]
            flipped = np.random.multinomial(1, p[y, :], 1)[0]
            noisy_targets[idx] = np.where(flipped == 1)[0]
            if noisy_targets[idx] != targets[idx]:
                noise_idx.append(idx)
    
    return noise_idx, samples, noisy_targets


def get_multi_ins_bags_labels(samples, targets, 
                              num_classes=10, 
                              target_classes=[3, 5, 7], 
                              class_map=None,
                              mean_bag_len=10, std_bag_len=1, num_bags=1000,
                              balanced_bags=True):
    
    assert np.max(target_classes) < num_classes, "Target classes should be less than num_classes"
    samples, targets = np.array(samples), np.array(targets)
    target_classes = sorted(target_classes)
            
    bags_list = []
    labels_list = []
    
    if len(target_classes) >= num_classes - 1:
        balanced_bags = False
        
    if balanced_bags:
    
        # define possible label combinations
        label_combinations = []
        for r in range(0, len(target_classes) + 1):
            label_combinations.extend(combinations(target_classes, r))
        # set the first combination to be the negative class
        negative_classes = tuple([i for i in range(num_classes) if i not in target_classes])
        label_combinations[0] = negative_classes
        
        # cosntruct target class mapping
        if class_map is None:
            class_map = {i: 0 for i in negative_classes}
            for i, target_class in enumerate(target_classes):
                class_map[target_class] = i + 1
                
        bags_per_combination = int(math.ceil(num_bags / len(label_combinations)))
        bags_per_combination = {comb: bags_per_combination for comb in label_combinations}
    
        # bags_per_combination = {comb: int(percentage * num_bags) for comb, percentage in zip(label_combinations, [0.1, 0.9])}
        
        for i, combination in enumerate(label_combinations):
            
            bags_count = 0
            target_bags_count = bags_per_combination[combination]
            comb_pos_cls_set = set(combination)
            if i == 0:
                comb_neg_cls_set = set([l for l in range(num_classes) if l not in comb_pos_cls_set])
            else:
                # comb_neg_cls_set = set([l for l in range(num_classes) if l in comb_pos_cls_set])
                comb_neg_cls_set = set(target_classes) - set(combination) 
            
            while bags_count < target_bags_count:
            
                bag_length = max(2, np.int(np.random.normal(mean_bag_len, std_bag_len, 1)))
                bag_indices = np.random.randint(0, len(samples), bag_length)
                labels_in_bag = targets[bag_indices]
                
                # check if labels in bag contains only pos_pos_cls_set and not comb_neg_cls_set
                labels_set = set(labels_in_bag)
                
                if not labels_set.isdisjoint(comb_neg_cls_set):
                    continue
                
                if len(comb_pos_cls_set) > len(labels_set):
                    if not (comb_pos_cls_set & labels_set):
                        continue
                else:
                    if not comb_pos_cls_set.issubset(labels_set):
                        continue
                
                # selected_indices = np.random.choice(bag_indices, bag_length, replace=False)
                bag_samples = np.squeeze(samples[bag_indices])
                labels_in_bag = np.squeeze(targets[bag_indices])
                
                # get bag_ins_label
                bag_ins_label = np.array(list(map(class_map.get, labels_in_bag.tolist())))
                
                bags_list.append(bag_samples)
                labels_list.append(bag_ins_label)
                bags_count += 1
    else:
        
        bags_count = 0
        ensure_target_classes = False
        if len(target_classes) < num_classes - 1:
            ensure_target_classes = True
        
        while bags_count < num_bags:
            bag_length = max(2, np.int(np.random.normal(mean_bag_len, std_bag_len, 1)))
            bag_indices = np.random.randint(0, len(samples), bag_length)
            labels_in_bag = targets[bag_indices]
            
            if ensure_target_classes and not (set(target_classes) & set(labels_in_bag)):
                continue
            
            bag_samples = np.squeeze(samples[bag_indices])
            labels_in_bag = np.squeeze(targets[bag_indices])
            bag_ins_label = np.array(list(map(class_map.get, labels_in_bag.tolist())))
            
            bags_list.append(bag_samples)
            labels_list.append(bag_ins_label)
            bags_count += 1
    print(labels_list)

    return bags_list, labels_list



def get_proportion_bags_labels(samples, targets, 
                               num_classes=10, 
                               target_classes=[3, 5, 7], 
                               class_map=None,
                               mean_bag_len=10, 
                               std_bag_len=1, 
                               num_bags=1000):
    
    assert np.max(target_classes) < num_classes, "Target classes should be less than num_classes"
    samples, targets = np.array(samples), np.array(targets)
    target_classes = sorted(target_classes)
    ensure_target_classes = False
    if len(target_classes) < num_classes:
        ensure_target_classes = True

    if class_map is None:
        class_map = {i: 0 for i in negative_classes}
        for i, target_class in enumerate(target_classes):
            class_map[target_class] = i + 1
    
    bags_count = 0
    bags_list = []
    labels_list = []
    
    while bags_count < num_bags:
    
        bag_length = max(2, np.int(np.random.normal(mean_bag_len, std_bag_len, 1)))
        bag_indices = np.random.randint(0, len(samples), bag_length)
        labels_in_bag = targets[bag_indices]
        
        if ensure_target_classes and not (set(target_classes) & set(labels_in_bag)):
            continue
        
        # selected_indices = np.random.choice(bag_indices, bag_length, replace=False)
        bag_samples = np.squeeze(samples[bag_indices])
        labels_in_bag = np.squeeze(targets[bag_indices])
        
        # get bag_ins_label
        bag_ins_label = np.array(list(map(class_map.get, labels_in_bag.tolist())))

        bags_list.append(bag_samples)
        labels_list.append(bag_ins_label)
        bags_count += 1

    return bags_list, labels_list


def get_pos_ulb_labels(samples, targets, 
                       num_classes=10, 
                       target_classes=[3, 5, 7], 
                       neg_classes=None,
                       num_pos_data=400, 
                       num_ulb_data=1000,
                       include_lb_to_ulb=True):
    
    assert np.max(target_classes) < num_classes, "Target classes should be less than num_classes"
    samples, targets = np.array(samples), np.array(targets)
    pos_classes = sorted(target_classes)
    if neg_classes is None or neg_classes == 'None':
        neg_classes = [i for i in range(num_classes) if i not in pos_classes]
    
    all_pos_idx = []
    all_neg_idx = []
    for c in range(num_classes):
        idx = np.where(targets == c)[0]
        if c in pos_classes:
            all_pos_idx.extend(idx)
        elif c in neg_classes:
            all_neg_idx.extend(idx)
    all_pos_idx = np.array(all_pos_idx)
    all_neg_idx = np.array(all_neg_idx)
    
    # calculate class prior
    class_prior = len(all_pos_idx) / (len(all_pos_idx) + len(all_neg_idx))
    
    # sample pos data
    pos_idx_idx = np.random.choice(len(all_pos_idx), num_pos_data, replace=False)
    pos_idx = all_pos_idx[pos_idx_idx]
    pos_samples, pos_targets = samples[pos_idx], targets[pos_idx]
    all_pos_idx = np.delete(all_pos_idx, pos_idx_idx)
    
    # sample ulb data
    if num_ulb_data is None or num_ulb_data == 'None':
        num_ulb_data = samples.shape[0] - num_pos_data
    num_ulb_pos_data = int(num_ulb_data * class_prior)
    num_ulb_neg_data = num_ulb_data - num_ulb_pos_data
    print("num_ulb_data: {}, num_ulb_pos_data: {}, num_ulb_neg_data: {}".format(num_ulb_data, num_ulb_pos_data, num_ulb_neg_data))
    
    if len(all_pos_idx) == 0:
        ulb_idx = None 
        ulb_samples = None
        ulb_targets = None
    else:
        if num_ulb_pos_data > len(all_pos_idx):
            ulb_pos_idx = np.random.choice(len(all_pos_idx), num_ulb_pos_data, replace=True)
        else:
            ulb_pos_idx = np.random.choice(all_pos_idx, num_ulb_pos_data, replace=False)
        if num_ulb_neg_data > len(all_neg_idx):
            ulb_neg_idx = np.random.choice(len(all_neg_idx), num_ulb_neg_data, replace=True)
        else:
            ulb_neg_idx = np.random.choice(all_neg_idx, num_ulb_neg_data, replace=False)
        ulb_idx = np.concatenate([ulb_pos_idx, ulb_neg_idx], axis=0)
        if include_lb_to_ulb and len(pos_idx) < len(samples):
            ulb_idx = np.concatenate([pos_idx, ulb_idx], axis=0)
        ulb_samples, ulb_targets = samples[ulb_idx], targets[ulb_idx]
    
    return pos_idx, pos_samples, pos_targets, ulb_idx, ulb_samples, ulb_targets, class_prior


def get_ulb_ulb_labels(samples, targets, 
                       num_classes=10, 
                       target_classes=[3, 5, 7], 
                       neg_classes=None,
                       cls_prior_ulb1=0.5,
                       cls_prior_ulb2=0.5,
                       num_ulb1_data=1000, 
                       num_ulb2_data=1000):
    
    assert np.max(target_classes) < num_classes, "Target classes should be less than num_classes"
    samples, targets = np.array(samples), np.array(targets)
    pos_classes = sorted(target_classes)
    if neg_classes is None or neg_classes == 'None':
        neg_classes = [i for i in range(num_classes) if i not in pos_classes]
    
    all_pos_idx = []
    all_neg_idx = []
    for c in range(num_classes):
        idx = np.where(targets == c)[0]
        if c in pos_classes:
            all_pos_idx.extend(idx)
        elif c in neg_classes:
            all_neg_idx.extend(idx)
    all_pos_idx = np.array(all_pos_idx)
    all_neg_idx = np.array(all_neg_idx)

    num_ulb1_pos_data = int(num_ulb1_data * cls_prior_ulb1)
    num_ulb1_neg_data = num_ulb1_data - num_ulb1_pos_data
    num_ulb2_pos_data = int(num_ulb2_data * cls_prior_ulb2)
    num_ulb2_neg_data = num_ulb2_data - num_ulb2_pos_data
    
    # salmple ulb1 index
    ulb1_pos_idx_idx = np.random.choice(len(all_pos_idx), num_ulb1_pos_data, replace=True)
    ulb1_pos_idx = all_pos_idx[ulb1_pos_idx_idx]
    ulb1_neg_idx_idx = np.random.choice(len(all_neg_idx), num_ulb1_neg_data, replace=True)
    ulb1_neg_idx = all_neg_idx[ulb1_neg_idx_idx]
    ulb1_idx = np.concatenate([ulb1_pos_idx, ulb1_neg_idx], axis=0)
    all_pos_idx = np.delete(all_pos_idx, ulb1_pos_idx_idx)
    all_neg_idx = np.delete(all_neg_idx, ulb1_neg_idx_idx)
    
    # sample ulb2 index
    ulb2_pos_idx = np.random.choice(all_pos_idx, num_ulb2_pos_data, replace=True)
    ulb2_neg_idx = np.random.choice(all_neg_idx, num_ulb2_neg_data, replace=True)
    ulb2_idx = np.concatenate([ulb2_pos_idx, ulb2_neg_idx], axis=0)
    
    # get data
    ulb1_samples, ulb1_targets = samples[ulb1_idx], targets[ulb1_idx]
    ulb2_samples, ulb2_targets = samples[ulb2_idx], targets[ulb2_idx]
    
    return ulb1_idx, ulb1_samples, ulb1_targets, ulb2_idx, ulb2_samples, ulb2_targets, cls_prior_ulb1, cls_prior_ulb2


def get_sim_dsim_ulb_labels(samples, targets, 
                            num_classes=10, 
                            target_classes=[3, 5, 7], 
                            neg_classes=None,
                            class_map=None,
                            num_pair_data=20000, 
                            num_ulb_data=10000,
                            class_prior=None):

    assert np.max(target_classes) < num_classes, "Target classes should be less than num_classes"
    samples, targets = np.array(samples), np.array(targets)
    pos_classes = sorted(target_classes)
    if neg_classes is None or neg_classes == 'None':
        neg_classes = [i for i in range(num_classes) if i not in pos_classes]

    if class_map is None:
        class_map = {i: 0 for i in neg_classes}
        for i, target_class in enumerate(target_classes):
            class_map[target_class] = 1

    all_pos_idx = []
    all_neg_idx = []
    for c in range(num_classes):
        idx = np.where(targets == c)[0]
        if c in pos_classes:
            all_pos_idx.extend(idx)
        elif c in neg_classes:
            all_neg_idx.extend(idx)
    all_pos_idx = np.array(all_pos_idx)
    all_neg_idx = np.array(all_neg_idx)
    
    # calculate class prior
    if class_prior is None:
        class_prior = len(all_pos_idx) / (len(all_pos_idx) + len(all_neg_idx))
    print("class_prior: {}".format(class_prior))
    
    # sample similar and dismilar labeled data
    if num_pair_data is None:
        num_pair_data = len(samples)
        
    num_sim_pos_data = int(num_pair_data  * (class_prior ** 2)) 
    num_sim_neg_data = int(num_pair_data * (1 - class_prior) ** 2)
    # num_sim_data = int(num_pair_data  * (class_prior ** 2 + (1 - class_prior) ** 2))
    num_sim_pos_data = num_sim_pos_data // 2 * 2
    num_sim_neg_data = num_sim_neg_data // 2 * 2
    num_sim_data = num_sim_pos_data + num_sim_neg_data

    num_dsim_data = int(num_pair_data  * class_prior * (1 - class_prior) * 2)
    num_dsim_data = num_dsim_data // 2 * 2
        
    print("num_sim_data: {}, num_sim_pos_data: {}, num_sim_neg_data: {}, num_dsim_data: {}".format(num_sim_data, num_sim_pos_data, num_sim_neg_data, num_dsim_data))
    
    if num_sim_pos_data > len(all_pos_idx):
        sim_pos_idx_idx = np.random.choice(len(all_pos_idx), num_sim_pos_data, replace=True)    
    else:
        sim_pos_idx_idx = np.random.choice(len(all_pos_idx), num_sim_pos_data, replace=False)    
    sim_pos_idx = all_pos_idx[sim_pos_idx_idx]
    if num_sim_neg_data > len(all_neg_idx):
        sim_neg_idx_idx = np.random.choice(len(all_neg_idx), num_sim_neg_data, replace=True)    
    else:
        sim_neg_idx_idx = np.random.choice(len(all_neg_idx), num_sim_neg_data, replace=False)    
    sim_neg_idx = all_neg_idx[sim_neg_idx_idx]
    all_pos_idx = np.delete(all_pos_idx, sim_pos_idx_idx)
    all_neg_idx = np.delete(all_neg_idx, sim_neg_idx_idx)
    
    dsim_pos_idx_idx = np.random.choice(len(all_pos_idx), num_dsim_data, replace=True)
    dsim_pos_idx = all_pos_idx[dsim_pos_idx_idx]
    dsim_neg_idx_idx = np.random.choice(len(all_neg_idx), num_dsim_data, replace=True)
    dsim_neg_idx = all_neg_idx[dsim_neg_idx_idx]
    all_pos_idx = np.delete(all_pos_idx, dsim_pos_idx_idx)
    all_neg_idx = np.delete(all_neg_idx, dsim_neg_idx_idx)
    
    if len(sim_pos_idx):
        sim_pos_samples, sim_pos_targets = samples[sim_pos_idx], targets[sim_pos_idx]
        sim_pos_targets = np.array(list(map(class_map.get, sim_pos_targets.tolist())))
        sim_pos_samples = sim_pos_samples.reshape(-1, 2, *samples.shape[1:])
        sim_pos_targets = sim_pos_targets.reshape(-1, 2, *sim_pos_targets.shape[1:])
        
        sim_neg_samples, sim_neg_targets = samples[sim_neg_idx], targets[sim_neg_idx]
        sim_neg_targets = np.array(list(map(class_map.get, sim_neg_targets.tolist())))
        sim_neg_samples = sim_neg_samples.reshape(-1, 2, *samples.shape[1:])
        sim_neg_targets = sim_neg_targets.reshape(-1, 2, *sim_neg_targets.shape[1:])
        
        sim_samples = np.concatenate([sim_pos_samples, sim_neg_samples], axis=0)
        sim_targets = np.concatenate([sim_pos_targets, sim_neg_targets], axis=0)
    else:
        sim_samples, sim_targets = np.array([]), np.array([])
    
    if len(dsim_pos_idx):
        dsim_idx =  list(chain(*zip(dsim_pos_idx, dsim_neg_idx))) 
        dsim_samples, dsim_targets = samples[dsim_idx], targets[dsim_idx]
        dsim_targets = np.array(list(map(class_map.get, dsim_targets.tolist())))
        dsim_samples = dsim_samples.reshape(-1, 2, *dsim_samples.shape[1:])
        dsim_targets = dsim_targets.reshape(-1, 2, *dsim_targets.shape[1:])
    else:
        dsim_samples, dsim_targets = np.array([]), np.array([])
    
    # sample ulb data
    if num_ulb_data is None or num_ulb_data == 'None':
        num_ulb_data = samples.shape[0] - int(num_sim_data * 2) - int(num_dsim_data * 2)
    num_ulb_pos_data = int(num_ulb_data * class_prior)
    num_ulb_neg_data = num_ulb_data - num_ulb_pos_data
    print("num_ulb_data: {}, num_ulb_pos_data: {}, num_ulb_neg_data: {}".format(num_ulb_data, num_ulb_pos_data, num_ulb_neg_data))
    if len(all_pos_idx) == 0:
        ulb_samples = None
        ulb_targets = None
    else:
        ulb_pos_idx = np.random.choice(all_pos_idx, num_ulb_pos_data, replace=True)
        ulb_neg_idx = np.random.choice(all_neg_idx, num_ulb_neg_data, replace=True)
        ulb_idx = np.concatenate([ulb_pos_idx, ulb_neg_idx], axis=0)
        ulb_samples, ulb_targets = samples[ulb_idx], targets[ulb_idx]
    
    return sim_samples, sim_targets, dsim_samples, dsim_targets, ulb_samples, ulb_targets, class_prior


def get_pairwise_comp_labels(samples, targets, 
                             num_classes=10, 
                             target_classes=[3, 5, 7], 
                             neg_classes=None,
                             class_map=None,
                             num_pair_data=10000,
                             class_prior=0.5):

    assert np.max(target_classes) < num_classes, "Target classes should be less than num_classes"
    samples, targets = np.array(samples), np.array(targets)
    pos_classes = sorted(target_classes)
    if neg_classes is None or neg_classes == 'None':
        neg_classes = [i for i in range(num_classes) if i not in pos_classes]

    if class_map is None:
        class_map = {i: 0 for i in neg_classes}
        for i, target_class in enumerate(target_classes):
            class_map[target_class] = 1

    all_pos_idx = []
    all_neg_idx = []
    for c in range(num_classes):
        idx = np.where(targets == c)[0]
        if c in pos_classes:
            all_pos_idx.extend(idx)
        elif c in neg_classes:
            all_neg_idx.extend(idx)
    all_pos_idx = np.array(all_pos_idx)
    all_neg_idx = np.array(all_neg_idx)
    
    # calculate different pairs
    num_pos_pairs = int((class_prior ** 2) * num_pair_data)
    num_neg_pairs = int(((1 - class_prior) ** 2) * num_pair_data)
    num_pos_neg_pairs = num_pair_data - num_pos_pairs - num_neg_pairs
    
    # sample pairs indexs
    if num_pos_pairs * 2 >= len(all_pos_idx) or num_neg_pairs * 2 >= len(all_neg_idx):
        replace = True
    else:
        replace = False 
    pos_pos_pairs_idx = np.random.choice(all_pos_idx, int(num_pos_pairs * 2), replace=replace)
    neg_neg_pairs_idx = np.random.choice(all_neg_idx, int(num_neg_pairs * 2), replace=replace)
    
    pos_neg_pair_pos_idx = np.random.choice(all_pos_idx, num_pos_neg_pairs, replace=replace)
    pos_neg_pair_neg_idx = np.random.choice(all_neg_idx, num_pos_neg_pairs, replace=replace)
    pos_neg_pair_idx = np.array(list(chain(*zip(pos_neg_pair_pos_idx, pos_neg_pair_neg_idx))))
    
    
    # sample positive pair data
    pos_pair_samples, pos_pair_targets = samples[pos_pos_pairs_idx], targets[pos_pos_pairs_idx]
    pos_pair_targets = np.array(list(map(class_map.get, pos_pair_targets.tolist())))
    pos_pair_samples = pos_pair_samples.reshape(-1, 2, *samples.shape[1:])
    pos_pair_targets = pos_pair_targets.reshape(-1, 2, *pos_pair_targets.shape[1:])
    
    # sample negative pair data
    neg_pair_samples, neg_pair_targets = samples[neg_neg_pairs_idx], targets[neg_neg_pairs_idx]
    neg_pair_targets = np.array(list(map(class_map.get, neg_pair_targets.tolist())))
    neg_pair_samples = neg_pair_samples.reshape(-1, 2, *samples.shape[1:])
    neg_pair_targets = neg_pair_targets.reshape(-1, 2, *neg_pair_targets.shape[1:])
    
    # sample positive negative pair data
    pos_neg_pair_samples, pos_neg_pair_targets = samples[pos_neg_pair_idx], targets[pos_neg_pair_idx]
    pos_neg_pair_targets = np.array(list(map(class_map.get, pos_neg_pair_targets.tolist())))
    pos_neg_pair_samples = pos_neg_pair_samples.reshape(-1, 2, *samples.shape[1:])
    pos_neg_pair_targets = pos_neg_pair_targets.reshape(-1, 2, *pos_neg_pair_targets.shape[1:])
    
    pair_samples = np.concatenate([pos_pair_samples, neg_pair_samples, pos_neg_pair_samples], axis=0)
    pair_targets = np.concatenate([pos_pair_targets, neg_pair_targets, pos_neg_pair_targets], axis=0)
    
    return pair_samples, pair_targets
    
    
def get_pos_conf_labels(samples, 
                        targets, 
                        num_classes=10, 
                        target_classes=[3, 5, 7], 
                        neg_classes=None,
                        num_data=20000,
                        data_dir='./data',
                        dataset_name='cifar10',
                        conf_model_name='clip',
                        zero_shot=True):
    
    assert np.max(target_classes) < num_classes, "Target classes should be less than num_classes"
    samples, targets = np.array(samples), np.array(targets)
    pos_classes = sorted(target_classes)
    if neg_classes is None or neg_classes == 'None':
        neg_classes = [i for i in range(num_classes) if i not in pos_classes]

    # zero_shot evaluation
    # check if already exists
    data_feat_file = os.path.join(data_dir, 'feat_files_pos_conf', f'{dataset_name}_{conf_model_name}.npz')
    if os.path.exists(data_feat_file):
        data = np.load(data_feat_file)
        sample_probs = data['sample_probs']
    else:
        model_source = conf_model_name.split('_')[0]
        model_name = '_'.join(conf_model_name.split('_')[1:])
        if zero_shot:
            if dataset_name in ['imagenet100', 'imagenet1k']:
                template = 'openai_imagenet_template'
            else:
                template = 'simple_template'
            data_classnames = classnames[dataset_name]
            sample_probs = zeroshot_eval(model_source, model_name, template, data_classnames, samples, targets)
        else:
            sample_probs = train_eval(model_name, dataset_name, samples, targets)
            # TODO: train a classifier on the dataset and get the confidence
            # Evaluate the confidence of the classifier on the samples
        os.makedirs(os.path.join(data_dir, 'feat_files_pos_conf'), exist_ok=True)
        np.savez(data_feat_file, sample_probs=sample_probs)

    all_pos_idx = []
    all_neg_idx = []
    for c in range(num_classes):
        idx = np.where(targets == c)[0]
        if c in pos_classes:
            all_pos_idx.extend(idx)
        elif c in neg_classes:
            all_neg_idx.extend(idx)
    all_pos_idx = np.array(all_pos_idx)
    all_neg_idx = np.array(all_neg_idx)
    
    # sample data
    if num_data // 2 > len(all_pos_idx):
        replace = True
    else:
        replace = False
    target_classes = np.array(target_classes)
    if num_data is None or num_data == 'None':
        pos_idx = all_pos_idx
        neg_idx = all_neg_idx
    else:
        pos_idx = np.random.choice(all_pos_idx, num_data // 2, replace=replace)
        neg_idx = np.random.choice(all_neg_idx, num_data // 2, replace=replace)
    pos_samples, pos_targets, pos_probs = samples[pos_idx], targets[pos_idx], sample_probs[pos_idx]
    pos_correct = (np.argmax(pos_probs, axis=1) == pos_targets).mean()
    pos_confs = pos_probs[:, target_classes]
    pos_confs = pos_confs.sum(axis=1)
    neg_samples, neg_targets, neg_probs = samples[neg_idx], targets[neg_idx], sample_probs[neg_idx]
    neg_correct = (np.argmax(neg_probs, axis=1) == neg_targets).mean()
    neg_confs = neg_probs[:, target_classes]
    neg_confs = neg_confs.sum(axis=1)
    print("pos_correct: {}, neg_correct: {}".format(pos_correct, neg_correct))
    
    selected_samples = np.concatenate([pos_samples, neg_samples], axis=0)
    selected_targets = np.concatenate([pos_confs, neg_confs], axis=0)
    # selected_targets = np.clip(selected_targets, a_min=1e-4, a_max=1-1e-4).astype(np.float32)
    
    return selected_samples, selected_targets


def get_single_cls_conf_labels(samples, 
                               targets, 
                               num_classes=10, 
                               target_classes=3, 
                               num_data=20000,
                               data_dir='./data',
                               dataset_name='cifar10',
                               conf_model_name='clip',
                               zero_shot=True):
    
    if isinstance(target_classes, int):
        target_classes = [target_classes]
    
    samples, targets = np.array(samples), np.array(targets)

    # zero_shot evaluation
    # check if already exists
    data_feat_file = os.path.join(data_dir, 'feat_files_v2', f'{dataset_name}_{conf_model_name}.npz')
    if os.path.exists(data_feat_file):
        data = np.load(data_feat_file)
        sample_probs = data['sample_probs']
    else:
        model_source = conf_model_name.split('_')[0]
        model_name = '_'.join(conf_model_name.split('_')[1:])
        if zero_shot:
            if dataset_name in ['imagenet100', 'imagenet1k']:
                template = 'openai_imagenet_template'
            else:
                template = 'simple_template'
            data_classnames = classnames[dataset_name]
            sample_probs = zeroshot_eval(model_source, model_name, template, data_classnames, samples, targets)
        else:
            sample_probs = train_eval(model_name, dataset_name, samples, targets)
            # TODO: train a classifier on the dataset and get the confidence
            # Evaluate the confidence of the classifier on the samples
        os.makedirs(os.path.join(data_dir, 'feat_files_v2'), exist_ok=True)
        np.savez(data_feat_file, sample_probs=sample_probs)

    for c in range(num_classes):
        if c not in target_classes:
            continue
        all_pos_idx = np.where(targets == c)[0]
    
    if num_data is None or num_data == 'None':
        pos_idx = all_pos_idx
    else:
        pos_idx = np.random.choice(all_pos_idx, num_data, replace=False)

    # sample data
    single_cls_samples = samples[pos_idx]
    single_cls_conf = sample_probs[pos_idx]
    # single_cls_conf = np.clip(single_cls_conf, a_min=1e-4, a_max=1-1e-4).astype(np.float32)
    return single_cls_samples, single_cls_conf



def get_sim_conf_labels(samples, 
                        targets, 
                        num_classes=10, 
                        target_classes=[3, 5, 7], 
                        neg_classes=None,
                        num_pair_data=20000,
                        class_prior=0.5,
                        data_dir='./data',
                        dataset_name='cifar10',
                        conf_model_name='clip',
                        zero_shot=True):

    assert np.max(target_classes) < num_classes, "Target classes should be less than num_classes"
    samples, targets = np.array(samples), np.array(targets)
    pos_classes = sorted(target_classes)
    if neg_classes is None or neg_classes == 'None':
        neg_classes = [i for i in range(num_classes) if i not in pos_classes]

    # zero_shot evaluation
    # check if already exists
    data_feat_file = os.path.join(data_dir, 'feat_files_sim_conf', f'{dataset_name}_{conf_model_name}.npz')
    if os.path.exists(data_feat_file):
        data = np.load(data_feat_file)
        sample_probs = data['sample_probs']
    else:
        model_source = conf_model_name.split('_')[0]
        model_name = '_'.join(conf_model_name.split('_')[1:])
        if zero_shot:
            if dataset_name in ['imagenet100', 'imagenet1k']:
                template = 'openai_imagenet_template'
            else:
                template = 'simple_template'
            data_classnames = classnames[dataset_name]
            sample_probs = zeroshot_eval(model_source, model_name, template, data_classnames, samples, targets)
        else:
            sample_probs = train_eval(model_name, dataset_name, samples, targets)
            # TODO: train a classifier on the dataset and get the confidence
            # Evaluate the confidence of the classifier on the samples
        os.makedirs(os.path.join(data_dir, 'feat_files_sim_conf'), exist_ok=True)
        np.savez(data_feat_file, sample_probs=sample_probs)
    
    all_pos_idx = []
    all_neg_idx = []
    for c in range(num_classes):
        idx = np.where(targets == c)[0]
        if c in pos_classes:
            all_pos_idx.extend(idx)
        elif c in neg_classes:
            all_neg_idx.extend(idx)
    all_pos_idx = np.array(all_pos_idx)
    all_neg_idx = np.array(all_neg_idx)

    # calculate class prior
    if class_prior is None:
        class_prior = len(all_pos_idx) / (len(all_pos_idx) + len(all_neg_idx))
    print("class_prior: {}".format(class_prior))
    
    # sample similar and dismilar labeled data
    if num_pair_data is None:
        num_pair_data = len(samples)

    target_classes = np.array(target_classes)
    num_pos_data = int(num_pair_data * class_prior * 2)
    num_neg_data = 2 * num_pair_data - num_pos_data
    if num_pos_data > len(all_pos_idx):
        pos_idx = np.random.choice(all_pos_idx, num_pos_data, replace=True)
    else:
        pos_idx = np.random.choice(all_pos_idx, num_pos_data, replace=False)
    if num_neg_data > len(all_neg_idx):
        neg_idx = np.random.choice(all_neg_idx, num_neg_data, replace=True)
    else:
        neg_idx = np.random.choice(all_neg_idx, num_neg_data, replace=False)
    
    pos_samples, pos_probs = samples[pos_idx], sample_probs[pos_idx]
    neg_samples, neg_probs = samples[neg_idx], sample_probs[neg_idx]
    train_samples, train_probs = np.concatenate([pos_samples, neg_samples], axis=0), np.concatenate([pos_probs, neg_probs], axis=0)
    
    all_train_idx = np.arange(len(train_samples))
    np.random.shuffle(all_train_idx)
    data1_idx = all_train_idx[:len(all_train_idx) // 2]
    data2_idx = all_train_idx[len(all_train_idx) // 2:]
    train_samples1, train_probs1 = train_samples[data1_idx], train_probs[data1_idx]
    train_conf1 = train_probs1[:, target_classes].sum(axis=1)
    train_samples2, train_probs2 = train_samples[data2_idx], train_probs[data2_idx]
    train_conf2 = train_probs2[:, target_classes].sum(axis=1)
    
    selected_samples = np.concatenate([train_samples1[:, np.newaxis], train_samples2[:, np.newaxis]], axis=1)
    selected_sim_confs = train_conf1 * train_conf2 + (1 - train_conf1) * (1 - train_conf2)
    selected_sim_confs = np.clip(selected_sim_confs, a_min=1e-4, a_max=1 - 1e-4).astype(np.float32)
    return selected_samples, selected_sim_confs


def get_conf_diff_labels(samples, 
                         targets, 
                         num_classes=10, 
                         target_classes=[3, 5, 7], 
                         neg_classes=None,
                         num_pair_data=20000,
                         class_prior=0.5,
                         data_dir='./data',
                         dataset_name='cifar10',
                         conf_model_name='clip',
                         zero_shot=True):

    assert np.max(target_classes) < num_classes, "Target classes should be less than num_classes"
    samples, targets = np.array(samples), np.array(targets)
    pos_classes = sorted(target_classes)
    if neg_classes is None or neg_classes == 'None':
        neg_classes = [i for i in range(num_classes) if i not in pos_classes]

    # zero_shot evaluation
    # check if already exists
    data_feat_file = os.path.join(data_dir, 'feat_files_conf_diff', f'{dataset_name}_{conf_model_name}.npz')
    if os.path.exists(data_feat_file):
        data = np.load(data_feat_file)
        sample_probs = data['sample_probs']
    else:
        model_source = conf_model_name.split('_')[0]
        model_name = '_'.join(conf_model_name.split('_')[1:])
        if zero_shot:
            if dataset_name in ['imagenet100', 'imagenet1k']:
                template = 'openai_imagenet_template'
            else:
                template = 'simple_template'
            data_classnames = classnames[dataset_name]
            sample_probs = zeroshot_eval(model_source, model_name, template, data_classnames, samples, targets)
        else:
            sample_probs = train_eval(model_name, dataset_name, samples, targets)
            # TODO: train a classifier on the dataset and get the confidence
            # Evaluate the confidence of the classifier on the samples
        os.makedirs(os.path.join(data_dir, 'feat_files_conf_diff'), exist_ok=True)
        np.savez(data_feat_file, sample_probs=sample_probs)
    
    all_pos_idx = []
    all_neg_idx = []
    for c in range(num_classes):
        idx = np.where(targets == c)[0]
        if c in pos_classes:
            all_pos_idx.extend(idx)
        elif c in neg_classes:
            all_neg_idx.extend(idx)
    all_pos_idx = np.array(all_pos_idx)
    all_neg_idx = np.array(all_neg_idx)

    # calculate class prior
    if class_prior is None:
        class_prior = len(all_pos_idx) / (len(all_pos_idx) + len(all_neg_idx))
    print("class_prior: {}".format(class_prior))
    
    # sample similar and dismilar labeled data
    if num_pair_data is None:
        num_pair_data = len(samples)
    
    target_classes = np.array(target_classes)
    num_pos_data = int(num_pair_data * class_prior * 2)
    num_neg_data = 2 * num_pair_data - num_pos_data
    if num_pos_data > len(all_pos_idx):
        pos_idx = np.random.choice(all_pos_idx, num_pos_data, replace=True)
    else:
        pos_idx = np.random.choice(all_pos_idx, num_pos_data, replace=False)
    if num_neg_data > len(all_neg_idx):
        neg_idx = np.random.choice(all_neg_idx, num_neg_data, replace=True)
    else:
        neg_idx = np.random.choice(all_neg_idx, num_neg_data, replace=False)
    
    pos_samples, pos_probs = samples[pos_idx], sample_probs[pos_idx]
    neg_samples, neg_probs = samples[neg_idx], sample_probs[neg_idx]
    train_samples, train_probs = np.concatenate([pos_samples, neg_samples], axis=0), np.concatenate([pos_probs, neg_probs], axis=0)
    
    all_train_idx = np.arange(len(train_samples))
    np.random.shuffle(all_train_idx)
    data1_idx = all_train_idx[:len(all_train_idx) // 2]
    data2_idx = all_train_idx[len(all_train_idx) // 2:]
    train_samples1, train_probs1 = train_samples[data1_idx], train_probs[data1_idx]
    train_conf1 = train_probs1[:, target_classes].sum(axis=1)
    train_samples2, train_probs2 = train_samples[data2_idx], train_probs[data2_idx]
    train_conf2 = train_probs2[:, target_classes].sum(axis=1)
    
    selected_samples = np.concatenate([train_samples1[:, np.newaxis], train_samples2[:, np.newaxis]], axis=1)
    selected_conf_diff = train_conf2 - train_conf1
    selected_conf_diff = np.clip(selected_conf_diff, a_min=1e-8, a_max=1 - 1e-8).astype(np.float32)
    return selected_samples, selected_conf_diff



if __name__ == '__main__':
    samples = np.random.randn(50000, 728)
    labels = np.random.randint(0, 10, (50000, ))
    get_pos_conf_labels(samples, labels, num_classes=10, target_classes=[0, 1, 2, 3, 4], conf_model_name='openclip_RN50_openai')