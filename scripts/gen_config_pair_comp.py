import os
import numpy as np


def create_configuration(cfg, cfg_file):
    cfg['save_name'] = "{alg}_{dataset}_target{target_classes}_neg{neg_classes}_pair{num_pair_data}_prior{class_prior}_{seed}".format(
        alg=cfg['algorithm'],
        dataset=cfg['dataset'],
        target_classes=len(cfg['target_classes']),
        neg_classes=len(cfg['neg_classes']) if cfg['neg_classes'] is not None else cfg['num_classes'] - len(cfg['target_classes']),
        num_pair_data=cfg['num_pair_data'],
        class_prior=cfg['class_prior'],
        seed=cfg['seed'],
    )

    # resume
    cfg['resume'] = True
    cfg['load_path'] = '{}/{}/latest_model.pth'.format(cfg['save_dir'], cfg['save_name'])

    alg_file = cfg_file + '/'
    if not os.path.exists(alg_file):
        os.mkdir(alg_file)

    print(alg_file + cfg['save_name'] + '.yaml')
    with open(alg_file + cfg['save_name'] + '.yaml', 'w', encoding='utf-8') as w:
        lines = []
        for k, v in cfg.items():
            line = str(k) + ': ' + str(v)
            lines.append(line)
        for line in lines:
            w.writelines(line)
            w.write('\n')


def create_config_pair_comp(seed, 
                            algorithm,
                            dataset, net, num_classes, num_epoch, batch_size, img_size, crop_ratio, optim, lr, weight_decay, 
                            target_classes=[9], neg_classes=[8],
                            num_pair_data=1000, class_prior=0.5,
                            port=1000, setting='classic_cv'):
    cfg = {}
    cfg['algorithm'] = algorithm

    # save config
    # cfg['save_dir'] = f'./saved_models/pair_comp/{setting}/{algorithm}'
    cfg['save_dir'] = f'/mnt/imprecise_label_learning_weak_supervision_haoc/projects/connectionise_imprecise_label_learning/240206_saved_models/pair_comp/{setting}'
    
    # cfg['save_dir'] = f'/mnt/imprecise_label_learning_weak_supervision_haoc/projects/connectionise_imprecise_label_learning/231229_saved_models/pair_comp/{setting}'
    cfg['save_name'] = None
    cfg['resume'] = False
    cfg['load_path'] = None
    cfg['overwrite'] = True
    cfg['use_tensorboard'] = True
    cfg['use_wandb'] = False
    
    # algorithm config
    cfg['num_train_iter'] = None
    cfg['num_eval_iter'] = None
    cfg['num_log_iter'] = 50
    cfg['batch_size'] = batch_size
    cfg['eval_batch_size'] = 128
    cfg['epoch'] = num_epoch
    
    # dataset config
    cfg['crop_ratio'] = crop_ratio
    cfg['img_size'] = img_size
    # cfg['data_dir'] = './data'
    cfg['data_dir'] = '/mnt/data/dataset/imprecise_datasets'
    # cfg['data_dir'] = '/media/Bootes/datasets'
    cfg['dataset'] = dataset
    cfg['num_classes'] = num_classes
    cfg['num_workers'] = 4
    

    # optim config
    cfg['optim'] = optim
    cfg['lr'] = lr
    cfg['momentum'] = 0.9
    cfg['weight_decay'] = weight_decay
    cfg['layer_decay'] = 1.0
    cfg['amp'] = False
    cfg['clip'] = 0.0

    # net config
    cfg['net'] = net
    cfg['net_from_name'] = False
    cfg['ema_m'] = 0.0


    # distributed config
    cfg['seed'] = seed
    cfg['world_size'] = 1
    cfg['rank'] = 0
    cfg['multiprocessing_distributed'] =  False # True
    cfg['dist_url'] = 'tcp://127.0.0.1:' + str(port)
    cfg['dist_backend'] = 'nccl'
    cfg['gpu'] = 0 # None
    
    
    if algorithm == 'pcomp_relu_pair_comp':
        cfg['activation'] = 'relu'
    elif algorithm == 'pcomp_abs_pair_comp':
        cfg['activation'] = 'abs'
    elif algorithm == 'pcomp_teacher_pair_comp':
        cfg['ema_m'] = 0.0
    elif algorithm == 'imp_pair_comp':
        cfg['strong_aug'] = False

    # setting config
    cfg['target_classes'] = target_classes
    cfg['neg_classes'] = neg_classes
    cfg['num_pair_data'] = num_pair_data
    cfg['class_prior'] = class_prior

    return cfg




def exp_classic_cv():
    config_file = r'./config/pair_comp/classic_cv'
    save_path = r'./saved_models/pair_comp/classic_cv'

    if not os.path.exists(config_file):
        os.makedirs(config_file)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    algorithms = ['imp_pair_comp', 'pcomp_unbiased_pair_comp', 'pcomp_relu_pair_comp', 'pcomp_abs_pair_comp', 'pcomp_teacher_pair_comp', 'rank_pruning_pair_comp']
    datasets = ['mnist', 'fmnist', 'cifar10', 'cifar100', 'stl10', 'imagenet100']
    # datasets = ['mnist', 'cifar10']
    settings_dict = {

        'mnist': [
            
                {'target_classes': [0, 1, 2, 3, 4], 'neg_classes': None, 'num_pair_data':25000, 'class_prior': 0.2,  'num_epoch': 30},
                {'target_classes': [0, 1, 2, 3, 4], 'neg_classes': None, 'num_pair_data':25000, 'class_prior': 0.5,  'num_epoch': 30},
                {'target_classes': [0, 1, 2, 3, 4], 'neg_classes': None, 'num_pair_data':25000, 'class_prior': 0.8,  'num_epoch': 30},

        ],

        'fmnist': [

                {'target_classes': [5, 7, 9], 'neg_classes': None, 'num_pair_data':25000, 'class_prior': 0.2,  'num_epoch': 30},
                {'target_classes': [5, 7, 9], 'neg_classes': None, 'num_pair_data':25000, 'class_prior': 0.5,  'num_epoch': 30},
                {'target_classes': [5, 7, 9], 'neg_classes': None, 'num_pair_data':25000, 'class_prior': 0.8,  'num_epoch': 30},

        ],

        'cifar10': [

                {'target_classes': [0, 1, 8, 9], 'neg_classes': None, 'num_pair_data':20000, 'class_prior': 0.2,  'num_epoch': 50},
                {'target_classes': [0, 1, 8, 9], 'neg_classes': None, 'num_pair_data':20000, 'class_prior': 0.5,  'num_epoch': 50},
                {'target_classes': [0, 1, 8, 9], 'neg_classes': None, 'num_pair_data':20000, 'class_prior': 0.8,  'num_epoch': 50},

        ],

        'svhn': [

                {'target_classes': [0, 1, 2, 3, 4], 'neg_classes': None, 'num_pair_data':20000, 'class_prior': 0.2,  'num_epoch': 50},
                {'target_classes': [0, 1, 2, 3, 4], 'neg_classes': None, 'num_pair_data':20000, 'class_prior': 0.5,  'num_epoch': 50},
                {'target_classes': [0, 1, 2, 3, 4], 'neg_classes': None, 'num_pair_data':20000, 'class_prior': 0.8,  'num_epoch': 50},
                
        ],

        'stl10': [

                {'target_classes': [0, 1, 8, 9], 'neg_classes': None, 'num_pair_data':5000, 'class_prior': 0.2,  'num_epoch': 50},
                {'target_classes': [0, 1, 8, 9], 'neg_classes': None, 'num_pair_data':5000, 'class_prior': 0.5,  'num_epoch': 50},
                {'target_classes': [0, 1, 8, 9], 'neg_classes': None, 'num_pair_data':5000, 'class_prior': 0.8,  'num_epoch': 50},
                
        ],

        'cifar100': [

                {'target_classes': [4, 30, 55, 72, 95] + [1, 32, 67, 73, 91] + [3, 42, 43, 88, 97] + [15, 19, 21, 31, 38] + [34, 63, 64, 66, 75] + [26, 45, 77, 79, 99] + [27, 29, 44, 78, 93] +  [36, 50, 65, 74, 80], 'neg_classes': None, 'num_pair_data':20000, 'class_prior': 0.2,  'num_epoch': 100},
                {'target_classes': [4, 30, 55, 72, 95] + [1, 32, 67, 73, 91] + [3, 42, 43, 88, 97] + [15, 19, 21, 31, 38] + [34, 63, 64, 66, 75] + [26, 45, 77, 79, 99] + [27, 29, 44, 78, 93] +  [36, 50, 65, 74, 80], 'neg_classes': None, 'num_pair_data':20000, 'class_prior': 0.5,  'num_epoch': 100},
                {'target_classes': [4, 30, 55, 72, 95] + [1, 32, 67, 73, 91] + [3, 42, 43, 88, 97] + [15, 19, 21, 31, 38] + [34, 63, 64, 66, 75] + [26, 45, 77, 79, 99] + [27, 29, 44, 78, 93] +  [36, 50, 65, 74, 80], 'neg_classes': None, 'num_pair_data':20000, 'class_prior': 0.8,  'num_epoch': 100},
                
        ],
        
    }
    # seeds = [2, 42, 2023]
    seeds = [42]
    dist_port = range(10001, 31120, 1)
    count = 0
    
    for dataset in datasets:
        
        if dataset == 'cifar10':
            # net = 'WideResNet'
            num_classes = 10
            optim = 'AdamW'
            lr = 0.001
            weight_decay = 1e-3
            net = 'wrn_28_2'
            img_size = 32
            batch_size = 64
            crop_ratio = 0.875

        elif dataset == 'mnist' or dataset == 'fmnist':
            # net = 'WideResNet'
            num_classes = 10
            optim = 'AdamW'
            lr = 0.0005
            weight_decay = 1e-4
            net = 'lenet5'
            img_size = 28
            batch_size = 64
            crop_ratio = 1.0
        
        elif dataset == 'svhn':
            
            num_classes = 10
            optim = 'AdamW'
            lr = 0.001
            weight_decay = 1e-3
            net = 'wrn_28_2'
            img_size = 32
            batch_size = 64
            crop_ratio = 0.875
        
        elif dataset == 'stl10':
            
            num_classes = 10
            optim = 'AdamW'
            lr = 0.001
            weight_decay = 1e-3
            net = 'resnet18'
            img_size = 96
            batch_size = 32
            crop_ratio = 0.875
            

        elif dataset == 'cifar100':
            
            
            num_classes = 100
            optim = 'AdamW'
            lr = 0.001
            weight_decay = 1e-3
            net = 'resnet18'
            img_size = 32 
            batch_size = 64
            crop_ratio = 0.875
        
        elif dataset == 'imagenet100':
            
            continue 
        
        for algorithm in algorithms:
            for setting_dict in  settings_dict[dataset]:
                
                for seed in seeds:
                    port = dist_port[count]
                    # prepare the configuration file
                    cfg = create_config_pair_comp(
                        seed=seed, algorithm=algorithm,
                        dataset=dataset, net=net, num_classes=num_classes, batch_size=batch_size, img_size=img_size, crop_ratio=crop_ratio, optim=optim, lr=lr, weight_decay=weight_decay, 
                        port=port, setting='classic_cv', **setting_dict)
                    count += 1
                    create_configuration(cfg, config_file)


if __name__ == '__main__':
    exp_classic_cv()
    # exp_rcr_cv()