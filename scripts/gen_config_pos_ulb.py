import os
import numpy as np


def create_configuration(cfg, cfg_file):
    cfg['save_name'] = "{alg}_{dataset}_target{target_classes}_neg{neg_classes}_pos{num_pos_data}_ulb{num_ulb_data}_{seed}".format(
        alg=cfg['algorithm'],
        dataset=cfg['dataset'],
        target_classes=len(cfg['target_classes']),
        neg_classes=len(cfg['neg_classes']) if cfg['neg_classes'] is not None else cfg['num_classes'] - len(cfg['target_classes']),
        num_pos_data=cfg['num_pos_data'],
        num_ulb_data=cfg['num_ulb_data'],
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


def create_config_pos_ulb(seed, 
                          algorithm,
                          dataset, net, num_classes, num_epoch, num_train_iter, batch_size, img_size, crop_ratio, optim, lr, weight_decay, 
                          target_classes=[9], neg_classes=[8],
                          num_pos_data=1000, num_ulb_data=10000,
                          port=1000, setting='classic_cv'):
    cfg = {}
    cfg['algorithm'] = algorithm

    # save config
    cfg['save_dir'] = f'./saved_models/pos_ulb/{setting}/{algorithm}'
    
    cfg['save_name'] = None
    cfg['resume'] = False
    cfg['load_path'] = None
    cfg['overwrite'] = True
    cfg['use_tensorboard'] = True
    cfg['use_wandb'] = False
    
    # algorithm config
    cfg['num_train_iter'] = num_train_iter
    cfg['num_eval_iter'] = None
    cfg['num_log_iter'] = 50
    cfg['batch_size'] = batch_size
    cfg['eval_batch_size'] = 128
    cfg['epoch'] = num_epoch
    
    # dataset config
    cfg['crop_ratio'] = crop_ratio
    cfg['img_size'] = img_size
    cfg['data_dir'] = './data'
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
    cfg['multiprocessing_distributed'] = True
    cfg['dist_url'] = 'tcp://127.0.0.1:' + str(port)
    cfg['dist_backend'] = 'nccl'
    cfg['gpu'] = None
    
    # algorithm specific config
    if algorithm == 'upu_pos_ulb':
        cfg['include_lb_to_ulb'] = False
        cfg['uratio'] = 1
    elif algorithm == 'nnpu_pos_ulb':
        cfg['include_lb_to_ulb'] = False
        cfg['uratio'] = 1
    elif algorithm == 'cvir_pos_ulb':
        cfg['include_lb_to_ulb'] = False
        cfg['uratio'] = 1
    elif algorithm == 'dist_pu_pos_ulb':
        cfg['include_lb_to_ulb'] = False
        cfg['uratio'] = 1
        cfg['mixup_alpha'] = 6.0
        cfg['loss_weight_ent'] = 0.004
        cfg['loss_weight_mixup'] = 5.0
        cfg['loss_weight_mixup_ent'] = 0.04
        cfg['warmup_epoch'] = cfg['epoch'] // 2
        cfg['warmup_lr'] = 5e-4
        cfg['warmup_weight_decay'] = 5e-3
    elif algorithm == 'count_loss_pos_ulb':
        cfg['uratio'] = 1
        cfg['include_lb_to_ulb'] = False
    elif algorithm == 'var_pu_pos_ulb':
        cfg['uratio'] = 1
        cfg['include_lb_to_ulb'] = False
        cfg['mixup_alpha'] = 0.3
        cfg['loss_weight_mixup'] = 0.03
        cfg['lr'] = 3e-5
    elif algorithm == 'imp_pos_ulb':
        cfg['uratio'] = 1
        cfg['include_lb_to_ulb'] = False
        cfg['strong_aug'] = False

    # setting config
    cfg['target_classes'] = target_classes
    cfg['neg_classes'] = neg_classes
    cfg['num_pos_data'] = num_pos_data
    cfg['num_ulb_data'] = num_ulb_data

    return cfg




def exp_classic_cv():
    config_file = r'./config/pos_ulb/classic_cv'
    save_path = r'./saved_models/pos_ulb/classic_cv'

    if not os.path.exists(config_file):
        os.makedirs(config_file)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    algorithms = ['upu_pos_ulb', 'nnpu_pos_ulb', 'cvir_pos_ulb', 'dist_pu_pos_ulb', 'var_pu_pos_ulb', 'count_loss_pos_ulb', 'imp_pos_ulb']
    # algorithms = ['imp_pos_ulb']
    datasets = ['mnist', 'fmnist', 'cifar10', 'cifar100', 'stl10']
    settings_dict = {

        'mnist': [
            
                {'target_classes': [0, 1, 2, 3, 4], 'neg_classes': None, 'num_pos_data':1000, 'num_ulb_data': None, 'num_train_iter': 50000, 'num_epoch': 50},
                {'target_classes': [0, 1, 2, 3, 4], 'neg_classes': None, 'num_pos_data':500, 'num_ulb_data': None, 'num_train_iter': 50000, 'num_epoch': 50},
                {'target_classes': [0, 1, 2, 3, 4], 'neg_classes': None, 'num_pos_data':100, 'num_ulb_data': None, 'num_train_iter': 50000, 'num_epoch': 50},

        ],

        'fmnist': [

                {'target_classes': [5, 7, 9], 'neg_classes': None, 'num_pos_data':1000, 'num_ulb_data': None, 'num_train_iter': 50000, 'num_epoch': 50},
                {'target_classes': [5, 7, 9],  'neg_classes': None, 'num_pos_data':500, 'num_ulb_data': None, 'num_train_iter': 50000, 'num_epoch': 50},
                {'target_classes': [5, 7, 9],  'neg_classes': None, 'num_pos_data':100, 'num_ulb_data': None, 'num_train_iter': 50000, 'num_epoch': 50},
                
        ],

        'cifar10': [
                {'target_classes': [0, 1, 8, 9], 'neg_classes': None, 'num_pos_data':2000, 'num_ulb_data': None, 'num_train_iter': 50000, 'num_epoch': 50},
                {'target_classes': [0, 1, 8, 9], 'neg_classes': None, 'num_pos_data':1000, 'num_ulb_data': None, 'num_train_iter': 50000, 'num_epoch': 50},
                {'target_classes': [0, 1, 8, 9], 'neg_classes': None, 'num_pos_data':500, 'num_ulb_data': None, 'num_train_iter': 50000, 'num_epoch': 50},
        ],
        
        'svhn': [
                {'target_classes': [0, 1, 2, 3, 4], 'neg_classes': None, 'num_pos_data':2000, 'num_ulb_data': None, 'num_train_iter': 50000, 'num_epoch': 50},
                {'target_classes': [0, 1, 2, 3, 4], 'neg_classes': None, 'num_pos_data':1000, 'num_ulb_data': None, 'num_train_iter': 50000, 'num_epoch': 50},
                {'target_classes': [0, 1, 2, 3, 4], 'neg_classes': None, 'num_pos_data':500, 'num_ulb_data': None, 'num_train_iter': 50000, 'num_epoch': 50},
        ],
        
        'stl10': [
                {'target_classes': [0, 1, 8, 9], 'neg_classes': None, 'num_pos_data':2000, 'num_ulb_data': None, 'num_train_iter': 100000, 'num_epoch': 50},
                {'target_classes': [0, 1, 8, 9], 'neg_classes': None, 'num_pos_data':1000, 'num_ulb_data': None, 'num_train_iter': 100000, 'num_epoch': 50},
                {'target_classes': [0, 1, 8, 9], 'neg_classes': None, 'num_pos_data':500, 'num_ulb_data': None, 'num_train_iter': 100000, 'num_epoch': 50},
        ],


        'cifar100': [
                {'target_classes': [4, 30, 55, 72, 95] + [1, 32, 67, 73, 91] + [3, 42, 43, 88, 97] + [15, 19, 21, 31, 38] + [34, 63, 64, 66, 75] + [26, 45, 77, 79, 99] + [27, 29, 44, 78, 93] +  [36, 50, 65, 74, 80], 'neg_classes': None, 'num_pos_data':4000, 'num_ulb_data': None, 'num_train_iter': 100000, 'num_epoch': 50},
                {'target_classes': [4, 30, 55, 72, 95] + [1, 32, 67, 73, 91] + [3, 42, 43, 88, 97] + [15, 19, 21, 31, 38] + [34, 63, 64, 66, 75] + [26, 45, 77, 79, 99] + [27, 29, 44, 78, 93] +  [36, 50, 65, 74, 80], 'neg_classes': None, 'num_pos_data':2000, 'num_ulb_data': None, 'num_train_iter': 100000, 'num_epoch': 50},
                {'target_classes': [4, 30, 55, 72, 95] + [1, 32, 67, 73, 91] + [3, 42, 43, 88, 97] + [15, 19, 21, 31, 38] + [34, 63, 64, 66, 75] + [26, 45, 77, 79, 99] + [27, 29, 44, 78, 93] +  [36, 50, 65, 74, 80], 'neg_classes': None, 'num_pos_data':1000, 'num_ulb_data': None, 'num_train_iter': 100000, 'num_epoch': 50},
        ]
        
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
                    cfg = create_config_pos_ulb(
                        seed=seed, algorithm=algorithm,
                        dataset=dataset, net=net, num_classes=num_classes, batch_size=batch_size, img_size=img_size, crop_ratio=crop_ratio, optim=optim, lr=lr, weight_decay=weight_decay, 
                        port=port, setting='classic_cv', **setting_dict)
                    count += 1
                    create_configuration(cfg, config_file)


if __name__ == '__main__':
    exp_classic_cv()
    # exp_rcr_cv()