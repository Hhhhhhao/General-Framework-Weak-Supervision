import os
import numpy as np


def create_configuration(cfg, cfg_file):
    cfg['save_name'] = "{alg}_{dataset}_target{target_classes}_bags{num_bags_train}_mean{mean_bag_len}_std{std_bag_len}_{seed}".format(
        alg=cfg['algorithm'],
        dataset=cfg['dataset'],
        target_classes=len(cfg['target_classes']),
        num_bags_train=cfg['num_bags_train'],
        mean_bag_len=cfg['mean_bag_len'],
        std_bag_len=cfg['std_bag_len'],
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


def create_config_proportion(seed, 
                             algorithm,
                             dataset, net, num_classes, num_epochs, batch_size, img_size, crop_ratio, optim, lr, weight_decay, 
                             target_classes=[9],
                             num_bags_train=10000, mean_bag_len=5, std_bag_len=1,
                             port=1000, setting='classic_cv'):
    cfg = {}
    cfg['algorithm'] = algorithm

    # save config
    cfg['save_dir'] = f'./saved_models/proportion/{setting}/{algorithm}'
    cfg['save_name'] = None
    cfg['resume'] = False
    cfg['load_path'] = None
    cfg['overwrite'] = True
    cfg['use_tensorboard'] = True
    cfg['use_wandb'] = False
    
    # algorithm config
    cfg['epoch'] = num_epochs
    cfg['num_eval_iter'] = None
    cfg['num_log_iter'] = 25
    cfg['batch_size'] = batch_size
    cfg['eval_batch_size'] = 128
    
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
    cfg['multiprocessing_distributed'] = False
    cfg['dist_url'] = 'tcp://127.0.0.1:' + str(port)
    cfg['dist_backend'] = 'nccl'
    cfg['gpu'] = 0

    if dataset in ['imagenet100', 'imagenet1k']:
        cfg['gpu'] = None
        cfg['multiprocessing_distributed'] = True
        cfg['num_log_iter'] = 10
        cfg['amp'] = True
        cfg['clip'] = 5.0
        cfg['data_dir'] = '/mnt/data/dataset'
    
    # algorithm specific config
    if algorithm == 'llp_vat_proportion':
        cfg['vat_xi'] = 1e-6
        cfg['vat_eps'] = 6.0
        cfg['vat_ip'] = 1
        cfg['prop_metric'] = 'ce'
        cfg['loss_weight_cons'] = 1.0
    elif algorithm == 'imp_porportion':
        cfg['strong_aug'] = False

    # setting config
    cfg['target_classes'] = target_classes
    cfg['mean_bag_len'] = mean_bag_len
    cfg['std_bag_len'] = std_bag_len
    cfg['num_bags_train'] = num_bags_train

    return cfg




def exp_classic_cv():
    config_file = r'./config/proportion/classic_cv'
    save_path = r'./saved_models/proportion/classic_cv'

    if not os.path.exists(config_file):
        os.makedirs(config_file)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    algorithms = ['imp_proportion', 'count_loss_proportion', 'uum_proportion', 'llp_vat_proportion', ]
    datasets = ['mnist', 'fmnist', 'cifar10', 'cifar100', 'stl10', 'imagenet100']
    settings_dict = {

        'mnist': [
                # {'target_classes': [9], 'num_bags_train': 1000, 'mean_bag_len': 5, 'std_bag_len': 1, 'num_epochs': 30},
                {'target_classes': [9], 'num_bags_train': 1000, 'mean_bag_len': 10, 'std_bag_len': 2, 'num_epochs': 30},
                {'target_classes': [9], 'num_bags_train': 500, 'mean_bag_len': 20, 'std_bag_len': 5, 'num_epochs': 30},
                {'target_classes': [9], 'num_bags_train': 250, 'mean_bag_len': 50, 'std_bag_len': 10, 'num_epochs': 30},
                
                {'target_classes': np.arange(10).tolist(), 'num_bags_train': 2000, 'mean_bag_len': 5, 'std_bag_len': 1, 'num_epochs': 30},
                {'target_classes': np.arange(10).tolist(), 'num_bags_train': 1000, 'mean_bag_len': 10, 'std_bag_len': 2, 'num_epochs': 30},
                {'target_classes': np.arange(10).tolist(), 'num_bags_train': 500, 'mean_bag_len': 20, 'std_bag_len': 10, 'num_epochs': 30},
        ],

        'fmnist': [
                # {'target_classes': [9], 'num_bags_train': 1000, 'mean_bag_len': 5, 'std_bag_len': 1, 'num_epochs': 30},
                # {'target_classes': [9], 'num_bags_train': 500, 'mean_bag_len': 10, 'std_bag_len': 2, 'num_epochs': 30},
                # {'target_classes': [9], 'num_bags_train': 250, 'mean_bag_len': 20, 'std_bag_len': 5, 'num_epochs': 30},
                {'target_classes': [9], 'num_bags_train': 1000, 'mean_bag_len': 10, 'std_bag_len': 2, 'num_epochs': 30},
                {'target_classes': [9], 'num_bags_train': 500, 'mean_bag_len': 20, 'std_bag_len': 5, 'num_epochs': 30},
                {'target_classes': [9], 'num_bags_train': 250, 'mean_bag_len': 50, 'std_bag_len': 10, 'num_epochs': 30},
                
                {'target_classes': np.arange(10).tolist(), 'num_bags_train': 2000, 'mean_bag_len': 5, 'std_bag_len': 1, 'num_epochs': 30},
                {'target_classes': np.arange(10).tolist(), 'num_bags_train': 1000, 'mean_bag_len': 10, 'std_bag_len': 2, 'num_epochs': 30},
                {'target_classes': np.arange(10).tolist(), 'num_bags_train': 500, 'mean_bag_len': 20, 'std_bag_len': 10, 'num_epochs': 30},
        ],

        'cifar10': [
                {'target_classes': [3], 'num_bags_train': 5000, 'mean_bag_len': 5, 'std_bag_len': 1, 'num_epochs': 100},
                {'target_classes': [3], 'num_bags_train': 2500, 'mean_bag_len': 10, 'std_bag_len': 2, 'num_epochs': 100},
                {'target_classes': [3], 'num_bags_train': 1250, 'mean_bag_len': 20, 'std_bag_len': 5, 'num_epochs': 100},
                
                {'target_classes': np.arange(10).tolist(), 'num_bags_train': 10000, 'mean_bag_len': 5, 'std_bag_len': 1, 'num_epochs': 100},
                {'target_classes': np.arange(10).tolist(), 'num_bags_train': 5000, 'mean_bag_len': 10, 'std_bag_len': 2, 'num_epochs': 100},
                {'target_classes': np.arange(10).tolist(), 'num_bags_train': 2500, 'mean_bag_len': 20, 'std_bag_len': 5, 'num_epochs': 100},
        ],
        
        'svhn': [
                {'target_classes': [3], 'num_bags_train': 5000, 'mean_bag_len': 5, 'std_bag_len': 1, 'num_epochs': 100},
                {'target_classes': [3], 'num_bags_train': 2500, 'mean_bag_len': 10, 'std_bag_len': 2, 'num_epochs': 100},
                {'target_classes': [3], 'num_bags_train': 1250, 'mean_bag_len': 20, 'std_bag_len': 5, 'num_epochs': 100},
                
                {'target_classes': np.arange(10).tolist(), 'num_bags_train': 10000, 'mean_bag_len': 5, 'std_bag_len': 1, 'num_epochs': 100},
                {'target_classes': np.arange(10).tolist(), 'num_bags_train': 5000, 'mean_bag_len': 10, 'std_bag_len': 2, 'num_epochs': 100},
                {'target_classes': np.arange(10).tolist(), 'num_bags_train': 2500, 'mean_bag_len': 20, 'std_bag_len': 5, 'num_epochs': 100},
        ],
        
        'stl10': [
                {'target_classes': [3], 'num_bags_train': 1000, 'mean_bag_len': 5, 'std_bag_len': 1, 'num_epochs': 100},
                {'target_classes': [3], 'num_bags_train': 500, 'mean_bag_len': 10, 'std_bag_len': 2, 'num_epochs': 100},
                # {'target_classes': [3], 'num_bags_train': 250, 'mean_bag_len': 20, 'std_bag_len': 5, 'num_epochs': 100},
                
                {'target_classes': np.arange(10).tolist(), 'num_bags_train': 2000, 'mean_bag_len': 5, 'std_bag_len': 1,  'num_epochs': 100},
                {'target_classes': np.arange(10).tolist(), 'num_bags_train': 1000, 'mean_bag_len': 10, 'std_bag_len': 2,  'num_epochs': 100},
                # {'target_classes': np.arange(10).tolist(), 'num_bags_train': 500, 'mean_bag_len': 20, 'std_bag_len': 5, 'num_epochs': 100},
        ],
        

        'cifar100': [
                # {'target_classes': [4, 30, 55, 72, 95] + [1, 32, 67, 73, 91] + [3, 42, 43, 88, 97] + [15, 19, 21, 31, 38] + [34, 63, 64, 66, 75] + [26, 45, 77, 79, 99] + [27, 29, 44, 78, 93] +  [36, 50, 65, 74, 80], 'num_bags_train': 5000, 'mean_bag_len': 5, 'std_bag_len': 1, 'num_epochs': 100},
                # {'target_classes': [4, 30, 55, 72, 95] + [1, 32, 67, 73, 91] + [3, 42, 43, 88, 97] + [15, 19, 21, 31, 38] + [34, 63, 64, 66, 75] + [26, 45, 77, 79, 99] + [27, 29, 44, 78, 93] +  [36, 50, 65, 74, 80], 'num_bags_train': 2500, 'mean_bag_len': 10, 'std_bag_len': 2, 'num_epochs': 100},
                # {'target_classes': [3], 'num_bags_train': 5000, 'mean_bag_len': 10, 'std_bag_len': 2, 'balanced_bags': True, 'num_epochs': 100},
                # {'target_classes': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90], 'num_bags_train': 1000, 'mean_bag_len': 50, 'std_bag_len': 10, 'num_epochs': 100},
                
                {'target_classes': np.arange(100).tolist(), 'num_bags_train': 10000, 'mean_bag_len': 5, 'std_bag_len': 1, 'num_epochs': 100},
                {'target_classes': np.arange(100).tolist(), 'num_bags_train': 5000, 'mean_bag_len': 10, 'std_bag_len': 2, 'num_epochs': 100},
                # {'target_classes': np.arange(100).tolist(), 'num_bags_train': 1000, 'mean_bag_len': 50, 'std_bag_len': 10, 'num_epochs': 100},
        ],

        'imagenet100': [
            {'target_classes': np.arange(100).tolist(), 'num_bags_train': 20000, 'mean_bag_len': 3, 'std_bag_len': 1, 'num_epochs': 100},
            {'target_classes': np.arange(100).tolist(), 'num_bags_train': 20000, 'mean_bag_len': 5, 'std_bag_len': 1, 'num_epochs': 100},
        ]
        
    }
    # seeds = [42, 231206, 3407]
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
            weight_decay = 1e-4
            net = 'wrn_28_2'
            img_size = 32
            batch_size = 4
            crop_ratio = 0.875

        elif dataset == 'mnist' or dataset == 'fmnist':
            # net = 'WideResNet'
            num_classes = 10
            optim = 'AdamW'
            lr = 0.0005
            weight_decay = 1e-4
            net = 'lenet5'
            img_size = 28
            batch_size = 2
            crop_ratio = 1.0
        
        elif dataset == 'svhn':
            
            num_classes = 10
            optim = 'AdamW'
            lr = 0.001
            weight_decay = 1e-4
            net = 'wrn_28_2'
            img_size = 32
            batch_size = 4
            crop_ratio = 0.875
        
        elif dataset == 'stl10':
            
            num_classes = 10
            optim = 'AdamW'
            lr = 0.001
            weight_decay = 1e-4
            net = 'resnet18'
            img_size = 96
            batch_size = 4
            crop_ratio = 0.875
            

        elif dataset == 'cifar100':
            
            
            num_classes = 100
            optim = 'AdamW'
            lr = 0.001
            weight_decay = 1e-4
            net = 'resnet18'
            img_size = 32 
            batch_size = 4
            crop_ratio = 0.875
        
        elif dataset == 'imagenet100':
            
            num_classes = 100
            optim = 'AdamW'
            lr = 0.001
            weight_decay = 1e-4
            net = 'resnet34' # 'resnet50'
            img_size = 224
            batch_size = 8
            crop_ratio = 0.875
        
        for algorithm in algorithms:
            for setting_dict in  settings_dict[dataset]:
                
                if algorithm == 'uum_proportion' and setting_dict['mean_bag_len'] > 5:
                    continue
                
                for seed in seeds:
                    port = dist_port[count]
                    # prepare the configuration file
                    cfg = create_config_proportion(
                        seed=seed, algorithm=algorithm,
                        dataset=dataset, net=net, num_classes=num_classes, batch_size=batch_size, img_size=img_size, crop_ratio=crop_ratio, optim=optim, lr=lr, weight_decay=weight_decay, 
                        port=port, setting='classic_cv', **setting_dict)
                    count += 1
                    create_configuration(cfg, config_file)


if __name__ == '__main__':
    exp_classic_cv()
    # exp_rcr_cv()