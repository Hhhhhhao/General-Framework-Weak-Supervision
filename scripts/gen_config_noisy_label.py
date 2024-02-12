import os


def create_configuration(cfg, cfg_file):
    cfg['save_name'] = "{alg}_{dataset}_{noise_type}_{noise_ratio}_{seed}".format(
        alg=cfg['algorithm'],
        dataset=cfg['dataset'],
        noise_type=cfg['noise_type'],
        noise_ratio=cfg['noise_ratio'],
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


def create_config(seed, dataset, net, epochs, batch_size, num_classes, noise_ratio, noise_type, img_size,crop_ratio, port, optim, lr, weight_decay):
    cfg = {}
    cfg['algorithm'] = 'imp_noisy_label'

    # save config
    cfg['save_dir'] = './saved_models/noisy_label/classic_cv'
    cfg['save_name'] = None
    cfg['resume'] = False
    cfg['load_path'] = None
    cfg['overwrite'] = True
    cfg['use_tensorboard'] = True
    cfg['use_wandb'] = False
    
    # algorithm config
    cfg['epoch'] = epochs
    cfg['num_eval_iter'] = None
    cfg['num_log_iter'] = 50
    cfg['batch_size'] = batch_size
    cfg['eval_batch_size'] = 256
    cfg['noise_ratio'] = noise_ratio
    cfg['noise_type'] = noise_type
    
    cfg['ema_m'] = 0.0
    cfg['crop_ratio'] = crop_ratio
    cfg['img_size'] = img_size

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

    # data config
    cfg['data_dir'] = './data'
    cfg['dataset'] = dataset
    cfg['num_classes'] = num_classes
    cfg['num_workers'] = 4

    # basic config
    cfg['seed'] = seed

    # distributed config
    cfg['world_size'] = 1
    cfg['rank'] = 0
    cfg['multiprocessing_distributed'] = True
    cfg['dist_url'] = 'tcp://127.0.0.1:' + str(port)
    cfg['dist_backend'] = 'nccl'
    cfg['gpu'] = None

    # other config
    cfg['overwrite'] = True
    cfg['amp'] = False
    cfg['strong_aug'] = True

    # em specific config
    cfg['average_entropy_loss'] = True
    
    # noise matrix config
    if dataset == 'cifar10':
        cfg['noise_matrix_scale'] = 1.0
    elif dataset == 'cifar100':
        cfg['noise_matrix_scale'] = 2.0
        if noise_ratio > 0.5 or noise_ratio == 'asym':
            cfg['noise_matrix_scale'] = 2.5
        elif noise_ratio == 0.5:
            cfg['noise_matrix_scale'] = 2.25
    elif dataset == 'webvision':
        cfg['noise_matrix_scale'] = 2.5
    elif dataset == 'clothing1m':
        cfg['noise_matrix_scale'] = 0.5
    elif dataset == 'cifar10n':
        cfg['noise_matrix_scale'] = 1.0
    elif dataset == 'cifar100n':
        cfg['noise_matrix_scale'] = 2.0
        

    return cfg




def exp_classic_cv():
    config_file = r'./config/noisy_label/classic_cv'
    save_path = r'./saved_models/noisy_label/classic_cv'

    if not os.path.exists(config_file):
        os.makedirs(config_file)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    datasets = ['cifar100', 'cifar10', 'webvision', 'clothing1m', 'cifar10_asym', 'cifar100_asym', 'cifar10n', 'cifar100n']
    noise_ratio_dict = {
        'cifar10_asym': {'ratio': [0.4], 'type':'asym'},
        'cifar10': {'ratio': [0.2, 0.5, 0.8], 'type':'sym'},
        'cifar100':{'ratio': [0.2, 0.5, 0.8], 'type':'sym'},
        'cifar100_asym': {'ratio': [0.4], 'type':'asym'},
        'webvision': {'ratio': [1.0], 'type':'ins'},
        'clothing1m': {'ratio': [1.0], 'type':'ins'},
        'cifar10n': {'ratio': ['clean_label', 'worse_label', 'aggre_label', 'random_label1', 'random_label2', 'random_label3'], 'type':'ins'},
        'cifar100n': {'ratio': ['clean_label', 'noisy_label'], 'type':'ins'},
    }
    # seeds = [666, 42, 123]
    seeds = [42]
    dist_port = range(10001, 11120, 1)
    count = 0
    
    for dataset in datasets:
        noise_setting =  noise_ratio_dict[dataset]
        noise_ratio_list, noise_type = noise_setting['ratio'],  noise_setting['type']
        for noise_ratio in noise_ratio_list:  
            for seed in seeds:
                
                # change the configuration of each dataset
                if dataset == 'cifar10' or dataset == 'cifar10_asym':
                    # net = 'WideResNet'
                    num_classes = 10
                    # weight_decay = 1e-3
                    # weight_decay = 1e-3
                    weight_decay = 1e-3
                    net = 'preact_resnet18'
                    img_size = 32
                    
                    epochs = 300
                    batch_size = 128
                    lr = 0.02
                    optim = 'SGD'
                    crop_ratio = 0.875
                    
                    dataset_name = 'cifar10'
                
                elif dataset == 'cifar10n':
                    
                    num_classes = 10
                    weight_decay = 1e-3
                    net = 'resnet34'
                    img_size = 32
                    
                    epochs = 300
                    batch_size = 128
                    lr = 0.02
                    optim = 'SGD'
                    crop_ratio = 0.875
                    
                    dataset_name = 'cifar10n'
                    
                    

                elif dataset == 'cifar100' or dataset == 'cifar100_asym':
                    # net = 'WideResNet'
                    num_classes = 100
                    weight_decay = 1e-3
                    net = 'preact_resnet18'
                    img_size = 32 
                    
                    epochs = 300
                    batch_size = 128
                    lr = 0.02
                    optim = 'SGD'
                    crop_ratio = 0.875
                    
                    dataset_name = 'cifar100'
                
                elif dataset == 'cifar100n':
                    
                    num_classes = 100
                    weight_decay = 1e-3
                    net = 'resnet34'
                    img_size = 32 
                    
                    epochs = 300
                    batch_size = 128
                    lr = 0.02
                    optim = 'SGD'
                    crop_ratio = 0.875
                    
                    dataset_name = 'cifar100n'
                    
                
                elif dataset == 'webvision':
                    
                    num_classes = 50
                    weight_decay = 5e-4
                    net = 'inception_resnet_v2'
                    img_size = 299
                    crop_ratio = 0.95
                    
                    epochs = 100
                    batch_size = 32
                    lr = 0.02
                    optim = 'SGD'
                    
                    dataset_name = 'webvision'
                
                elif dataset == 'clothing1m':
                    num_classes = 14
                    weight_decay = 1e-3
                    net = 'resnet50_pretrained'
                    img_size = 224
                    crop_ratio = 0.875
                    
                    epochs = 15
                    batch_size = 64
                    lr = 0.002
                    optim = 'SGD'
                    
                    dataset_name = 'clothing1m'
                
                
                port = dist_port[count]
                # prepare the configuration file
                cfg = create_config(seed, dataset_name, net, epochs, batch_size, num_classes, noise_ratio, noise_type, img_size, crop_ratio, port, optim, lr, weight_decay)
                count += 1
                create_configuration(cfg, config_file)         



if __name__ == '__main__':
    exp_classic_cv()