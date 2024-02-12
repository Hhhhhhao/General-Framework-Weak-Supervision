import os


def create_configuration(cfg, cfg_file):
    cfg['save_name'] = "{alg}_{dataset}_lb{num_lb}_n{noise_ratio}_p{partial_ratio}_{seed}".format(
        alg=cfg['algorithm'],
        dataset=cfg['dataset'],
        num_lb=cfg['num_labels'],
        noise_ratio=cfg['noise_ratio'],
        partial_ratio=cfg['partial_ratio'],
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


def create_config(seed, dataset, net, num_classes, num_labels, partial_ratio, noise_ratio, img_size, port, weight_decay):
    cfg = {}
    cfg['algorithm'] = 'imp_partial_noisy_ulb'

    # save config
    
    cfg['save_dir'] = r'./saved_models/partial_noisy_ulb/classic_cv'
    cfg['save_name'] = None
    cfg['resume'] = False
    cfg['load_path'] = None
    cfg['overwrite'] = True
    cfg['use_tensorboard'] = True
    cfg['use_wandb'] = False
    
    # algorithm config
    cfg['epoch'] = 150
    cfg['num_train_iter'] = 300 * 1024 # 2 ** 19
    cfg['num_eval_iter'] = 1024
    cfg['num_log_iter'] = 256
    cfg['num_labels'] = num_labels
    cfg['batch_size'] = 64
    cfg['eval_batch_size'] = 256
    
    cfg['ema_m'] = 0.999
    cfg['crop_ratio'] = 0.875
    cfg['img_size'] = img_size

    # optim config
    cfg['optim'] = 'SGD'
    cfg['lr'] = 0.02
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
    # cfg['data_dir'] = '/mnt/data/dataset/imprecise_datasets'
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

    
    # em specific config
    cfg['uratio'] = 3
    cfg['include_lb_to_ulb'] = True
    cfg['average_entropy_loss'] = True
    cfg['noise_ratio'] = noise_ratio
    cfg['partial_ratio'] = partial_ratio
    
    if dataset == 'cifar10':
        cfg['noise_matrix_scale'] = 1.0
    else:
        cfg['noise_matrix_scale'] = 2.0

    return cfg




def exp_classic_cv():
    config_file = r'./config/partial_noisy_ulb/classic_cv'
    save_path = r'./saved_models/partial_noisy_ulb/classic_cv'

    if not os.path.exists(config_file):
        os.makedirs(config_file)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    datasets = ['cifar100', 'cifar10']
    num_labels_dict = {
        'cifar10': [1000, 5000, 50000],
        'cifar100': [5000, 10000, 50000],
        # 'svhn': [40, 250, 1000],
        # 'stl10': [40, 250, 1000],
    }
    noise_ratio_dict = {
        'cifar10': [0.1, 0.2, 0.3],
        'cifar100': [0.1, 0.2, 0.3],
    }
    partial_ratio_dict = {
        'cifar10': [0.1, 0.3, 0.5],
        'cifar100': [0.01, 0.05, 0.1],
    } 
    # seeds = [0, 1, 2]
    # seeds = [0, 42, 123]
    
    seeds = [42]
    dist_port = range(10001, 11120, 1)
    count = 0
    
    for dataset in datasets:
        for num_labels in num_labels_dict[dataset]:
            for noise_ratio in noise_ratio_dict[dataset]:
                for partial_ratio in partial_ratio_dict[dataset]:
                    for seed in seeds:
                        
                        # change the configuration of each dataset
                        if dataset == 'cifar10':
                            # net = 'WideResNet'
                            num_classes = 10
                            weight_decay = 1e-3
                            net = 'resnet18'
                            img_size = 32

                        elif dataset == 'cifar100':
                            # net = 'WideResNet'
                            num_classes = 100
                            weight_decay = 1e-3
                            # depth = 28
                            # widen_factor = 8
                            net = 'resnet18'
                            # net = 'wrn_28_2'
                            img_size = 32 

                        
                        port = dist_port[count]
                        # prepare the configuration file
                        cfg = create_config(seed, dataset, net, num_classes, num_labels, partial_ratio, noise_ratio, img_size, port, weight_decay)
                        count += 1
                        create_configuration(cfg, config_file)
                        


if __name__ == '__main__':
    exp_classic_cv()