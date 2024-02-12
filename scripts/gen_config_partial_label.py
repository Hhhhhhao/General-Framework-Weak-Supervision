import os


def create_configuration(cfg, cfg_file):
    cfg['save_name'] = "{alg}_{dataset}_{partial_ratio}_{seed}".format(
        alg=cfg['algorithm'],
        dataset=cfg['dataset'],
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


def create_config(seed, alg, dataset, net, num_classes, partial_ratio, num_epochs, batch_size, img_size, port, optim, lr, weight_decay, setting='classic_cv'):
    cfg = {}
    cfg['algorithm'] = alg

    # save config
    cfg['save_dir'] = f'./saved_models/partial_label/{setting}'
    cfg['save_name'] = None
    cfg['resume'] = False
    cfg['load_path'] = None
    cfg['overwrite'] = True
    cfg['use_tensorboard'] = True
    cfg['use_wandb'] = False
    
    # algorithm config
    cfg['epoch'] = num_epochs
    cfg['num_eval_iter'] = None
    cfg['num_log_iter'] = 50
    cfg['batch_size'] = batch_size
    cfg['eval_batch_size'] = 256
    cfg['partial_ratio'] = partial_ratio
    
    cfg['ema_m'] = 0.0
    cfg['crop_ratio'] = 0.875
    cfg['img_size'] = img_size

    # optim config
    cfg['optim'] = optim
    cfg['lr'] = lr
    cfg['momentum'] = 0.9
    cfg['weight_decay'] = weight_decay
    cfg['layer_decay'] = 1.0
    cfg['amp'] = False
    cfg['clip_grad'] = 1.0

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
    cfg['strong_aug'] = True

    if dataset in ['imagenet100', 'imagenet1k']:
        cfg['gpu'] = None
        cfg['multiprocessing_distributed'] = True
        cfg['num_log_iter'] = 25
        cfg['amp'] = True
        cfg['clip'] = 5.0
        cfg['data_dir'] = '/mnt/data/dataset'
        cfg['eval_batch_size'] = 128
        # cfg['data_dir'] = '/media/Bootes/datasets'
    
    cfg['average_entropy_loss'] = False
    if alg == 'lws_partial_label':
        cfg['lw'] = 1.0
    elif alg == 'imp_partial_label':
        cfg['average_entropy_loss'] = True 

    # other config
    cfg['overwrite'] = True
    cfg['amp'] = False

    # em specific config


    return cfg
                

def exp_rcr_cv():
    config_file = r'./config/partial_label/classic_cv'
    save_path = r'./saved_models/partial_label/classic_cv'

    if not os.path.exists(config_file):
        os.makedirs(config_file)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # algorithms = ['imp_partial_label', 'rcr_partial_label', 'lws_partial_label', 'cc_partial_label', 'proden_partial_label']
    algorithms = ['imp_partial_label']
    # algorithms = [ 'pico_partial_label', 'lws_partial_label', 'cc_partial_label', 'proden_partial_label']
    datasets = ['mnist', 'fmnist', 'cifar100', 'cifar10', 'stl10', 'imagenet100']
    partial_ratio_dict = {
        'mnist': [0.1, 0.3, 0.5, 0.7],
        'fmnist': [0.1, 0.3, 0.5, 0.7],
        'cifar10': [0.1, 0.3, 0.5, 0.7],
        'cifar100': [0.01, 0.05, 0.1, 0.2],
        'stl10': [0.1, 0.3],
        'imagenet100': [0.01, 0.05]
    }
    # seeds = [2, 42, 2023]
    seeds = [42]
    dist_port = range(10001, 31120, 1)
    count = 0
    
    for alg in algorithms:
        for dataset in datasets:
            for num_labels in partial_ratio_dict[dataset]:
                for seed in seeds:
                    
                    # change the configuration of each dataset
                    if dataset == 'cifar10':
                        # net = 'WideResNet'
                        num_classes = 10
                        lr = 0.1
                        weight_decay = 1e-4
                        net = 'wrn_34_10'
                        img_size = 32
                        num_epochs = 200
                        batch_size = 64
                        optim = 'SGD'

                    # change the configuration of each dataset
                    if dataset == 'mnist' or dataset == 'fmnist':
                        # net = 'WideResNet'
                        num_classes = 10
                        lr = 0.1
                        weight_decay = 1e-4
                        net = 'lenet5'
                        img_size = 28
                        num_epochs = 200
                        batch_size = 64
                        optim = 'SGD'

                    elif dataset == 'cifar100':
                        # net = 'WideResNet'
                        num_classes = 100
                        lr = 0.1
                        weight_decay = 1e-4
                        net = 'wrn_34_10'
                        img_size = 32
                        num_epochs = 200
                        batch_size = 64
                        optim = 'SGD'

                    elif dataset == 'stl10':
                        # net = 'WideResNet'
                        
                        optim = 'AdamW'
                        num_classes = 10
                        lr = 0.001
                        weight_decay = 1e-4
                        net = 'resnet18'
                        img_size = 96
                        num_epochs = 200
                        batch_size = 64
                    
                    elif dataset == 'imagenet100':
                
                        num_classes = 100
                        optim = 'AdamW'
                        lr = 0.001
                        weight_decay = 1e-4
                        net = 'resnet34' # 'resnet50'
                        img_size = 224
                        batch_size = 32
                        num_epochs = 200
                    
                    port = dist_port[count]
                    # prepare the configuration file
                    cfg = create_config(seed, alg, dataset, net, num_classes, num_labels, num_epochs, batch_size, img_size, port, optim, lr, weight_decay, setting='rcr_cv')
                    count += 1
                    create_configuration(cfg, config_file)



if __name__ == '__main__':
    # exp_classic_cv()
    exp_rcr_cv()