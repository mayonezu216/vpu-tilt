import argparse
from vpu_GNN import *

parser = argparse.ArgumentParser(description='Pytorch Variational Positive Unlabeled Learning')
parser.add_argument('--dataset', default='urine',
                    choices=['cifar10', 'fashionMNIST', 'stl10', 'avila', 'pageblocks', 'grid','urine'])
parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('--val-iterations', type=int, default=30)
parser.add_argument('--batch-size', type=int, default= 100)
parser.add_argument('--num_labeled', type=int, default=3000, help="number of labeled positive samples")
parser.add_argument('--learning-rate', type=float, default=1.25e-5)
parser.add_argument('--epochs', type=int, default=0)
parser.add_argument('--mix-alpha', type=float, default=0.3, help="parameter in Mixup")
parser.add_argument('--lam', type=float, default=0.03, help="weight of the regularizer")
parser.add_argument('--th', type=float, default=0.5, help="threshold of decision")

args = parser.parse_args()
if args.dataset == 'urine':
    from model.model_urine_crossvit import NetworkPhi
    from dataset.dataset_urine import get_urine_loaders as get_loaders
    parser.add_argument('--positive_label_list', type=list, default=[0])

elif args.dataset == 'cifar10':
    from model.model_cifar import NetworkPhi
    from dataset.dataset_cifar import get_cifar10_loaders as get_loaders

    parser.add_argument('--positive_label_list', type=list, default=[0, 1, 8, 9])
elif args.dataset == 'fashionMNIST':
    from model.model_fashionmnist import NetworkPhi
    from dataset.dataset_fashionmnist import get_fashionMNIST_loaders as get_loaders

    parser.add_argument('--positive_label_list', type=list, default=[1, 4, 7])
elif args.dataset == 'stl10':
    from model.model_stl import NetworkPhi
    from dataset.dataset_stl import get_stl10_loaders as get_loaders

    parser.add_argument('--positive_label_list', type=list, default=[0, 2, 3, 8, 9])
elif args.dataset == 'pageblocks':
    from model.model_vec import NetworkPhi
    from dataset.dataset_pageblocks import get_pageblocks_loaders as get_loaders

    parser.add_argument('--positive_label_list', type=list, default=[2, 3, 4, 5])
elif args.dataset == 'grid':
    from model.model_vec import NetworkPhi
    from dataset.dataset_grid import get_grid_loaders as get_loaders

    parser.add_argument('--positive_label_list', type=list, default=[1])
elif args.dataset == 'avila':
    from model.model_vec import NetworkPhi
    from dataset.dataset_avila import get_avila_loaders as get_loaders

    parser.add_argument('--positive_label_list', type=list, default=['A'])
else:
    assert False
args = parser.parse_args()

def get_mean_std(train_loader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in train_loader:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    # print(num_batches)
    # print(channels_sum)
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    return mean, std


def main(config):
    # set up cuda if it is available
    # if torch.cuda.is_available():
    #     CUDA_VISIBLE_DEVICES = 2, 1
        # torch.cuda.set_device(config.gpu)

    # set up the loaders
    if config.dataset == 'urine':
        x_loader, p_loader, val_x_loader, val_p_loader, test_loader = get_loaders(batch_size=config.batch_size,positive_label_list=config.positive_label_list)

    if config.dataset in ['cifar10', 'fashionMNIST', 'stl10']:
        x_loader, p_loader, val_x_loader, val_p_loader, test_loader, idx = get_loaders(batch_size=config.batch_size,
                                                                                       num_labeled=config.num_labeled,
                                                                                       positive_label_list=config.positive_label_list)
    elif config.dataset in ['avila', 'pageblocks', 'grid']:
        x_loader, p_loader, val_x_loader, val_p_loader, test_loader = get_loaders(batch_size=config.batch_size,
                                                                                  num_labeled=config.num_labeled,
                                                                                  positive_label_list=config.positive_label_list)
    loaders = (p_loader, x_loader, val_p_loader, val_x_loader, test_loader)
    # mean, std = get_mean_std(p_loader)
    # print(mean, std)
    # mean, std = get_mean_std(x_loader)
    # print(mean, std)

    # x_iter = iter(test_loader)
    # data_x, label = x_iter.next()
    # print(label)
    # assert 2==3
    # please read the following information to make sure it is running with the desired setting
    print('==> Preparing data')
    print('    # train data: ', len(x_loader.dataset))
    print('    # labeled train data: ', len(p_loader.dataset))
    print('    # test data: ', len(test_loader.dataset))
    print('    # val x data:', len(val_x_loader.dataset))
    print('    # val p data:', len(val_p_loader.dataset))

    # something about saving the model
    checkpoint = get_checkpoint_path(config)
    if not os.path.isdir(checkpoint):
        mkdir_p(checkpoint)

    # call VPU
    run_vpu(config, loaders, NetworkPhi)

if __name__ == '__main__':
    main(args)
