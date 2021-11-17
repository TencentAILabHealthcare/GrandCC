import argparse
import os

import torch


### Parser

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp', type=str, default='test', help='model_name')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument('--which_graph', type=str, default='similarity', help='HumanNet or similarity matrix.')
    parser.add_argument('--layer_norm', type=str, default='True', help='Whether to use layer normalization.')
    parser.add_argument('--drop_edge', default=0.8, type=float, help='the percent of preserving edges')

    parser.add_argument('--external_file', type=str, default='test',
                        help='The file name of the external test data uploaded by the user.')

    ### whether include the unlabelled data to form semi-supervised learning
    parser.add_argument('--semi', type=str, default='False', help='Whether use unlabelled data.')
    parser.add_argument('--labeled_bs', type=int, default=8, help='Number of labeled samples in each batch.')

    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--batch_size', type=int, default=8, help='Number of batches to train/test for. Default: 64')
    parser.add_argument('--num_epochs', type=int, default=8, help='Number of epochs for training')
    parser.add_argument('--model_dir', type=str, default='./save_models/')
    parser.add_argument('--results_dir', type=str, default='./results/')

    parser.add_argument('--lambda_reg', type=float, default=3e-4)
    parser.add_argument('--lambda_nll', type=float, default=1)

    ### For mean teacher
    parser.add_argument('--lambda_consistency', type=float, default=1)
    parser.add_argument('--consistency', type=float, default=1, help='consistency')
    parser.add_argument('--consistency_rampup', type=float, default=10, help='consistency_rampup')

    parser.add_argument('--label_dim', type=int, default=4, help='number of classes')
    parser.add_argument('--input_dim', type=int, default=1, help="input_size for omic vector")
    parser.add_argument('--num_nodes', type=int, default=5973, help="number of nodes in the graph")
    parser.add_argument('--which_layer', type=str, default='cat_all', help='which GAT layer as the input of fc layers.')
    parser.add_argument('--act_type', type=str, default="none", help='activation function')

    parser.add_argument('--optimizer_type', type=str, default='adam')
    parser.add_argument('--lr_policy', default='linear', type=str, help='5e-4 for Adam | 1e-3 for AdaBound')
    parser.add_argument('--lr', default=0.0001, type=float, help='5e-4 for Adam | 1e-3 for AdaBound')
    parser.add_argument('--final_lr', default=0.1, type=float, help='Used for AdaBound')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Used for Adam. L2 Regularization on weights.')
    parser.add_argument('--fc_dropout', default=0, type=float, help='Dropout rate')
    parser.add_argument('--GAT_dropout', default=0, type=float, help='Dropout rate for the GAT layer')
    parser.add_argument('--alpha', default=0.2, type=float, help='Used in the leaky relu')
    parser.add_argument('--patience', default=0.005, type=float)
    parser.add_argument('--gpu_ids', type=str, default='3,4,5', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

    opt = parser.parse_known_args()[0]
    print_options(parser, opt)

    return opt


def print_options(parser, opt):
    """Print and save options

    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    # save to the disk
    mkdirs(opt.model_dir)
    mkdirs(opt.results_dir)
    file_name = os.path.join(opt.model_dir, '{}_opt.txt'.format('train'))
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
