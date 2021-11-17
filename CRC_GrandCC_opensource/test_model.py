"""
Usage:
python3 test_model.py --which_graph similarity --exp GNN_sim_graph

The pretrained model is saved at './save_models/GNN_sim_graph.pt'
"""

import os
import logging
import numpy as np
import random
import pickle

import torch
from torch_geometric.utils import dense_to_sparse, to_dense_batch

# Env
from utils import *
from model_GAT_v4 import *
from options import parse_args
from train_test_new import test

from load_CRCSC_data import load_features_labels, CRC_Dataset, construct_graph

### 1. Initializes parser and device
opt = parse_args()
device = torch.device('cuda:0')
num_splits = 5
results = []


def load_graph(opt):

    if opt.which_graph == "similarity":
        logging.info("Graph initialization with similarity graph.")
        root_path = './similarity_graphs/'
        edge_index = np.array(pd.read_csv(root_path + "TCGA_sim_graph.csv").iloc[:, 1:]).astype(float)
        edge_index = torch.LongTensor(edge_index)
        # print(edge_index.shape)
        logging.info("Number of edges: {:04d}\n".format(edge_index.shape[1]))

    return edge_index


if __name__ == "__main__":

    tr_features, norm_tr_features, te_features, tr_labels, te_labels, keep_idx, \
        all_test_cohorts, tr_sample_ids, te_sample_ids = load_features_labels(retain_dim=opt.num_nodes)
    edge_index = load_graph(opt)

    print("train data:", tr_features.shape, tr_labels.shape)
    print("test data:", te_features.shape, te_labels.shape)

    train_dataset = CRC_Dataset(feature=tr_features, label=tr_labels, edge=edge_index)
    test_dataset = CRC_Dataset(feature=te_features, label=te_labels, edge=edge_index)

    load_path = opt.model_dir + opt.exp + '.pt'
    model_ckpt = torch.load(load_path, map_location=device)

    #### Loading Env
    model_state_dict = model_ckpt['model_state_dict']
    if hasattr(model_state_dict, '_metadata'):
        del model_state_dict._metadata

    model = MLA_GNN(opt).cuda()

    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    print('Loading the model from %s' % load_path)
    model.load_state_dict(model_state_dict)

    ### 3.2 Test the model.
    loss_train, grad_acc_train, pred_train, tr_features, tr_fc_features = test(opt, model, train_dataset)
    loss_test, grad_acc_test, pred_test, te_features, te_fc_features = test(opt, model, test_dataset)
    train_probs = np.exp(pred_train[0])/np.expand_dims(np.sum(np.exp(pred_train[0]), axis=1), axis=1)
    test_probs = np.exp(pred_test[0])/np.expand_dims(np.sum(np.exp(pred_test[0]), axis=1), axis=1)
    print("test preds:", test_probs)
    print("train fc features:", tr_fc_features.shape)
    print("test fc features:", te_fc_features.shape)

    pd.DataFrame(tr_fc_features, index=tr_sample_ids).to_csv('./results/GNN_sim_graph_train_features.csv')
    pd.DataFrame(te_fc_features, index=te_sample_ids).to_csv('./results/GNN_sim_graph_test_features.csv')

    all_metrics = compute_cohort_metrics(pred_test[0], np.uint(pred_test[1]), all_test_cohorts)
    print(all_metrics)
    train_results = {'sample_id': tr_sample_ids, 'CMS1_prob': train_probs[:, 0], 'CMS2_prob': train_probs[:, 1],
                    'CMS3_prob': train_probs[:, 2], 'CMS4_prob': train_probs[:, 3],
                    'GNN_pred': np.argmax(pred_train[0], axis=1), 'CMS_network': np.uint(pred_train[1])}
    test_results = {'sample_id': te_sample_ids, 'CMS1_prob': test_probs[:, 0], 'CMS2_prob': test_probs[:, 1],
                    'CMS3_prob': test_probs[:, 2], 'CMS4_prob': test_probs[:, 3], 
                    'GNN_pred': np.argmax(pred_test[0], axis=1), 'CMS_network': np.uint(pred_test[1])}
    pd.DataFrame(train_results).to_csv('./results/GNN_sim_graph_preds_train.csv')
    pd.DataFrame(test_results).to_csv('./results/GNN_sim_graph_preds.csv')

    print("[Final] Apply model to training set: Loss: %.10f, Acc: %.4f" % (loss_train, grad_acc_train))
    print("[Final] Apply model to testing set: Loss: %.10f, Acc: %.4f" % (loss_test, grad_acc_test))
    results.append(grad_acc_test)

