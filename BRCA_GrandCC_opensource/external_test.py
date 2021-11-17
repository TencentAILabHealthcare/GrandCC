"""
Usage:
python3 external_test.py --which_graph similarity --exp GNN_sim_graph --num_nodes 3000 --external_file BRCA_test_data_8459genes
python3 external_test.py --which_graph similarity --exp GNN_sim_graph --num_nodes 3000 --external_file BRCA_test_data_1057genes

The pretrained model is saved at './save_models/GNN_sim_graph.pt'
"""

import logging
import numpy as np
import torch


# Env
from utils import *
from model_GAT_v4 import *
from options import parse_args
from train_test_new import test

from load_BRCA_data import load_external_features, BRCA_Dataset, construct_graph

### 1. Initializes parser and device
opt = parse_args()
device = torch.device('cuda:0')
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

    te_features, te_labels, te_sample_ids, match_flag = load_external_features(
        file_name = opt.external_file, retain_dim=opt.num_nodes)

    if match_flag:
        edge_index = load_graph(opt)

        test_dataset = BRCA_Dataset(feature=te_features, label=te_labels, edge=edge_index)

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
        loss_test, grad_acc_test, pred_test, te_features, te_fc_features = test(opt, model, test_dataset)
        test_probs = np.exp(pred_test[0])/np.expand_dims(np.sum(np.exp(pred_test[0]), axis=1), axis=1)
        print("test preds:", test_probs)

        test_results = {'sample_id': te_sample_ids, 'Basal_prob': test_probs[:, 0], 'LumA_prob': test_probs[:, 1],
                        'LumB_prob': test_probs[:, 2], 'Her2_prob': test_probs[:, 3], 'Normal_prob': test_probs[:, 4],
                        'GNN_pred': np.argmax(pred_test[0], axis=1)}
        pd.DataFrame(test_results).to_csv('./results/external_preds.csv')
