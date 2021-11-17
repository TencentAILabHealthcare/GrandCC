"""
Load the data from different cohorts separately.
"""
import logging
import torch
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from statsmodels import robust
from collections import Counter

from torch_geometric.utils import dense_to_sparse, to_dense_batch
from torch_geometric.data import Data, DataLoader
from torch.utils.data import Dataset


def construct_graph(opt, features, keep_idx):

    features = features.squeeze().detach().numpy()
    print("================", features.shape)
    ######################################
    ########### Similarity graph #########
    ######################################
    if opt.which_graph == "similarity":
        logging.info("Graph initialization with similarity matrix.")
        similarity_matrix = cosine_similarity(features.T) # max: 1, min: 0.28
        adj_thresh = opt.adj_thresh
        adj_matrix = torch.LongTensor(np.where(similarity_matrix > adj_thresh, 1, 0))
        edge_index = dense_to_sparse(torch.LongTensor(adj_matrix))[0]
        print("edge index:", edge_index)
        logging.info("Number of edges: {:04d}\n".format(edge_index.shape[1]))
        logging.info('Number of singleton nodes: {:04d}\n'.format(
            torch.sum(torch.sum(adj_matrix, dim=1) == 1).detach().numpy()))

        edge_matrix = edge_index.cpu().detach().numpy()
        pd.DataFrame(edge_matrix).to_csv('./similarity_graphs/' + 'TCGA_sim_graph.csv')
        print(edge_matrix)

    return adj_matrix, edge_index



def load_train_data():
    root_path = './BRCA_data_label/intersect_genes/'
    file_path = root_path + 'TCGA_10574.txt'
    train_data = pd.read_table(file_path, sep=' ', header=0)
    sample_id = list(train_data.index)
    gene_names = list(train_data)
    train_data = np.array(train_data)
    norm_train_data = preprocessing.MinMaxScaler().fit_transform(train_data)
    all_labels = pd.read_table(root_path + 'PAM50_label.txt', sep=' ', header=0)\
        .replace(['Basal', 'LumA', 'LumB', 'Her2', 'Normal'], [0, 1, 2, 3, 4])
    train_labels = np.array(all_labels.loc[all_labels['sample'].isin(sample_id)]['PAM50'])
    # print(sample_id)
    # print(gene_names)
    print("Train features:", train_data.shape)
    print("Train labels:", train_labels.shape)
    print(Counter(train_labels))

    return train_data, norm_train_data, train_labels, sample_id, gene_names


def load_test_data(train_genes, tr_mean, tr_std, cohorts=np.array(['NKI', 'TRANSBIG', 'UNT', 'UPP', 'VDX'])):

    root_path = './BRCA_data_label/intersect_genes/'
    all_test_data, all_test_labels, all_sample_ids, all_test_cohorts = None, None, None, None
    all_labels = pd.read_table(root_path + 'PAM50_label.txt', sep=' ', header=0)\
        .replace(['Basal', 'LumA', 'LumB', 'Her2', 'Normal'], [0, 1, 2, 3, 4])

    for i in range(cohorts.shape[0]):
        test_file = root_path + cohorts[i] + '_10574.txt'
        test_data = pd.read_table(test_file, sep=' ', header=0)
        test_genes = list(test_data)
        if train_genes != test_genes:
            print("genes in the train and test set doesn't match.")

        sample_id = np.array(list(test_data.index))
        test_labels = np.array(all_labels.loc[all_labels['sample'].isin(sample_id)]['PAM50'])
        test_cohort = np.array(all_labels.loc[all_labels['cohort'] == cohorts[i]]['cohort'])
        test_data = np.array(test_data)
        print(Counter(test_labels))

        all_sample_ids = sample_id if all_sample_ids is None else np.concatenate((all_sample_ids, sample_id))
        all_test_data = test_data if all_test_data is None else np.concatenate((all_test_data, test_data), 0)
        all_test_labels = test_labels if all_test_labels is None else np.concatenate((all_test_labels, test_labels))
        all_test_cohorts = test_cohort if all_test_cohorts is None else np.concatenate((all_test_cohorts, test_cohort))

    print("Test samples:", all_sample_ids)
    print("Test data:", all_test_data.shape)

    return all_test_data, all_test_labels, all_sample_ids, all_test_cohorts




### This is for the data uploaded by users.
def load_user_data(file_name, train_genes):
    """
    The data uploaded by the user must be in "txt" file, and the genes must be represented with entrez ids.
    If the overlap between the train_genes and test_genes is less than 70%, the model will exit without making predictions.
    If the overlap between the train_genes and test_genes is larger than 70%, the model will match the genes order and
    impute the missing genes with all 0.
    """
    test_file = './external_dataset/' + str(file_name) + '.txt'
    user_data = pd.read_table(test_file, sep=' ', header=0)
    test_genes = list(user_data)

    overlap = set(train_genes).intersection(set(test_genes))
    print("Number of overlapped genes", len(list(overlap)))

    new_user_data = pd.DataFrame(columns=train_genes)

    if len(list(overlap)) < 0.7*len(train_genes):
        match_flag = False
    else:
        match_flag = True

        for i in range(len(train_genes)):
            # print(train_genes[i], train_genes[i] in overlap)
            if train_genes[i] in overlap:
                new_user_data[train_genes[i]] = user_data[train_genes[i]]
            else:
                new_user_data[train_genes[i]] = np.zeros(user_data.shape[0])

    print("Can we use this data?", match_flag)

    sample_ids = np.array(list(user_data.index))
    test_labels = np.zeros(len(list(user_data.index)))

    return np.array(new_user_data), test_labels, sample_ids, match_flag




def load_features_labels(retain_dim=10574):

    tr_feat, norm_tr_feat, tr_labels, tr_sample_ids, gene_names = load_train_data()
    tr_mean, tr_std = np.mean(tr_feat, axis=0), np.var(tr_feat, axis=0)
    te_feat, te_labels, te_sample_ids, all_test_cohorts = load_test_data(train_genes=gene_names,
                                                                         tr_mean=tr_mean, tr_std=tr_std)

    mad_score = robust.mad(tr_feat, axis=0)
    sort_idx = np.argsort(-mad_score)
    keep_idx = sort_idx[:retain_dim]
    print(keep_idx)

    tr_features = torch.FloatTensor(tr_feat[:, keep_idx].reshape(-1, retain_dim, 1)) #.requires_grad_()
    norm_tr_features = torch.FloatTensor(norm_tr_feat[:, keep_idx].reshape(-1, retain_dim, 1))
    te_features = torch.FloatTensor(te_feat[:, keep_idx].reshape(-1, retain_dim, 1)) #.requires_grad_()
    tr_labels, te_labels = torch.LongTensor(tr_labels), torch.LongTensor(te_labels)

    return tr_features, norm_tr_features, te_features, tr_labels, te_labels, keep_idx, \
           all_test_cohorts, tr_sample_ids, te_sample_ids



def load_external_features(file_name, retain_dim=3000):

    tr_feat, norm_tr_feat, tr_labels, tr_sample_ids, gene_names = load_train_data()

    # print("train genes", gene_names)
    mad_score = robust.mad(tr_feat, axis=0)
    sort_idx = np.argsort(-mad_score)
    keep_idx = sort_idx[:retain_dim]
    remain_genes = list(np.array(gene_names)[keep_idx])
    # print("remaining genes:", remain_genes)

    te_feat, te_labels, te_sample_ids, match_flag = load_user_data(file_name, train_genes=remain_genes)

    if match_flag:
        te_features = torch.FloatTensor(te_feat.reshape(-1, retain_dim, 1)) #.requires_grad_()
        te_labels = torch.LongTensor(te_labels)

    else:
        te_features = None

    return te_features, te_labels, te_sample_ids, match_flag

class BRCA_Dataset(Dataset):
    def __init__(self, feature, label, edge):
        super(BRCA_Dataset, self).__init__()
        self.feature = feature
        self.label = label
        self.edge = edge

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        data = Data(x=self.feature[idx], edge_index=self.edge, y=self.label[idx])
        return data



if __name__ == "__main__":
    load_features_labels()
