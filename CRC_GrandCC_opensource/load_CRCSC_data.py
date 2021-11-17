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

from torch_geometric.utils import dense_to_sparse, to_dense_batch
from torch_geometric.data import Data, DataLoader
from torch.utils.data import Dataset


def construct_graph(opt, features, keep_idx):

    ######################################
    ########### Similarity graph #########
    ######################################
    if opt.which_graph == "similarity":
        logging.info("Graph initialization with similarity matrix.")
        similarity_matrix = cosine_similarity(features.T)
        # print(np.max(similarity_matrix))
        # pd.DataFrame(similarity_matrix, index=feat_name, columns=feat_name).to_csv("similarity.csv")
        # print("mean similarity:", np.mean(similarity_matrix))
        adj_thresh = 0.9766250264658827
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
    root_path = './CRC_dataset/'
    file_path = root_path + 'tcga.txt'
    train_data = pd.read_table(file_path, sep=' ', header=0)
    sample_id = list(train_data.index)
    gene_names = list(train_data)
    train_data = np.array(train_data)
    norm_train_data = preprocessing.MinMaxScaler().fit_transform(train_data)

    all_labels = pd.read_table(root_path + 'labels.txt', sep=' ', header=0)\
        .replace(['CMS1', 'CMS2', 'CMS3', 'CMS4'], [0, 1, 2, 3])
    train_labels = np.array(all_labels.loc[all_labels['sample'].isin(sample_id)]['CMS_network'])
    print("Train features:", train_data.shape)

    return train_data, norm_train_data, train_labels, sample_id, gene_names


def load_test_data(train_genes, cohorts=np.array(['gse13067', 'gse13294', 'gse14333', 'gse17536', 'gse20916', 'gse2109',
                                                  'gse35896', 'gse37892', 'gse39582', 'kfsyscc', 'petacc3'])):
    root_path = './CRC_dataset/'

    all_test_data, all_test_labels, all_sample_ids, all_test_cohorts = None, None, None, None
    all_labels = pd.read_table(root_path + 'labels.txt', sep=' ', header=0)\
        .replace(['CMS1', 'CMS2', 'CMS3', 'CMS4'], [0, 1, 2, 3])

    for i in range(cohorts.shape[0]):
        test_file = root_path + cohorts[i] + '.txt'
        test_data = pd.read_table(test_file, sep=' ', header=0)
        test_genes = list(test_data)
        # print("Assert the train and test set use the same genes", train_genes == test_genes)
        if train_genes != test_genes:
            print("genes in the train and test set doesn't match.")

        sample_id = np.array(list(test_data.index))
        test_labels = np.array(all_labels.loc[all_labels['sample'].isin(sample_id)]['CMS_network'])
        test_cohort = np.array(all_labels.loc[all_labels['cohort'] == cohorts[i]]['cohort'])

        all_sample_ids = sample_id if all_sample_ids is None else np.concatenate((all_sample_ids, sample_id))
        all_test_data = np.array(test_data) if all_test_data is None \
            else np.concatenate((all_test_data, np.array(test_data)), 0)
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
    print("Ratio of overlaps", len(list(overlap)))

    new_user_data = pd.DataFrame(columns=train_genes)

    if len(list(overlap)) < 0.7*len(train_genes):
        match_flag = False
    else:
        match_flag = True

        for i in range(len(train_genes)):
            if train_genes[i] in overlap:
                new_user_data[train_genes[i]] = user_data[train_genes[i]]
            else:
                new_user_data[train_genes[i]] = 0.0

    print("Can we use this data?", match_flag)

    sample_ids = np.array(list(user_data.index))
    test_labels, test_cohorts = np.zeros(len(list(user_data.index))), None

    return np.array(new_user_data), test_labels, sample_ids, test_cohorts, match_flag


def load_external_features(file_name):
    _, _, _, _, gene_names = load_train_data()
    te_feat, te_labels, te_sample_ids, all_test_cohorts, match_flag = load_user_data(file_name, train_genes=gene_names)

    if match_flag:
        te_features = torch.FloatTensor(te_feat.reshape(-1, 5973, 1))
        te_labels = torch.LongTensor(te_labels)

    else:
        te_features = None

    return te_features, te_labels, all_test_cohorts, te_sample_ids, match_flag

def load_features_labels(retain_dim=5973):

    tr_feat, norm_tr_feat, tr_labels, tr_sample_ids, gene_names = load_train_data()
    te_feat, te_labels, te_sample_ids, all_test_cohorts = load_test_data(train_genes=gene_names)
    # print("original test features:", te_feat)


    if retain_dim < 5973:
        print("selecting features")
        mad_score = robust.mad(tr_feat, axis=0)
        sort_idx = np.argsort(-mad_score)
        keep_idx = sort_idx[:retain_dim]
        tr_feat, norm_tr_feat, te_feat = tr_feat[:, keep_idx], norm_tr_feat[:, keep_idx], te_feat[:, keep_idx]

    elif retain_dim == 5973:
        keep_idx = np.array(range(5973))

    tr_features = torch.FloatTensor(tr_feat.reshape(-1, retain_dim, 1)) #.requires_grad_()
    norm_tr_features = torch.FloatTensor(norm_tr_feat.reshape(-1, retain_dim, 1))
    te_features = torch.FloatTensor(te_feat.reshape(-1, retain_dim, 1)) #.requires_grad_()
    tr_labels, te_labels = torch.LongTensor(tr_labels), torch.LongTensor(te_labels)

    return tr_features, norm_tr_features, te_features, tr_labels, te_labels, \
           keep_idx, all_test_cohorts, tr_sample_ids, te_sample_ids

class CRC_Dataset(Dataset):
    def __init__(self, feature, label, edge):
        super(CRC_Dataset, self).__init__()
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

