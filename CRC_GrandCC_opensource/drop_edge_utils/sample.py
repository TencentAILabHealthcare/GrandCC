import numpy as np
import torch
import scipy.sparse as sp
from .normalization import fetch_normalization


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def preprocess_adj(normalization, adj, cuda):
    adj_normalizer = fetch_normalization(normalization)
    r_adj = adj_normalizer(adj)
    r_adj = sparse_mx_to_torch_sparse_tensor(r_adj).float()
    if cuda:
        r_adj = r_adj.cuda()
    return r_adj


def stub_sampler(train_adj, train_features, normalization, cuda):
    """
    The stub sampler. Return the original data.
    """
    r_adj = preprocess_adj(normalization, train_adj, cuda)
    fea = preprocess_fea(train_features, cuda)
    return r_adj, fea


def randomedge_sampler(train_adj, train_features, percent, normalization, cuda=True):
    """
    Randomly drop edge and preserve percent% edges.
    train_adj: [2, bs*num_edges]
    """
    "Opt here"
    if percent >= 1.0:
        return stub_sampler(train_adj, train_features, normalization, cuda)

    print(train_features.shape)
    train_adj = train_adj.cpu().numpy()
    train_adj = sp.coo_matrix((np.ones(train_adj.shape[1]), (train_adj[0, :], train_adj[1, :])),
                                shape=(train_features.shape[0], train_features.shape[0]), dtype=np.float32)
    print("adjacency:", train_adj.data, train_adj.row, train_adj.col)

    nnz = train_adj.nnz
    print("Number of edges", nnz)
    perm = np.random.permutation(nnz)
    preserve_nnz = int(nnz * percent)
    perm = perm[:preserve_nnz]
    print("Preserved edges:", perm.shape[0])
    r_adj = sp.coo_matrix((train_adj.data[perm],
                           (train_adj.row[perm],
                            train_adj.col[perm])),
                          shape=train_adj.shape)
    r_adj = preprocess_adj(normalization, r_adj, cuda)
    print(r_adj)

    return r_adj


def randomedge_sampler(train_adj, percent):
    """
    Randomly drop edge and preserve percent% edges, without adjacency normalization.
    train_adj: [2, bs*num_edges]
    """
    nnz = train_adj.shape[1]
    # print("Number of edges", nnz)
    perm = np.random.permutation(nnz)
    preserve_nnz = int(nnz * percent)
    # perm = np.sort(perm[:preserve_nnz])
    perm = perm[:preserve_nnz]
    # print("Preserved edges:", perm)

    r_adj = train_adj[:, perm]
    # print(r_adj)
    # print(r_adj.shape)

    return r_adj


def degree_sampler(self, percent, normalization, cuda):
    """
    Randomly drop edge wrt degree (high degree, low probility).
    """
    if percent >= 0:
        return self.stub_sampler(normalization, cuda)
    if self.degree_p is None:
        degree_adj = self.train_adj.multiply(self.degree)
        self.degree_p = degree_adj.data / (1.0 * np.sum(degree_adj.data))
    # degree_adj = degree_adj.multi degree_adj.sum()
    nnz = self.train_adj.nnz
    preserve_nnz = int(nnz * percent)
    perm = np.random.choice(nnz, preserve_nnz, replace=False, p=self.degree_p)
    r_adj = sp.coo_matrix((self.train_adj.data[perm],
                           (self.train_adj.row[perm],
                            self.train_adj.col[perm])),
                          shape=self.train_adj.shape)
    r_adj = self._preprocess_adj(normalization, r_adj, cuda)
    fea = self._preprocess_fea(self.train_features, cuda)
    return r_adj, fea

