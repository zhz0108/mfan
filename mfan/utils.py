from typing import TextIO

import numpy as np
import scipy.sparse as sp
import torch
import pickle as pkl
import networkx as nx
import sys

from torch_geometric.datasets import AttributedGraphDataset
from torch_geometric.utils import to_dense_adj, to_undirected


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def accuracy(output, labels):
    # print ('output', output.size())
    preds = output.max(1)[1].type_as(labels)
    # print ('preds', preds.size())
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_indices(file_name: str) -> list[int]:
    indices = []

    with open(file_name, 'r') as file:
        for line in file:
            index = int(line)

            indices.append(index)

    return indices


# noinspection PyPep8Naming
def normalize_symmetric(A: torch.Tensor, check_zero: bool = False) -> torch.Tensor:
    D = torch.sum(A, dim=1, keepdim=True)

    if check_zero:
        for i in range(A.size()[0]):
            if D[i, 0] == 0:
                D[i, 0] = 1

    return A / torch.sqrt(D * D.t())


# noinspection PyPep8Naming
def normalize_row(A: torch.Tensor, check_zero: bool = False) -> torch.Tensor:
    if not A.is_sparse:
        D = torch.sum(A, dim=1, keepdim=True)

        if check_zero:
            for i in range(A.size()[0]):
                if D[i, 0] == 0:
                    D[i, 0] = 1

        return A / D
    else:
        D = torch.sparse.sum(A, dim=1).to_dense()

        D_inv = 1 / D
        D_inv = torch.sparse_coo_tensor(A.coalesce().indices(), D_inv[A.coalesce().indices()[0, :]], size=A.size(), device=A.device)

        return A * D_inv


def load_data(dataset_str, root):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(f"{root}/ind.{dataset_str}.{names[i]}", 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(f"{root}/ind.{dataset_str}.test.index")
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    if dataset_str == 'citeseer':
        for i in range(labels.shape[0]):
            if np.array_equal(labels[i], np.zeros(labels.shape[1])):
                labels[i][0] = 1

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj).to_dense()

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def load_data2(dataset_str):
    if dataset_str == 'wiki':
        data = AttributedGraphDataset(root='/tmp', name='Wiki').data

        labels = data.y
    elif dataset_str == 'facebook':
        data = AttributedGraphDataset(root='/tmp', name='Facebook').data

        labels = torch.argmax(data.y, dim=1)
    else:
        data = None

        labels = None

    adj = to_dense_adj(to_undirected(data.edge_index))[0]
    features = data.x

    idx = list(range(adj.size()[0]))
    np.random.shuffle(idx)

    idx_train = idx[0: adj.size()[0] // 5]
    idx_val = idx_train
    idx_test = idx[adj.size()[0] // 5: adj.size()[0]]

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test
