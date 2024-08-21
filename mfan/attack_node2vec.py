from __future__ import division
from __future__ import print_function

import argparse
from pickle import load

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import dense_to_sparse
from tqdm import tqdm

from MFAN import MFAN
from utils import load_indices, normalize_symmetric, normalize_row, load_data, load_data2


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'citeseer', 'facebook', 'wiki', 'pubmed'], help='Graph dataset.')
parser.add_argument('--K', type=int, default=2)
parser.add_argument('--xi', type=int, default=10)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

if args.dataset == 'cora' or args.dataset == 'citeseer' or args.dataset == 'pubmed':
    adj, features, labels, idx_train, idx_val, idx_test = load_data(args.dataset, 'data')
else:
    adj, features, labels, idx_train, idx_val, idx_test = load_data2(args.dataset)

N = adj.size()[0]


def train(model, loader, optimizer):
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def test(model):
    model.eval()
    z = model()

    clf = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=150).fit(z[idx_train].detach().cpu().numpy(), labels[idx_train].detach().cpu().numpy())

    acc = clf.score(z[idx_test].detach().cpu().numpy(), labels[idx_test].detach().cpu().numpy())

    return acc


@torch.no_grad()
def predict(model):
    model.eval()
    z = model()

    clf = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=150) \
        .fit(z[idx_train].detach().cpu().numpy(), labels[idx_train].detach().cpu().numpy())

    y = clf.predict(z.detach().cpu().numpy())

    return y


# noinspection PyPep8Naming
def fooling_ratio(A: torch.Tensor, X: torch.Tensor, victim_model: Node2Vec, attack_model: MFAN, indices: list[int]) -> float:
    A_n = normalize_symmetric(A + torch.eye(N, device=device))

    X_n = normalize_row(X, check_zero=True)

    Y = predict(victim_model)
    Y = torch.tensor(Y)

    num_nodes = len(indices)

    num_fooled_nodes = 0

    P_binary = attack_model.binarize(attack_model.P_best)

    P_indices = torch.empty((attack_model.xi, attack_model.K), dtype=torch.int, device=device)
    for k in range(attack_model.K):
        P_indices[:, k] = torch.argwhere(P_binary[:, k]).squeeze(dim=1)

    W = attack_model.g_best(X_n, A_n)

    for i in tqdm(indices):
        w = W[i]

        p = P_binary[:, torch.argmax(w).item()]

        A_p = MFAN.perturb(A, i, p)

        edge_index_p = dense_to_sparse(A_p)[0]

        model_p = Node2Vec(
            edge_index_p,
            embedding_dim=128,
            walk_length=20,
            context_size=10,
            walks_per_node=10,
            num_negative_samples=1,
            p=1.0,
            q=1.0,
            sparse=True,
        ).to(device)

        loader_p = model_p.loader(batch_size=128, shuffle=True)

        optimizer_p = torch.optim.SparseAdam(list(model_p.parameters()), lr=0.01)

        for epoch in range(1, 101):
            train(model_p, loader_p, optimizer_p)

        Y_p = predict(model_p)
        Y_p = torch.tensor(Y_p)

        if Y_p[i] != Y[i]:
            num_fooled_nodes += 1

    fr = num_fooled_nodes / num_nodes

    return fr


def main():
    dataset = args.dataset
    K = args.K
    xi = args.xi

    A = adj.to(device)
    X = features.to(device)

    indices = load_indices(f'indices/indices-dataset={dataset}.txt')

    edge_index = dense_to_sparse(A)[0]

    victim = Node2Vec(
        edge_index,
        embedding_dim=128,
        walk_length=20,
        context_size=10,
        walks_per_node=10,
        num_negative_samples=1,
        p=1.0,
        q=1.0,
        sparse=True,
    ).to(device)

    loader = victim.loader(batch_size=128, shuffle=True)

    optimizer = torch.optim.SparseAdam(list(victim.parameters()), lr=0.01)

    for epoch in range(1, 101):
        train(victim, loader, optimizer)

    print(f'dataset = {dataset}, K = {K}, xi = {xi}')
    print()

    with open(f'saved-models/model_dataset={dataset}_K={K}_xi={xi}.pickle', 'rb') as file_model:
        mfan = load(file_model)

        fr = fooling_ratio(A, X, victim, mfan, indices)

        print(f'Fooling Ratio: {fr:.2%}')
        print()


if __name__ == '__main__':
    main()
