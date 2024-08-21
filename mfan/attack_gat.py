from __future__ import division, print_function

import argparse
from pickle import load
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F

from gat import GAT, SpGAT
from MFAN import MFAN
from utils import accuracy, load_indices, normalize_symmetric, normalize_row, load_data, load_data2


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=0,
                    help='Random seed.')
parser.add_argument('--hidden', type=int, default=8,
                    help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8,
                    help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2,
                    help='Alpha for the leaky_relu.')
parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'citeseer', 'facebook', 'wiki', 'pubmed'],
                    help='Graph dataset.')
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

if args.dataset == 'facebook':
    victim = SpGAT(nfeat=features.shape[1],
                   nhid=args.hidden,
                   nclass=int(labels.max()) + 1,
                   dropout=args.dropout,
                   nheads=args.nb_heads,
                   alpha=args.alpha)
else:
    victim = GAT(nfeat=features.shape[1],
                 nhid=args.hidden,
                 nclass=int(labels.max()) + 1,
                 dropout=args.dropout,
                 nheads=args.nb_heads,
                 alpha=args.alpha)

if torch.cuda.is_available():
    victim.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def compute_test():
    victim.eval()
    output = victim(normalize_row(features, check_zero=True), normalize_symmetric(adj + torch.eye(adj.shape[0], device=device)))
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return output


victim.load_state_dict(torch.load(f'gat/saved-models/gat-dataset={args.dataset}.pkl'))

compute_test()


# noinspection PyPep8Naming
def fooling_ratio(A: torch.Tensor, X: torch.Tensor, victim_model: Union[GAT, SpGAT], attack_model: MFAN, indices: list[int]) -> float:
    A_n = normalize_symmetric(A + torch.eye(N, device=device))

    X_n = normalize_row(X, check_zero=True)

    Y = victim_model(X_n, A_n).argmax(dim=1)

    num_nodes = len(indices)

    num_fooled_nodes = 0

    P_binary = attack_model.binarize(attack_model.P_best)

    P_indices = torch.empty((attack_model.xi, attack_model.K), dtype=torch.int, device=device)
    for k in range(attack_model.K):
        P_indices[:, k] = torch.argwhere(P_binary[:, k]).squeeze(dim=1)

    W = attack_model.g_best(X_n, A_n)

    for i in indices:
        w = W[i]

        p = P_binary[:, torch.argmax(w).item()]

        A_p = MFAN.perturb(A, i, p)

        A_p_n = normalize_symmetric(A_p + torch.eye(N, device=device))

        Y_p = victim_model(X_n, A_p_n)

        if torch.argmax(Y_p[i]) != Y[i]:
            num_fooled_nodes += 1

    fr = num_fooled_nodes / num_nodes

    return fr


def main():
    dataset = args.dataset
    K = args.K
    xi = args.xi

    A = adj
    X = features

    indices = load_indices(f'indices/indices-dataset={dataset}.txt')

    print(f'dataset = {dataset}, K = {K}, xi = {xi}')
    print()

    with open(f'saved-models/model_dataset={dataset}_K={K}_xi={xi}.pickle', 'rb') as file_model:
        mfan = load(file_model)

        fr = fooling_ratio(A, X, victim, mfan, indices)

        print(f'Fooling Ratio: {fr:.2%}')
        print()


if __name__ == '__main__':
    main()
