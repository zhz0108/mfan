from __future__ import division, print_function

import argparse
from pickle import dump, load

import numpy as np
import torch
from torch.nn.functional import nll_loss

from gcn import GCN
from MFAN import MFAN
from utils import accuracy, load_indices, normalize_symmetric, normalize_row, load_data, load_data2


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=0,
                    help='Random seed.')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'citeseer', 'facebook', 'wiki'],
                    help='Graph dataset.')
parser.add_argument('--K', type=int, default=2)
parser.add_argument('--xi', type=int, default=10)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

if args.dataset == 'cora' or args.dataset == 'citeseer':
    adj, features, labels, idx_train, idx_val, idx_test = load_data(args.dataset, 'data')
else:
    adj, features, labels, idx_train, idx_val, idx_test = load_data2(args.dataset)

N = adj.size()[0]

victim = GCN(nfeat=features.shape[1], nhid=args.hidden, nclass=labels.max().item() + 1, dropout=args.dropout)

if torch.cuda.is_available():
    victim.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def test():
    victim.eval()
    output = victim(normalize_row(features, check_zero=True), normalize_symmetric(adj + torch.eye(adj.shape[0], device=device)))
    loss_test = nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return output


victim.load_state_dict(torch.load(f'gcn/saved-models/gcn-dataset={args.dataset}.pt'))

test()


# noinspection PyPep8Naming
def fooling_ratio(A, X, victim_model: GCN, attack_model: MFAN, indices):
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

    # indices = list(range(N))
    # np.random.shuffle(indices)

    print(f'dataset = {dataset}, K = {K}, xi = {xi}')
    print()

    fooling_ratios_train = []
    fooling_ratios_test = []

    trials = 5

    for trial in range(trials):
        print(f'Trail: {trial + 1}')

        indices_train = indices[trial * (N // trials): trial * (N // trials) + (N // trials)]
        indices_test = list(np.setdiff1d(indices, indices_train))

        mfan = MFAN(victim, args.dataset, A, X, K, xi)
        mfan.train(indices_train)

        with open(f'saved-models/model_dataset={dataset}_K={K}_xi={xi}_trial={trial + 1}.pickle', 'wb') as file_model:
            dump(mfan, file_model)

        with open(f'saved-models/model_dataset={dataset}_K={K}_xi={xi}_trial={trial + 1}.pickle', 'rb') as file_model:
            mfan_loaded = load(file_model)

            fr_train = fooling_ratio(A, X, victim, mfan_loaded, indices_train)
            fr_test = fooling_ratio(A, X, victim, mfan_loaded, indices_test)

            print(f'Training Set Fooling Ratio: {fr_train:.2%}')
            print(f'Test Set Fooling Ratio: {fr_test:.2%}')
            print()

            fooling_ratios_train.append(fr_train)
            fooling_ratios_test.append(fr_test)

    print(f'Average Training Set Fooling Ratio: {sum(fooling_ratios_train) / trials:.2%}, ')
    print(f'Average Test Set Fooling Ratio: {sum(fooling_ratios_test) / trials:.2%}')


if __name__ == '__main__':
    main()
