from __future__ import division, print_function

import math
from typing import List
from copy import deepcopy

import numpy as np
import torch
import tqdm
from torch.nn.functional import nll_loss

from gcn import GCN, GCN2
from utils import normalize_symmetric, normalize_row


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# noinspection PyPep8Naming
class MFAN:
    def __init__(self, model: GCN, dataset: str, A: torch.Tensor, X: torch.Tensor, K: int, xi: int):
        self.model = model
        self.A = A
        self.X = X
        self.K = K
        self.xi = xi

        self.N = A.shape[0]

        self.A_n = normalize_symmetric(self.A + torch.eye(self.N, device=device))
        self.X_n = normalize_row(self.X, check_zero=True)
        self.Y = model(self.X_n, normalize_symmetric(self.A + torch.eye(self.N, device=device))).argmax(dim=1)
        self.Y2 = model(self.X_n, normalize_row(self.A + torch.eye(self.N, device=device))).argmax(dim=1)

        self.P = None
        self.g = None
        self.P_best = None
        self.g_best = None

        self.max_attempts = 10
        self.max_epoch = 120
        self.B = 32

        self.lr_p = 0.01
        self.lr_g = 0.005
        self.init_la = 0.1 / self.xi * self.K
        if dataset == 'facebook':
            self.init_la = 0.02 / self.xi * self.K
        self.init_T = 1

    def __getstate__(self):
        return {'K': self.K, 'xi': self.xi, 'P_best': self.P_best, 'g_best': self.g_best}

    def __setstate__(self, state):
        self.K = state['K']
        self.xi = state['xi']

        self.P_best = state['P_best']
        self.g_best = state['g_best']

    def train(self, indices_train: list[int]):
        indices_train = indices_train[:]

        for attempt in range(self.max_attempts):
            print(f'Attempt: {attempt + 1}')

            self.P = torch.rand((self.N, self.K), device=device)
            self.g = GCN2(nfeat=self.X.shape[1], nhid=16, nclass=self.K, dropout=0.5).to(device).eval()

            Q = [[] for k in range(self.K)]

            flag = False

            active_epoch = [0 for k in range(self.K)]

            la = self.init_la
            T = self.init_T

            optimizer_p = torch.optim.Adam([self.P], lr=self.lr_p)
            optimizer_g = torch.optim.Adam(self.g.parameters(), lr=self.lr_g)

            num_epochs = self.max_epoch
            num_steps = math.ceil(len(indices_train) / self.B)

            self.update_Q(Q, indices_train)

            fooling_ratio_train_best = 0
            fooling_ratio_train_last = 0

            pbar = tqdm.tqdm(range(num_epochs), ncols=256, bar_format='{l_bar}{bar:20}{r_bar}')

            for epoch in pbar:
                np.random.shuffle(indices_train)

                if 0 < epoch <= 60 and epoch % 20 == 0:
                    la *= 5

                if epoch > 60 and epoch % 5 == 0:
                    T /= 2

                for step in range(num_steps):
                    message = f'Epoch #{epoch + 1} Step #{step + 1}'

                    indices_step = indices_train[step * self.B: min(step * self.B + self.B, len(indices_train))]

                    self.P.requires_grad = True

                    optimizer_g.zero_grad()

                    loss = self.loss(indices_step, la, T)

                    loss.backward()

                    optimizer_p.step()

                    self.P.detach_()

                    self.P = clip(self.P)

                    optimizer_p.param_groups[0]['params'] = [self.P]

                    if self.K > 1:
                        self.g.train()

                        optimizer_g.zero_grad()

                        loss = self.loss(indices_step, la, T)

                        loss.backward()

                        optimizer_g.step()

                        self.g.eval()

                        self.update_Q(Q, indices_train)

                        if attempt != self.max_attempts - 1 and step == 0:
                            for kk in range(self.K):
                                if len(Q[kk]) != 0:
                                    active_epoch[kk] = epoch
                                else:
                                    if epoch >= active_epoch[kk] + 30:
                                        flag = True

                    message += f', Loss: {loss:.2f}'

                    if self.K > 1:
                        message += f', Cluster Size: [{", ".join([f"{len(Q[k])}" for k in range(self.K)])}]'

                    if epoch == 0 and step == 0 or step == num_steps - 1:
                        fooling_ratio_train = self.fooling_ratio(indices_train)
                        fooling_ratio_train_last = fooling_ratio_train

                        if fooling_ratio_train_best <= fooling_ratio_train:
                            fooling_ratio_train_best = fooling_ratio_train

                            self.P_best = self.P.clone()
                            self.g_best = deepcopy(self.g)
                    else:
                        fooling_ratio_train = fooling_ratio_train_last

                    message += f', Training Set Fooling Ratio: ' \
                               f'{round(fooling_ratio_train * len(indices_train))} / ' \
                               f'{len(indices_train)} = ' \
                               f'{fooling_ratio_train:.2%}'

                    pbar.set_postfix_str(message)

                    if flag:
                        break

                if flag:
                    break

                epoch += 1

            if flag:
                continue

            if not flag:
                break

    def binarize(self, P: torch.Tensor) -> torch.Tensor:
        P_binary = torch.empty_like(P)

        if P.dim() == 1:
            indices = torch.argsort(P, descending=True)

            indices = indices[0: self.xi]

            P_binary[:] = 0
            P_binary[indices] = 1
        else:
            for k in range(self.K):
                p = P[:, k]

                indices = torch.argsort(p, descending=True)

                indices = indices[0: self.xi]

                P_binary[:, k] = 0
                P_binary[indices, k] = 1

        return P_binary

    def fooling_ratio(self, indices: List[int]) -> float:
        P = self.P
        g = self.g

        num_nodes = len(indices)

        num_fooled_nodes = 0

        P_binary = self.binarize(P)

        P_indices = torch.empty((self.xi, self.K), dtype=torch.int, device=device)
        for k in range(self.K):
            P_indices[:, k] = torch.argwhere(P_binary[:, k]).squeeze(dim=1)

        W = g(self.X_n, self.A_n)

        for i in indices:

            w = W[i]

            p = P_binary[:, torch.argmax(w).item()]

            A_p = self.perturb(self.A, i, p)

            A_p = normalize_symmetric(A_p + torch.eye(self.N, device=device))

            Y_p = self.model(self.X_n, A_p)

            if torch.argmax(Y_p[i]) != self.Y[i]:
                num_fooled_nodes += 1

        fr = num_fooled_nodes / num_nodes

        return fr

    def loss(self, indices: List[int], la: float, T: float) -> torch.Tensor:
        U = torch.zeros((self.N, self.K), device=device)

        for i in indices:
            for k in range(self.K):
                p = self.P[:, k]

                A_p = self.perturb(self.A, i, p)

                A_p = normalize_row(A_p + torch.eye(self.N, device=device))

                Y_p = self.model(self.X_n, A_p)

                u = self.utility(Y_p[i], self.Y2[i])

                U[i, k] = u

        if self.K == 1:
            W = torch.ones((self.N, self.K), device=device)
        else:
            W = self.g(self.X_n, self.A_n)

        l = torch.sum(W[indices, :] * U[indices, :]) / len(indices) * self.B

        W_W = sigmoid(self.P.detach(), self.xi, T)

        W_sorted, W_indices = torch.sort(W_W, dim=0)

        w_min = W_sorted[0]

        W2 = torch.nn.functional.normalize(torch.sum(W[indices, :], dim=0), p=1, dim=0).detach()

        W_W2 = torch.clamp(torch.sum(W_W * self.P, dim=0) - w_min * self.xi, min=0)

        l += la * torch.sum(W2 * W_W2)

        l = l * len(indices) / self.B

        return l

    def update_Q(self, Q, train_indices: List[int]):
        W = self.g(self.X_n, self.A_n)

        w_indices = torch.argmax(W, dim=1)

        for k in range(self.K):
            Q[k] = np.intersect1d(torch.argwhere(w_indices == k).squeeze(dim=1).tolist(), train_indices).tolist()

    @staticmethod
    def perturb(A: torch.Tensor, i: int, p: torch.Tensor) -> torch.Tensor:
        N = A.size()[0]

        ONES = torch.ones((N, N), device=device)
        ONES_0 = torch.ones((N, N), device=device).fill_diagonal_(0)

        P = torch.zeros((N, N), device=device)

        P[i, :] = p
        P[:, i] = p

        A_p = (ONES - P) * A + P * (ONES_0 - A)

        return A_p

    # noinspection PyShadowingBuiltins
    @staticmethod
    def utility(input, target):
        u = -torch.clamp(nll_loss(input, target), max=10)

        return u


# noinspection PyPep8Naming
def clip(P: torch.Tensor) -> torch.Tensor:
    P = torch.clamp(P, min=0, max=1)

    return P


# noinspection PyPep8Naming
def sigmoid(P: torch.Tensor, xi: int, T: float) -> torch.Tensor:
    P_sorted, P_indices = torch.sort(P, dim=0, descending=True)

    delta = (P_sorted[xi - 1] + P_sorted[xi]) / 2

    W = torch.sigmoid(-(P - delta) / T)

    return W
