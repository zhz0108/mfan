import torch.nn as nn
import torch.nn.functional as F

from .layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj, sparse=False):
        x = F.relu(self.gc1(x, adj, sparse))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj, sparse)
        return F.log_softmax(x, dim=1)


class GCN2(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN2, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj, sparse=False):
        x = F.relu(self.gc1(x, adj, sparse))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj, sparse)
        return F.softmax(x, dim=1)
