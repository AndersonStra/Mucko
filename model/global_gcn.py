import torch
import torch.nn.functional as F
from torch import nn
import dgl
import networkx as nx
import dgl.function as fn


gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')


class GlobalGCN(nn.Module):
    def __init__(self, config, in_dim, out_dim):
        super(GlobalGCN, self).__init__()
        self.config = config

        self.gcn1 = GCNLayer(in_dim, out_dim)

    def forward(self, bg):
        bg = self.gcn1(bg)
        return bg


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g):
        g.update_all(gcn_msg, gcn_reduce)
        g.ndata['h'] = F.relu(self.linear(g.ndata['h']))
        return g
