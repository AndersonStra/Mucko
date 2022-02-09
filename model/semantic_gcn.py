import torch
import torch.nn.functional as F
from torch import nn
import dgl
import networkx as nx


class SemanticGCN(nn.Module):
    def __init__(self, config, in_dim, out_dim, rel_dim):
        super(SemanticGCN, self).__init__()
        self.config = config

        self.gcn1 = SemanticGCNLayer(in_dim, out_dim, rel_dim)

    def forward(self, bg):
        bg = self.gcn1(bg)
        return bg


class SemanticGCNLayer(nn.Module):
    def __init__(self, in_dims, out_dims, rel_dims):
        super(SemanticGCNLayer, self).__init__()
        self.node_fc = nn.Linear(in_dims, in_dims)
        self.rel_fc = nn.Linear(rel_dims, rel_dims)
        self.apply_fc = nn.Linear(in_dims + rel_dims + in_dims, out_dims)

    def forward(self, g):
        g.apply_nodes(func=self.apply_node)
        g.update_all(message_func=self.message, reduce_func=dgl.function.sum('msg', 'h_sum'))
        h = g.ndata['h']
        h_sum = g.ndata['h_sum']
        h = torch.cat([h, h_sum], dim=1)  # shape(2*outdim)
        h = self.apply_fc(h)
        h = F.relu(h)
        g.ndata['h'] = h
        return g

    def apply_node(self, nodes):
        h = self.node_fc(nodes.data['h'])
        return {'h': h}

    def message(self, edges):
        z1 = z1 = edges.src['att'] * edges.src['h'] + edges.dst['att'] * edges.dst['h']
        z2 = edges.data['att'] * self.rel_fc(edges.data['rel'])
        msg = torch.cat([z1, z2], dim=1)
        return {'msg': msg}


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, rel_dim):
        super(GATLayer, self).__init__()
        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g):
        # equation (1)
        z = self.fc(g.ndata['h'])
        g.ndata['z'] = z
        # equation (2)
        g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        g.update_all(self.message_func, self.reduce_func)
        return g
