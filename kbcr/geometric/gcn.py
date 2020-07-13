# -*- coding: utf-8 -*-

import torch
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops

from torch_geometric.nn.inits import glorot, zeros


class EdgeGCNConv(MessagePassing):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 edge_dim: int,
                 improved: bool = False,
                 bias: bool = True,
                 **kwargs):
        super().__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.improved = improved

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.edge_update = Parameter(torch.Tensor(out_channels + edge_dim, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.edge_update)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr, edge_weight=None):
        x = torch.matmul(x, self.weight)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=x.dtype, device=edge_index.device)

        fill_value = 1 if not self.improved else 2
        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value, x.size(0))

        self_loop_edges = torch.zeros(x.size(0), edge_attr.size(1)).to(edge_index.device)
        edge_attr = torch.cat([edge_attr, self_loop_edges], dim=0)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=x.size(0))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, norm=norm)

    def message(self, x_j, edge_attr, norm):
        x_j = torch.cat([x_j, edge_attr], dim=-1)

        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        aggr_out = torch.mm(aggr_out, self.edge_update)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)
