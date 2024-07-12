import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import GCN as standardGCN
from torch_geometric.nn import GCNConv

"""
Implementation of a simple GCN and a GCN with residual connections for benchmarking purposes.
"""


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(GCN, self).__init__()
        self.gcn = standardGCN(in_channels=in_channels, hidden_channels=hidden_channels, num_layers=num_layers, out_channels=out_channels, dropout=dropout)

    def forward(self, data):

        return self.gcn(data.x, data.edge_index)


class GCNwithResConn(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, num_layers_after_conv=0):
        super(GCNwithResConn, self).__init__()

        self.in_channels = in_channels  
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

        self.enc = nn.Linear(in_channels, hidden_channels)

        layers = [GCNConv(hidden_channels, hidden_channels) for i in range(num_layers)]

        decoder_layers = [nn.Linear(hidden_channels, hidden_channels) for i in range(num_layers_after_conv)]

        self.final_dec = nn.Linear(hidden_channels, out_channels)
        self.layers = nn.ModuleList(layers)
        self.decoder_layers = nn.ModuleList(decoder_layers)

    def forward(self, data):

        x, edge_index = data.x, data.edge_index

        x = self.enc(x)

        for i in range(self.num_layers):
            x = x + self.layers[i](x, edge_index)
        
        for i in range(len(self.decoder_layers)):
            x = self.decoder_layers[i](x)

        x = self.final_dec(x)
        return x

    