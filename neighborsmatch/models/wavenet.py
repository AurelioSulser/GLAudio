from typing import Any, Dict, List, Optional, Union
import torch

from torch.nn import Linear, Parameter, Module, ModuleList
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import to_undirected

from torch.nn import functional as F


class WaveLayer(MessagePassing):
    
    def __init__(self, h=0.1, normalized_lap=False, self_loops=False):
        super().__init__(aggr='add')
        self.h = h
        self.normalized_lap = normalized_lap
        self.self_loops = self_loops


    def forward(self, x, vel, edge_index):
        row, col = edge_index
        deg = degree(col, x.size(-2), dtype=x.dtype)
        inv_sqrt_deg = deg.pow(-0.5)
        inv_sqrt_deg[inv_sqrt_deg == float('inf')] = 0
        norm = inv_sqrt_deg[row] * inv_sqrt_deg[col]

        # Compute D*X (assumes that V is of dimension ... x N x d)
        vel_update = torch.einsum('...i,...ij->...ij', deg, x)
        
        
        # Computes -A*X or D0.5AD0.5*X    
        if self.normalized_lap:
            vel_update = x
            mes = self.propagate(edge_index, x=x, norm=norm)
        else:
            vel_update = torch.einsum('...i,...ij->...ij', deg, x)
            mes = self.propagate(edge_index, x=x)

        # V_update = L*X=(D-A)*X or N*X = (I - D0.5 A D0.5)*X
        vel_update = vel_update + mes
        
        # add x if self loops
        if self.self_loops:
            vel_update = vel_update + x

        #V_new = V_old - hL*X
        vel_new = vel - self.h*vel_update

        #X_new = X_old + h*V_new
        x_new = x + self.h*vel_new

        return x_new, vel_new
    
    def message(self, x_j, norm=None):
        if norm is None:
            return -x_j
        else:
            return - x_j * norm.view(-1, 1)
            
    

class ElementwiseLinear(Module):

    def __init__(self, input_features):
        super(ElementwiseLinear, self).__init__()
        self.weights = Parameter(torch.Tensor(input_features))
        torch.nn.init.zeros_(self.weights)  # Initialize weights

    def forward(self, input):
        return self.weights * input
       


class NeuralOscillatorLayer(Module):

    def __init__(self, input_dim, output_dim, h=0.1, activation='relu') -> None:
        super().__init__()
        self.h = h
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.a = ElementwiseLinear(self.output_dim)
        self.B = Linear(in_features=input_dim, out_features=output_dim, bias=True)
        
        #set activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        else:
            self.activation = nn.ReLU()
        


    def forward(self, z, z_prev_layer, u):

        u_new = u + self.h*self.activation(self.a(z) + self.B(z_prev_layer))
        z_new = z + self.h*u_new

        return z_new, u_new

class NeuralOscillator(Module):

    def __init__(self, L, input_dim, hidden_dim, output_dim, N, h, normalize, self_loops, activation):
        super().__init__()
        self.h = h
        self.L = L
        self.N = N
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.layers = ModuleList([NeuralOscillatorLayer(input_dim=hidden_dim, output_dim=hidden_dim, h=self.h, activation=activation)])
        
        # added for encoding of the data structure with key and value
        self.layer0_keys = nn.Embedding(num_embeddings=input_dim + 1, embedding_dim=hidden_dim)
        self.layer0_values = nn.Embedding(num_embeddings=input_dim + 1, embedding_dim=hidden_dim)
        self.intital_encoder = nn.Linear(in_features=input_dim, out_features=hidden_dim)   

        for _ in range(self.L-1):
            self.layers.append(NeuralOscillatorLayer(input_dim=hidden_dim, h=self.h, output_dim=hidden_dim))

        self.final = Linear(in_features=hidden_dim, out_features=output_dim +1 , bias=False) # out_features=output_dim + 1

        self.wave_layer = WaveLayer(h=0.1, normalized_lap=normalize,self_loops=self_loops)
        self.linear = Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.activation = nn.ReLU()


    def forward(self, data):

        x, edge_index, roots = data.x, data.edge_index, data.root_mask

        edge_index = to_undirected(edge_index)
        data.edge_index = edge_index

        x_key, x_val = x[:, 0], x[:, 1]
        x_key_embed = self.layer0_keys(x_key)
        x_val_embed = self.layer0_values(x_val)
        
        X = x_key_embed + x_val_embed
        X = self.linear(X)
        X = self.activation(X)

        Y = torch.zeros_like(X).to(X.device)        
        hidden_size = list(X.shape[:-1])+[self.hidden_dim]

        Z = [torch.zeros(size=hidden_size).to(X.device) for _ in range(self.L)]
        U = [torch.zeros(size=hidden_size).to(X.device) for _ in range(self.L)]

        for _ in range(self.N):
            X, Y = self.wave_layer(X,Y, edge_index)
            Z[0], U[0] = self.layers[0](Z[0], X, U[0])
            for j in range(self.L-1,0, -1):
                Z[j], U[j] = self.layers[j](Z[j], Z[j-1], U[j])
                # print(f'j={j} Z={Z[j]}')

        Z_root_nodes = Z[-1][roots]
        out = self.final(Z_root_nodes)

        return out