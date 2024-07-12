import math
import os
from tempfile import TemporaryDirectory

from typing import Any, Dict, List, Optional, Union, Tuple
import torch
import math

from torch.nn import Linear, Parameter, Module, ModuleList
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import global_mean_pool, global_add_pool
import dgl


def load_activation(activation):
    """
        Load activation function from string
        activation: name of activation function (relu, tanh, sigmoid, elu, leaky_relu, gelu)
    """
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'elu':
        return nn.ELU()
    elif activation == 'leaky_relu':
        return nn.LeakyReLU()
    elif activation == 'gelu':
        return nn.GELU()
    else:
        raise NotImplementedError

class VariableLayerGCN(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim=30, out_dim=30, num_layers=4, activation='relu'):
        """
            Initialize GCN with small initial encoder, mulitple GCN layers and a final decoder. Only used for ZINC experiment.
            input_dim: dimension of input features
            hidden_dim: dimension of hidden layers
            out_dim: dimension of output features
            num_layers: number of GCN layers
            activation: name of activation function
        """

        super(VariableLayerGCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        
        # First initial encoder layer
        self.intital_encoder1 = Linear(in_features=input_dim, out_features=hidden_dim, bias=True)

        # Activation function
        self.activation = load_activation(activation)

        # Second intial encoder layer
        self.intital_encoder2 = Linear(in_features=hidden_dim, out_features=hidden_dim) 
        
        # Hidden layers
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Output layer
        self.decoder1 = Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True) 
        self.decoder2 = Linear(in_features=hidden_dim, out_features=out_dim) 



    def forward(self, data):
        x, edge_index, pe = data.x.float(), data.edge_index, data.pe
        x = torch.cat((x, pe), dim=-1)

        x = self.activation(self.intital_encoder1(x))
        x = self.intital_encoder2(x)


        # Apply GCN layers
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.activation(x)
            x = nn.functional.dropout(x, p=0.2, training=self.training)
        
        # Apply last layer without ReLU or dropout
        x = self.activation(self.decoder1(x))
        x = self.decoder2(x)

        return x


class WaveLayer(MessagePassing):
    
    def __init__(self, h=0.1, normalized_lap=False, self_loops=False, learn_spectrum=False, hidden_dim=0):
        """
            Initialize WaveLayer corresponding to Eq. III. in the report
            h: step size
            normalized_lap: if True, use normalized laplacian
            self_loops: if True, add self loops
            learn_spectrum: if True, adds an additional learn-able linear layer to the wave equation. We tested this but it seemed to make learning numerically unstable
            hidden_dim: dimension of hidden layer (only used to construct linear layer, when learn_spectrum is True)
        """
        super().__init__(aggr='add')
        self.h = h
        self.normalized_lap = normalized_lap
        self.self_loops = self_loops

        self.learn_spectrum = learn_spectrum
        if self.learn_spectrum:
            assert hidden_dim > 0
            self.linear = Linear(in_features=hidden_dim, out_features=hidden_dim)
            self.linear.weight.data.copy_(torch.eye(hidden_dim))


    def forward(self, x, vel, edge_index):
        """
            Forward pass of WaveLayer. Corresponds to one time step of Eq. III. in the report
            x: input features
            vel: velocity state used to integration of second order ODE
            edge_index: edge index of graph
        """
        row, col = edge_index
        deg = degree(col, x.size(-2), dtype=x.dtype)
        inv_sqrt_deg = deg.pow(-0.5)
        inv_sqrt_deg[inv_sqrt_deg == float('inf')] = 0
        norm = inv_sqrt_deg[row] * inv_sqrt_deg[col]

        if self.learn_spectrum:
            x = self.linear(x)

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
        """
        Initialize ElementwiseLinear layer. This l
        """
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
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise NotImplementedError


    def forward(self, z, z_prev_layer, u):
 
        u_new = u + self.h*self.activation(self.a(z) + self.B(z_prev_layer))
        z_new = z + self.h*u_new

        return z_new, u_new

class glaudioNeuralOscillator(Module):

    def __init__(self, L, input_dim, hidden_dim, output_dim, N, normalize=False, activation='relu', h=0.1, dropout_rate=0.0,
                 self_loops=False, pooling=False, initial_processing=False, post_processing=False, learn_spectrum=False,  hidden_gcn_dim=0):
        """
            Initialize glaudioNeuralOscillator. This is the main model used in the experiments. It consists of the wave equation encoder and the neural oscillator decoder.
        """
        super().__init__()
        self.h = h
        self.L = L
        self.N = N
        self.hidden_gcn_dim = hidden_gcn_dim
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.pooling = pooling

        #  here we experimented with an additional GCN procesuing local encodings which we concatenate to the input of the neural oscillator
        if hidden_gcn_dim != 0:
            self.gcn = VariableLayerGCN(input_dim+16, hidden_dim=hidden_gcn_dim, out_dim=hidden_gcn_dim) #add plus 16 to input dim for positional encoding

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



        self.layers = ModuleList([NeuralOscillatorLayer(input_dim=hidden_dim+hidden_gcn_dim, output_dim=hidden_dim+hidden_gcn_dim, h=self.h, activation=activation)])

        for _ in range(self.L-1):
            self.layers.append(NeuralOscillatorLayer(input_dim=hidden_dim+hidden_gcn_dim, h=self.h, output_dim=hidden_dim+hidden_gcn_dim))

        self.final = Linear(in_features=hidden_dim+hidden_gcn_dim, out_features=output_dim, bias=True)
        self.intital_encoder1 = Linear(in_features=input_dim, out_features=hidden_dim) 



        self.wave_layer = WaveLayer(h=0.1, normalized_lap=normalize,self_loops=self_loops, learn_spectrum=learn_spectrum, hidden_dim=hidden_dim)

        self.initial_processing = initial_processing
        self.post_processing = post_processing
        # if post_processing is true, we postprocess the output with a MLP with one hidden layer
        if self.post_processing:
            self.post_processing_1 = Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True)
            self.post_processing_2 = Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True)

        # if initial processing is true, we add an additional linear layer non-linearity to the input
        if self.initial_processing:
            self.intital_encoder2 = Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True)


    def forward(self, data):

        #add dropout to input
        input = data.x.float()
        input = nn.functional.dropout(input, p=self.dropout_rate, training=self.training)
        edge_index = data.edge_index

        X = self.intital_encoder1(input)
        if self.initial_processing:
            X = self.activation(X)
            X = self.intital_encoder2(X)

        Y = torch.zeros_like(X).to(X.device)

        X = nn.functional.dropout(X, p=self.dropout_rate, training=self.training)

        if self.hidden_gcn_dim != 0:
            gcn_pred = self.gcn(data)
            X_tilde = torch.cat((X, gcn_pred), dim=-1)
        else:
            X_tilde = X

        hidden_size = list(X_tilde.shape[:-1])+[self.hidden_dim+self.hidden_gcn_dim]

        Z = [torch.zeros(size=hidden_size).to(X_tilde.device) for _ in range(self.L)]
        U = [torch.zeros(size=hidden_size).to(X_tilde.device) for _ in range(self.L)]

        for _ in range(self.N):
            # propagates the wave equation by one time step
            X, Y = self.wave_layer(X,Y, edge_index)

            # concatenate local encoding of GCN to imput signal of neural oscillator
            if self.hidden_gcn_dim != 0:
                X_tilde = torch.cat((X, gcn_pred), dim=-1)
            else:
                X_tilde = X

            Z[0], U[0] = self.layers[0](Z[0], X_tilde, U[0])
            # propagate the encoded signal at the current time step through all neural oscillator layers
            for j in range(self.L-1,0, -1):        
                Z[j], U[j] = self.layers[j](Z[j], Z[j-1], U[j])
        
        # postprocess output with MLP if post_processing is true
        if self.post_processing:
            out = self.activation(self.post_processing_1(Z[-1]))
            out = self.activation(self.post_processing_2(out))
            out = self.final(out)
        else:
            out = self.final(Z[-1])

        # apply pooling to layer to comine node-wise outsputs to graph-wise output if pooling is true
        if self.pooling:
            out = global_add_pool(out, data.batch).squeeze(-1)

        return out


class glaudioLSTM(Module):

    def __init__(self, input_dim, hidden_dim, output_dim, N, normalize=False, activation='relu', h=0.1, dropout_rate=0.0, self_loops=False, pooling=False):
        """
            Initialize glaudioLSTM. This is an experimental model in which we use a LSTM instead of a neural oscillator to decode the wave signal.
        """
        super().__init__()

        self.h = h
        self.N = N

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.pooling = pooling

        self.rnn = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, proj_size=0, dropout=dropout_rate, num_layers=1, bidirectional=True)
        
        self.final1 = Linear(in_features=2*hidden_dim, out_features=hidden_dim, bias=True)
        self.final2 = Linear(in_features=hidden_dim, out_features=output_dim, bias=True)
        
        self.intital_encoder1 = Linear(in_features=input_dim, out_features=hidden_dim, bias=True) 
        
        self.act_f = load_activation(activation)
        
        
        self.intital_encoder2 = Linear(in_features=hidden_dim, out_features=hidden_dim) 
        self.wave_layer = WaveLayer(h=self.h, normalized_lap=normalize,self_loops=self_loops)



    def forward(self, data):

        #add dropout to input
        input = data.x.float()
        #input = nn.functional.dropout(input, p=self.dropout_rate_high, training=self.training)
        #input = self.norm1(input)
        edge_index = data.edge_index
        

        X = self.intital_encoder1(input)
        Y = torch.zeros_like(X).to(X.device)
        X = self.act_f(X)
        X = self.intital_encoder2(X)

        #X = self.norm2(X)

        #X = nn.functional.dropout(X, p=self.dropout_rate_high, training=self.training)

        #hidden_size = list(X.shape[:-1])+[self.hidden_dim]

        X_tilde = X.unsqueeze(0)

        for _ in range(self.N-1):
            X, Y = self.wave_layer(X,Y, edge_index)
            X_tilde = torch.cat((X_tilde, X.unsqueeze(0)), dim=0)
        
        out_vec = self.rnn(X_tilde)[0]

        out = self.act_f(self.final1(out_vec[-1]))

        out = self.final2(out)

        if self.pooling:
            out = global_mean_pool(out, data.batch).squeeze(-1)

        return out
    

class glaudioTransformer(Module):

    def __init__(self, input_dim, output_dim, N, hidden_dim = 100, n_head = 4, normalize=False, activation='relu', h=0.1, dropout_rate=0.0, self_loops=False, pooling=False):
        """
            Initialize glaudioTransformer. This is an experimental model in which we use a LSTM instead of a neural oscillator to decode the wave signal.
        """
        super().__init__()

        self.h = h
        self.N = N

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.pooling = pooling
        self.dropout_rate = dropout_rate

        self.pos_encoder = PositionalEncoding(d_model=hidden_dim, dropout = self.dropout_rate)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4)
        self.rnn = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.final1 = Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True)
        self.final2 = Linear(in_features=hidden_dim, out_features=output_dim, bias=True)
        
        self.intital_encoder1 = Linear(in_features=input_dim, out_features=hidden_dim, bias=True) 
        
        self.act_f = load_activation(activation)
        
        
        self.intital_encoder2 = Linear(in_features=hidden_dim, out_features=hidden_dim) 
        self.wave_layer = WaveLayer(h=self.h, normalized_lap=normalize,self_loops=self_loops)

        self.pos_encoder = PositionalEncoding(d_model=hidden_dim, dropout=0.1)

    def forward(self, data):

        #add dropout to input
        input = data.x.float()
        input = nn.functional.dropout(input, p=0.5, training=self.training)
        #input = nn.functional.dropout(input, p=self.dropout_rate_high, training=self.training)
        #input = self.norm1(input)
        edge_index = data.edge_index
        

        X = self.intital_encoder1(input)
        Y = torch.zeros_like(X).to(X.device)
        #X = self.act_f(X)
        #X = self.intital_encoder2(X)
        #X = self.norm2(X)

        X = nn.functional.dropout(X, p=self.dropout_rate, training=self.training)

        #hidden_size = list(X.shape[:-1])+[self.hidden_dim]
        X_tilde = X.unsqueeze(0)

        for _ in range(self.N-1):
            X, Y = self.wave_layer(X,Y, edge_index)
            X_tilde = torch.cat((X_tilde, X.unsqueeze(0)), dim=0)
        
        #X_tilde = nn.functional.dropout(X_tilde, p=self.dropout_rate, training=self.training)
        X_tilde = self.pos_encoder(X_tilde)
        #print(X_tilde.shape)
        out_vec = self.rnn(X_tilde)[0]

        out = self.act_f(self.final1(out_vec))
        #out = self.act_f(self.final1(out_vec[-1]))
        
        out = self.final2(out)

        if self.pooling:
            out = global_mean_pool(out, data.batch).squeeze(-1)

        return out
    

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
