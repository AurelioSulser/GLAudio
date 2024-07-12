from torch_geometric.data import Data
from torch_geometric.utils import get_laplacian
import torch
import numpy as np
from torch_geometric.utils import to_torch_coo_tensor

# construct a tree with 2^depth leaves
# depth = 2

"""
    Small script to compute the SVD of the Laplacian of a tree with depth 2 (i.e. 4 leaves).
"""


# depth 2 tree
edge_index = torch.tensor([[0,1],
                           [1,0],
                           [0,2],
                           [2,0],
                           [1,3],
                           [3,1],
                           [1,4],
                           [4,1],
                           [2,5],
                           [5,2],
                           [2,6],
                           [6,2]
                           ]).transpose(0,1)

laplacian_index, laplacian_weights = get_laplacian(edge_index)

# get svd of laplacian

L = to_torch_coo_tensor(edge_index=laplacian_index, edge_attr=laplacian_weights).to_dense().type(torch.float64)

U, S, V  = torch.svd(L)

print(S)

for i in range(V.shape[0]):
    for j in range(V.shape[1]-1):
        print("{:.1e}".format(V[i,j].item()), end=" & ")
    print("{:.1e}".format(V[i,-1].item()), "\\\\")


#depth 3 tree
    
edge_index = torch.tensor([[0,1],
                           [1,0],
                           [0,2],
                           [2,0],
                           [1,3],
                           [3,1],
                           [1,4],
                           [4,1],
                           [2,5],
                           [5,2],
                           [2,6],
                           [6,2],
                           [3,7],
                           [7,3],
                           [3,8],
                           [8,3],
                           [4,9],
                           [9,4],
                           [4,10],
                           [10,4],
                           [5,11],
                           [11,5],
                           [5,12],
                           [12,5],
                           [6,13],
                           [13,6],
                           [6,14],
                           [14,6]                      
                           ]).transpose(0,1)

laplacian_index, laplacian_weights = get_laplacian(edge_index)

# get svd of laplacian

L = to_torch_coo_tensor(edge_index=laplacian_index, edge_attr=laplacian_weights).to_dense().type(torch.float64)

U, S, V  = torch.svd(L)

for i in range(V.shape[0]):
    for j in range(V.shape[1]-1):
        print("{:.1e}".format(V[i,j].item()), end=" & ")
    print("{:.1e}".format(V[i,-1].item()), "\\\\")