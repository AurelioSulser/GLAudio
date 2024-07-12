import torch
from torch_geometric.datasets import LRGBDataset
from torch_geometric.utils import to_undirected
import numpy as np
from src.utils.structure import check_and_create_folder
import os
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import RemoveDuplicatedEdges
from torch_geometric.data import Data
from omegaconf import DictConfig, OmegaConf
import hydra

re = RemoveDuplicatedEdges()

# def pe(data):
#     """
#     Computes random walk positional encoding for the graph and adds it to the data object.
#     :param data: data object containing the graph
#     return: data object with positional encoding added
#     """
#     edge_index = data.edge_index
#     pe = dgl.random_walk_pe(dgl.graph((edge_index[0],edge_index[1]), num_nodes=data.num_nodes), k=16)
#     data.pe = pe
#     return data


def perturbe(data):
    """
    Method to perturb the graph structure by adding edges with probability p=0.5/N, where N is the number of nodes in the graph.
    :param data: data object containing the graph
    return: data object with perturbed graph
    """
    # add edges with probability p to data
    edge_index = data.edge_index
    N = data.num_nodes
    p = 0.5/N
    E = edge_index.shape[1]
    edge_index = edge_index.numpy()
    edge_index = edge_index.tolist()
    for i in range(N):
        for j in range(i+1,N):
            if np.random.rand() < p:
                edge_index[0].append(i)
                edge_index[1].append(j)
                edge_index[0].append(j)
                edge_index[1].append(i)
                E += 2
    edge_index = torch.tensor(edge_index)
    data.edge_index = edge_index
    
    del data.edge_attr
    data = re(data)

    # remove duplicate edges
    edge_index = to_undirected(data.edge_index)
    data.edge_index = edge_index

    return data


def get_lrgb_data(cfg,sweep=False, add_perturbation=False):
    """
    Method to load the LRGB (peptides-struct) dataset. If present, the dataset is loaded from the data folder. If the dataset is not present, it is downloaded from the web.
    
    :param cfg: configuration object containing the dataset name as a field named 'name'
    :param sweep: whether we are running a wandb sweep or a hydra experiment. This affects the working directory and thus the path to the data folder.
    :param add_perturbation: whether to add perturbations to the graph structure. This is done by adding edges with probability p=0.5/N, where N is the number of nodes in the graph.
                                We briefly tested this but did not include it in our report.

    return: train, test and validation datasets
    """

    if sweep:
        ocwd = os.getcwd()
    else:
        ocwd = hydra.utils.get_original_cwd()

    if add_perturbation:
        path = ocwd + "/data/" + cfg.name + "-perturbed"
    else:
        path = ocwd + "/data/" + cfg.name

    check_and_create_folder(path)

    pretransform = perturbe if add_perturbation else None
    transform = pe if cfg.hidden_gcn_dim > 0 else None

    train_dataset = LRGBDataset(root=path, name=cfg.name, split='train', pre_transform=pretransform, transform=transform)
    test_dataset = LRGBDataset(root=path, name=cfg.name, split='test', pre_transform=pretransform, transform=transform)
    val_dataset = LRGBDataset(root=path, name=cfg.name, split='val', pre_transform=pretransform, transform=transform)
    
    return train_dataset, test_dataset, val_dataset

if __name__ == "__main__":
    cfg = DictConfig({'name': 'Peptides-struct', 'hidden_gcn_dim': 0})
    train, test, val = get_lrgb_data(cfg, sweep=True, add_perturbation=False)
