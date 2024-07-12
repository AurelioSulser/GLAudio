import torch
from torch_geometric.datasets import ZINC
from torch_geometric.utils import to_undirected
import numpy as np
from src.utils.structure import check_and_create_folder

from omegaconf import DictConfig, OmegaConf
import hydra
import os


# def pe(data):
#     edge_index = data.edge_index
#     pe = dgl.random_walk_pe(dgl.graph((edge_index[0],edge_index[1])), k=16)
#     data.pe = pe
#     return data
 

def get_zinc_data(cfg,sweep=False):
    """
    Method to load the ZINC dataset.
    :param cfg: config object containing the dataset name as a field named 'name'
    :param sweep: flag to indicate if the method is called during a wandb sweep. This affects how to correctly load the current working directory.

    return: train, test and validation datasets  
    """

    if sweep:
        ocwd = os.getcwd()
    else:
        ocwd = hydra.utils.get_original_cwd()
    
    path = ocwd + "/data/" + cfg.name

    check_and_create_folder(path)

    # train_dataset = ZINC(root=path, subset=True, split='train', transform=pe)
    # test_dataset = ZINC(root=path, subset=True, split='test', transform=pe)
    # val_dataset = ZINC(root=path, subset=True, split='val', transform=pe)

    train_dataset = ZINC(root=path, subset=True, split='train')
    test_dataset = ZINC(root=path, subset=True, split='test')
    val_dataset = ZINC(root=path, subset=True, split='val')
    
    return train_dataset, test_dataset, val_dataset

if __name__ == "__main__":

    cfg = DictConfig({'name': 'zinc'})
    train, test, val = get_zinc_data(cfg, sweep=True)