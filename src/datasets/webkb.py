import torch
from torch_geometric.datasets import WebKB
from torch_geometric.utils import to_undirected
import numpy as np
from src.utils.structure import check_and_create_folder

from omegaconf import DictConfig, OmegaConf
import hydra
import os

def get_data(cfg, sweep=False):
    """
    Method to load the WebKB dataset.
    :param cfg: config object containing the dataset name as a field named 'name'
    :param sweep: flag to indicate if the method is called during a wandb sweep. This affects how to correctly load the current working directory.
    return: data object containing the dataset
    """
    if sweep:
        ocwd = os.getcwd()
    else:
        ocwd = hydra.utils.get_original_cwd()


    path = ocwd + "/data/" + cfg.name

    check_and_create_folder(path)

    dataset = WebKB(path, name=cfg.name)

    split = cfg.split

    data = dataset[0]
    splits_file = np.load(f'{path}/{cfg.name}/raw/{cfg.name}_split_0.6_0.2_{split}.npz')
    train_mask = splits_file['train_mask']
    val_mask = splits_file['val_mask']
    test_mask = splits_file['test_mask']

    data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
    data.val_mask = torch.tensor(val_mask, dtype=torch.bool)
    data.test_mask = torch.tensor(test_mask, dtype=torch.bool)
    data.edge_index = to_undirected(data.edge_index)

    return data