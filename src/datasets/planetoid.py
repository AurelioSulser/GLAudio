import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected
import numpy as np
from src.utils.structure import check_and_create_folder
from omegaconf import DictConfig, OmegaConf
import hydra
import os


def get_planetoid_data(cfg, sweep=False):
        """
        Method to load the planetoid dataset.
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

        dataset = Planetoid(path, name=cfg.name)
        data  = dataset[0]
        data.edge_index = to_undirected(data.edge_index)
        
        return data