# glaudio

## Prerequisites

This project is based on PyTorch and the PyTorch Geometric library which both might need manual installation. 
Further needed libraries are listed below. Before running the code, please ensure the following python libraries are installed:
* matplotlib
* torch
* torchvision
* torchaudio
* torch-geometric
* numpy
* hydra-core
* dgl
* dglgo
* scikit-learn
* attrdict
* omegaconf

We use Python 3.9.18. 

## Experiments
Our repository contains the code to test glaudio on the following datasets:
* Planetoid (Cora, CiteSeer, PubMed)
* WebKB (Wisconsin, Texas, Cornell)
* Zinc
* Long Range Graph Benchmark: Peptides-struct
* NeighborsMatch

To configure the experiments, we used hydra. Configuration files can be found in the `conf` directory. The master config can be found under `conf/config.yaml`. Configuration files containing the fixed constants and hyperparameters for the individual experiments can be found in the directory `conf/dataset`.

### Running an experiment (except NeighborMatch)
In order to start an experiment, specifiy the following three options in the master config file:
* `dataset`: The name of the experiment.  Possible names are `citeseer`, `cora`, `cornell`, `peptides-struct`, `peptides-struct_lstm`, `pubmed`, `texas`, `wisconsin`, `zinc_gcn`, `zinc_lstm` and `zinc`.
* `model` The name of the model. Possible names are `glaudio`, `glaudio_lstm`, `gcn` and `gcn_res_conn`.
* `num_runs`: The number of runs to average evaluation metrics over. Note that the time requirements for a single experiment varies drastically between datasets.

Then, start the experiment using the command;
```
$ python main.py
```

Alternatively, the config values can also be set for a single experiment via passed arguments:

```
$ python main.py dataset=NAME_EXPERIMENT model=NAME_MODEL num_runs=NUMBER_RUNS
```

### Running NeighborsMatch

The code to run neighborsmatch can be found in the subdirectory `neighborsmatch/`. It is based on the original repository https://github.com/tech-srl/bottleneck by Uri Alon and Eran Yahav. 
To run a single experiment with depth 2 and the hyperparameters from the report, run the following command:

```
$ python neighborsmatch/main.py
```


To run different depths or set parameters, you can give options. For example, to train glaudio with depth=4, run:
```
$ python neighborsmatch/main.py --depth 4
```
Available options are:
- depth of Neighbormatch task: --depth
- hidden dimension: --dim
- maximum epochs: --max_epochs
- step size for evaluting epochs: --eval_every
- batch size for training: --batch_size
- L: --num_layers
- N: --N
- h: --h

For further information on how to choose the hyperparameters L, N, and h, please refer to model architecture and model hyperparameters in the report.

