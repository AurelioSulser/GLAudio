
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import get_original_cwd
from src.datasets.webkb import get_data
from src.datasets.zinc import get_zinc_data
from src.datasets.planetoid import get_planetoid_data
import os
import torch
from src.models.glaudio import glaudioNeuralOscillator
from src.models.graphcon import GraphCON_GCN
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
import wandb

NAME = "CiteSeer"

if NAME == "PubMed":
    PROJECT_NAME = "sweep-pubmed"
    OUTPUT_DIM = 3
elif NAME == "Cora":
    PROJECT_NAME = "sweep-cora"
    OUTPUT_DIM = 7
elif NAME == "CiteSeer":
    PROJECT_NAME = "sweep-citeseer"
    OUTPUT_DIM = 6
else:
    raise NotImplementedError

def train(model, data, optimizer, cfg):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, data, mask):
    model.eval()
    with torch.no_grad():
        logits = model(data)
        test_mask = mask
        pred = logits[test_mask].max(1)[1]
        acc = pred.eq(data.y[test_mask]).sum().item() / test_mask.sum().item()
    return acc

def run_experiment(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_accs = []
    val_accs = []

    val_acc_at_best_val_loss_es = []
    test_acc_at_best_val_loss_es = []

    for i in range(10):
        
        if cfg.name in ['wisconsin', 'texas', 'cornell']:
            data = get_data(cfg,sweep=True)
        elif cfg.name in ["PubMed", "Cora", "CiteSeer"]:
            data  = get_planetoid_data(cfg, sweep=True)
        elif cfg.name == 'zinc':
            data = get_zinc_data(cfg, sweep=True)
        else:
            raise NotImplementedError

        data.to(device)

        #test graphcon
        model = load_model(cfg,data)
        model.to(device)

        wandb.log({"num_parameters": sum(p.numel() for p in model.parameters())})

        optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        lrscheduler = ReduceLROnPlateau(optimizer, factor=cfg.lr_factor, patience=cfg.lr_patience, min_lr=cfg.min_lr, verbose=False)

        best_val_loss = 10000000
        val_acc_at_best_val_loss = 0
        test_acc_at_best_val_loss = 0

        for epoch in range(cfg.num_epochs):
            loss = train(model, data, optimizer, cfg)
            with torch.no_grad():    
                val_loss = F.cross_entropy(model(data)[data.val_mask], data.y[data.val_mask])

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                val_acc_at_best_val_loss = evaluate(model, data, data.val_mask)
                test_acc_at_best_val_loss = evaluate(model, data, data.test_mask)

            lrscheduler.step(val_loss)
            #print(f'Epoch: {epoch}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

        accuracy = evaluate(model, data, data.test_mask)
        print(f"Epoch {cfg.num_epochs}: Test Accuracy: {accuracy}, Test Accuracy at best val loss: {test_acc_at_best_val_loss}")
        test_accs.append(accuracy)

        val_acc = evaluate(model, data, data.val_mask)
        val_accs.append(val_acc)
        print(f"Epoch {cfg.num_epochs}: Validation Accuracy: {val_acc}, Validation Accuracy at best val loss: {val_acc_at_best_val_loss}")

        test_acc_at_best_val_loss_es.append(test_acc_at_best_val_loss)
        val_acc_at_best_val_loss_es.append(val_acc_at_best_val_loss)

    avg_acc = sum(val_accs)/len(val_accs)
    test_acc = sum(test_accs)/len(test_accs)   

    avg_val_acc_at_best_val_loss = sum(val_acc_at_best_val_loss_es)/len(val_acc_at_best_val_loss_es)
    avg_test_acc_at_best_val_loss = sum(test_acc_at_best_val_loss_es)/len(test_acc_at_best_val_loss_es)

    wandb.log({"avg_val_accuracy": avg_acc}) 
    wandb.log({"avg_test_accuracy": test_acc})
    wandb.log({"avg_val_accuracy_at_best_val_loss": avg_val_acc_at_best_val_loss})
    wandb.log({"avg_test_accuracy_at_best_val_loss": avg_test_acc_at_best_val_loss})

def load_model(cfg, data):
    if cfg.model == 'graphcon':
            model = GraphCON_GCN(nfeat=data.num_node_features,
                                nhid=cfg.hidden_dim, 
                                nclass=cfg.output_dim, 
                                dropout=cfg.dropout_rate, 
                                nlayers=2,
                                dt=1, alpha=0,
                                gamma=0,
                                res_version=2)
        
    elif cfg.model == 'glaudio':
        model = glaudioNeuralOscillator(
            L=cfg.L, 
            input_dim=data.num_node_features,
            hidden_dim=cfg.hidden_dim,
            output_dim=cfg.output_dim,
            N=cfg.N,
            normalize=cfg.normalize,
            self_loops=cfg.self_loops,
            dropout_rate=cfg.dropout_rate,
            h=cfg.h)
    elif cfg.model == 'gcn':
        raise NotImplementedError
    else:   
        raise NotImplementedError
    return model

if __name__ == "__main__":
    
        # Import the W&B Python Library and log into W&B
    wandb.login()

    def main():
        with wandb.init(project=PROJECT_NAME):
            # access all HPs through wandb.config, so logging matches execution!
            cfg = wandb.config
            run_experiment(cfg) 

    # 2: Define the search space    
    sweep_configuration = {
        "method": "random",
        "metric": {"goal": "maximize", "name": "avg_val_accuracy"},
        "parameters": {
            "N": {"values": [1, 2, 5, 10, 10, 50, 100, 150, 200]},
            "L": {"values": [1, 2, 3, 5]},
            "h": {"values": [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]},
            "hidden_dim": {"values": [16, 24, 32]},
            "lr": {"values": [1e-3, 2.5e-3, 5e-3, 1e-2]},
            "weight_decay": {"values": [0, 1e-3, 2.5e-3, 5e-3]},
            "dropout_rate": {"values": [0.0, 0.1, 0.2, 0.3, 0.4]},
            "normalize": {"values": [True, False]},
            "self_loops": {"value": False},
            "activation" : {"values": ["relu", "leaky_relu"]},
            #fixed parameters
            
            "model" : {"value": "glaudio"},
            "name" : {"value": NAME},
            "split" : {"value": "0"},
            "output_dim" : {"value": OUTPUT_DIM},
            "num_epochs" : {"value": 300},
            "lr_factor" : {"value": 0.5},
            "lr_patience" : {"value": 10},
            "min_lr" : {"value": 1e-5},                
        }                           
    }



    # 3: Start the sweep
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=PROJECT_NAME)
    wandb.agent(sweep_id=sweep_id, function=main, count=500)



