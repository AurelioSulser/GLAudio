
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import get_original_cwd
from src.datasets.webkb import get_data
from src.datasets.zinc import get_zinc_data
from src.datasets.planetoid import get_planetoid_data
from src.datasets.lrgb import get_lrgb_data
import os
import torch
from src.models.glaudio import glaudioNeuralOscillator, glaudioLSTM
from src.models.graphcon import GraphCON_GCN
from src.models.gcn import GCN, GCNwithResConn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb

from sklearn.metrics import f1_score
from src.utils.weightedCE import weighted_cross_entropy

PROJECT_NAME = "sweep-peptides-struct"
OUTPUT_DIM = 11
NAME = "Peptides-struct"
perturbe = False


def get_loss_function(loss_fn):
    if loss_fn == 'cross_entropy':
        lf = F.cross_entropy
    elif loss_fn == 'weighted_cross_entropy':
        lf = weighted_cross_entropy
    elif loss_fn == 'l1':
        lf = torch.nn.L1Loss()
    else:
        raise NotImplementedError
    return lf

def train(model, data, optimizer, mask=None, loss_fn='cross_entropy', gradient_clipping=False):
    model.train()
    optimizer.zero_grad()
    out = model(data)

    lf = get_loss_function(loss_fn)

    if mask is None:
        loss = lf(out, data.y)
    else:
        loss = lf(out[mask], data.y[mask])

    loss.backward()
    if gradient_clipping:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()


def evaluate(model, data, mask):
    model.eval()
    with torch.no_grad():
        logits = model(data)
        test_mask = mask
        pred = logits[test_mask].max(-1)[1]
        acc = pred.eq(data.y[test_mask]).sum().item() / test_mask.sum().item()
    return acc



def run_experiment(cfg):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    final_val_accs  = []
    final_test_accs = []

    for i in range(1):
        
        #test graphcon
        model = load_model(cfg)
        model.to(device)

        if cfg.name in ['wisconsin', 'texas', 'cornell']:
            data = get_data(cfg)
            data.to(device)
            val_accuracy, test_accuracy = single_graph_training(cfg, data, model, device)
            final_val_accs.append(val_accuracy)
            final_test_accs.append(test_accuracy)
        elif cfg.name in ["PubMed", "Cora", "CiteSeer"]:
            data  = get_planetoid_data(cfg)
            data.to(device)
            val_accuracy, test_accuracy = single_graph_training(cfg, data, model, device)
            final_val_accs.append(val_accuracy)
            final_test_accs.append(test_accuracy)
        elif cfg.name == 'zinc':
            data = get_zinc_data(cfg)
            raise NotImplementedError
        elif cfg.name == 'PascalVOC-SP':
            data = get_lrgb_data(cfg, sweep=True)
            val_accuracy, test_accuracy = multi_graph_training(cfg, data, model, device)
            final_val_accs.append(val_accuracy)
            final_test_accs.append(test_accuracy)
        elif cfg.name == 'Peptides-struct':
            if perturbe:
                data = get_lrgb_data(cfg, sweep=True, add_perturbation=True)
            else:
                data = get_lrgb_data(cfg, sweep=True)
            val_loss, test_loss = multi_graph_training(cfg, data, model, device)
            final_val_accs.append(val_loss)
            final_test_accs.append(test_loss)
        else:
            raise NotImplementedError
        
    
    print(f"Test results for model {cfg.model} and dataset {cfg.name}:")
    print(f'Average final validation accuracy (or loss): {sum(final_val_accs)/len(final_val_accs)}  std: {torch.tensor(final_val_accs).std()}')
    print(f'Average final test accuracy (or loss): {sum(final_test_accs)/len(final_test_accs)}  std: {torch.tensor(final_test_accs).std()}')

    wandb.log({"final_val_loss": sum(final_val_accs)/len(final_val_accs), "final_test_loss": sum(final_test_accs)/len(final_test_accs)})

def single_graph_training(cfg, data, model, device):
    """
    Training routine for single graph datasets
    """
    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    lrscheduler = ReduceLROnPlateau(optimizer, factor=cfg.lr_factor, patience=cfg.lr_patience, min_lr=cfg.min_lr, verbose=True)

    for epoch in range(cfg.num_epochs):
        loss = train(model, data, optimizer, mask=data.train_mask, loss_fn=cfg.loss_fn, gradient_clipping=cfg.gradient_clipping)
        train_acc = evaluate(model, data, data.train_mask)
        val_acc = evaluate(model, data, data.val_mask)
        with torch.no_grad():    
            val_loss = F.cross_entropy(model(data)[data.val_mask], data.y[data.val_mask])

        print(f'Epoch: {epoch}, Train Loss: {loss:.4f}, Val Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
        
        if cfg.reduce_on_plateau:
            lrscheduler.step(val_loss)

    val_accuracy = evaluate(model, data, data.val_mask)
    test_accuracy = evaluate(model, data, data.test_mask)
    print(f"Result after final epoch {cfg.num_epochs}: Validation Accuracy: {val_accuracy} Test Accuracy: {test_accuracy}")
    
    return val_accuracy, test_accuracy

def multi_graph_training(cfg, data, model, device):
    """
    Training routine for multi graph datasets
    data is a tuple of train, test, val datasets
    """
    train_dataset, test_dataset, val_dataset = data
    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    #TODO Implement learning rate scheduler

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    if cfg.task == 'node_classification':

        def eval(loader, model):
            model.eval()
            correct = 0
            n_samples = 0
            predictions = []
            targets = []
            
            with torch.no_grad():
                for batch in loader:
                    batch.to(device)
                    logits = model(batch)
                    pred = logits.max(-1)[1]
                    correct+= pred.eq(batch.y).sum().item()
                    n_samples += len(batch.y)

                    predictions.append(pred)
                    targets.append(batch.y)

            predictions = torch.concatenate(predictions, dim=0)
            targets = torch.concatenate(targets, dim=0)
            for i in range(18):
                print(f'Class {i} has {torch.sum(predictions==i)} predicted samples')

            acc = correct/n_samples
            f1 = f1_score(targets.cpu(), predictions.cpu(), average='macro')
            return acc, f1

        for epoch in range(cfg.num_epochs):
            for i, batch in enumerate(train_loader):
                batch.to(device)
                loss = train(model, batch, optimizer, loss_fn=cfg.loss_fn, gradient_clipping=cfg.gradient_clipping)
                if i % 20 == 0:
                    print(f'Epoch: {epoch}, Loss: {loss:.4f}')
            train_acc, train_f1 = eval(train_loader, model)
            val_acc, val_f1 = eval(val_loader, model)
            print(f'Epoch: {epoch}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}')

        val_acc, val_f1 = eval(val_loader, model)
        test_acc, test_f1 = eval(test_loader, model)
        print(f"Result after final epoch {cfg.num_epochs}: Validation Accuracy: {val_acc} Test Accuracy: {test_acc}")
        print(f"Result after final epoch {cfg.num_epochs}: Validation F1: {val_f1} Test F1: {test_f1}")

        return val_acc, test_acc

    elif cfg.task == 'node_regression':
        raise NotImplementedError
    
    elif cfg.task == 'graph_regression':

        def eval(loader, model):
            model.eval()
            with torch.no_grad():
                error = 0
                for batch in loader:
                    batch.to(device)
                    out = model(batch)
                    val_loss = get_loss_function(cfg.loss_fn)(out, batch.y).item()*len(batch.y)
                    error += val_loss
                return error/len(loader.dataset)

        for epoch in range(cfg.num_epochs):
            for i, batch in enumerate(train_loader):
                batch.to(device)
                loss = train(model, batch, optimizer, loss_fn=cfg.loss_fn)
                if i % 20 == 0:
                    print(f'Epoch: {epoch}, Loss: {loss:.4f}')
            train_loss = eval(train_loader, model)
            val_loss = eval(val_loader, model)
            test_loss = eval(test_loader, model)
            wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "test_loss": test_loss})
            print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}')

        val_loss = eval(val_loader, model)
        test_loss = eval(test_loader, model)

        print(f'Result after final epoch {cfg.num_epochs}: Validation Loss: {val_loss} Test Loss: {test_loss}')

        return val_loss, test_loss

    elif cfg.task == 'graph_classification':
        raise NotImplementedError

def load_model(cfg):
    pooling = False
    if cfg.task in ['graph_classification', 'graph_regression']:
        pooling = True

    #TODO ADD POOLING FOR GRAPHCON AND GCNs
    if cfg.model == 'graphcon':
        model = GraphCON_GCN(nfeat=cfg.num_node_features,
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
            input_dim=cfg.num_node_features,
            hidden_dim=cfg.hidden_dim,
            output_dim=cfg.output_dim,
            N=cfg.N,
            normalize=cfg.normalize,
            activation=cfg.activation,
            self_loops=cfg.self_loops,
            dropout_rate=cfg.dropout_rate,
            h=cfg.h,
            pooling=pooling,
            initial_processing=cfg.initial_processing,
            post_processing=cfg.post_processing,
            learn_spectrum=cfg.learn_spectrum,
            hidden_gcn_dim=cfg.hidden_gcn_dim)
    elif cfg.model == 'glaudio_lstm':
        model = glaudioLSTM(input_dim=cfg.num_node_features,
                            hidden_dim=cfg.hidden_dim,
                            output_dim=cfg.output_dim,
                            N=cfg.N,
                            normalize=cfg.normalize,
                            activation=cfg.activation,
                            h = cfg.h,
                            dropout_rate=cfg.dropout_rate,
                            self_loops=cfg.self_loops,
                            pooling=pooling)
        
    elif cfg.model == 'gcn':
        model = GCN(in_channels=cfg.num_node_features, hidden_channels=cfg.hidden_dim, out_channels=cfg.output_dim, num_layers=cfg.N, dropout=cfg.dropout_rate)
    elif cfg.model == 'gcn_res_conn':
        model = GCNwithResConn(in_channels=cfg.num_node_features, hidden_channels=cfg.hidden_dim, out_channels=cfg.output_dim, num_layers=cfg.N, num_layers_after_conv=2)
    else:   
        raise NotImplementedError
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded model {cfg.model} with {n_params} parameters")

    wandb.log({"num_parameters": n_params})

    return model


if __name__ == "__main__":
    wandb.login()

    def main():
        with wandb.init(project=PROJECT_NAME):
            # access all HPs through wandb.config, so logging matches execution!
            cfg = wandb.config
            run_experiment(cfg)

    # 2: Define the search space    
    sweep_configuration = {
        "method": "grid",
        "metric": {"goal": "minimize", "name": "val_loss"},
        "parameters": {
            "N": {"values": [20]},
            "L": {"values": [3]},
            "h": {"values": [0.5]}, 
            "hidden_dim": {"values": [192]},
            "lr": {"values": [2e-3]},
            "weight_decay": {"values": [0.0]},
            "dropout_rate": {"value": 0.0},
            "normalize": {"values": [True]},
            "self_loops": {"value": False},
            "activation" : {"value": "gelu"},
            "initial_processing": {"value": True},
            "post_processing": {"value": False},
            "perturbe": {"value": perturbe},
            "learn_spectrum" : {"value": False},
            "hidden_gcn_dim" : {"value": 0},
            "gradient_clipping" : {"value": False}, 
            
            #fixed parameters
            "num_node_features" : {"value": 9},
            "batch_size" : {"value": 64},
            "task" : {"value": "graph_regression"},
            "model" : {"value": "glaudio_lstm"},
            "name" : {"value": NAME},
            "split" : {"value": "0"},
            "loss_fn": {"value": "l1"},
            "output_dim" : {"value": OUTPUT_DIM},
            "num_epochs" : {"value": 300},
            "lr_factor" : {"value": 0.5},
            "lr_patience" : {"value": 10},
            "min_lr" : {"value": 1e-5},             
        }                           
    }



    # 3: Start the sweep
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=PROJECT_NAME)
    wandb.agent(sweep_id=sweep_id, function=main)


