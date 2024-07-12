from main import load_model
from main import single_graph_training
from src.datasets.planetoid import get_planetoid_data
from omegaconf import DictConfig  
import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle

"""
This is a short script to test the impact of N on the performance of the model.
We fixed for different experiments all hyperparameters except N and we vary N and h = T/N accordingly.
"""

pub_med_configuration = {
            "name": "PubMed",
            "task": "node_classification",
            "num_node_features": 500,
            "output_dim": 3,
            "model": "glaudio",
            "num_runs": 10,

            "num_epochs": 300,
            "weight_decay": 0,
            "split": 0,
            "activation": "leaky_relu",
            "gradient_clipping": False,

            "normalize": False,
            "self_loops": False,
            "hidden_dim": 24,
            "dropout_rate": 0.3,
            "L": 1,
            "N": 50,
            "h": 0.05,
            "T": 2.5,
            "hidden_gcn_dim": 0,
            "initial_processing": False,
            "post_processing": False,

            "reduce_on_plateau": True,
            "lr" : 0.0025,
            "lr_patience": 10,
            "lr_factor": 0.5,
            "min_lr": 0.0001,

            "loss_fn": "cross_entropy"           
        }  

cora_configuration = {
            "name": "Cora",
            "task": "node_classification",
            "num_node_features": 1433,
            "output_dim": 7,
            "model": "glaudio",
            "num_runs": 10,

            "num_epochs": 300,
            "weight_decay": 0.005,
            "split": 0,
            "activation": "leaky_relu",
            "gradient_clipping": False,

            "normalize": True,
            "self_loops": False,
            "hidden_dim": 32,
            "dropout_rate": 0.2,
            "L": 2,
            "N": 200,
            "h": 0.02,
            "T": 4,
            "hidden_gcn_dim": 0,
            "initial_processing": False,
            "post_processing": False,

            "reduce_on_plateau": True,
            "lr" : 0.001,
            "lr_patience": 10,
            "lr_factor": 0.5,
            "min_lr": 0.0001,

            "loss_fn": "cross_entropy"           
        }

citeseer_configuration = {
            "name": "CiteSeer",
            "task": "node_classification",
            "num_node_features": 3703,
            "output_dim": 6,
            "model": "glaudio",
            "num_runs": 10,

            "num_epochs": 300,
            "weight_decay": 0.005,
            "split": 0,
            "activation": "leaky_relu",
            "gradient_clipping": False,

            "normalize": False,
            "self_loops": False,
            "hidden_dim": 24,
            "dropout_rate": 0.2,
            "L": 1,
            "N": 150,
            "h": 0.01,
            "T": 1.5,
            "hidden_gcn_dim": 0,
            "initial_processing": False,
            "post_processing": False,

            "reduce_on_plateau": True,
            "lr" : 0.0025,
            "lr_patience": 10,
            "lr_factor": 0.5,
            "min_lr": 0.0001,

            "loss_fn": "cross_entropy"           
        }
    
configs = [pub_med_configuration, cora_configuration, citeseer_configuration]

N_values = [1, 2, 4, 8, 16, 32, 64, 128]

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":

    reports = []
    
    for configuration in configs:

        report = []

        for N in N_values:

            print("N = ", N, " Start training...")

            vals = []
            tests = []

            for i in range(configuration["num_runs"]):    
                configuration["N"] = N
                configuration["h"] = configuration["T"]/N

                conf = DictConfig(configuration)
                model = load_model(conf)
                model.to(device)

                data = get_planetoid_data(conf, sweep=True)
                data.to(device)
                

                val_acc, test_acc = single_graph_training(conf, data, model, verbose=False)
                
                vals.append(val_acc)
                tests.append(test_acc)

            avgval = sum(vals)/len(vals)
            avgtest = sum(tests)/len(tests)

            report.append([N, avgval, avgtest])
            print("N = ", N, " done")
        
        reports.append((configuration["name"], report))

    with open("plots/reports.pkl", "wb") as f:
        pickle.dump(reports, f)



    for name, report in reports:
        # print report as a table
        print("Report for: ", name)
        print("N \t val_acc \t test_acc")
        for row in report:
            print(row[0], "\t", row[1], "\t \t", row[2])

    plt.figure()

    # plot a line graph with with two lines: val_acc and test_acc, x-axis: N
    for name, report in reports:
        report = np.array(report)
        N = report[:, 0]
        val_acc = report[:, 1]
        test_acc = report[:, 2]

        plt.plot(N, test_acc, label=name, marker="s")

    # use log scale for x-axis with base 2
    plt.xscale("log", base=2)


    x_ticks = np.array(N_values)
    # use N as x-axis scale
    plt.xticks(ticks=x_ticks, labels=x_ticks.astype(int))

    plt.legend()
    plt.xlabel("Time Steps")
    plt.ylabel("Test Accuracy")

    plt.tight_layout()


    plt.savefig("plots/report.eps", format="eps")
    plt.savefig("plots/report.svg", format="svg")
    plt.savefig("plots/report.png", format="png")


    

