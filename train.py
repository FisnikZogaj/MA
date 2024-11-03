import pickle
import os
import argparse
import ast
import numpy as np
import torch.nn
import time

# Custom Modules
from GNN_Models import *  # here the models are stored
from ParametricGraphModels.ADC_SBM import from_config
from tqdm import tqdm

def run_experiment(graph_config: dict, architecture: str, seed: int, ts: str):
    """
    train.py script, that gets executed via main.py.

    :param graph_config: serialized graph Object
    :param architecture:
    :param seed:
    :param ts:
    :return:
    """

    # 1)  -----  Generate the Graph with a specified Configuration ------
    # print("*" * 40)
    g = from_config(graph_config, seed)

    # ----------- Define Hyperparameters for training ----------------

    num_targets = g.y_out_dim # important flag for task determination. Number of targets |tau|
    num_input_features = g.x.shape[1]
    wgth_dcy = 5e-4
    lrn_rt = 0.01
    hc1 = 16
    drp1 = .4
    attention_heads = 8

    # Linux vs Windows
    #base = r"C:\Users\zogaj\PycharmProjects\MA\ExperimentLogs"
    base = r"/home/zogaj/MA/ExperimentLogs"  # Linux Server

    final_path = os.path.join(base, ts, architecture, graph_config["name"])
    final_gchar_path = os.path.join(base, ts, "GraphCharacteristics", graph_config["name"])

    # --------------------------- Load Model ------------------------

    if architecture == "GCN":
        model = TwoLayerGCN(hidden_channels=hc1,
                            input_channels=num_input_features,
                            output_channels=num_targets)

    elif architecture == "SAGE":
        model = TwoLayerGraphSAGE(hidden_channels=hc1,
                                  input_channels=num_input_features,
                                  output_channels=num_targets)

    elif architecture == "GAT":
        model = TwoLayerGAT(input_channels=num_input_features,
                            hidden_channels=hc1,
                            heads=attention_heads,
                            output_channels=num_targets)

    elif architecture == "MLP":
        model = TwoLayerMLP(input_channels=num_input_features,
                            hidden_channels=hc1,
                            output_channels=num_targets)

    else:  
        pass

    criterion = torch.nn.CrossEntropyLoss()


    # ------------- Train The model --------------------

    def train(data):
        """
        Note that all the parameter-inputs must be uniform across all models.
        :param data: DataObject
        :return: loss
        """
        model.train()
        optimizer.zero_grad()
        mask = data.train_mask

        out = model(data.x,
                    data.edge_index,
                    drpt=drp1)

        loss = criterion(out[mask],
                         data.y[mask])
        loss.backward()
        optimizer.step()
        return loss

    def test(data, mask) -> float:
        """
        :param data: A torch.Data object.
        :param mask: Controlling whether test or validation data is used.
        :return:
        """
        model.eval()
        out = model(data.x,
                    data.edge_index,
                    drpt=drp1)

        pred = out.argmax(dim=1)
        correct = pred[mask] == data.y[mask]
        acc = int(correct.sum()) / int(mask.sum())

        return acc

    def full_training_early_stop(data, n_epochs, patience):
        """
        Run a full train_loop, with early stopping. Will run the full Epochs, but track the early stop
        and calculate test accuracy at that point.
        (Control-flow could be problematic)
        :param data:
        :param n_epochs:
        :param patience:
        :return:
        """

        val_acc_track = np.zeros(n_epochs)
        loss_track = np.zeros(n_epochs)
        iteration_times = np.zeros(n_epochs)
        early_stop = None

        best_val_acc = -np.inf
        epochs_without_improvement = 0
        pseudo_break = False

        for epoch in tqdm(range(1, n_epochs)):  # [1, ..., 100] - len() -> 100

            start_time = time.time()
            loss = train(data)
            end_time = time.time()
            iteration_times[epoch-1] = end_time - start_time

            val_acc = test(data, data.val_mask)
            val_acc_track[epoch-1] = val_acc
            loss_track[epoch-1] = loss


            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement == patience and not pseudo_break:
                early_stop = epoch
                pseudo_break = True  # won't trigger if-clause anymore
                test_accuracy = test(data, data.test_mask)


        if early_stop is None:
            # Training was never aborted due to early stopping, thus it stays None.
            test_accuracy = test(data, data.test_mask)

        loss_track = loss_track
        val_acc_track = val_acc_track

        return loss_track, val_acc_track, test_accuracy, early_stop, iteration_times


    optimizer = torch.optim.Adam(model.parameters(), lr=lrn_rt, weight_decay=wgth_dcy)

    loss_track, val_acc_track, test_accuracy, final_epoch, epoch_times = (
        full_training_early_stop(data=g.DataObject, n_epochs=151, patience=10))  # range(151) -> 150 actual runs!

    # ---------------- Save all results -----------------

    train_output = {
        "model": architecture,
        "Scenario": graph_config["name"],
        "loss_track": loss_track,
        "val_acc_track": val_acc_track,
        "test_accuracy": test_accuracy,
        "epoch_times": epoch_times,
        "final_epoch": final_epoch
    }


    output_path_1 = os.path.join(final_path, f"output{seed}.pkl")
    with open(output_path_1, 'wb') as file:
        print("@ ", architecture, "+", graph_config["name"], "+", seed, "->", output_path_1, "\n")
        pickle.dump(train_output, file)


    # ---- Further save Graph characteristics here ----

    GraphCharacteristics = {
        "h_hat": g.edge_homophily(),
        "h": g.simple_edge_homophily(),
        "class_balance": np.bincount(g.y),
        "wilks_lambda": g.manova_x()
    }

    output_path_2 = os.path.join(final_gchar_path, f"output{seed}.pkl")
    if not os.path.exists(output_path_2):
        with open(output_path_2, 'wb') as file:
            pickle.dump(GraphCharacteristics, file)
            print("@ ", graph_config["name"], "+", seed, "->", output_path_2, "\n")
    else:
        print("Already exists -> pass")
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training Script')
    parser.add_argument('--config', type=str, required=True, help='Config dict of the Graph')
    parser.add_argument('--architecture', type=str, required=True, help='What model to run: [GCN, SAGE, GAT, MLP]')
    parser.add_argument('--seed', type=int, required=True, help='reproducibility seed')
    parser.add_argument('--timestamp', type=str, required=True, help='When has main been executed')
    args = parser.parse_args()

    # read in str()-representation to return actual dict-type
    config = ast.literal_eval(args.config)

    run_experiment(graph_config=config,
                   architecture=args.architecture,
                   seed=args.seed,
                   ts=args.timestamp)

    # ---- for debugging ------

    # from config import Scenarios
    # run_experiment(graph_config=Scenarios.perfect,
    #                 architecture="MLP",
    #                 seed=1,
    #                 ts="debug")



