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
from xgboost import XGBClassifier

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
    base = r"C:\Users\zogaj\PycharmProjects\MA\ExperimentLogs"
    #base = r"/home/zogaj/MA/ExperimentLogs"  # Linux Server

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

    else:  # architecture == "XGBoost"
        pass

    if g.y_out_dim > 2:
        criterion = torch.nn.CrossEntropyLoss()
        mtrc = "CEL"

    elif g.y_out_dim == 2:
        criterion = torch.nn.BCELoss()
        mtrc = "BCEL"

    else:
        criterion = torch.nn.MSELoss()
        mtrc = "MSE"

    # ------------- Train The model --------------------

    # The layer itself is agnostic to whether nodes are labeled or not;
    # it computes the convolution on the node features and the adjacency matrix, propagating information throughout the graph.
    # The supervision (i.e., where labels are considered for the loss) is managed outside the GCNConv layer.
    # For example, in a typical training loop using torch_geometric, one would mask the loss function to only include labeled nodes.

    # Splitting based on 3 generated graphs would blur the idea of Inductive and transductive learning. (Source for why we split data?)

    # Inductive: Training on one graph and evaluating on a completely separate graph.
    # Transductive: Splitting the same graph into train and test sets while ensuring the test
    # set contains nodes or edges unseen during training.
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

        if mtrc == "CEL":
            pred = out.argmax(dim=1)
            correct = pred[mask] == data.y[mask]
            acc = int(correct.sum()) / int(mask.sum())

        elif mtrc == "BCEL":
            raise NotImplementedError

        elif mtrc == "MSE":
            acc = np.mean(np.power(out[mask] - data.y[mask], 2))

        return acc

    nepchs = 101  #if architecture != "GCN" else 201

    def full_training_early_stop(data, n_epochs=nepchs, patience=10):
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
        early_stop = None

        # Initialize early stopping variables
        best_val_acc = -np.inf
        epochs_without_improvement = 0
        pseudo_break = False

        iteration_times = []
        for epoch in tqdm(range(1, n_epochs)): # [1, ..., 100] - len() -> 101

            start_time = time.time()
            loss = train(data)
            end_time = time.time()
            iteration_times.append(end_time - start_time)

            val_acc = test(data, data.val_mask)
            val_acc_track[epoch-1] = val_acc
            loss_track[epoch-1] = loss

            # print(f'Epoch: {epoch + 1:03d}, Loss: {loss:.4f}, Val: {val_acc:.4f}')

            # Check if the current epoch has the best validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement == patience and not pseudo_break:
                # print(f"Early stopping triggered at epoch {epoch + 1}")
                test_accuracy = test(data, data.test_mask)
                early_stop = epoch
                pseudo_break = True  # won't trigger if-clause anymore
                test_accuracy = test(data, data.test_mask)
                # break

        if early_stop is None:
            # Training was never aborted due to early stopping, thus it stayed None.
            test_accuracy = test(data, data.test_mask)

        loss_track = loss_track  # [:epoch]
        val_acc_track = val_acc_track  # [:epoch]

        return loss_track, val_acc_track, test_accuracy, early_stop, iteration_times

    if architecture in ["GCN", "GAT", "SAGE", "MLP"]:
        optimizer = torch.optim.Adam(model.parameters(), lr=lrn_rt, weight_decay=wgth_dcy)
        loss_track, val_acc_track, test_accuracy, final_epoch, epoch_times = (
            full_training_early_stop(g.DataObject, 100, 25))

    else:  # XGBoost, not Manually specified
        raise AssertionError("No valid model selected.")

    #     model = XGBClassifier(
    #         objective='multi:softmax',
    #         num_class=g.y_out_dim,
    #         use_label_encoder=False,
    #         eval_metric=['mlogloss', "merror"],
    #         n_estimators=100,  # Number of boosting rounds (epochs)
    #         learning_rate=lrn_rt,  # Step size shrinkage
    #         max_depth=10,  # Maximum depth of trees
    #         min_child_weight=3,  # Minimum sum of instance weight
    #         subsample=0.8,  # Fraction of samples
    #         colsample_bytree=0.8,  # Fraction of features
    #         gamma=1,  # Minimum loss reduction to split a node
    #         verbosity=0
    #     )
    #
    #     X = g.DataObject.x
    #     y = g.DataObject.y
    #
    #     train_mask = g.DataObject.train_mask
    #     test_mask = g.DataObject.test_mask
    #     val_mask = g.DataObject.val_mask
    #     eval_set = [(X[train_mask], y[train_mask]), (X[val_mask], y[val_mask])]  # 0, 1 index
    #
    #     model.fit(X[train_mask], y[train_mask], eval_set=eval_set, verbose=False)
    #     evals_result = model.evals_result()
    #     y_pred = model.predict(X[test_mask])
    #
    #     test_accuracy = np.mean(y_pred == np.array(y[test_mask]))
    #     loss_track = evals_result['validation_0']['mlogloss']
    #     val_acc_track = 1 - np.array(evals_result['validation_1']['merror'])  # accuracy
    #
    #     counter = 0
    #     final_epoch = 100
    #     for i, e in enumerate(np.diff(val_acc_track)):
    #         if e == 0:
    #             counter += 1
    #         elif counter > 10:
    #             final_epoch = i
    #             break
    #         else:
    #             counter = 0
    #
    # print("Training successfully completed!",  "\n")

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


    output_path = os.path.join(final_path, f"output{seed}.pkl")
    with open(output_path, 'wb') as file:
        print("@ ", architecture, "+", graph_config["name"], "+", seed, "->", output_path, "\n")
        pickle.dump(train_output, file)


    # ---- Further save Graph characteristics here ----
    # If already exists, pass
    GraphCharacteristics = {
        "h_hat": g.edge_homophily(),
        "class_balance": np.bincount(g.y),
        "lab_corr": g.label_correlation(),
        "wilks_lambda": g.manova_x()
        # "tec": g.target_edge_counter(),
        # "pur": g.purity(),
    }

    output_path = os.path.join(final_gchar_path, f"output{seed}.pkl")
    if not os.path.exists(output_path):
        with open(output_path, 'wb') as file:
            pickle.dump(GraphCharacteristics, file)
            print("@ ", graph_config["name"], "+", seed, "->", output_path, "\n")
    else:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training Script')
    parser.add_argument('--config', type=str, required=True, help='Config dict of the Graph')
    parser.add_argument('--architecture', type=str, required=True, help='What model to run: [GCN, SAGE, GAT, "XGBoost]')
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



