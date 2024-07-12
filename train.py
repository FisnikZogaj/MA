import pickle
import numpy as np
import os
import argparse
import ast

# Custom Modules
from GNN_Models import *  # here the models are stored
from ParametricGraphModels.ADC_SBM import from_config

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
    g = from_config(graph_config, seed)
    # g.rich_plot_graph()
    # print(g.purity(plot_it=True))
    # input("Precede with this graph ?")

    # ----------- Define Hyperparameters for training ----------------

    num_targets = g.num_y_labels
    num_input_features = g.x.shape[1]
    wgth_dcy = 5e-4
    lrn_rt = 0.01
    hc1 = 16
    hc2 = 8
    drp1 = .4
    drp2 = .1
    attention_heads = 8

    # --------------------------- Load Model ------------------------

    if architecture == "GCN":
        model = TwoLayerGCN(hidden_channels=hc1,
                            input_channels=num_input_features,
                            output_channels=num_targets)  # initialize here

    elif architecture == "SAGE":
        model = TwoLayerGraphSAGE(hidden_channels=hc1,
                                  input_channels=num_input_features,
                                  output_channels=num_targets)  # initialize here

    elif architecture == "GAT":
        model = TwoLayerGAT(input_channels=num_input_features,
                            hidden_channels=hc1,
                            heads=num_input_features,
                            output_channels=num_targets)  # initialize here
    else:
        raise ValueError(f"Model Architecture must be one of [GCN, SAGE, GAT]. Received: '{architecture}'.")

    optimizer = torch.optim.Adam(model.parameters(), lr=lrn_rt, weight_decay=wgth_dcy)
    criterion = torch.nn.CrossEntropyLoss()

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
        model.train()
        optimizer.zero_grad()
        # Note that all the inputs must be uniform across all models
        out = model(data.x, # Put Train Masks here?
                    data.edge_index, # and here?
                    drpt=drp1,
                    drpt2=drp2)

        mask = data.train_mask # & (data.y != -1)
        loss = criterion(out[mask],
                         data.y[mask]) # only calculate loss on train?
        loss.backward()
        optimizer.step()
        return loss

    def test(data, mask):
        model.eval()
        out = model(data.x,
                    data.edge_index)
        pred = out.argmax(dim=1)
        correct = pred[mask] == data.y[mask]
        acc = int(correct.sum()) / int(mask.sum())
        return acc

    def full_training(data, n_epochs=101):
        val_acc_track = np.zeros(n_epochs)
        #test_acc_all = np.zeros(n_epochs)
        loss_track = np.zeros(n_epochs)

        for epoch in range(n_epochs):
            loss = train(data)
            val_acc = test(data, data.val_mask)
            #test_acc = test(data, data.test_mask)
            val_acc_track[epoch] = val_acc
            #test_acc_all[epoch] = test_acc
            loss_track[epoch] = loss
            print(f'Epoch: {epoch + 1:03d}, Loss: {loss:.4f}, Val: {val_acc:.4f}')
        test_accuracy = test(data, data.test_mask)

        return loss_track, val_acc_track, test_accuracy

    def full_training_early_stop(data, n_epochs=101, patience=10):
        """
        Run a full train_loop, with early stopping. Will run the full Epochs, but track the early stop
        and calculate test accuracy at that point.
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
        # best_epoch = 0
        pseudo_break = False

        for epoch in range(n_epochs):
            loss = train(data)
            val_acc = test(data, data.val_mask)
            val_acc_track[epoch] = val_acc
            loss_track[epoch] = loss

            # print(f'Epoch: {epoch + 1:03d}, Loss: {loss:.4f}, Val: {val_acc:.4f}')

            # Check if the current epoch has the best validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # best_epoch = epoch
                epochs_without_improvement = 0  # Reset the patience counter
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience and not pseudo_break:
                # print(f"Early stopping triggered at epoch {epoch + 1}")
                test_accuracy = test(data, data.test_mask)
                early_stop = epoch + 1
                pseudo_break = True # won't trigger if-clause anymore
                # break

        if early_stop is not None:
            # then it was never overwritten
            test_accuracy = test(data, data.test_mask)

        loss_track = loss_track #[:epoch]
        val_acc_track = val_acc_track #[:epoch]

        return loss_track, val_acc_track, test_accuracy, early_stop # epoch,


    loss_track, val_acc_track, test_accuracy, final_epoch = (
        full_training_early_stop(g.DataObject, 100, 25))

    # ---------------- Save all results -----------------
    # Saved at ExperimentLogs/ts/hypp.txt. Only once tho. Be additinal param ...
    hyperparams = {
        "out_dim": num_targets,
        "in_dim": num_input_features,
        "weight_decay": wgth_dcy,
        "hidden_layer1_dim": hc1,
        "hidden_layer2_dim": hc2,
        "drop_out1": drp1,
        "drop_out2": drp2,
        "learn_rate": lrn_rt,
        "attention_heads": attention_heads
    }

    train_output = {
        "model": architecture,
        "loss_track": loss_track,
        "val_acc_track": val_acc_track,
        "test_accuracy": test_accuracy,
        "final_epoch": final_epoch
    }

    base = r"C:\Users\zogaj\PycharmProjects\MA\ExperimentLogs"
    final_path = os.path.join(base, ts, architecture, graph_config["name"])
    # architecture_path = os.path.join(stamped, architecture)
    # final_path = os.path.join(architecture_path, graph_config["name"])

    output_path = os.path.join(final_path, f"output{seed}.pkl")
    with open(output_path, 'wb') as file:
        pickle.dump(train_output, file)

    # Further save Graph characteristics here

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training Script')
    parser.add_argument('--config', type=str, required=True, help='Config dict of the Graph')
    parser.add_argument('--architecture', type=str, required=True, help='What model to run: [GCN, SAGE,GAT]')
    parser.add_argument('--seed', type=int, required=True, help='reproducibility seed')
    parser.add_argument('--timestamp', type=str, required=True, help='When has main been executed')
    args = parser.parse_args()

    # read in str()-representation to return actual dict-type
    config = ast.literal_eval(args.config)

    run_experiment(graph_config=config,
                   architecture=args.architecture,
                   seed=args.seed,
                   ts=args.timestamp)
