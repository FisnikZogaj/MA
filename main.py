# Python Packages
import pickle
import torch
import numpy as np
from GNN_Models import ThreeLayerGCN
import torch_geometric
from datetime import datetime
import os
from matplotlib import pyplot as plt

# Custom Modules
from ParametricGraphModels.ADC_SBM import from_config
from config import MultiClassClassification as MCC

ts = datetime.now().strftime("%Y-%m-%d %H:%M").translate(str.maketrans({" ": ".", ":": ".", "-": "."}))

# 1)  -----  Generate the Graph with a specified Configuration ------
graph_params = MCC.overlap_assort_seperated
# ... Different configs
g = from_config(graph_params, rs=26)

g.rich_plot_graph()
print(g.purity(plot_it=True))
input("Precede with this graph ?")

# ----------- Define Hyperparameters for training ----------------

num_targets = graph_params["n_targets"]
num_input_features = g.x.shape[1]
wgth_dcy = 5e-4
lrn_rt = 0.01
hc1 = 16
hc2 = 8
drp1 = .4
drp2 = .1

print("number of targets generated : ", num_targets, "specified: ", graph_params["n_targets"])

# --------------------------- Load Model ------------------------

model = ThreeLayerGCN(hidden_channels=16,
                      hidden_channels2=8,
                      input_channels=num_input_features,
                      output_channels=num_targets)  # initialize here
# model2 ...
# model3 ...

optimizer = torch.optim.Adam(model.parameters(), lr=lrn_rt, weight_decay=wgth_dcy)
criterion = torch.nn.CrossEntropyLoss()


# ------------- Train The model --------------------
# Are all Node-labels known?
# then it's not semi-supervised node classification
# Splitting based on 3 generated graphs would blur the idea of Inductive and transductive learning.
# Source for why we split data?
# Inductive: Training on one graph and evaluating on a completely separate graph.
# Transductive: Splitting the same graph into train and test sets while ensuring the test
# set contains nodes or edges unseen during training.
def train(data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, drop=drp1, drop2=drp2) # Why train full model but
    loss = criterion(out[data.train_mask], data.y[data.train_mask]) # only calculate loss on train?
    loss.backward()
    optimizer.step()
    return loss


def test(data, mask):
    model.eval()
    out = model(data.x, data.edge_index)
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
    val_acc_track = np.zeros(n_epochs)
    loss_track = np.zeros(n_epochs)

    # Initialize early stopping variables
    best_val_acc = -np.inf
    epochs_without_improvement = 0
    best_epoch = 0

    for epoch in range(n_epochs):
        loss = train(data)
        val_acc = test(data, data.val_mask)
        val_acc_track[epoch] = val_acc
        loss_track[epoch] = loss

        print(f'Epoch: {epoch + 1:03d}, Loss: {loss:.4f}, Val: {val_acc:.4f}')

        # Check if the current epoch has the best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_without_improvement = 0  # Reset the patience counter
        else:
            epochs_without_improvement += 1

        # Check for early stopping
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    test_accuracy = test(data, data.test_mask)

    loss_track = loss_track[:epoch]
    val_acc_track = val_acc_track[:epoch]

    return loss_track, val_acc_track, test_accuracy, epoch


loss_track, val_acc_track, test_accuracy, final_epoch = (
    full_training_early_stop(g.DataObject, 100, 25))

# ---------------- Save all results -----------------

hyperparams = {
    "out_dim": num_targets,
    "in_dim": num_input_features,
    "weight_decay": wgth_dcy,
    "hidden_layer1_dim": hc1,
    "hidden_layer2_dim": hc2,
    "drop_out1": drp1,
    "drop_out2": drp2,
    "learn_rate": lrn_rt
}

train_output = {
    "loss_track": loss_track,
    "val_acc_track": val_acc_track,
    "test_accuracy": test_accuracy,
    "final_epoch": final_epoch
}

plt.plot(np.arange(final_epoch) + 1, loss_track)
plt.show()
plt.plot(np.arange(final_epoch) + 1, val_acc_track)
plt.show()
print(test_accuracy)


def save_res(params: dict, sub_name: str, train_output: dict, hyperparameters: dict, graph: any):
    # Timestamped filenames advised
    global ts

    # 1) Create Subdirectory within ExperimentLogs
    base = r"C:\Users\zogaj\PycharmProjects\MA\ExperimentLogs"
    subdir_path = os.path.join(base, sub_name)
    os.makedirs(subdir_path, exist_ok=True)

    graph_params_path = os.path.join(subdir_path, "params.txt")
    with open(graph_params_path, 'w') as file:
        for k, v in params.items():
            file.write(f"{k}: {v}\n")
        file.write("\n" + "*"*40 + "\n")
        for k, v in hyperparameters.items():
            file.write(f"{k}: {v}\n")

    graph_path = os.path.join(subdir_path, "graph.pkl")
    with open(graph_path, 'wb') as file:
        pickle.dump(graph, file)

    output_path = os.path.join(subdir_path, "output.pkl")
    with open(output_path, 'wb') as file:
        pickle.dump(train_output, file)

    print(f"Configuration written to {graph_params_path}")
