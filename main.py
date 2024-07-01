import pickle
import torch
import numpy as np
from ParametricGraphModels.ADC_SBM import Adc_sbm, setB
from GNN_Models import GCN
import torch_geometric





with open(r'C:\Users\zogaj\PycharmProjects\MA\SyntheticGraphs\g1.pkl', 'rb') as f:
    g = pickle.load(f)

num_targets = len(np.unique(g.y))
num_input_features = g.x.shape[1]
print("ny: ", num_targets, "nx: ", g.x.shape[1])

model = GCN(hidden_channels=16,
            hidden_channels2=8,
            input_channels=num_input_features,
            output_channels=num_targets)  # initialize here
# model2 ...
# model3 ...

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

def train(data:torch_geometric.data.data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)  # = model.forward(data.x, data.edge_index) !

    loss = criterion(out[data.train_mask],
                     data.y[data.train_mask])  # pred - y

    loss.backward()
    optimizer.step()
    return loss

data = g.DataObject


for epoch in range(1, 101):
    loss = train(g.DataObject)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')


def mod_evaluation():

    model.eval()
    out = model(data.x, data.edge_index)

    pred = out.argmax(dim=1)

    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    return test_acc


test_acc = mod_evaluation()
print(f'Test Accuracy: {test_acc:.4f}')