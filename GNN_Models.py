import torch
from torch_geometric.nn import GATConv, SAGEConv, GCNConv
import torch.nn.functional as F
from torch.nn import Linear
# Note: softmax at the end of each forward pass not necessary because CLE expects raw scores.

# ------------------------------------------------------------------
# ----------------------------- GCN --------------------------------
# ------------------------------------------------------------------


class TwoLayerGCN(torch.nn.Module):
    def __init__(self, hidden_channels, input_channels, output_channels):
        super().__init__()
        # torch.manual_seed(26)
        self.conv1 = GCNConv(input_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, output_channels)

    def forward(self, x, edge_index, drpt):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=drpt, training=self.training)

        x = self.conv2(x, edge_index)

        return x

# ------------------------------------------------------------------
# ----------------------------- SAGE -------------------------------
# ------------------------------------------------------------------

class TwoLayerGraphSAGE(torch.nn.Module):
    def __init__(self, hidden_channels, input_channels, output_channels):
        super().__init__()
        # torch.manual_seed(26)
        self.conv1 = SAGEConv(input_channels, hidden_channels, aggr='mean')  # 'lstm', 'max', 'pool'?
        self.conv2 = SAGEConv(hidden_channels, output_channels, aggr='mean')

    def forward(self, x, edge_index, drpt):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=drpt, training=self.training)

        x = self.conv2(x, edge_index)

        return x

# ------------------------------------------------------------------
# ----------------------------- GAT --------------------------------
# ------------------------------------------------------------------

class TwoLayerGAT(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, heads, output_heads=1):
        super().__init__()
        #torch.manual_seed(26)
        self.conv1 = GATConv(input_channels, hidden_channels, heads)
        self.conv2 = GATConv(hidden_channels * heads, output_channels, output_heads)

    def forward(self, x, edge_index, drpt):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=drpt, training=self.training)

        x = self.conv2(x, edge_index)

        return x


class TwoLayerMLP(torch.nn.Module):
    def __init__(self, hidden_channels, input_channels, output_channels):
        super().__init__()
        # torch.manual_seed(26)
        self.lin1 = Linear(input_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, output_channels)

    def forward(self, x, edge_index, drpt):  # edge_index must be included, although not needed
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=drpt, training=self.training)

        x = self.lin1(x)

        return x





if __name__ == '__main__':

    from ParametricGraphModels.ADC_SBM import *
    from config import Scenarios
    print(Scenarios.noise["community_sizes"])
    g = from_config(Scenarios.noise)

    num_targets = g.num_y_labels
    num_input_features = g.x.shape[1]
    wgth_dcy = 5e-4
    lrn_rt = 0.01
    hc1 = 16
    hc2 = 8
    drp1 = .4
    drp2 = .1
    attention_heads = 8

    model = TwoLayerGAT()

    optimizer = torch.optim.Adam(model.parameters(), lr=lrn_rt, weight_decay=wgth_dcy)
    criterion = torch.nn.CrossEntropyLoss()

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
                         data.y[mask]) # only calculate loss on train (due to graph structure)
        loss.backward()
        optimizer.step()
        return loss
