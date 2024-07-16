import torch
from torch_geometric.nn import GATConv, SAGEConv, GCNConv
import torch.nn.functional as F

# ------------------------------------------------------------------
# ----------------------------- GCN --------------------------------
# ------------------------------------------------------------------

class ThreeLayerGCN(torch.nn.Module):
    def __init__(self, hidden_channels, hidden_channels2, input_channels, output_channels):
        super().__init__()
        torch.manual_seed(26)
        self.conv1 = GCNConv(input_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels2)
        self.conv3 = GCNConv(hidden_channels2, output_channels)

    def forward(self, x, edge_index, drpt=0, drpt2=0):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=drpt, training=self.training)

        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=drpt2, training=self.training)

        x = self.conv3(x, edge_index)

        return x

class TwoLayerGCN(torch.nn.Module):
    def __init__(self, hidden_channels, input_channels, output_channels):
        super().__init__()
        torch.manual_seed(26)
        self.conv1 = GCNConv(input_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, output_channels)

    def forward(self, x, edge_index, drpt=0, drpt2=0):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=drpt, training=self.training)

        x = self.conv2(x, edge_index)
        x = x.relu()

        return x

# ------------------------------------------------------------------
# ----------------------------- SAGE -------------------------------
# ------------------------------------------------------------------

class TwoLayerGraphSAGE(torch.nn.Module):
    def __init__(self, hidden_channels, input_channels, output_channels):
        super(TwoLayerGraphSAGE, self).__init__()
        torch.manual_seed(26)
        self.conv1 = SAGEConv(input_channels, hidden_channels)  # First GraphSAGE layer
        self.conv2 = SAGEConv(hidden_channels, output_channels)  # Second GraphSAGE layer

    def forward(self, x, edge_index, drpt=0, drpt2=0):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=drpt, training=self.training)

        x = self.conv2(x, edge_index)
        x = x.relu()

        return x

# ------------------------------------------------------------------
# ----------------------------- GAT --------------------------------
# ------------------------------------------------------------------
class TwoLayerGAT(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, heads, output_heads=1):
        super().__init__()
        torch.manual_seed(26)
        self.conv1 = GATConv(input_channels, hidden_channels, heads)
        self.conv2 = GATConv(hidden_channels * heads, output_channels, output_heads)

    def forward(self, x, edge_index, drpt=0, drpt2=0):
        x = F.dropout(x, p=drpt, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)

        x = F.dropout(x, p=drpt2, training=self.training)
        x = self.conv2(x, edge_index)

        # Optionally apply an activation function if needed for the final output
        # x = F.elu(x)

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
