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
    def __init__(self, hidden_channels, heads, out_dim):
        super().__init__()
        torch.manual_seed(26)
        self.conv1 = GATConv(hidden_channels, hidden_channels, heads)
        self.conv2 = GATConv(heads*hidden_channels, out_dim, heads)

    def forward(self, x, edge_index, drpt=0, drpt2=0):
        x = F.dropout(x, p=drpt, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)

        x = F.dropout(x, p=drpt2, training=self.training)
        x = self.conv2(x, edge_index)
        return x


if __name__ == '__main__':
    from ParametricGraphModels.ADC_SBM import ADC_SBM
    from config import Scenarios
    print(Scenarios.noise["community_sizes"])

