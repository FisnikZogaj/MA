import networkx as nx
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


from torch_geometric.nn import GATConv, GraphSAGE, GCNConv
import torch.nn.functional as F
from torch_geometric.utils import to_networkx


class ThreeLayerGCN(torch.nn.Module):
    def __init__(self, hidden_channels, hidden_channels2, input_channels, output_channels):
        super().__init__()
        torch.manual_seed(26)
        self.conv1 = GCNConv(input_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels2)
        self.conv3 = GCNConv(hidden_channels2, output_channels)

    def forward(self, x, edge_index, drop=0, drop2=0):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=drop, training=self.training)

        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=drop2, training=self.training)

        x = self.conv3(x, edge_index)

        return x


if __name__ == '__main__':
    from ParametricGraphModels.ADC_SBM import Adc_sbm
    from config import synthetic_dataset_configure
    print(synthetic_dataset_configure["community_sizes"])

