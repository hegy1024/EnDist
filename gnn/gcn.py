import torch
import torch.nn as nn

from torch.nn import ReLU, Linear, BatchNorm1d
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool

class NodeGCN(nn.Module):
    """
    A node classification model for nodes classification described in https://arxiv.org/abs/1903.03894.
    This model consists of 3 stacked Graph conv layers followed by a linear layer.
    """
    def __init__(self, input_dims, out_dims, hidden_dims: int = 20):
        super(NodeGCN, self).__init__()
        self.embedding_size = hidden_dims * 3
        self.conv1 = GCNConv(input_dims, hidden_dims)
        self.relu1 = ReLU()
        self.conv2 = GCNConv(hidden_dims, hidden_dims)
        self.relu2 = ReLU()
        self.conv3 = GCNConv(hidden_dims, hidden_dims)
        self.relu3 = ReLU()
        self.lin = Linear(self.embedding_size, out_dims)

    def readout(self, embeds):
        return self.lin(embeds)

    def forward(self, x, edge_index, edge_weights=None, **kwargs):
        embeds = self.embedding(x, edge_index, edge_weights)[0]
        return self.readout(embeds)

    def embedding(self, x, edge_index, edge_weights=None):
        stack = []

        out1 = self.conv1(x, edge_index, edge_weights)
        out1 = torch.nn.functional.normalize(out1, p=2, dim=1)
        out1 = self.relu1(out1)
        stack.append(out1)

        out2 = self.conv2(out1, edge_index, edge_weights)
        out2 = torch.nn.functional.normalize(out2, p=2, dim=1)
        out2 = self.relu2(out2)
        stack.append(out2)

        out3 = self.conv3(out2, edge_index, edge_weights)
        out3 = torch.nn.functional.normalize(out3, p=2, dim=1)
        out3 = self.relu3(out3)
        stack.append(out3)

        embedding = torch.cat(stack, dim=1)

        return embedding,

class GraphGCN(nn.Module):
    """
    A graph classification model for datasets described in https://arxiv.org/abs/1903.03894.
    This model consists of 3 stacked GCN layers followed by a linear layer.
    In between the GCN outputs and linear layers are pooling operations in both mean and max.
    """
    def __init__(self, input_dims, out_dims, hidden_dims: int = 20):
        super(GraphGCN, self).__init__()
        self.embedding_size = hidden_dims * 3
        self.conv1 = GCNConv(input_dims, hidden_dims)
        self.relu1 = ReLU()
        self.conv2 = GCNConv(hidden_dims, hidden_dims)
        self.relu2 = ReLU()
        self.conv3 = GCNConv(hidden_dims, hidden_dims)
        self.relu3 = ReLU()
        self.lin = Linear(hidden_dims * 2, out_dims)

    def readout(self, embeds, batch):
        out1 = global_max_pool(embeds, batch)
        out2 = global_mean_pool(embeds, batch)
        input_lin = torch.cat([out1, out2], dim=-1)

        return self.lin(input_lin)

    def forward(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None:  # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        embeds = self.embedding(x, edge_index, edge_weights)[1]

        return self.readout(embeds, batch)

    def embedding(self, x, edge_index, edge_weights=None):
        stack = []

        out1 = self.conv1(x, edge_index, edge_weights)
        out1 = torch.nn.functional.normalize(out1, p=2, dim=1)
        out1 = self.relu1(out1)
        stack.append(out1)

        out2 = self.conv2(out1, edge_index, edge_weights)
        out2 = torch.nn.functional.normalize(out2, p=2, dim=1)
        out2 = self.relu2(out2)
        stack.append(out2)

        out3 = self.conv3(out2, edge_index, edge_weights)
        out3 = torch.nn.functional.normalize(out3, p=2, dim=1)
        out3 = self.relu3(out3)
        stack.append(out3)

        embedding = torch.cat(stack, dim=1)

        return embedding, out3