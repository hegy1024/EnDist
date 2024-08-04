from torch_geometric.nn import GCNConv, VGAE, GAE, InnerProductDecoder
from torch_geometric.utils import negative_sampling

from utils.typing_utils import *
from explain.utils import set_masks, clear_masks


class VGAEncoder(nn.Module):
    r"""
    Encoder for VGAE of proxy generator.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int = 16):
        super(VGAEncoder, self).__init__()
        self.gcn_shared = GCNConv(in_channels, 2 * hidden_channels)
        self.gcn_out    = GCNConv(2 * hidden_channels, 2 * out_channels)

    def forward(self, x: Tensor, edge_index: Tensor, **kwargs):
        x = F.relu(self.gcn_shared(x, edge_index, **kwargs))
        # return mu & logvar
        return self.gcn_out(x, edge_index, **kwargs).chunk(2, dim=1)


class GAEncoder(nn.Module):
    r"""
    Encoder for GAE of proxy generator.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int = 16):
        super(GAEncoder, self).__init__()
        self.gcn_shared = GCNConv(in_channels, hidden_channels)
        self.gcn_out    = GCNConv(hidden_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor, **kwargs):
        x = F.relu(self.gcn_shared(x, edge_index, **kwargs))
        # return z
        return self.gcn_out(x, edge_index, **kwargs)

class ProxyGenerator(nn.Module):
    r"""
    An implementation code for ProxyGenerator in ICML2024 paper
        `Interpreting Graph Neural Networks with In-Distributed Proxies`,
    Code are based on the algorithm description of paper.
    Check https://arxiv.org/abs/2402.02036 for more details.
    """
    coeffs = {'beta': 1., 'lambda': 1., 'alpha': 0.7}

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, **kwargs):
        super(ProxyGenerator, self).__init__()
        self.exp_subgraph_generator     = GAE(encoder=GAEncoder(in_channels,
                                                                hidden_channels,
                                                                out_channels),
                                              decoder=InnerProductDecoder())
        self.non_exp_subgraph_generator = VGAE(encoder=VGAEncoder(in_channels,
                                                                  hidden_channels,
                                                                  out_channels),
                                               decoder=InnerProductDecoder())
        self.coeffs.update(**kwargs)

    def __loss__(self, data: Data, adj_pred: Tensor) -> Munch:
        # kl_loss project
        kl_loss = self.non_exp_subgraph_generator.kl_loss() / data.num_nodes
        # dist loss project
        num_pos_sample_edges = data.all_edge_label.sum().item()
        weights = torch.ones_like(data.all_edge_label).float()
        weights[data.all_edge_label == 1] = self.coeffs['beta'] / num_pos_sample_edges
        weights[data.all_edge_label == 0] = 1 / (data.all_edge_label.size(0) - num_pos_sample_edges)

        dist_loss = F.binary_cross_entropy(adj_pred,
                                           data.all_edge_label.float(),
                                           weight=weights,
                                           reduction='sum')

        return Munch(kl_loss=kl_loss * self.coeffs['lambda'], dist_loss=dist_loss)

    def train_loop(self, data: Data) -> Munch:
        assert (edge_mask := data.get('pred_edge_mask')) is not None
        set_masks(self.exp_subgraph_generator,
                  edge_mask.sigmoid(),
                  data.edge_index,
                  apply_sigmoid=False)
        set_masks(self.non_exp_subgraph_generator,
                  1 - edge_mask.sigmoid(),
                  data.edge_index,
                  apply_sigmoid=False)

        z_exp     = self.exp_subgraph_generator.encode(data.x, data.edge_index)
        z_non_exp = self.non_exp_subgraph_generator.encode(data.x, data.edge_index)

        edge_mask1 = self.exp_subgraph_generator.decode(z_exp, data.all_edge_index)
        edge_mask2 = self.non_exp_subgraph_generator.decode(z_non_exp, data.all_edge_index)

        gen_edge_mask = self.mix(edge_mask1, edge_mask2)
        clear_masks(self.exp_subgraph_generator)
        clear_masks(self.non_exp_subgraph_generator)

        loss_dict = self.__loss__(data, gen_edge_mask)
        return loss_dict

    def mix(self, edge_mask1: Tensor, edge_mask2: Tensor) -> Tensor:
        return torch.max(edge_mask1, edge_mask2)

    def forward(self, data: Data, edge_mask: Tensor) -> Tensor:
        set_masks(self.exp_subgraph_generator,
                  edge_mask.sigmoid(),
                  data.edge_index,
                  apply_sigmoid=False)

        set_masks(self.non_exp_subgraph_generator,
                  1 - edge_mask.sigmoid(),
                  data.edge_index,
                  apply_sigmoid=False)

        z_exp     = self.exp_subgraph_generator.encode(data.x, data.edge_index)
        z_non_exp = self.non_exp_subgraph_generator.encode(data.x, data.edge_index)

        edge_mask1 = self.exp_subgraph_generator.decode(z_exp, data.all_edge_index)
        edge_mask2 = self.non_exp_subgraph_generator.decode(z_non_exp, data.all_edge_index)

        gen_edge_mask = self.mix(edge_mask1, edge_mask2)
        clear_masks(self.exp_subgraph_generator)
        clear_masks(self.non_exp_subgraph_generator)
        return gen_edge_mask