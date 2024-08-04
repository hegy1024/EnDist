import random
import torch

from torch import Tensor
from torch.types import Device
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.utils import k_hop_subgraph, remove_self_loops, dense_to_sparse, to_dense_adj

from utils.typing_utils import *


def adj2edge_index(adj: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Translate adj matrix to edge_index.
    :param adj: The adj matrix of graph.
    :return: Tensor(adj of graph), Tensor(weight of edge)
    """
    if not isinstance(adj, Tensor): adj = torch.tensor(adj)
    assert adj.shape[0] == adj.shape[1], AssertionError('It is not a graph!')
    edge_index, edge_mask = dense_to_sparse(adj)

    return edge_index, edge_mask


def edge_index2adj(edge_index, mask=None, num_nodes=None) -> Tensor:
    """
    Translate edge_index to adj matrix.
    :param edge_index: edge_index of graph.
    :param mask: the edge weights of graph.
    :param num_nodes: the number of nodes.
    :return: Tensor(sparse adj of graph).
    """
    if num_nodes is None:
        num_nodes = edge_index.max().item() + 1
    if mask is None:
        mask = torch.ones(edge_index.size(1), device=edge_index.device, dtype=torch.float)

    adj = to_dense_adj(edge_index, edge_attr=mask, max_num_nodes=num_nodes)
    return adj.squeeze(dim=0)

def perturbation(data: Data, strategy: str = 'normal', alpha: float = 0.,
                 beta: float = 1., sigma: Optional[float] = None, **kwargs) -> Data:
    """对边的权重进行扰动，助于后续训练"""
    E = data.edge_index.size(1)
    edge_mask = torch.empty(E)
    if strategy == 'uniform':
        edge_mask.uniform_(alpha, beta)
    elif strategy == 'normal':
        edge_mask.normal_(alpha, beta)
    elif strategy == 'mix':
        edge_mask.uniform_(alpha, beta)
        if (rate := kwargs.get('rate')) is None: rate = 0.5
        indices = torch.tensor(random.sample(range(E), int(E * rate)), dtype=torch.long)
        edge_mask[indices] = torch.normal(alpha, beta, indices.size())
    elif strategy == 'none':
        # do nothing
        edge_mask = torch.ones_like(data.edge_index)
    else:
        raise ValueError(f"Do not support the strategy{strategy}")
    if sigma:
        """按照概率取值"""
        prob_adj_matrix = edge_index2adj(data.edge_index, edge_mask.sigmoid(), data.num_nodes)
        bernoulli_adj = torch.where(
            prob_adj_matrix > 1 / 2,
            torch.full_like(prob_adj_matrix, sigma),
            torch.full_like(prob_adj_matrix, 0.),
        )
        adj_upper = torch.bernoulli(bernoulli_adj).triu(1)
        adj_lower = adj_upper.transpose(-1, -2)
        data.edge_index = adj2edge_index(adj_upper + adj_lower)[0]
    else:
        data.edge_mask = edge_mask
    return data

def select_data(dataset: Union[Data, Dataset, InMemoryDataset], index: Union[int, Tensor] = 0,
                num_hops: Optional[int] = None, relabel_nodes: bool = True,
                remove_self_loop: bool = False, device: Optional[Device] = None, remove_noise_node: bool = False) -> Data:
    r"""
    Select data from :class:`torch_geometric.data.Dataset` for given graph idx.

    Args:
        dataset: A given dataset.
        device:  idx of device
        index:   index of target graph
        num_hops: number of hops for node classification graph
        relabel_nodes:    whether to relabel nodes
        remove_self_loop: whether to remove self loops
        remove_noise_node: whether to remove noised node
    """
    if device is None:
        device = torch.device('cpu')
    if num_hops is not None:
        # get k hop subgraph
        if not isinstance(data := dataset, Data): data = dataset.get(0)

        subgraph = k_hop_subgraph(
            node_idx      = index,
            edge_index    = data.edge_index,
            num_hops      = num_hops,
            relabel_nodes = relabel_nodes
        )
        data_ = Data(
            x             = data.x[subgraph[0]] if relabel_nodes else data.x,
            y             = data.y[subgraph[0]] if relabel_nodes else data.y[index],
            ori_node_idx  = subgraph[0],
            edge_index    = subgraph[1],
            corn_node_id  = subgraph[2] if relabel_nodes else index,
            edge_mask     = data.edge_mask[subgraph[-1]] if data.get("edge_mask") is not None else None
        )
    else:
        # get graph within index
        data_ = dataset.get(index)
        data_.ori_node_idx = torch.arange(data_.num_nodes)

    if remove_self_loop:
        edge_index, edge_mask = data_.edge_index, data_.get("edge_mask")
        data_.edge_index, data_.edge_mask = remove_self_loops(edge_index, edge_mask)
    if remove_noise_node:
        un_noise_node = torch.nonzero(data_.x.sum(dim=-1)).view(-1)
        data_.x       = data_.x[un_noise_node]
        edge_indices  = torch.bitwise_and(torch.isin(data_.edge_index[0], un_noise_node),
                                          torch.isin(data_.edge_index[1], un_noise_node))
        edge_index    = data_.edge_index[:, edge_indices]
        edge_mask     = data_.edge_mask[edge_indices]
        data_.edge_index, data_.edge_mask = edge_index, edge_mask

    return data_.to(device)

def clean_data(data: Data) -> Data:
    r"""
    For a given data, deleting its noise nodes and edges, return a new data.
    """
    node_mask = data.x.sum(dim=-1) != 0
    src, tgt  = data.edge_index
    edge_mask = torch.bitwise_and(node_mask[src], node_mask[tgt])
    return Data(
        x = data.x[node_mask],
        y = data.y,
        edge_index = data.edge_index[:, edge_mask],
        edge_mask  = data.edge_mask[edge_mask],
        ground_truth_mask = data.ground_truth_mask[edge_mask])

def to_edge_subgraph(edge_index: Tensor, pred_edges: Tensor, threshold: float = 0.5) -> Tuple[Tensor, Tensor]:
    r"""
    For given edge weight, sample subgraph.
    """
    edge_indices = pred_edges >= threshold
    if edge_indices.sum() == 0: return None
    edge_index   = pyg.utils.to_undirected(edge_index[:, edge_indices])
    node_mapping = {x: i for i, x in enumerate(edge_index.flatten().unique().cpu().tolist())}
    ori_node_map = {x: i for i, x in node_mapping.items()}
    edge_index = torch.tensor([[node_mapping[x] for x in edge_index[0].cpu().tolist()],
                               [node_mapping[x] for x in edge_index[1].cpu().tolist()]], dtype=torch.long)

    return edge_index, ori_node_map

class SubgraphBuilding(object):
    r"""
    Graph building/Perturbation
    `graph_build_zero_filling` and `graph_build_split` are adapted from the DIG library
    """
    def __init__(self, subgraph_building_method: str):
        self.subgraph_building_method = subgraph_building_method

    @staticmethod
    def graph_build_zero_filling(X, edge_index, node_mask: torch.Tensor):
        """subgraph building through masking the unselected nodes with zero features"""
        ret_X = X * node_mask.unsqueeze(1)
        return ret_X, edge_index

    @staticmethod
    def graph_build_split(X, edge_index, node_mask: torch.Tensor):
        """subgraph building through spliting the selected nodes from the original graph"""
        ret_X = X
        row, col = edge_index
        edge_mask = (node_mask[row] == 1) & (node_mask[col] == 1)
        ret_edge_index = edge_index[:, edge_mask]
        return ret_X, ret_edge_index

    @staticmethod
    def graph_build_remove(X, edge_index, node_mask: torch.Tensor):
        """subgraph building through removing the unselected nodes from the original graph"""
        ret_X = X[node_mask == 1]
        ret_edge_index, _ = pyg.utils.subgraph(node_mask.bool(), edge_index, relabel_nodes=True)
        return ret_X, ret_edge_index

    def __call__(self, *args, **kwargs):
        if self.subgraph_building_method == "zero_filling":
            return self.graph_build_zero_filling(*args, **kwargs)
        elif self.subgraph_building_method == "split":
            return self.graph_build_split(*args, **kwargs)
        elif self.subgraph_building_method == "remove":
            return self.graph_build_remove(*args, **kwargs)
        else:
            raise NotImplementedError(f"the method {self.subgraph_building_method} doesn't exist")


class MaskedDataset(Dataset):
    def __init__(self, data: Data, masks: List[Tensor], subgraph_building_method: str):
        super().__init__()

        self.num_nodes = data.num_nodes
        self.x = data.x
        self.edge_index = data.edge_index
        self.device = data.x.device
        self.y = data.y

        if not isinstance(masks, list):
            masks = [masks]
        self.masks = torch.cat(masks, dim=0).float().to(self.device)
        self.subgraph_building_func = SubgraphBuilding(subgraph_building_method)

    def len(self):
        return self.masks.shape[0]

    def get(self, idx):
        masked_x, masked_edge_index = self.subgraph_building_func(
            self.x, self.edge_index, self.masks[idx]
        )
        masked_data = Data(x=masked_x, edge_index=masked_edge_index)

        return masked_data