import os
import torch
import os.path as osp
import numpy as np
import scipy.sparse as sp
import pickle as pkl

from typing import List, Union
from torch import Tensor
from numpy.random import RandomState
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse

def read_syn_node_dataset(file, prefix):
    """读取四个合成节点分类数据集。"""
    file_path = osp.join(file, prefix)
    assert os.path.exists(file_path), (
        FileNotFoundError(f'File path {file_path} do not exist, please check your path'))

    with open(file_path, 'rb') as f:
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, edge_label_matrix = pkl.load(f)

    #  Translate adj to edge_index
    ed = sp.coo_matrix(adj)
    edge_index = np.vstack((ed.row, ed.col))

    #  Translate ground-truth to edge_index
    em = sp.coo_matrix(edge_label_matrix + edge_label_matrix.T)
    gt_edge_index = np.vstack((em.row, em.col))
    #  Get ground-truth mask
    edge_mask = list(map(lambda x: sum((x == gt_edge_index.T).all(axis=1)), edge_index.T))

    #  get label
    label = y_train + y_test + y_val
    y = label.argmax(axis=1)

    #  split train & val & test data
    idx = torch.arange(0, adj.shape[0])
    train_idx = idx[train_mask]
    test_idx = idx[test_mask]
    val_idx = idx[val_mask]

    data = Data(
        x=torch.tensor(features, dtype=torch.float),
        y=torch.tensor(y, dtype=torch.long),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_mask=torch.tensor(edge_mask, dtype=torch.float),
        num_classes=y_train.shape[1]
    )

    sizes = {
        'num_node_attributes': data.x.size(-1),
        'num_node_labels': y_train.shape[1],
    }

    index = {
        'train_idx': train_idx,
        'test_idx': test_idx,
        'val_idx': val_idx
    }

    return data, index, sizes


def read_ba2_dataset(path: str, shuffle: bool = True):
    assert os.path.exists(path), FileNotFoundError(f"The file with path {path} does not exist!")
    with open(path, 'rb') as fin:
        adjs, features, labels = pkl.load(fin)

    # process datasets
    num_graphs = adjs.shape[0]
    indices = np.arange(0, num_graphs)
    if shuffle:
        # Make sure that the permutation is always the same, even if we set the seed different
        prng = RandomState(42)
        indices = prng.permutation(indices)

    # Create shuffled data
    adjs = adjs[indices]
    features = features[indices].astype('float32')
    labels = labels[indices]  # sequence of [0, 1]

    features = torch.tensor(features)
    y = torch.argmax(torch.tensor(labels), dim=-1)

    # Divide datasets to three sub_dataset.
    idx = torch.arange(0, num_graphs)

    # Transforms adj into edge index
    edge_index_list = list(map(lambda adj: adj2edge_index(adj)[0], adjs))

    # Load ground truth mask
    # Judge the index of node belongs to [20, 25)
    motif_nodes = set(range(20, 25))
    edge_mask_list = []
    for edge_index in edge_index_list:
        edge_mask = list(map(lambda x: int(x[0] in motif_nodes and x[1] in motif_nodes), edge_index.T.tolist()))
        edge_mask_list.append(torch.tensor(edge_mask, dtype=torch.float))
    node_label = torch.cat((torch.zeros(20), torch.ones(5))).to(torch.long)
    # Get datasets
    dataset = list(
        map(
            lambda data: Data(
                x=data[0],
                edge_index=data[1],
                y=data[2],
                edge_mask=data[3],
                node_labels=node_label
            ),
            zip(features, edge_index_list, y, edge_mask_list)
        )
    )

    index = {
        'train_idx': idx[: int(0.8 * num_graphs)],
        'test_idx': idx[int(0.8 * num_graphs): int(0.9 * num_graphs)],
        'val_idx': idx[int(0.9 * num_graphs):]
    }

    sizes = {
        'num_node_features': features.shape[-1],
        'num_classes': 2,
        'num_node_classes': 2
    }

    return dataset, index, sizes

def adj2edge_index(adj) -> [Tensor, Tensor]:
    """
    Translate adj matrix to edge_index.
    :param adj: The adj matrix of graph.
    :return: Tensor, Tensor
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
    :return: sparse adj of graph.
    """
    if num_nodes is None:
        num_nodes = edge_index.max().item() + 1
    if mask is None:
        mask = torch.ones(edge_index.size(1), device=edge_index.device, dtype=torch.float)

    adj = to_dense_adj(edge_index, edge_attr=mask, max_num_nodes=num_nodes)
    return adj.squeeze(dim=0)

def remove_self_loops(data: Data) -> Data:
    """
    Remove the self loops of given data.
    """
    edge_index = data.edge_index
    indices = edge_index[0] != edge_index[1]
    new_edge_index = edge_index[:, indices]
    node_indices = torch.unique(new_edge_index)
    new_data = Data(
        x=data.x[node_indices],
        edge_index=new_edge_index,
        y=data.y
    )
    if 'edge_attr' in data.keys:
        new_data.__setattr__('edge_attr', data.edge_attr[indices])
    if 'edge_mask' in data.keys:
        new_data.__setattr__('edge_mask', data.edge_mask[indices])
    if 'node_labels' in data.keys:
        new_data.__setattr__('node_labels', data.node_labels[node_indices])
    return new_data