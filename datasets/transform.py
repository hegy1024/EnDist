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

def padding_nodes(data: Data, max_num_nodes: int) -> Data:
    r"""
    Padding data to max num nodes if the num of nodes less than it.
    """
    cur_num_nodes = data.num_nodes
    feats_      = torch.zeros((max_num_nodes - cur_num_nodes, data.num_node_features), dtype=torch.float)
    edge_index_ = torch.arange(cur_num_nodes, max_num_nodes).repeat(2, 1)
    edge_attr_  = torch.zeros((edge_index_.size(1), data.num_edge_features), dtype=torch.float)
    edge_mask_  = torch.full((edge_index_.size(1), ), 2.)
    node_mask_  = torch.full((feats_.size(0), ), 2.)
    new_data = Data(
        x=torch.cat([data.x, feats_], dim=0),
        edge_index=torch.cat([data.edge_index, edge_index_], dim=-1),
        edge_attr=torch.cat([data.edge_attr, edge_attr_], dim=0),
        edge_mask=torch.cat([data.edge_mask, edge_mask_], dim=-1),
        node_mask=torch.cat([data.node_mask, node_mask_], dim=-1),
        y=data.y
    )

    return new_data


def pre_transform_mutag(data, N):
    """
    对于图分类数据集，若图的节点个数低于N，则将其节点个数扩充到N。
    Args:
        data: 待处理的数据
        N: 需要扩充的最多节点数
    """
    device = data.x.device
    data.to('cpu')
    n = data.num_nodes
    m = data.num_node_features
    edge_index_ = torch.arange(n, N).repeat(2, 1)
    feats_ = torch.zeros((N - n, data.num_node_features), dtype=torch.float)
    # edge_attr_ = torch.zeros((edge_index_.size(1), data.num_edge_features), dtype=torch.float)
    # edge_mask_ = torch.full((edge_index_.size(1), ), 2.)
    # node_labels_ = torch.full((N - n, ), m)

    new_data = Data(
        x=torch.cat([data.x, feats_], dim=0),
        edge_index=torch.cat([data.edge_index, edge_index_], dim=-1),
        # edge_attr=torch.cat([data.edge_attr, edge_attr_], dim=0),
        # edge_mask=torch.cat([data.edge_mask, edge_mask_], dim=-1),
        # node_labels=torch.cat([data.node_labels, node_labels_], dim=-1),
        y=data.y
    ).to(device)

    return new_data

def pre_process_mutag(data):
    """
    获取Mutag数据集的ground truth。
    :param data: Graph data in MUTAG 188.
    """
    gt_mask = torch.zeros(data.num_edges)
    if int(data.y):
        edges = data.edge_index.T
        indices = [i for i, (st, ed) in enumerate(edges)
                   if torch.argmax(data.x[st]) in (1, 2) and torch.argmax(data.x[ed]) in (1, 2)]
        indices = torch.tensor(indices)
        gt_mask[indices] = 1.0

    data.edge_mask = gt_mask
    return data

def normalize(data):
    """
    标准化数据的特征。
    """
    data.x /= data.x.sum(dim=-1).unsqueeze(dim=-1)
    return data