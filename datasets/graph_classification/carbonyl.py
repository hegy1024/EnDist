import random
import sys
import os.path as osp

import torch
from numpy.random import RandomState

from datasets.base import BaseDataset
from datasets.extract_google_datasets import read_data

sys.path.append(osp.split(sys.path[0])[0])
from utils.typing_utils import *

class AlkaneCarbonyl(BaseDataset):
    r"""
    Process alkane_carbonyl dataset where from https://github.com/mims-harvard/GraphXAI,
    I just rearrange their code and adjust the output in this code. Thanks for their extraordinary work.
    AlkaneCarbonyl:
        A chemistry dataset for graph-level classification. Task is two-class classification, 0 = has neither alkane
        nor carbonyl functional groups, 1 = has alkane and carbonyl. Ground-truth explanations are alkane and
        carbonyl functional groups. If there are multiple combinations of alkane and carbonyl groups, then there will be
        multiple ground-truth explanations, i.e., one for each combination.
    """

    name = 'alkane_carbonyl'

    def __init__(
        self,
        root_path:     str,
        transform:     Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter:    Optional[Callable] = None
    ):
        root = osp.join(root_path, self.name)
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices, self.index, self.sizes = torch.load(self.processed_paths[0])

    @property
    def type(self):
        return 'graph'

    @property
    def num_node_classes(self) -> int:
        return self.sizes['num_node_labels']

    @property
    def num_edge_classes(self) -> int:
        return self.sizes['num_edge_labels']

    @property
    def train_idx(self):
        return self.index['train_idx']

    @property
    def test_idx(self):
        return self.index['test_idx']

    @property
    def val_idx(self):
        return self.index['val_idx']

    def process(self):
        data_list, self.slices, sizes = read_data(self.name, self.raw_dir, down_sample=True)
        max_num_nodes = max(data.num_nodes for data in data_list)
        if self.pre_filter:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform:
            data_list = [self.pre_transform(d, max_num_nodes) for d in data_list]

        # Divide dataset to train_dataset test_dataset and val_dataset.
        num_graphs = len(self)
        indices = np.arange(0, num_graphs)
        prng = RandomState(42)
        indices = prng.permutation(indices)
        data_list_ = [data_list[i] for i in indices]
        idx = torch.arange(0, num_graphs)
        index = {
            'train_idx': idx[: int(0.8 * num_graphs)],
            'test_idx': idx[int(0.8 * num_graphs): int(0.9 * num_graphs)],
            'val_idx': idx[int(0.9 * num_graphs):]
        }
        # collect dataset
        self.data, self.slices = self.collate(data_list_)
        self._data_list = None
        torch.save((self._data, self.slices, index, sizes), self.processed_paths[0])


class FluorideCarbonyl(BaseDataset):
    r"""
    Process fluoride_carbonyl dataset where from https://github.com/mims-harvard/GraphXAI,
    I just rearrange their code and adjust the output in this code. Thanks for their extraordinary work.
    FluorideCarbonyl:
        A chemistry dataset for graph-level classification. Task is two-class classification, 0 = has neither fluoride
        nor carbonyl functional groups, 1 = has both fluoride and carbonyl. Ground-truth explanations are fluoride and
        carbonyl functional groups. If there are multiple combinations of alkane and carbonyl groups, then there will be
        multiple ground-truth explanations, i.e., one for each combination.
    """

    name = 'fluoride_carbonyl'

    def __init__(
        self,
        root_path: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None
    ):
        root = osp.join(root_path, self.name)
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices, self.index, self.sizes = torch.load(self.processed_paths[0])

    @property
    def type(self):
        return 'graph'

    @property
    def num_node_classes(self) -> int:
        return self.sizes['num_node_labels']

    @property
    def num_edge_classes(self) -> int:
        return self.sizes['num_edge_labels']

    @property
    def train_idx(self):
        return self.index['train_idx']

    @property
    def test_idx(self):
        return self.index['test_idx']

    @property
    def val_idx(self):
        return self.index['val_idx']

    def process(self):
        data_list, self.slices, sizes = read_data(self.name, self.raw_dir, down_sample=False)
        max_num_nodes = max(data.num_nodes for data in data_list)
        if self.pre_filter:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform:
            data_list = [self.pre_transform(d, max_num_nodes) for d in data_list]

        # Divide dataset to train_dataset test_dataset and val_dataset.
        num_graphs = len(self)
        indices = np.arange(0, num_graphs)
        prng = RandomState(42)
        indices = prng.permutation(indices)
        data_list_ = [data_list[i] for i in indices]
        idx = torch.arange(0, num_graphs)
        index = {
            'train_idx': idx[: int(0.8 * num_graphs)],
            'test_idx': idx[int(0.8 * num_graphs): int(0.9 * num_graphs)],
            'val_idx': idx[int(0.9 * num_graphs):]
        }
        # collect dataset
        self.data, self.slices = self.collate(data_list_)
        self._data_list = None
        torch.save((self._data, self.slices, index, sizes), self.processed_paths[0])