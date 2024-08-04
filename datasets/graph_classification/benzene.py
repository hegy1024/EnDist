import sys
import os.path as osp

import torch
from numpy.random import RandomState

from torch_geometric.utils import coalesce

from datasets.base import BaseDataset
from datasets.extract_google_datasets import read_data

sys.path.append(osp.split(sys.path[0])[0])
from utils.typing_utils import *

class Benzene(BaseDataset):
    r"""
    Process benzene dataset where from https://github.com/mims-harvard/GraphXAI, thanks for their extraordinary work.
    Benzene:
        Chemistry dataset for graph-level classification. Task is two-class classification,
        0 = does not have a benzene ring, 1 = has a benzene ring. Ground-truth explanations
        are benzene rings. If there are multiple rings in one sample, there will be multiple
        ground-truth explanations: one for each benzene ring present.
    """

    ATOM_TYPES = [
        'C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'Na', 'Ca', 'I', 'B', 'H', '*'
    ]
    seed = 10
    name = 'benzene'

    def __init__(
        self,
        root_path:     str,
        transform:     Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter:    Optional[Callable] = None
    ):
        root = osp.join(root_path, "benzene")
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

if __name__ == '__main__':
    path_list = os.getcwd().split('/')
    path = osp.join('/'.join(path_list[:-2]), 'data')
    dataset = Benzene(path)
    print(dataset)