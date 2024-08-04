import sys
import os.path as osp

from torch_geometric.utils import one_hot
from torch_geometric.utils import coalesce
from numpy.random import RandomState

from datasets.base import BaseDataset

sys.path.append(osp.split(sys.path[0])[0])
from utils.typing_utils import *


class Mutagenicity(BaseDataset):
    """
    A class for processing the Mutagenicity graph classification datasets.

    Mutagenicity:
        Mutagenicity is a real-world molecular dataset commonly used for graph classification explanations.
        Each graph in Mutagenicity represents a molecule, with nodes representing atoms
        and edges representing bonds between atoms. The labels for the graphs
        are based on the chemical functionalities of the corresponding molecules.
    """
    files = [
        'A', 'graph_indicator', 'node_labels',
        'edge_labels', 'graph_labels', 'edge_gt'
    ]
    name = 'MUTAG'

    def __init__(
        self,
        root_path: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None
    ):
        root = osp.join(root_path, 'Mutagenicity')
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

    def read_file(self, name, data_type=torch.long):
        path_ = f'{self.raw_paths[0]}{name}.txt'
        src = torch.tensor(np.loadtxt(path_, delimiter=','), dtype=data_type)
        return src

    def read_data(self):
        """
        Reading dataset from the storage file path.
        """
        # Load the sparse adj.
        edge_index = self.read_file(self.files[0]).T - 1
        # Load batch information, denote which graph the node belong
        batch = self.read_file(self.files[1]) - 1
        # Load node labels.
        node_labels = self.read_file(self.files[2])
        features = one_hot(node_labels, 14).to(torch.float)
        # Load edge labels
        edge_labels = self.read_file(self.files[3])
        edge_attr = one_hot(edge_labels).to(torch.float)
        # Load graph labels
        graph_labels = self.read_file(self.files[4])
        # Load ground truth
        edge_mask = self.read_file(self.files[-1])

        edge_attrs = [edge_attr, edge_mask]
        num_nodes = features.size(0)
        # Sorts and removes duplicated entries from edge indices
        edge_index, edge_attrs = coalesce(edge_index, edge_attrs, num_nodes)

        data = Data(
            x=features,
            edge_index=edge_index,
            edge_mask=edge_attrs[1],
            edge_attr=edge_attrs[0],
            node_labels=node_labels,
            y=graph_labels
        )

        data, slices = self.split(data, batch)
        sizes = {
            'num_node_labels': features.size(-1),
            'num_edge_labels': edge_attr.size(-1),
            'num_classes': 2,
        }

        return data, slices, sizes

    def split(self, data, batch):
        """
        Split the one huge data with the batch.
        """
        node_slice = torch.cumsum(torch.bincount(batch), dim=0)
        node_slice = torch.cat([torch.tensor([0]), node_slice])  # the initial index of node for every graph

        row, _ = data.edge_index
        edge_slice = torch.cumsum(torch.bincount(batch[row]), dim=0)
        edge_slice = torch.cat([torch.tensor([0]), edge_slice])  # the initial index of edge for every graph

        # Edge indices should start at zero for every graph.
        data.edge_index -= node_slice[batch[row]].unsqueeze(dim=0)
        slices = {
            'edge_index': edge_slice,
            'x': node_slice,
            'node_labels': node_slice,
            'edge_attr': edge_slice,
            'edge_mask': edge_slice,
            'y': torch.arange(0, batch[-1] + 2, dtype=torch.long)
        }

        return data, slices

    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        """
        Collect the file name of datasets to be processed.
        """
        return 'Mutagenicity_'

    def process(self):
        """
        Processing the Mutagenicity dataset.
        """

        self.data, self.slices, sizes = self.read_data()

        data_list = [self.get(idx) for idx in range(len(self))]

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

        self.data, self.slices = self.collate(data_list_)
        self._data_list = None

        torch.save((self._data, self.slices, index, sizes), self.processed_paths[0])