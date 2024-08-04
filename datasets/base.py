from typing import Union, List, Tuple
from torch_geometric.data import InMemoryDataset

class BaseDataset(InMemoryDataset):
    """
    A base class to process datasets which we need in GNN explaining.
    """
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        """
        Which file the processed datasets should be saved.
        """
        return ['data.pt']

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'

