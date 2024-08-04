import sys
import os.path as osp

from torch_geometric.utils import coalesce

sys.path.append(osp.split(sys.path[0])[0])
from utils.typing_utils import *

def edge_mask_from_node_mask(node_mask: torch.Tensor, edge_index: torch.Tensor):
    r"""
    Convert edge_mask to node_mask

    Args:
        node_mask (torch.Tensor): Boolean mask over all nodes included in edge_index. Indices must
            match to those in edge index. This is straightforward for graph-level prediction, but
            converting over subgraph must be done carefully to match indices in both edge_index and
            the node_mask.
        edge_index (torch.Tensor): adj of graph.
    """

    node_numbers = node_mask.nonzero(as_tuple=True)[0]

    iter_mask = torch.zeros((edge_index.shape[1],))

    # See if edges have both ends in the node mask
    for i in range(edge_index.shape[1]):
        iter_mask[i] = (edge_index[0, i] in node_numbers) and (edge_index[1, i] in node_numbers)

    return iter_mask.long()

def read_file(data_name: str, dir_path: str):
    r"""
    Code from https://github.com/mims-harvard/GraphXAI, I just rearrange this.
    Returns:
        all_graphs (list of `torch_geometric.data.Data`): List of all graphs in the
            dataset
        explanations (list of `Explanation`): List of all ground-truth explanations for
            each corresponding graph. Ground-truth explanations consist of multiple
            possible explanations as some of the molecular prediction tasks consist
            of multiple possible pathways to predicting a given label.
        zinc_ids (list of ints): Integers that map each molecule back to its original
            ID in the ZINC dataset.
    """
    data_path = osp.join(dir_path, f'{data_name}.npz')
    data = np.load(data_path, allow_pickle=True)

    att, X, y, df = data['attr'], data['X'], data['y'], data['smiles']
    y_list = [y[i][0] for i in range(y.shape[0])]

    X = X[0]

    # Unique zinc identifiers:
    zinc_ids = df[:, 1]

    all_graphs = []
    explanations = []

    for i in range(len(X)):
        x = torch.from_numpy(X[i]['nodes'])
        edge_attr = torch.from_numpy(X[i]['edges'])
        y = torch.tensor([y_list[i]], dtype=torch.long)
        # Get edge_index:
        e1 = torch.from_numpy(X[i]['receivers']).long()
        e2 = torch.from_numpy(X[i]['senders']).long()

        edge_index = torch.stack([e1, e2])

        data_i = Data(
            x=x,
            y=y,
            edge_attr=edge_attr,
            edge_index=edge_index
        )

        all_graphs.append(data_i)  # Add to larger list

        # Get ground-truth explanation:
        node_imp = torch.from_numpy(att[i][0]['nodes']).float()

        # Error-check:
        assert att[i][0]['n_edge'] == X[i]['n_edge'], 'Num: {}, Edges different sizes'.format(i)
        assert node_imp.shape[0] == x.shape[0], 'Num: {}, Shapes: {} vs. {}'.format(i, node_imp.shape[0],
                                                                                    x.shape[0]) \
                                                + '\nExp: {} \nReal:{}'.format(att[i][0], X[i])

        i_exps = []
        node_imp_ = torch.zeros(data_i.num_nodes).long()
        edge_imp_ = torch.zeros(data_i.num_edges).long()
        for j in range(node_imp.shape[1]):
            # if there have different ground truth, put all of them
            node_imp_ = torch.bitwise_or(node_imp_, node_imp[:, j].long())
            edge_imp_ = torch.bitwise_or(edge_imp_,
                                         edge_mask_from_node_mask(node_imp[:, j].bool(), edge_index=edge_index))
        # i_exps.append(node_imp_, edge_imp_])
        explanations.append([node_imp_, edge_imp_])

    return all_graphs, explanations, zinc_ids

def read_data(name: str, dir_path: str, down_sample: bool):
    data_list, exp_list, zinc_idx = read_file(name, dir_path)
    if down_sample:
        # down_samples because of extreme imbalance
        zero_bin = [i for i, data in enumerate(data_list) if data.y == 0]
        one_bin  = [i for i, data in enumerate(data_list) if data.y == 1]
        random.seed(2024)
        keep_idxs = random.sample(zero_bin, k = 2 * len(one_bin)) + one_bin
        data_list = [data_list[i] for i in keep_idxs]
        exp_list  = [exp_list[i] for i in keep_idxs]

    node_slice, edge_slice = [0], [0]
    for data, [node_imp, edge_imp] in zip(data_list, exp_list):
        edge_attrs = [data.edge_attr, edge_imp]
        edge_index, edge_attrs = coalesce(data.edge_index, edge_attrs, data.num_nodes)
        data.edge_index = edge_index
        data.edge_attr = edge_attrs[0]
        data.edge_mask = edge_attrs[1]
        data.node_mask = node_imp
        node_slice.append(data.num_nodes)
        edge_slice.append(data.num_edges)
    node_slice = torch.cumsum(torch.tensor(node_slice, dtype=torch.long), dim=0)
    edge_slice = torch.cumsum(torch.tensor(edge_slice, dtype=torch.long), dim=0)
    slices = {
        'edge_index': edge_slice,
        'x': node_slice,
        'edge_attr': edge_slice,
        'edge_mask': edge_slice,
        'node_mask': node_slice,
        'y': torch.arange(0, len(data_list) + 1, dtype=torch.long)
    }
    sizes = {
        'num_node_labels': data_list[0].x.size(-1),
        'num_edge_labels': data_list[0].edge_attr.size(-1),
        'num_classes': 2
    }
    return data_list, slices, sizes