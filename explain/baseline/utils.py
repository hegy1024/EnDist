from utils.typing_utils import *
from torch_geometric.utils import coalesce

def mixup_graph(data1: Data, data2: Data, yita: int = 5, sorted_edge: bool = True) -> Data:
    link_edge_index = [[], []]
    yita = yita / (data1.num_nodes * data2.num_nodes)
    for i in range(data1.num_nodes):
        for j in range(data1.num_nodes, data2.num_nodes + data1.num_nodes):
            if random.random() < yita:
                link_edge_index[0].append(i)
                link_edge_index[1].append(j)
    # to undirected edge
    link_edge_index = pyg.utils.to_undirected(torch.tensor(link_edge_index))
    x = torch.cat((data1.x, data2.x), dim=0)
    mask1 = torch.tensor([1] * data1.num_edges + [0] * data2.num_edges + [0] * link_edge_index.size(1))
    mask2 = torch.tensor([0] * data1.num_edges + [1] * data2.num_edges + [1] * link_edge_index.size(1))
    edge_index = torch.cat((data1.edge_index, data2.edge_index + data1.num_nodes, link_edge_index), dim=-1).long()
    if sorted_edge:
        edge_index, [mask1, mask2] = coalesce(edge_index, [mask1, mask2], reduce='min')

    return Data(
        x=x,
        edge_index=edge_index,
        x1=data1.x,
        x2=data2.x,
        edge_index1=data1.edge_index,
        edge_index2=data2.edge_index,
        mask1=mask1,
        mask2=mask2,
        edge_mask=data1.edge_mask,
        # batch=torch.tensor([0] * data1.num_nodes + [1] * data2.num_nodes),
        ori_node_idx1=data1.ori_node_idx,
        ori_node_idx2=data2.ori_node_idx,
        corn_node_id =[data1.get('corn_node_id'), data2.get('corn_node_id')]
    )