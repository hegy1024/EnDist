import time
from sklearn.metrics import roc_auc_score

from utils.typing_utils import *
from utils.data_utils import MaskedDataset
from explain import Explanation

def get_explanation_syn(
    data: Data,
    edge_mask: Tensor,
    max_nodes: Optional[int]  = None,
    max_edges: Optional[int]  = None,
    node_list: Optional[List] = None,
    edge_list: Optional[List] = None
):
    """Create an explanation graph from the edge_mask.
    Args:
        data (PyG data object): the initial graph as Data object
        edge_mask (Tensor): the explanation mask(Hard mask)
        node_list:  node indices
        max_nodes: max number of nodes
        max_edges: max number of edges
        edge_list: edge list of explanation subgraph
    Returns:
        G_masked (networkx graph): explanatory subgraph
    """
    G = nx.Graph()

    if edge_list is not None:
        G.add_edges_from(edge_list)
        return G

    ### remove self loops
    edge_index = data.edge_index.cpu().numpy()
    assert edge_index.shape[-1] == edge_mask.shape[-1]
    self_loop_mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, self_loop_mask]
    edge_mask = edge_mask[self_loop_mask]

    order = (-edge_mask).argsort()

    if node_list is not None:
        G.add_nodes_from(node_list)

    for i in order:
        u, v = edge_index[:, i]
        if max_edges is not None and G.number_of_edges() >= max_edges:
            break

        if max_nodes is not None and G.number_of_nodes() >= max_nodes and \
                (u not in G.nodes or v not in G.nodes):
            continue

        if edge_mask[i] == 0:
            continue

        G.add_edge(u, v)

    return G

class FidelityAlpha(object):
    r"""
        Idea `robust fidelity metrics` came from https://arxiv.org/pdf/2310.01820v1.pdf.
        Code is based on the paper `MEGA: Explaining Graph Neural Networks via Structure-aware Interaction Index`,
        I just rearrange theirs code.
    """
    def __init__(
        self,
        model:       nn.Module,
        alpha:       float = 0.8,
        num_samples: int = 2000,
        subgraph_building_method: str = 'split'
    ):
        self.model = model
        self.alpha = alpha
        self.num_samples = num_samples
        self.subgraph_building_method = subgraph_building_method

    @staticmethod
    def alpha_sampling(G: nx.Graph, alpha: float) -> nx.Graph:
        r"""
        Random sample a graph under parameter alpha.
        """
        H = nx.Graph()

        for u in G.nodes:
            if np.random.rand() <= alpha:
                H.add_node(u)
        for u, v in G.edges:
            if u in H.nodes and v in H.nodes:
                H.add_edge(u, v)

        return H

    @staticmethod
    def graph_minus(G1: nx.Graph, G2: nx.Graph) -> nx.Graph:
        r"""
        Get graph with explanation graph.
        """
        H = nx.Graph()
        for u in G1.nodes:
            if u not in G2.nodes:
                H.add_node(u)

        for u, v in G1.edges:
            if u in H.nodes and v in H.nodes:
                H.add_edge(u, v)
        return H

    @staticmethod
    def graph_plus(G1: nx.Graph, G2: nx.Graph, edge_list) -> nx.Graph:
        r"""
        Get graph without explanation graph.
        """
        H = nx.Graph()
        for u in G1.nodes:
            H.add_node(u)
        for u in G2.nodes:
            H.add_node(u)

        for u, v in edge_list:
            if u in H.nodes and v in H.nodes:
                H.add_edge(u, v)

        return H

    def compute_payoffs(self, data: Explanation, masks: List[Tensor]):
        r"""
        Getting prediction.
        """
        masked_dataset = MaskedDataset(data, masks, self.subgraph_building_method)

        masked_dataset = DataLoader(masked_dataset, batch_size=128, shuffle=False)
        masked_payoff_list = []
        for masked_data in masked_dataset:
            masked_data.to(data.x.device)
            logits = self.model(masked_data.x, masked_data.edge_index, batch=masked_data.batch)
            preds  = logits.softmax(-1)[:, data.y]
            masked_payoff_list.append(preds)

        return torch.cat(masked_payoff_list).detach().cpu().numpy()

    def fidelity_alpha_plus(self, data: Explanation, expl_graph: nx.Graph, ori_prob: Optional[Tensor] = None) -> float:
        graph    = pyg.utils.to_networkx(data, to_undirected=True)
        if ori_prob is None:
            ori_prob = self.model(data.x, data.edge_index).softmax(dim=-1)[:, data.y].cpu().item()
        mask_lst = []

        for _ in range(self.num_samples):
            expl_graph_plus = self.alpha_sampling(expl_graph, self.alpha)
            non_expl_graph  = self.graph_minus(graph, expl_graph_plus)
            node_mask       = torch.zeros(data.num_nodes)
            node_mask[torch.LongTensor(list(non_expl_graph.nodes))] = 1
            mask_lst.append(node_mask)

        excluded_masks = torch.vstack(mask_lst)
        excluded_probs = self.compute_payoffs(data, excluded_masks)
        scores = ori_prob - excluded_probs

        if len(scores) != 0:
            return np.mean(scores)
        else:
            return 0

    def fidelity_alpha_minus(self, data: Explanation, expl_graph: nx.Graph, ori_prob: Optional[Tensor] = None) -> float:
        graph = pyg.utils.to_networkx(data, to_undirected=True)
        if ori_prob is None:
            ori_prob = self.model(data.x, data.edge_index).softmax(dim=-1)[:, data.y].cpu().item()
        mask_lst = []

        for _ in range(self.num_samples):
            non_expl_graph       = self.graph_minus(graph, expl_graph)
            non_expl_graph_minus = self.alpha_sampling(non_expl_graph, 1 - self.alpha)
            expl_graph_minus     = self.graph_plus(non_expl_graph_minus, expl_graph, graph.edges)
            node_mask            = torch.zeros(data.num_nodes)
            node_mask[torch.LongTensor(list(expl_graph_minus.nodes))] = 1
            mask_lst.append(node_mask)

        included_masks = torch.vstack(mask_lst)
        included_probs = self.compute_payoffs(data, included_masks)
        scores = ori_prob - included_probs

        if len(scores) != 0:
            return np.mean(scores)
        else:
            return 0

    def __call__(self, data: Explanation, **kwargs) -> Tuple[float, float, float]:
        r"""

        Args:
            data: pyg Data
            **kwargs:
                node_list: List[int]
                max_nodes: int
                max_edges: int
                ori_pred: float
                edge_list: List[Tuple[int, int]]
        """
        assert (kwargs.get('max_nodes') is not None or
                kwargs.get('edge_list') is not None or
                kwargs.get('ground_truth_mask') is not None)

        expl_graph = get_explanation_syn(data, data.edge_mask, **kwargs)
        fid_plus   = self.fidelity_alpha_plus( data, expl_graph, kwargs.get('ori_prob'))
        fid_minus  = self.fidelity_alpha_minus(data, expl_graph, kwargs.get('ori_prob'))
        fid_delta  = fid_plus - fid_minus

        return fid_delta, fid_plus, fid_minus


class ExplanationEvaluator(object):
    r"""
    Collect explanations information, and then compute the metrics.
    """
    def __init__(
        self,
        model:       nn.Module,
        alpha:       float = 0.8,
        num_samples: int = 2000,
        subgraph_building_method: str = 'split'
    ):
        self.alpha_func = FidelityAlpha(model, alpha, num_samples, subgraph_building_method)
        self.ori_func   = FidelityAlpha(model, 1, 1, subgraph_building_method)
        # time consume
        self.st_time = dict()
        self.ed_time = dict()
        # for roc auc
        self.target_masks : List[Tensor] = []
        self.predict_masks: List[Tensor] = []
        self.roc_auc_score: Optional[float] = None
        # for others
        self.related_preds = defaultdict(list)
        self.results_dict  = {}

    def fidelity(self, data: Explanation, mode: str = 'alpha', **kwargs):
        if mode == 'alpha':
            results = self.alpha_func(data, **kwargs)
            self.related_preds['fid_delta'].append(results[0])
            self.related_preds['fid_plus'].append(results[1])
            self.related_preds['fid_minus'].append(results[2])
        elif mode == 'ori':
            results = self.ori_func(data, **kwargs)
            self.related_preds['ori_fid_delta'].append(results[0])
            self.related_preds['ori_fid'].append(results[1])
            self.related_preds['ori_fid_inv'].append(results[2])
        else:
            raise NotImplementedError

    def get_average(self, metric: str) -> Tuple[float, float]:
        if metric not in self.related_preds.keys():
            raise ValueError(f"there is no metric named: {metric}")

        score = np.array(self.related_preds[metric])
        return score.mean().item(), score.std().item()

    def get_summarized_results(self):
        for metric in self.related_preds.keys():
            self.results_dict[metric] = self.get_average(metric)

    def collect(self, data: Explanation, metric: str = 'acc', **kwargs):

        if metric == 'fid' and kwargs.get("compute_fid", True):
            self.fidelity(data, mode='alpha', **kwargs)
            self.fidelity(data, mode='ori', **kwargs)

        if self.roc_auc_score is None:
            # remove self loops
            self_loops = data.edge_index[0] != data.edge_index[1]
            if ((edge_mask := data.get('edge_mask')) is not None and
                (gt_mask := data.get('ground_truth_mask')) is not None and
                (gt_mask == 1).sum() != 0):
                self.predict_masks.append(edge_mask[self_loops])
                self.target_masks.append(gt_mask[self_loops])

    def eval_start(self, mode: str = 'train'):
        self.st_time[mode] = time.time()

    def eval_finish(self, mode: str = 'train'):
        self.ed_time[mode] = time.time()

    def time_consume(self, mode: str):
        assert self.st_time.get(mode) is not None and self.ed_time.get(mode) is not None
        return self.ed_time[mode] - self.st_time[mode]

    def accuracy_one_element(self, idx: int) -> float:
        try:
            return roc_auc_score(self.target_masks[idx].cpu(), self.predict_masks[idx].cpu())
        except:
            return 0.

    @property
    def fidelity_plus(self):
        return self.results_dict['fid_plus']

    @property
    def fidelity_minus(self):
        return self.results_dict['fid_minus']

    @property
    def fidelity_delta(self):
        return self.results_dict['fid_delta']

    @property
    def fidelity_ori(self):
        return self.results_dict['ori_fid']

    @property
    def fidelity_ori_inv(self):
        return self.results_dict['ori_fid_inv']

    @property
    def fidelity_ori_delta(self):
        return self.results_dict['ori_fid_delta']

    @property
    def accuracy(self) -> float:
        r"""
        Accuracy, reflecting the alignment between explanation results and human understanding.
        """
        if self.roc_auc_score: return self.roc_auc_score

        tgt = torch.cat(self.target_masks, dim=-1)
        src = torch.cat(self.predict_masks, dim=-1)
        assert tgt.size(0) == src.size(0)

        self.roc_auc_score = roc_auc_score(tgt.cpu(), src.cpu())
        return self.roc_auc_score