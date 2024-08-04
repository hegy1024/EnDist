from torch.nn import ReLU
from torch_geometric.nn import Linear
from torch_geometric.utils import k_hop_subgraph

from explain.algorithm.base import InstanceExplainAlgorithm
from utils.typing_utils import *
from explain.utils import set_masks, clear_masks


class PGExplainer(InstanceExplainAlgorithm):
    r"""
    An implementation of PGExplainer in
    `Parameterized Explainer for Graph Neural Network <https://arxiv.org/abs/2011.04573>'.
    """
    coeffs = {
        'reg_size': 0.005,
        'reg_ent': 1.0,
        'temp0': 5.0,
        'temp1': 2.0,
        'EPS': 1e-15,
        'edge_reduction': 'sum',
        'bias': 0.00001,
        'knn_loss': 1.
    }

    def __init__(self, device, epochs: int = 100, gnn_task: str = 'node', **kwargs):
        super().__init__()
        self.device   = device
        self.gnn_task = gnn_task
        self.coeffs.update(kwargs)

        self.explainer = nn.Sequential(
            Linear(-1, 64),
            ReLU(),
            Linear(64, 1)
        )

        self.temp_schedule = lambda e: (self.coeffs['temp0']
                                        * pow((self.coeffs['temp1'] / self.coeffs['temp0']), e / epochs))

    def __loss__(
        self,
        masked_preds:    List[Tensor],
        original_labels: List[Tensor],
        masks:           List[Tensor],
        edge_batch:      Tensor,
        apply_sigmoid:   bool
    ) -> Munch:
        """
        Returns the loss score based on the given mask.

        Notes: for MixUpExplainer, there are more than one element for masked_preds, original_preds and masks

        Args:
            masked_preds:    Prediction based on the current explanation
            original_labels: Prediction label based on the original graph
            masks:      Current explanation
            edge_batch: which graph every edge belong to
            apply_sigmoid: whether to use sigmoid
        Returns: loss dict
        """
        loss_dict = Munch()
        for masked_pred, original_label, mask in zip(masked_preds, original_labels, masks):
            mask_ = mask.sigmoid() if apply_sigmoid else mask
            # Cross Entropy loss
            ce_loss = F.cross_entropy(masked_pred, original_label, reduction='sum')

            # Regulation loss1: size loss
            # for batch graph
            size_loss = (scatter(mask_, edge_batch, dim=-1, reduce=self.coeffs['edge_reduction']).sum()
                         * self.coeffs['reg_size'])
            # Regulation loss2: ent loss
            mask_ = mask_ * 0.99 + 0.005
            ent  = - mask_ * torch.log(mask_) - (1 - mask_) * torch.log(1 - mask_)
            # for batch graph
            ent_loss = scatter(ent, edge_batch, dim=-1, reduce='mean').sum() * self.coeffs['reg_ent']

            loss_dict_ = Munch(
                cross_entropy = ce_loss,
                size_loss     = size_loss,
                ent_loss      = ent_loss
            )
            # Update loss dict
            for u, v in loss_dict_.items():
                loss_dict.__setitem__(u, loss_dict.get(u, 0) + v)

        if self._explain_backward_hook:
            for hook in self._explain_backward_hook.values():
                hook(loss_dict, masks, masked_preds)

        assert len(loss_dict.values()) is not None
        return loss_dict

    def _create_inputs(
        self,
        embeddings: Tensor,
        edge_index: Tensor,
        index:      Optional[Tensor] = None,
        edge_batch: Optional[Tensor] = None,
        hard_mask:  Optional[Tensor] = None
    ):
        """
        Creating the input of MLP in pgexplainer.
        Corresponding to the creating input in paper of pgexplainer.
        """
        src, trg = edge_index
        src_embeds, trg_embeds = embeddings[src], embeddings[trg]

        if hard_mask is not None:
            src_embeds = src_embeds * hard_mask[..., None]
            trg_embeds = trg_embeds * hard_mask[..., None]

        if self.gnn_task == 'node':
            num_edges_per_graph = edge_batch.bincount()
            node_embed = embeddings[index].repeat_interleave(num_edges_per_graph, dim=0)
            inputs = torch.cat([src_embeds, trg_embeds, node_embed], dim=-1)
        else:
            inputs = torch.cat([src_embeds, trg_embeds], dim=-1)

        return inputs

    def _concrete_sample(self, logits, temperature=1.0, training=True):
        """
        Implementation of the reparameterize trick to
        obtain a sample graph while maintaining the possibility to backprop.
        :param logits: Weights provided by the graph_generator
        :param temperature: annealing temperature to make the procedure more deterministic
        :param training: If set to false, the sampling will be entirely deterministic
        :return: sample graph
        """
        if training:
            bias = self.coeffs['bias']
            eps = (1 - 2 * bias) * torch.rand_like(logits) + bias
            gate_inputs = torch.log(eps + self.coeffs['EPS']) - torch.log(1 - eps + self.coeffs['EPS'])
            gate_inputs = (gate_inputs.to(self.device) + logits) / temperature
            edge_mask = gate_inputs
        else:
            edge_mask = logits

        return edge_mask

    @property
    def pretrain(self):
        return True

    def load_parameters(self, path: str, dataset: str, ood: bool = False):
        """
        Load the pre-trained parameters of explainer.
        :param path: save path of parameters.
        :param dataset: dataset to train explainer
        :param ood: whether to use ood hook
        """
        file_path = osp.join(path, f"pge+{'ood' if ood else 'src'}/{dataset}.pkl")
        assert osp.exists(file_path)
        self.load_state_dict(torch.load(file_path))

    def train_loop(
        self,
        data:             Data,
        model_to_explain: nn.Module,
        epoch:            int,
        use_edge_weight:  bool = False,
        apply_sigmoid:    bool = True,
        **kwargs
    ) -> Munch:
        r"""
        Train pgexplainer for given graph.
        Args:
            data:             a PYG Data which storage information of graph to be explained
            model_to_explain: GNN model to be explained
            epoch:            current epoch
            use_edge_weight:   whether to use edge weight when get prediction
            apply_sigmoid:     whether to use sigmoid when compute loss
            **kwargs:

        Returns:
            A dict/munch which storage loss information
        """
        x, edge_index = data.x, data.edge_index
        if (embeds := data.get('ori_embeddings')) is None:
            embeds = [model_to_explain.embedding(x, edge_index)]

        if (batch := data.get('batch')) is None:
            batch = torch.zeros(data.num_nodes).long().to(self.device)
        edge_batch = batch[edge_index[0]]

        temperature = self.temp_schedule(epoch)

        def get_explainer_pred(embed: Tensor, hard_mask: Tensor) -> Tensor:
            # Sample possible explanation
            expl_inputs = self._create_inputs(
                        embed,
                        edge_index,
                        data.get('corn_node_id'),
                        edge_batch,
                        hard_mask).unsqueeze(dim=0)
            logits      = self.explainer(expl_inputs)[0]
            mask        = self._concrete_sample(logits, temperature).squeeze()
            return mask

        def get_gnn_pred(mask: Tensor) -> Tensor:
            # Get gnn prediction
            if not use_edge_weight:
                set_masks(model_to_explain, mask, edge_index)
                pred = model_to_explain(x, edge_index, batch=batch)
                clear_masks(model_to_explain)
            else:
                pred = model_to_explain(x, edge_index, edge_weights=mask.sigmoid(), batch=batch)
            return pred
        if not isinstance(embeds, List): embeds = [embeds]

        hard_masks = data.get('hard_masks', [None] * len(embeds))
        masks = [get_explainer_pred(embed, hard_mask) for embed, hard_mask in zip(embeds, hard_masks)]
        masked_pred: List[Tensor] = []
        if self._explain_forward_hook:
            for hook in self._explain_forward_hook.values():
                masks, masked_pred = hook(masks)
        else:
            masked_pred = [get_gnn_pred(mask) for mask in masks]

        if self.gnn_task == 'node':
            corn_node_id = data.get('corn_node_id')
            assert len(masked_pred) == len(corn_node_id)
            masked_pred = [pred[node_id] for pred, node_id in zip(masked_pred, corn_node_id)]

        if not isinstance(target_label := data.get('target_label'), List): target_label = [target_label]
        loss_dict = self.__loss__(masked_pred, target_label, masks, edge_batch, apply_sigmoid)

        return loss_dict

    def forward(
        self,
        data: Data,
        **kwargs
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        r"""
        Generating explanation for given graph.
        Args:
            data: a PYG Data which storage information of graph to be explained.
        Returns:
            Union[Tensor, Tensor]: edge_mask, feat_mask
        """
        assert (embeds := data.get('ori_embeddings')) is not None

        edge_index = data.edge_index
        if (corn_node_id := data.get('corn_node_id')) is not None and kwargs.get("sample_subgraph"):
            edge_index = k_hop_subgraph(
                corn_node_id,
                kwargs.get('num_hops', 3),
                data.edge_index,
                relabel_nodes=False
            )[1]

        if (batch := data.get('batch')) is None:
            batch = torch.zeros(data.num_nodes).long().to(self.device)
        edge_batch = batch[edge_index[0]]

        def get_explainer_pred(embed: Tensor, hard_mask: Tensor) -> Tensor:
            # Sample possible explanation
            expl_inputs = self._create_inputs(
                        embed,
                        edge_index,
                        corn_node_id,
                        edge_batch,
                        hard_mask).unsqueeze(dim=0)
            logits      = self.explainer(expl_inputs)[0]
            mask        = self._concrete_sample(logits, training=False).squeeze()
            return mask

        if not isinstance(embeds, List): embeds = [embeds]
        hard_masks = data.get('hard_masks', [None] * len(embeds))
        masks      = [get_explainer_pred(embed, hard_mask)
                      for embed, hard_mask in zip(embeds, hard_masks)]

        return masks, None

        # expl_inputs = self._create_inputs(
        #     embeds,
        #     edge_index,
        #     corn_node_id,
        #     edge_batch
        # ).unsqueeze(dim=0)
        #
        # logits = self.explainer(expl_inputs)
        # mask = self._concrete_sample(logits, training=False).squeeze()
        #
        # return mask, None

    @property
    def name(self):
        return 'pgexplainer'