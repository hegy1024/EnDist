import torch
from torch.nn import ReLU
from torch_geometric.nn import Linear

from utils.typing_utils import *

from .base import InstanceExplainAlgorithm
from explain.utils import set_masks, clear_masks


class KFactExplainer(InstanceExplainAlgorithm):
    r"""
    This code is a sample version of KFactExplainer: stack k-local explainer model like PGExplainer/GNNExplainer
    with a global explainer(a MLP layer). Different from mentioned as source paper, I use super_parameter k to
    control the number of local explainer.
    """

    coeffs = {
        'reg_size': 0.005,
        'reg_ent': 1.0,
        'temp0': 5.0,
        'temp1': 2.0,
        'EPS': 1e-15,
        'edge_reduction': 'sum',
        'bias': 0.,
        'knn_loss': 1.
    }

    def __init__(
        self,
        device,
        epochs:   int = 100,
        gnn_task: str = 'node',
        k:        int = 20,
        **kwargs
    ):
        super().__init__()
        self.device   = device
        self.gnn_task = gnn_task
        self.coeffs.update(kwargs)

        # global explainer
        self.global_explainer = nn.Sequential(
            Linear(-1, 64),
            ReLU(),
            Linear(64, k)
        )

        # local explainers
        self.local_explainers = nn.ModuleList()
        for _ in range(k):
            self.local_explainers.append(
                nn.Sequential(
                    Linear(-1, 64),
                    ReLU(),
                    Linear(64, 1)
                )
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
    ):
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
            # edge_reduction = getattr(torch, self.coeffs['edge_reduction'])
            # size_loss = edge_reduction(mask) * self.coeffs['reg_size']
            # for batch graph
            size_loss = (scatter(mask_, edge_batch, dim=-1, reduce=self.coeffs['edge_reduction']).sum()
                         * self.coeffs['reg_size'])
            # Regulation loss2: ent loss
            mask_ = mask_ * 0.99 + 0.005
            ent  = - mask_ * torch.log(mask_) - (1 - mask_) * torch.log(1 - mask_)
            # ent_loss = ent.mean() * self.coeffs['reg_ent']
            # for batch graph
            ent_loss = scatter(ent, edge_batch, dim=-1, reduce='mean').sum() * self.coeffs['reg_ent']

            loss_dict_ = Munch(
                cross_entropy=ce_loss,
                size_loss=size_loss,
                ent_loss=ent_loss
            )
            # Update loss dict
            for u, v in loss_dict_.items():
                loss_dict.__setitem__(u, loss_dict.get(u, 0) + v)

        if self._explain_backward_hook:
            for hook in self._explain_backward_hook.values():
                hook(loss_dict, masks, masked_preds)

        return loss_dict

    def _create_inputs(self, embeddings, edge_index, index, edge_batch: Tensor):
        """
        Creating the input of MLP in pgexplainer.
        Corresponding to the creating input in paper of pgexplainer.
        """
        src, trg = edge_index
        src_embeds, trg_embeds = embeddings[src], embeddings[trg]
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

        # return mask without normalized
        return edge_mask

    def train_loop(
        self,
        data:             Data,
        model_to_explain: nn.Module,
        epoch:            int,
        use_edge_weight: bool = False,
        apply_sigmoid: bool = True,
        **kwargs
    ) -> Munch:
        x, edge_index = data.x, data.edge_index
        if (embeds := data.get('ori_embeddings')) is None:
            embeds = [model_to_explain.embedding(x, edge_index)]

        if (batch := data.get('batch')) is None:
            batch = torch.zeros(data.num_nodes).long().to(self.device)
        edge_batch = batch[edge_index[0]]
        temperature = self.temp_schedule(epoch)

        def get_explainer_pred(embed: Tensor) -> Tensor:
            # Sample possible explanation
            expl_inputs = self._create_inputs(
                        embed,
                        edge_index,
                        data.get('corn_node_id'),
                        edge_batch).unsqueeze(dim=0)
            local_logits = torch.stack(
                [local_explainer(expl_inputs)[0].view(-1) for local_explainer in self.local_explainers],
                dim=0
            )
            graph_embeds = torch.cat(
                [pyg.nn.global_mean_pool(embed, batch),
                 pyg.nn.global_max_pool(embed, batch),
                 pyg.nn.global_add_pool(embed, batch)],
                dim=-1
            )
            global_logits = (self.global_explainer(graph_embeds)
                             .sigmoid()
                             .T
                             .repeat_interleave(edge_batch.bincount(), dim=-1))
            logits       = (global_logits * local_logits).sum(dim=0)
            mask         = self._concrete_sample(logits, temperature).squeeze()
            return mask

        def get_gnn_pred(mask: Tensor) -> Tensor:
            if not use_edge_weight:
                set_masks(model_to_explain, mask, edge_index)
                pred = model_to_explain(x, edge_index, batch=batch)
                clear_masks(model_to_explain)
            else:
                pred = model_to_explain(x, edge_index, edge_weights=mask.sigmoid(), batch=batch)
            return pred

        if not isinstance(embeds, List): embeds = [embeds]
        masks = [get_explainer_pred(embed) for embed in embeds]
        masked_pred: List[Tensor] = []
        if self._explain_forward_hook:
            for hook in self._explain_forward_hook.values():
                masks, masked_pred = hook(masks)
        else:
            masked_pred = [get_gnn_pred(mask) for mask in masks]
        # if self._explain_forward_hook:
        #     for hook in self._explain_forward_hook.values():
        #         masked_pred = hook(logits)
        # elif not use_edge_weight:
        #     set_masks(model_to_explain, mask, edge_index)
        #     masked_pred = model_to_explain(x, edge_index, batch=batch)
        #     clear_masks(model_to_explain)
        # else:
        #     masked_pred = model_to_explain(x, edge_index, edge_weights=mask.sigmoid(), batch=batch)

        if self.gnn_task == 'node':
            if not isinstance(corn_node_id := data.get('corn_node_id'), List):
                corn_node_id = [corn_node_id]
            assert len(masked_pred) == len(corn_node_id)
            masked_pred = [pred[node_id] for pred, node_id in zip(masked_pred, corn_node_id)]

        if not isinstance(target_label := data.get('target_label'), List):
            target_label = [target_label]
        loss_dict = self.__loss__(masked_pred, target_label, masks, edge_batch, apply_sigmoid)

        return loss_dict

    def forward(
        self,
        data: Data,
        **kwargs
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        r"""
        Prediction in testing.
        """
        assert (embeds := data.get('ori_embeddings')) is not None

        if not isinstance(embeds, list):
            embeds = [embeds]

        edge_index = data.edge_index

        if (corn_node_id := data.get('corn_node_id')) is not None and kwargs.get("sample_subgraph"):
            edge_index = pyg.utils.k_hop_subgraph(
                corn_node_id,
                kwargs.get('num_hops', 3),
                data.edge_index,
                relabel_nodes=False
            )[1]

        if (batch := data.get('batch')) is None:
            batch = torch.zeros(data.num_nodes).long().to(self.device)
        edge_batch = batch[edge_index[0]]

        def get_explainer_pred(embed: Tensor) -> Tensor:
            # Sample possible explanation
            expl_inputs = self._create_inputs(
                        embed,
                        edge_index,
                        data.get('corn_node_id'),
                        edge_batch).unsqueeze(dim=0)
            local_logits = torch.stack(
                [local_explainer(expl_inputs)[0].view(-1) for local_explainer in self.local_explainers],
                dim=0
            )
            graph_embeds = torch.cat(
                [pyg.nn.global_mean_pool(embed, batch),
                 pyg.nn.global_max_pool(embed, batch),
                 pyg.nn.global_add_pool(embed, batch)],
                dim=-1
            )
            global_logits = (self.global_explainer(graph_embeds)
                             .sigmoid()
                             .T
                             .repeat_interleave(edge_batch.bincount(), dim=-1))
            logits       = (global_logits * local_logits).sum(dim=0)
            mask         = self._concrete_sample(logits, training=False).squeeze()
            return mask

        masks = [get_explainer_pred(embed) for embed in embeds]

        # edge mask，feat mask
        return masks, None

    @property
    def name(self):
        return 'kfactexplainer'