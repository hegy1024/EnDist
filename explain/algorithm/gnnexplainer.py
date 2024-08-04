from math import sqrt

import torch

from explain.utils import set_masks, clear_masks
from utils.typing_utils import *
from explain.algorithm.base import InstanceExplainAlgorithm

class GNNExplainer(InstanceExplainAlgorithm):
    r"""
    An implementation of GNNExplainer in
    `Gnnexplainer: Generating explanations for graph neural networks <https://arxiv.org/abs/1903.03894>.`_.
    """
    coeffs = Munch(
        reg_size=0.005,
        reg_ent=1.,
        EPS=1e-15,
        edge_reduction='sum',
        feat_reduction='mean',
        feat_ent=0.1,
        feat_size=1.,
        knn_loss=1.
    )

    def __init__(
        self,
        device:       Device,
        gnn_task:     str = 'node',
        explain_type: ExplainTask = ExplainTask.EDGE,
        **kwargs
    ):
        super().__init__()
        self.device = device
        self.gnn_task = gnn_task
        self.explain_type = explain_type
        self.coeffs.update(kwargs)

    def __loss__(
        self,
        masked_preds:    List[Tensor],
        original_labels: List[Tensor],
        edge_batch:    Tensor,
        batch:         Tensor,
        edge_masks:    Optional[List[Tensor]] = None,
        feat_masks:    Optional[Tensor] = None,
        apply_sigmoid: bool = True
    ) -> Munch:
        """
        Return loss of given between mask_pred and original_label.
        Args:
            masked_preds:   preds of masked edge
            original_labels: preds of original edge
            edge_batch:    which graph are edge belongs to
            batch:         which graph are node belongs to
            edge_masks:     explanation edge mask of graph
            feat_masks:     explanation feat mask of graph
            apply_sigmoid:  whether to use sigmoid
        Returns:
            Type[munch]:  loss information
        """
        loss_dict = Munch()
        for masked_pred, original_label, edge_mask in zip(masked_preds, original_labels, edge_masks):
            # the cross_entropy loss of preds
            ce_loss = F.cross_entropy(masked_pred, original_label, reduction='sum')

            # regulation loss
            reg_loss1 = reg_loss2 = 0
            if edge_mask is not None:
                mask = edge_mask.sigmoid() if apply_sigmoid else edge_mask
                # reg loss 1: size loss
                reg_loss1 = (scatter(mask, edge_batch, dim=-1, reduce=self.coeffs['edge_reduction']).sum()
                             * self.coeffs['reg_size'])
                # reg loss 2: ent loss
                ent = (- mask * torch.log(mask + self.coeffs['EPS'])
                       - (1 - mask) * torch.log(1 - mask + self.coeffs['EPS']))
                reg_loss2 = scatter(ent, edge_batch, dim=-1, reduce='mean').sum() * self.coeffs['reg_ent']

            if feat_masks is not None:
                mask = feat_masks.sigmoid()
                # reg loss 1
                reg_loss1 += (scatter(mask, batch, dim=-1, reduce=self.coeffs['feat_reduction']).sum()
                              * self.coeffs['reg_size'])
                # reg loss 2
                ent = (- mask * torch.log(mask + self.coeffs['EPS'])
                       - (1 - mask) * torch.log(1 - mask + self.coeffs['EPS']))
                reg_loss2 += scatter(ent, batch, dim=-1, reduce='mean').sum() * self.coeffs['reg_ent']

            loss_dict_ = Munch(
                cross_entropy = ce_loss,
                size_loss     = reg_loss1,
                ent_loss      = reg_loss2
            )
            for u, v in loss_dict_.items():
                loss_dict.__setitem__(u, loss_dict.get(u, 0) + v)

        if self._explain_backward_hook:
            for hook in self._explain_backward_hook.values():
                hook(loss_dict, edge_masks, masked_preds)

        return loss_dict

    @property
    def pretrain(self):
        """Whether to pretrain model"""
        return False

    def train_loop(self, *args):
        pass

    def initialize_mask(self, x, edge_index, N=None) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """
        The method for initializing the mask of edge and node features.
        Args:
            x: The features metrix of graph.
            edge_index: The adj of graph.
            N: The number of node in graph, which can be pre-defined, default value is None.
        Returns:
            initialize edge mask
        """
        if N is None: N = x.size(0)
        E = edge_index.size(1)

        std = nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))

        if self.explain_type == ExplainTask.EDGE:
            edge_mask = nn.Parameter(torch.randn(E).to(self.device) * std)
            feat_mask = None
        elif self.explain_type == ExplainTask.FEATURE:
            edge_mask = None
            feat_mask = nn.Parameter(torch.randn_like(x).to(self.device) * std)
        else:
            edge_mask = nn.Parameter(torch.randn(E).to(self.device) * std)
            feat_mask = nn.Parameter(torch.randn_like(x).to(self.device) * std)

        return edge_mask, feat_mask

    def forward(
        self,
        data:             Data,
        model_to_explain: nn.Module,
        use_edge_weight:  bool = False,
        apply_sigmoid:    bool = True,
        **kwargs
    ) -> Munch:
        r"""
        Train loop for gnnexplainer and generator.
        Args:
            data:             a PYG Data which storage information of graph to be explained
            model_to_explain: GNN model to be explained
            use_edge_weight:  whether to use edge weight when get prediction
            apply_sigmoid:    whether to use sigmoid when compute loss
        Returns:
            A dict/munch which storage loss information
        """
        x, edge_index = data.x, data.edge_index
        if (batch := data.get('batch')) is None:
            batch  = torch.zeros(data.num_nodes).long().to(self.device)
        edge_batch = batch[edge_index[0]]

        if (feat_mask := data.get('pred_feat_mask')) is not None:
            x = torch.mul(x, feat_mask.sigmoid())
        if not isinstance(edge_masks := data.get('pred_edge_mask'), List):
            edge_masks = [edge_masks]

        def get_gnn_pred(mask: Tensor) -> Tensor:
            if not use_edge_weight:
                set_masks(model_to_explain, mask, edge_index)
                pred = model_to_explain(x, edge_index, batch=batch)
                clear_masks(model_to_explain)
            else:
                pred = model_to_explain(x, edge_index, edge_weights=mask.sigmoid(), batch=batch)
            return pred

        masked_pred: List[Tensor] = []
        if self._explain_forward_hook:
            for hook in self._explain_forward_hook.values():
                edge_masks, masked_pred = hook(edge_masks)
        else:
            masked_pred = [get_gnn_pred(edge_mask) for edge_mask in edge_masks]

        if self.gnn_task == 'node':
            corn_node_id = data.get('corn_node_id')
            if not isinstance(corn_node_id, List): corn_node_id = [corn_node_id]
            assert len(corn_node_id) == len(masked_pred)
            masked_pred = [pred[node_id] for pred, node_id in zip(masked_pred, corn_node_id)]

        if not isinstance(target_label := data.get('target_label'), List): target_label = [target_label]
        loss_dict = self.__loss__(
            masked_pred,
            target_label,
            edge_batch,
            batch,
            edge_masks,
            data.get('pred_feat_mask'),
            apply_sigmoid
        )

        return loss_dict

    @property
    def name(self):
        return 'gnnexplainer'