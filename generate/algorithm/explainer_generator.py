import torch_geometric.nn.inits

from .base import DiffusionAlgorithm
from utils.typing_utils import *
from utils.model_utils import get_embeddings

from generate.utils import reparameterize

class ExplanationGenerator(DiffusionAlgorithm):
    coeffs = Munch(
        knn_loss=1.,
        knn_loss_cf=1.,
        ent_loss=1.,
        bias=0.00001,
        temp0=5.0,
        temp1=2.0,
    )

    def __init__(
        self,
        denoise_fn: Callable,
        device:     Device = torch.device('cpu'),
        epochs:     int = 10,
        **kwargs
    ):
        super(ExplanationGenerator, self).__init__()
        self.denoise_fn = denoise_fn
        self.device     = device

        self.coeffs.update(**kwargs)

        self.temp_schedule = lambda e: (self.coeffs['temp0']
                                        * pow((self.coeffs['temp1'] / self.coeffs['temp0']), e / epochs))

    def __loss__(self, graphs: Data, logits_list: List[Tensor]):
        loss_infos = Munch()

        pos_logits_list = [
            torch.stack([logits[pos_samples_mask]
                         for pos_samples_mask in graphs.pos_samples_mask])
            for logits in logits_list
        ]

        neg_logits_list = [
            torch.stack([logits[neg_samples_mask]
                         for neg_samples_mask in graphs.neg_samples_mask])
            for logits in logits_list
        ]
        # get the corn node labels
        target_label_list = [logits[graphs.get('corn_node_id')].argmax(-1).view(-1) for logits in logits_list]
        # labels of positive samples is 1, and 0 for negative samples
        labels = torch.stack(
            [torch.tensor([1.] * int(pos_samples_mask.sum()) + [0.] * int(neg_samples_mask.sum()),
                          device=self.device)
             for pos_samples_mask, neg_samples_mask
                 in zip(graphs.pos_samples_mask, graphs.neg_samples_mask)]
        )
        knn_loss_list = []
        # get preds of positive samples and negative samples, then compute bce loss
        for pos_logits, neg_logits, target_label in zip(pos_logits_list, neg_logits_list, target_label_list):
            preds = torch.stack(
                [torch.cat([pos_logits[i, :, target], neg_logits[i, :, target]], dim=0)
                 for i, target in enumerate(target_label)], dim=0)
            knn_loss_list.append(sum(F.binary_cross_entropy_with_logits(pred, label)
                                     for pred, label in zip(preds, labels)))

        loss_infos.knn_loss = sum(knn_loss_list) * self.coeffs.knn_loss

        return loss_infos

    def _q_pred(self, num_edges: int):
        pass

    def _p_pred(self, encoder: nn.Module, edge_states_t: Tensor, batch_graph: Data, t: float = 0.):
        edge_weights      = reparameterize(edge_states_t, temperature=t, training=self.training)
        # print(edge_weights)
        with torch.no_grad():
            batch_node_embeds = get_embeddings(
                encoder,
                use_hook      = True,
                x             = batch_graph.x,
                edge_index    = batch_graph.edge_index,
                edge_weights  = edge_weights,
                batch         = batch_graph.get('batch')
            )[-1]
        return self.denoise_fn(edge_states_t, batch_graph.edge_index, batch_node_embeds, batch_graph.batch)

    def reset(self):
        pygnn.inits.reset(self.denoise_fn)

    def train_loop(
        self,
        encoder:            nn.Module,
        batch_graph:        Data,
        cur_epoch:          int = -1
    ) -> Munch:
        assert cur_epoch != -1
        assert hasattr(batch_graph, 'pos_samples_mask') and hasattr(batch_graph, 'neg_samples_mask')

        denoised_edge_0_to_t = [
            self._p_pred(encoder, noised_edge_t, batch_graph, self.temp_schedule(cur_epoch))
            for noised_edge_t in batch_graph.noise_edge_mask_0_to_t_list[-1:]
        ]

        edge_weights = [reparameterize(denoised_edge_mask_t,
                                       bias=self.coeffs.bias,
                                       temperature=self.temp_schedule(cur_epoch),
                                       training=self.training)
                        for denoised_edge_mask_t in denoised_edge_0_to_t]

        logits = [encoder(batch_graph.x,
                          batch_graph.edge_index,
                          edge_weights = edge_weight,
                          batch        = batch_graph.get('batch'))
                  for edge_weight in edge_weights]
        loss_infos  = self.__loss__(batch_graph, logits)
        return loss_infos

    def forward(
        self,
        encoder:     nn.Module,
        batch_graph: Data,
        edge_mask:   Optional[Tensor] = None
    ) -> Tensor:
        if edge_mask is None:
            # get only the last noised graph when test
            if hasattr(batch_graph, 'edge_mask'):
                edge_mask = batch_graph.edge_mask
            else:
                edge_mask = self._q_pred(batch_graph.num_edges)[-1]
        return self._p_pred(encoder, edge_mask, batch_graph)