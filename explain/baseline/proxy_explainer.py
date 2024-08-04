import torch.optim.optimizer

from utils.io_utils import check_dir
from utils.data_utils import select_data, to_edge_subgraph
from utils.optimizer_utils import Optimizer
from utils.loss_utils import LossInfos
from utils.model_utils import get_embeddings, get_preds
from explain.utils import set_masks, clear_masks
from explain.explanation import Explanation
from explain.algorithm import PGExplainer
from explain.algorithm.base import InstanceExplainAlgorithm
from .generator import ProxyGenerator

from utils.typing_utils import *

class ProxyExplainer(object):
    r"""
    An implementation code for ProxyExplainer in ICML2024 paper:
        `Interpreting Graph Neural Networks with In-Distributed Proxies`,
    This implementation are based on the description of paper and official implementation,
    thanks for the extraordinary work.
    (Check https://arxiv.org/abs/2402.02036 and https://github.com/realMoana/ProxyExplainer for more details.)
    """
    def __init__(
        self,
        model:     nn.Module,
        dataset:   Union[Dataset, InMemoryDataset, List[Data]],
        cfgs:      Union[Munch, EasyDict],
        algorithm: Optional[InstanceExplainAlgorithm] = None,
        use_edge_weight: bool = False
    ):
        self.model   = model
        self.dataset = dataset
        self.cfgs    = cfgs
        self.use_edge_weight = use_edge_weight

        self.algorithm = algorithm.to(cfgs.device)
        self.algorithm.to(cfgs.device)
        self.generator = ProxyGenerator(
            in_channels     = dataset.num_features,
            hidden_channels = cfgs.generator_cfgs.hidden_channels,
            out_channels    = cfgs.generator_cfgs.out_channels
        ).to(cfgs.device)

    def prepare_for_dataset(self, indices: List[int], filt: bool = False) -> List[Data]:
        if self.dataset.type == 'node':
            # get k hop subgraph
            dataset = [select_data(self.dataset, idx, num_hops=self.cfgs.num_hops, relabel_nodes=True)
                       for idx in indices]
        else:
            dataset = [select_data(self.dataset, idx, remove_noise_node=True) for idx in indices]

        if filt:
            dataset = [data for data in dataset if (data.edge_mask == 1).sum() > 0]

        for data in tqdm(dataset, desc="Process data"):
            node_idx  = torch.arange(data.edge_index.flatten().max() + 1)
            num_nodes = torch.ones_like(node_idx) * data.num_nodes
            all_edge_index  = torch.stack((torch.repeat_interleave(node_idx, num_nodes),
                                                  node_idx.repeat(1, data.num_nodes).view(-1)),
                                          dim=0)
            all_edge_label  = torch.stack([torch.any(torch.bitwise_and(data.edge_index[0] == u,
                                                                       data.edge_index[1] == v))
                                           for u, v in all_edge_index.T],
                                          dim=0).long()
            all_edge_index,      all_edge_label      = pyg.utils.remove_self_loops(all_edge_index, all_edge_label)
            data.all_edge_index, data.all_edge_label = pyg.utils.coalesce(all_edge_index, all_edge_label)

        return dataset

    def prepare_for_gnne(self, indices: Union[Tensor, List[int]]) -> List[Data]:
        # define hook function
        def explain_forward_hook(edge_masks: List[Tensor], *args, **kwargs):
            preds = []
            for edge_mask in edge_masks:
                self.generator.eval()
                gen_edge_mask = self.generator(data, edge_mask)
                set_masks(self.model, gen_edge_mask, data.all_edge_index, apply_sigmoid=False)
                logits = self.model(data.x, data.all_edge_index, batch=data.get('batch'))
                clear_masks(self.model)
                preds.append(logits)
            return edge_masks, preds

        # define hook function for knn loss in explainer
        def explain_backward_hook(*args, **kwargs):
            pass

        dataset = self.prepare_for_dataset(indices, filt=True)
        # create generator optimizer
        self.generator.train()
        generator_optim = torch.optim.Adam(self.generator.parameters(),
                                           lr=self.cfgs.generator_cfgs.optimizer_cfgs.lr,
                                           weight_decay=0.00005)

        generator_loss_infos = LossInfos(self.cfgs.generator_cfgs.epochs, size=len(indices))

        # create explainer optimizer list and prepare dataset
        for data in dataset:
            data.to(self.cfgs.device)
            if self.dataset.type == 'node':
                num_nodes = self.dataset[0].num_nodes
            else:
                num_nodes = data.num_nodes
            edge_mask, feat_mask = self.algorithm.initialize_mask(data.x, data.edge_index, num_nodes)
            params = []
            if edge_mask is not None:
                params.append(edge_mask)
            if feat_mask is not None:
                params.append(feat_mask)
            assert params is not None
            data.optimizer = Optimizer(self.cfgs.explainer_cfgs.optimizer_cfgs,
                                       self.cfgs.explainer_cfgs.scheduler_cfgs,
                                       params)
            data.pred_edge_mask = edge_mask
            data.pred_feat_mask = feat_mask

            data.target_label = get_preds(
                self.model,
                data.x,
                data.edge_index,
                index=data.get('corn_node_id'),
                task_type=self.dataset.type,
                batch=data.batch,
                return_type='label'
            )

        explainer_loss_infos = LossInfos(self.cfgs.explainer_cfgs.epochs, size=len(indices))

        explain_backward_handle = self.algorithm.register_explain_backward_hook(explain_backward_hook)
        explain_forward_handle  = self.algorithm.register_explain_forward_hook(explain_forward_hook)

        for epoch in range(self.cfgs.explainer_cfgs.epochs):
            generator_loss_infos.clear()
            self.generator.train()
            # step1. train generator
            for gen_epoch in range(self.cfgs.generator_cfgs.epochs):
                generator_optim.zero_grad()
                loss = torch.FloatTensor([0]).to(self.cfgs.device)
                for data in dataset:
                    data.to(self.cfgs.device)
                    generator_loss_dict = self.generator.train_loop(data)
                    # generator_optim.compute_gradients(generator_loss_dict.values(), mode='sum')
                    loss += sum(generator_loss_dict.values())
                    generator_loss_infos.update(gen_epoch, generator_loss_dict)

                loss.backward()
                generator_optim.step()
                print("Generator Loss: ", end='')
                generator_loss_infos.print(epoch=gen_epoch)
            # fix ood generator into explainer
            # step2. training explainer
            for data in dataset:
                data.optimizer.zero_grad()
                loss_dict = self.algorithm(data, self.model, self.use_edge_weight)
                data.optimizer.compute_gradients(loss_dict.values(), mode='sum')
                explainer_loss_infos.update(epoch, loss_dict)
                data.optimizer.step()
            print(f"Explainer Loss: ", end='')
            explainer_loss_infos.print(epoch=epoch)

        if explain_forward_handle is not None:
            explain_forward_handle.remove()
        if explain_backward_handle is not None:
            explain_backward_handle.remove()

        return dataset

    def prepare(
        self,
        indices: Union[Tensor, List[int]],
        pretrained: bool = False,
        save_params: bool = False,
        sifting:     bool = False,
        save_name: str = "best_model",
        seed:      int = None
    ):
        r"""
        Training proxy explainer.
        Args:
            indices:     train dataset
            pretrained:  whether been pretrained
            save_params: whether to save params of explainer
        """

        # define hook function
        def explain_forward_hook(edge_masks: List[Tensor], *args, **kwargs):
            preds = []
            for edge_mask in edge_masks:
                self.generator.eval()
                gen_edge_mask = self.generator(data, edge_mask)
                set_masks(self.model, gen_edge_mask, data.all_edge_index, apply_sigmoid=False)
                logits = self.model(data.x, data.all_edge_index, batch=data.get('batch'))
                clear_masks(self.model)
                preds.append(logits)
            return edge_masks, preds

        # define hook function for knn loss in explainer
        def explain_backward_hook(*args, **kwargs):
            pass

        if pretrained:
            print(f"Trying to load the parameters which had be pretrained!")
            self.load_parameters(save_name)
            print(f"The parameters has been loaded!")
            return

        if not self.algorithm.pretrain:
            print(f"There is no need to pretrain explainer model!")
            return

        dataset = self.prepare_for_dataset(indices, filt=sifting)
        self.algorithm.train()
        explainer_optim = torch.optim.Adam(self.algorithm.parameters(),
                                           lr=self.cfgs.explainer_cfgs.optimizer_cfgs.lr,
                                           weight_decay=0.00005)
        self.generator.train()
        generator_optim = torch.optim.Adam(self.generator.parameters(),
                                           lr=self.cfgs.generator_cfgs.optimizer_cfgs.lr,
                                           weight_decay=0.00005)

        generator_loss_infos = LossInfos(self.cfgs.generator_cfgs.epochs, size=len(indices))
        explainer_loss_infos = LossInfos(self.cfgs.explainer_cfgs.epochs, size=len(indices))

        # step1. get embedding for all data
        if self.dataset.type == 'node':
            with torch.no_grad():
                all_embeddings = get_embeddings(
                    self.model,
                    use_hook=self.cfgs.use_hook,
                    x=self.dataset.x.to(self.cfgs.device),
                    edge_index=self.dataset.edge_index.to(self.cfgs.device)
                )[-1]
        for data in tqdm(dataset, desc="Process Dataset"):
            data.to(self.cfgs.device)
            # get embeddings
            if self.dataset.type == 'node':
                ori_embeddings = all_embeddings[data.ori_node_idx]
            else:
                with torch.no_grad():
                    ori_embeddings = get_embeddings(
                        self.model,
                        use_hook=self.cfgs.use_hook,
                        x=data.x,
                        edge_index=data.edge_index
                    )[-1]
            data.ori_embeddings = ori_embeddings
            # get target label
            data.target_label = get_preds(
                self.model,
                data.x,
                data.edge_index,
                index=data.get('corn_node_id'),
                task_type=self.dataset.type,
                return_type='label'
            )

        explain_forward_handle  = self.algorithm.register_explain_forward_hook(explain_forward_hook)
        explain_backward_handle = self.algorithm.register_explain_backward_hook(explain_backward_hook)

        for epoch in range(self.cfgs.explainer_cfgs.epochs):
            # explainer pred step
            self.algorithm.eval()
            # step2. explainer pred for every batch graph
            for data in dataset:
                # explainer preds step: collect information of explainer
                with torch.no_grad():
                    edge_mask, _        = self.algorithm(data)
                    data.pred_edge_mask = edge_mask[0]
            ########################
            # generator train step #
            ########################
            generator_loss_infos.clear()
            # self.generator.reset()
            self.generator.train()
            # step3. train generator
            for gen_epoch in range(self.cfgs.generator_cfgs.epochs):
                generator_loss = torch.FloatTensor([0]).to(self.cfgs.device)
                generator_optim.zero_grad()
                for data in dataset:
                    generator_loss_dict = self.generator.train_loop(data)
                    generator_loss += sum(generator_loss_dict.values())

                generator_loss.backward()
                generator_optim.step()

            if self.cfgs.save_generator:
                self.save_parameters(f"{seed}_{epoch}")

            self.generator.eval()
            # step4. train explainer
            # explainer train loop
            self.algorithm.train()
            explainer_optim.zero_grad()
            explainer_loss = torch.FloatTensor([0]).to(self.cfgs.device)
            ########################
            # explainer train step #
            ########################
            for data in dataset:
                explainer_loss_dict = self.algorithm.train_loop(data,
                                                                self.model,
                                                                epoch,
                                                                self.use_edge_weight)

                explainer_loss += sum(explainer_loss_dict.values())
                # explainer_optim.compute_gradients(explainer_loss_dict.values(), mode='sum')
                explainer_loss_infos.update(epoch, explainer_loss_dict)

            print("Explainer Loss: ", end='')
            explainer_loss_infos.print(epoch=epoch)
            explainer_loss.backward(retain_graph=True)
            explainer_optim.step()

        explain_forward_handle.remove()
        explain_backward_handle.remove()

        # if save_params:
        #     self.save_parameters(save_name)
    def sample_gen_graph(self, data: Data) -> Data:
        self.generator.eval()
        pred_edges   = self.generator(data, data.edge_mask)
        edge_index, node_mapping = to_edge_subgraph(data.all_edge_index, pred_edges)

        return Data(
            x = torch.stack([data.x[node_mapping[i]] for i in edge_index.flatten().unique().tolist()]),
            y = data.y,
            edge_index = edge_index)

    def save_parameters(self, save_name: str = "best_model"):
        r"""
        Save parameters of explainer and generator.
        """
        model_name = f"proxy_{self.algorithm.name}_{save_name}.pkl"
        file_name = osp.join(self.cfgs.model_path, "explainers", self.dataset.name, "proxy_explainer")
        check_dir(file_name)
        save_path = osp.join(file_name, model_name)
        model_state_dict = {
            "explainer": self.algorithm.state_dict() if self.algorithm.pretrain else None,
            "generator": self.generator.state_dict()
        }
        torch.save(model_state_dict, save_path)
        print(f"model is saving at {save_path}")

    def load_parameters(self, save_name: str = "best_model"):
        r"""
        Load parameters of explainer and generator.
        """
        model_name = f"proxy_{self.algorithm.name}_{save_name}.pkl"

        file_name = osp.join(self.cfgs.model_path, "explainers", self.dataset.name, "proxy_explainer")
        save_path = osp.join(file_name, model_name)

        assert osp.exists(save_path), Exception(f"{save_path} is not exist!")

        model_state_dict = torch.load(save_path)

        if self.algorithm.pretrain:
            self.algorithm.load_state_dict(model_state_dict["explainer"])
        self.generator.load_state_dict(model_state_dict["generator"])
        self.algorithm.to(self.cfgs.device)
        self.generator.to(self.cfgs.device)

    def __call__(self, idxs: Union[int, List[int], Tensor], sifting: bool = True) -> List[Explanation]:
        r"""
        Given index or list of index wait to be explained, return the explanation with sigmoid operation.
        Args:
            idxs: index(s) of graph or node to be explained
        Returns: a list of explanation
        """
        self.algorithm.eval()
        self.generator.eval()

        explanations: List[Explanation] = []

        if self.algorithm.name == 'gnnexplainer':
            dataset = self.prepare_for_gnne(idxs)
            for data in dataset:
                edge_mask = data.get('pred_edge_mask').sigmoid().detach()
                if sifting:
                    if (data.get('edge_mask') == 1).sum() == 0: continue
                explanations.append(
                    Explanation(
                        x                 = data.x,
                        y                 = data.target_label,
                        corn_node_id      = data.get('corn_node_id'),
                        edge_index        = data.edge_index,
                        edge_mask         = edge_mask,
                        ground_truth_mask = data.edge_mask,
                    )
                )
        else:
            def get_explanation(idx):
                if self.dataset.type == 'node':
                    all_embeddings = get_embeddings(
                        self.model,
                        use_hook=self.cfgs.use_hook,
                        x=self.dataset.x.to(self.cfgs.device),
                        edge_index=self.dataset.edge_index.to(self.cfgs.device)
                    )[-1]
                    data = select_data(self.dataset,
                                       idx,
                                       self.cfgs.num_hops,
                                       relabel_nodes=True).to(self.cfgs.device)
                    embeds = all_embeddings[data.ori_node_idx]
                else:
                    data   = select_data(self.dataset, idx).to(self.cfgs.device)
                    embeds = get_embeddings(
                        self.model,
                        self.cfgs.use_hook,
                        data.x,
                        data.edge_index
                    )[-1]
                if sifting:
                    if (data.get('edge_mask') == 1).sum() == 0: return None

                target_label = get_preds(
                    self.model,
                    data.x,
                    data.edge_index,
                    task_type   = self.dataset.type,
                    return_type = 'label'
                )

                data.target_label    = target_label
                data.ori_embeddings  = embeds
                edge_mask, feat_mask = self.algorithm(data)

                edge_mask = edge_mask[0].detach().sigmoid()

                explanation = Explanation(
                    x                 = data.x,
                    y                 = target_label,
                    corn_node_id      = data.get('corn_node_id'),
                    edge_index        = data.edge_index,
                    ground_truth_mask = data.edge_mask,
                    edge_mask         = edge_mask
                )
                return explanation

            for idx in idxs:
                if (explanation := get_explanation(idx)) is not None:
                    explanations.append(explanation)

        return explanations

    def __repr__(self):
        return f"proxy_{self.algorithm.name}"