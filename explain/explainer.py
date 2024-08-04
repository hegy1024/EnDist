from torch_geometric.nn.inits import reset

from utils.typing_utils import *
from utils.io_utils import check_dir
from utils.data_utils import select_data, to_edge_subgraph
from utils.optimizer_utils import Optimizer
from utils.loss_utils import knn, LossInfos
from explain.algorithm.base import InstanceExplainAlgorithm
from generate.algorithm import DiffusionAlgorithm
from .utils import get_embeddings, get_preds
from .explanation import Explanation
from .utils import set_masks, clear_masks, unbatch_data

class EnDistExplainer(object):
    def __init__(
        self,
        model:        nn.Module,
        algorithm:    InstanceExplainAlgorithm,
        generator:    DiffusionAlgorithm,
        dataset:      Union[Dataset, InMemoryDataset, List[Data]],
        cfgs:         Union[Munch, EasyDict],
        explain_type: ExplainTask,
        use_edge_weight: bool = False
    ):
        self.model           = model
        self.algorithm       = algorithm.to(cfgs.device)
        self.generator       = generator.to(cfgs.device)
        self.dataset         = dataset
        self.cfgs            = cfgs
        self.explain_type    = explain_type
        self.use_edge_weight = use_edge_weight

    def get_knn(self, batch_graph: Data):
        batch_graph.to(self.cfgs.device)
        if self.dataset.type == 'node':
            with torch.no_grad():
                logits = self.model(batch_graph.x, batch_graph.edge_index)
            pos_samples_list, neg_samples_list = [], []  # save pos/neg samples
            batch_graph.corn_node_id[1:] += batch_graph.batch.bincount().cumsum(0)[:-1]  # update corn node id per graph
            for tgt in logits[batch_graph.corn_node_id]:
                pos_samples, neg_samples = knn(tgt[None, :], logits, k=self.cfgs.generator_cfgs.K)
                # random sample object of other label
                num_labels = self.cfgs.num_labels - 1
                # number of samples
                if self.dataset.type == 'graph':
                    size = min(self.cfgs.generator_cfgs.K, int(sum(pos_samples))) // num_labels
                else:
                    size = min(self.cfgs.generator_cfgs.K, int(sum(pos_samples)))
                for label in range(batch_graph.y.max() + 1):
                    if label == tgt.argmax(-1): continue
                    candidates = (logits.argmax(-1) == label).nonzero().view(-1).tolist()
                    indices    = random.sample(candidates, min(size, len(candidates)))
                    neg_samples[indices] = True

                pos_samples_list.append(pos_samples)
                neg_samples_list.append(neg_samples)
        else:
            pos_samples_list, neg_samples_list = [], []
            with torch.no_grad():
                logits = self.model(batch_graph.x, batch_graph.edge_index, batch=batch_graph.batch)
            for tgt in logits:
                pos_samples, neg_samples = knn(tgt[None, ...], logits, k=self.cfgs.generator_cfgs.K)
                # random samples object of negative samples
                num_labels = batch_graph.y.max().item()
                # number of neg samples
                size = min(self.cfgs.generator_cfgs.K, int(sum(pos_samples))) // num_labels
                for label in range(num_labels + 1):
                    if label == tgt.argmax(-1): continue
                    indices = random.sample((logits.argmax(-1) == label).nonzero().view(-1).tolist(), size)
                    neg_samples[indices] = True

                pos_samples_list.append(pos_samples)
                neg_samples_list.append(neg_samples)

        batch_graph.pos_samples_mask = torch.stack(pos_samples_list, dim=0)
        batch_graph.neg_samples_mask = torch.stack(neg_samples_list, dim=0)

    def prepare_for_dataset(self, indices: List[int], sifting: bool = False, need_knn: bool = True) -> List[Data]:
        if self.dataset.type == 'node':
            # get k hop subgraph
            dataset = [select_data(self.dataset, idx, num_hops=self.cfgs.num_hops, relabel_nodes=True)
                       for idx in indices]
        else:
            dataset = [select_data(self.dataset, idx) for idx in indices]

        if sifting:
            dataset = [data for data in dataset if (data.edge_mask == 1).sum() > 0]

        loader_list = list(DataLoader(dataset, batch_size=self.cfgs.batch_size, shuffle=self.cfgs.shuffle))

        if need_knn:
            # print("pre-processing data...")
            for data in loader_list:
                data.noise_edge_mask_0_to_t_list = [torch.ones_like(data.edge_mask).float().to(self.cfgs.device) * 10]
                data.degree_0   = pyg.utils.degree(data.edge_index[0], num_nodes=data.num_nodes)
                data.max_degree = data.degree_0.max()
                self.get_knn(data)
            # print("done")
        return loader_list

    def prepare_for_gnne(self, indices: Union[Tensor, List[int]]):
        # define hook function
        # the corn of implementation of DDPGExplainer is the hook function
        def explain_forward_hook(edge_masks: List[Tensor], *args, **kwargs):
            self.generator.eval()
            preds = []
            for temp_edge_mask in edge_masks:
                gen_edge_mask = self.generator(self.model, data, temp_edge_mask)
                set_masks(self.model, gen_edge_mask, data.edge_index)
                logits = self.model(data.x, data.edge_index, batch=data.get('batch'))
                clear_masks(self.model)
                preds.append(logits)

            return edge_masks, preds

        def explain_backward_hook(loss_dict: Munch, edge_masks: List[Tensor], *args, **kwargs):
            edge_mask = edge_masks[0]
            set_masks(self.model, edge_mask, data.edge_index)
            preds = self.model(data.x, data.edge_index, batch=data.get('batch'))
            clear_masks(self.model)
            pos_samples_preds = torch.stack([preds[pos_samples_mask] for pos_samples_mask in data.pos_samples_mask])
            neg_samples_preds = torch.stack([preds[neg_samples_mask] for neg_samples_mask in data.neg_samples_mask])
            samples_labels    = torch.stack(
                [torch.tensor([1.] * int(pos_samples_mask.sum()) + [0.] * int(neg_samples_mask.sum()),
                              device=self.cfgs.device)
                 for pos_samples_mask, neg_samples_mask in zip(data.pos_samples_mask, data.neg_samples_mask)]
            )
            target_label = preds[data.get('corn_node_id')].argmax(-1).view(-1)
            preds = torch.stack(
                [torch.cat([pos_samples_preds[i, :, target], neg_samples_preds[i, :, target]], dim=0)
                 for i, target in enumerate(target_label)]
            )
            loss_dict.knn_loss = sum(F.binary_cross_entropy_with_logits(pred, label)
                                     for pred, label in zip(preds, samples_labels))

        if isinstance(indices, Tensor):
            indices = indices.tolist()

        dataset = self.prepare_for_dataset(indices)

        if self.cfgs.ood_explain:
            # create generator optimizer
            self.generator.train()
            generator_optim      = Optimizer(self.cfgs.generator_cfgs.optimizer_cfgs,
                                             self.cfgs.generator_cfgs.scheduler_cfgs,
                                             list(self.generator.parameters()))
        # create explainer optimizer list and prepare dataset
        self.process_data(dataset)

        explain_forward_handle  = None
        explain_backward_handle = None

        # add backward hook func to explainer
        if self.cfgs.knn_loss:
            explain_backward_handle = self.algorithm.register_explain_backward_hook(explain_backward_hook)

        outer_loop = tqdm(range(self.cfgs.explainer_cfgs.epochs), desc="Explainer Training")
        for epoch in outer_loop:
            # fix ood generator into explainer
            # add hook func to explainer
            if epoch in self.cfgs.hook_point and self.cfgs.ood_explain:
                explain_forward_handle = self.algorithm.register_explain_forward_hook(explain_forward_hook)

            # step1. training explainer
            outer_loop.set_description(f"Explainer Training Start on {epoch}")
            for data in dataset:
                data.optimizer.zero_grad()
                loss_dict = self.algorithm(data,
                                           self.model,
                                           self.use_edge_weight)
                data.optimizer.compute_gradients(loss_dict.values(), mode='sum')
                data.optimizer.step()
            # remove hook func from explainer
            if explain_forward_handle is not None:
                explain_forward_handle.remove()

            # step 2. training generator model if mode is ood_explain
            if self.cfgs.ood_explain:
                for data in dataset:
                    # collect explanation distribution information
                    data.noise_edge_mask_0_to_t_list.append(copy.deepcopy(data.pred_edge_mask[0].detach()))

                if epoch in self.cfgs.generator_train_point:
                    ########################
                    # generator train step #
                    ########################
                    self.generator.train()
                    # step3. train generator
                    inner_loop = tqdm(range(self.cfgs.generator_cfgs.epochs), desc="Generator Training", leave=False)
                    for gen_epoch in inner_loop:
                        inner_loop.set_description(f"Generator Training on {gen_epoch}")
                        generator_optim.zero_grad()
                        for data in dataset:
                            data.to(self.cfgs.device)
                            generator_loss_dict = self.generator.train_loop(self.model,
                                                                            data,
                                                                            gen_epoch)
                            generator_optim.compute_gradients(generator_loss_dict.values(), mode='sum')
                        generator_optim.step()
                        if gen_epoch == self.cfgs.generator_cfgs.epochs - 1:
                            inner_loop.set_description(f"Generator Training Finish")
            if epoch == self.cfgs.explainer_cfgs.epochs - 1:
                outer_loop.set_description(f"Explainer Training Finish")
        # remove backward hook function
        if explain_backward_handle is not None:
            explain_backward_handle.remove()

        return dataset

    def prepare(self, indices: Union[Tensor, List[int]], pretrained:  bool = False, **kwargs):
        """
        training explainer model and generator model
        :param indices: list index of train graph / node
        :param pretrained: whether has been pretrained
        """
        if pretrained:
            print(f"Trying to load the parameters which had be pretrained!")
            self.load_parameters(kwargs.get('save_name'))
            print(f"The parameters has been loaded!")
            return

        if not self.algorithm.pretrain:
            print(f"There is no need to pretrain explainer model!")
            return

        # define forward hook function for explainer
        def explain_forward_hook(edge_masks: List[Tensor], *args, **kwargs):
            preds = []
            self.generator.eval()
            for edge_mask in edge_masks:
                gen_edge_mask = self.generator(self.model, data, edge_mask)
                set_masks(self.model, gen_edge_mask, data.edge_index)
                logits = self.model(data.x,
                                    data.edge_index,
                                    batch=data.get('batch'))
                clear_masks(self.model)
                preds.append(logits)
            return edge_masks, preds

        # define backward hook function for knn loss in explainer
        def explain_backward_hook(loss_dict: Munch, edge_masks: List, *args, **kwargs):
            edge_mask = edge_masks[0]
            set_masks(self.model, edge_mask, data.edge_index)
            preds = self.model(data.x, data.edge_index, batch=data.get('batch'))
            clear_masks(self.model)
            pos_samples_preds = torch.stack([preds[pos_samples_mask] for pos_samples_mask in data.pos_samples_mask])
            neg_samples_preds = torch.stack([preds[neg_samples_mask] for neg_samples_mask in data.neg_samples_mask])
            samples_labels    = torch.stack(
                [torch.tensor([1.] * int(pos_samples_mask.sum()) + [0.] * int(neg_samples_mask.sum()),
                               device=self.cfgs.device)
                 for pos_samples_mask, neg_samples_mask in zip(data.pos_samples_mask, data.neg_samples_mask)]
            )
            target_label = preds[data.get('corn_node_id')].argmax(-1).view(-1)
            preds = torch.stack(
                [torch.cat([pos_samples_preds[i, :, target], neg_samples_preds[i, :, target]], dim=0)
                 for i, target in enumerate(target_label)]
            )
            loss_dict.knn_loss = sum(F.binary_cross_entropy_with_logits(pred, label)
                                     for pred, label in zip(preds, samples_labels))

        if isinstance(indices, Tensor):
            indices = indices.tolist()

        # split dataset from indices
        dataset = self.prepare_for_dataset(indices)

        self.algorithm.train()
        explainer_optim          = Optimizer(self.cfgs.explainer_cfgs.optimizer_cfgs,
                                             self.cfgs.explainer_cfgs.scheduler_cfgs,
                                             list(self.algorithm.parameters()))

        # for ood explain mode
        if self.cfgs.ood_explain:
            self.generator.train()
            generator_optim      = Optimizer(self.cfgs.generator_cfgs.optimizer_cfgs,
                                             self.cfgs.generator_cfgs.scheduler_cfgs,
                                             list(self.generator.parameters()))
        else:
            generator_optim      = None

        # step1. get embedding for all data
        self.process_data(dataset)

        # register hook function
        explain_forward_handle  = None
        explain_backward_handle = None

        if self.cfgs.knn_loss:
            explain_backward_handle = self.algorithm.register_explain_backward_hook(explain_backward_hook)

        outer_loop = tqdm(range(self.cfgs.explainer_cfgs.epochs), desc="Explainer Training")
        for epoch in outer_loop:
            if self.cfgs.ood_explain:
                # explainer pred step
                self.algorithm.eval()
                # step2. explainer pred for every batch graph
                for data in dataset:
                    # explainer preds step: collect information of explainer
                    with torch.no_grad():
                        edge_mask_t, _ = self.algorithm(data)
                        data.noise_edge_mask_0_to_t_list.append(edge_mask_t[0].detach())

                if epoch in self.cfgs.generator_train_point:
                    ########################
                    # generator train step #
                    ########################
                    self.generator.train()
                    # step3. train generator
                    inner_loop = tqdm(range(self.cfgs.generator_cfgs.epochs), desc="Generator Training", leave=False)
                    for gen_epoch in inner_loop:
                        generator_optim.zero_grad()
                        tot_loss = 0
                        for data in dataset:
                            generator_loss_dict = self.generator.train_loop(self.model,
                                                                            data,
                                                                            gen_epoch)
                            generator_optim.compute_gradients(generator_loss_dict.values(), mode='sum')
                            tot_loss += float(sum(generator_loss_dict.values()))
                        inner_loop.set_description(f"Epoch {gen_epoch} Loss {tot_loss:.4f}")
                        generator_optim.step()
                if self.cfgs.save_generator:
                    self.save_parameters(f"{kwargs.get('save_name')}_{kwargs.get('seed')}_{epoch}")

            # fix ood generator into explainer
            if self.cfgs.ood_explain and epoch in self.cfgs.hook_point:
                explain_forward_handle = self.algorithm.register_explain_forward_hook(explain_forward_hook)
            # step4. train explainer
            # explainer train loop
            self.algorithm.train()
            explainer_optim.zero_grad()
            ########################
            # explainer train step #
            ########################
            tot_loss = 0
            for data in dataset:
                explainer_loss_dict = self.algorithm.train_loop(data,
                                                                self.model,
                                                                epoch,
                                                                self.use_edge_weight)

                explainer_optim.compute_gradients(explainer_loss_dict.values(), mode='sum')
                tot_loss += float(sum(explainer_loss_dict.values()))
            explainer_optim.step()
            outer_loop.set_description(f"Epoch {epoch + 1} Loss {tot_loss:.4f}")
            # self.save_parameters(f"explainer_{seed}_{epoch}_new_new")
            # remove forward handle
            if explain_forward_handle is not None:
                explain_forward_handle.remove()
        # remove backward handle
        if explain_backward_handle is not None:
            explain_backward_handle.remove()

    def process_data(self, dataset: List[Data]) -> List[Data]:
        r"""
        Processing dataset for given dataset,
        for PGExplainer we do 1) get embedding, 2) get target label
        for GNNExplainer we do 1) get target label, 2) initialize optimizer
        """
        if self.dataset.type == 'node':
            with torch.no_grad():
                all_embeddings = get_embeddings(
                    self.model,
                    use_hook     = self.cfgs.use_hook,
                    x            = self.dataset.x.to(self.cfgs.device),
                    edge_index   = self.dataset.edge_index.to(self.cfgs.device)
                )[-1]
        else:
            all_embeddings       = None

        for data in dataset:
            data.to(self.cfgs.device)
            if self.algorithm.name == 'gnnexplainer':
                self._process_data_for_gnne(data)
            else:
                self._process_data_for_pge(data, all_embeddings)

    @staticmethod
    def split_explanation(dataset: List[Data], sifting: bool = False) -> List[Explanation]:
        r"""
        Splitting explanation information from given Data object.
        """
        explanations: List[Explanation] = []
        for data in dataset:
            for data_ in unbatch_data(data):
                # only test for data with ground truth
                if sifting:
                    if (data_.get('edge_mask') == 1).sum() == 0: continue
                # get explanation edge mask
                edge_mask = data_.get('pred_edge_mask').detach().sigmoid()
                explanations.append(
                    Explanation(
                        x                 = data_.x,
                        y                 = data_.target_label,
                        corn_node_id      = data_.get('corn_node_id'),
                        edge_index        = data_.edge_index,
                        edge_mask         = edge_mask,
                        ground_truth_mask = data_.edge_mask,
                    )
                )

        return explanations

    def get_explanation(self, dataset: List[Data]) -> List[Data]:
        r"""
        Get explanation subgraph for pgexplainer and kfactexplainer.
        """
        if not isinstance(dataset, list):
            dataset = [dataset]
        for data in dataset:
            edge_mask, _ = self.algorithm(data)
            data.pred_edge_mask = edge_mask

    def save_parameters(self, save_name: str = "best_model"):
        file_name  = osp.join(self.cfgs.model_path, "explainers_parameters", self.dataset.name,
                              f"{self.cfgs.mode}_{self.algorithm.name}")
        check_dir(file_name)
        save_path  = osp.join(file_name, f"{save_name}.pkl")

        model_state_dict = {
            "explainer": self.algorithm.state_dict() if self.algorithm.pretrain else None,
            "generator": self.generator.state_dict()
        }

        torch.save(model_state_dict, save_path)
        # print(f"model is saving at {save_path}")

    def load_parameters_old(self, save_name: str = "best_model"):
        file_name = osp.join(self.cfgs.model_path, "explainers_parameters", self.dataset.name,
                             f"{self.cfgs.mode}_{self.algorithm.name}")
        save_path = osp.join(file_name, f"{save_name}.pkl")
        assert osp.exists(save_path), Exception(f"{save_path} is not exist!")

        model_state_dict = torch.load(save_path, map_location=self.cfgs.device)
        if self.algorithm.pretrain:
            self.algorithm.load_state_dict(model_state_dict["explainer"])
        self.generator.load_state_dict(model_state_dict["generator"])

    def load_parameters(self, save_name: str = "best_model"):
        base_name = [f"{self.cfgs.mode}_{self.cfgs.explainer.value}_without_consistency",
                     f"{self.cfgs.mode}_{self.cfgs.explainer.value}_within_consistency"][int(self.cfgs.knn_loss)]
        base_name += "_within_generator" if self.cfgs.ood_explain and self.cfgs.mode != "raw" else "_without_generator"
        save_name = f"{base_name}_{save_name}"
        file_name = osp.join(self.cfgs.model_path, "explainers_parameters", self.dataset.name,
                             f"{self.cfgs.mode}_{self.algorithm.name}")
        save_path = osp.join(file_name, f"{save_name}_new.pkl")

        assert osp.exists(save_path), Exception(f"{save_path} is not exist!")

        model_state_dict = torch.load(save_path, map_location=self.cfgs.device)
        if self.algorithm.pretrain:
            self.algorithm.load_state_dict(model_state_dict["explainer"])
        self.generator.load_state_dict(model_state_dict["generator"])

        print(f"model has been loaded at {save_path}")

    def get_masked_pred(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor, **kwargs):
        self.model.eval()
        set_masks(self.model, edge_weight, edge_index, apply_sigmoid=False)
        preds = self.model(x, edge_index, **kwargs)
        clear_masks(self.model)

        return preds.softmax(dim=-1).detach().cpu().view(-1).tolist()

    @torch.no_grad()
    def sample_gen_graph(self, data: Data) -> Data:
        self.generator.eval()
        pred_edges = self.generator(self.model, data)
        new_data = data.clone()
        new_data.edge_mask = pred_edges
        return new_data

    def get_generate_graph(self, data: Data) -> Data:
        self.generator.eval()
        gen_edge_mask = self.generator(self.model, data).sigmoid()
        # get real graph
        degree        = pyg.utils.degree(data.edge_index[0][torch.bernoulli(gen_edge_mask).bool()],
                                         num_nodes=data.num_nodes)
        node_index = degree.nonzero().view(-1)
        edge_index = pyg.utils.subgraph(node_index, data.edge_index,
                                        relabel_nodes=True, num_nodes=data.num_nodes)[0]
        return Data(
            x              = data.x[node_index],
            edge_index     = edge_index,
            ori_edge_index = data.edge_index,
            ori_x          = data.x,
            gen_edge_mask  = gen_edge_mask)

    def reset_params(self):
        reset(self.algorithm)
        reset(self.generator)

    def _process_data_for_gnne(self, data: Data):
        r"""
        for GNNExplainer we do 1) get target label, 2) initialize optimizer
        """
        data.to(self.cfgs.device)
        if self.dataset.type == 'node':
            num_nodes = self.dataset[0].num_nodes
        else:
            num_nodes = data.num_nodes

        # initialize edge mask
        edge_mask, feat_mask = self.algorithm.initialize_mask(data.x, data.edge_index, num_nodes)
        params = []
        if edge_mask is not None:
            params.append(edge_mask)
        if feat_mask is not None:
            params.append(feat_mask)
        assert params is not None

        # initialize optimizer
        data.optimizer = Optimizer(self.cfgs.explainer_cfgs.optimizer_cfgs,
                                   self.cfgs.explainer_cfgs.scheduler_cfgs,
                                   params)
        data.pred_edge_mask = [edge_mask]

        # get target label
        data.target_label = get_preds(
            self.model,
            data.x,
            data.edge_index,
            index=data.get('corn_node_id'),
            task_type=self.dataset.type,
            batch=data.batch,
            return_type='label'
        )

    def _process_data_for_pge(self, data: Data, all_embeddings: Optional[Tensor] = None):
        r"""
        for PGExplainer and KfactExplainer we do 1) get embedding, 2) get target label
        """
        # get embeddings
        if self.dataset.type == 'node':
            ori_embeddings = all_embeddings[data.ori_node_idx]
        else:
            with torch.no_grad():
                ori_embeddings = get_embeddings(
                    self.model,
                    use_hook   = self.cfgs.use_hook,
                    x          = data.x,
                    edge_index = data.edge_index
                )[-1]
        data.ori_embeddings = ori_embeddings

        # get target label
        data.target_label   = get_preds(
            self.model,
            data.x,
            data.edge_index,
            index       = data.get('corn_node_id'),
            task_type   = self.dataset.type,
            batch       = data.batch,
            return_type = 'label'
        )

    def __call__(self, indices: Union[int, List[int], Tensor], sifting: bool = True) -> List[Explanation]:
        r"""
        Given index or list of index wait to be explained, return the explanation without sigmoid operation.
        Args:
            indices: index(s) of graph or node to be explained
            sifting: whether to split data with ground truth

        Return: a list of explanation
        """
        self.algorithm.eval()
        self.generator.eval()

        if isinstance(indices, int): indices = [indices]

        if self.algorithm.name == 'gnnexplainer':
            dataset = self.prepare_for_gnne(indices)
        else:
            dataset = self.prepare_for_dataset(indices, sifting=sifting, need_knn=False)
            self.process_data(dataset)
            self.get_explanation(dataset)

        return self.split_explanation(dataset, sifting=sifting)

    def __repr__(self):
        if self.cfgs.mode == 'ed':
            return f"EnDist_{self.algorithm.name}"
        else:
            return f"raw_{self.algorithm.name}"