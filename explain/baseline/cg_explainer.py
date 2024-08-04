from abc import ABC


from explain.utils import pack_explanatory_subgraph, combine_mask
from gnn import GraphGCN, NodeGCN
from utils.typing_utils import *
from utils.io_utils import check_dir
from utils.data_utils import select_data
from utils.optimizer_utils import Optimizer
from explain.algorithm.base import InstanceExplainAlgorithm
from explain.utils import get_embeddings, get_preds
from explain.explanation import Explanation

class AddTrainableMask(ABC):
    _tensor_name: str

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, module, inputs):
        r"""
        Compute new weights of network.
        """
        setattr(module, self._tensor_name, self.apply_mask(module))

    def apply_mask(self, module):
        r"""
        Add network mask to weights of module.
        Args:
            module: Module need to add mask.
        """
        mask_train: Tensor  = getattr(module, self._tensor_name + "_mask_train")
        mask_fixed: Tensor  = getattr(module, self._tensor_name + "_mask_fixed")
        orig_weight: Tensor = getattr(module, self._tensor_name + "_orig_weight")
        # compute new weights of network
        pruned_weight = mask_train * mask_fixed * orig_weight

        return pruned_weight

    @classmethod
    def apply(cls, module, name, mask_train, mask_fixed, *args, **kwargs):
        r"""
        Rename relate weights of module and
        """
        # instance the class method
        method = cls(*args, **kwargs)
        method._tensor_name = name
        # get original weights
        orig: Tensor = getattr(module, name)
        # apply mask of network
        module.register_parameter(name + "_mask_train", mask_train)
        module.register_parameter(name + "_mask_fixed", mask_fixed)
        module.register_parameter(name + "_orig_weight", orig)
        # delete the original weights
        del module._parameters[name]
        # set new weights of network
        setattr(module, name, method.apply_mask(module))
        # add hook func to module to auto compute masked network
        module.register_forward_pre_hook(method)

        return method


class NetworkMaskTrainer(object):
    def __init__(self, cfgs, data: Data, dataset: Dataset):
        self.cfgs = cfgs
        self.data = data

        # Initialize the GNN model.
        if self.cfgs.gnn_cfgs.paper == "gcn":
            if dataset.type == 'node':
                self.model = NodeGCN(
                    dataset.num_node_features,
                    dataset.num_classes,
                    hidden_dims=self.cfgs.gnn_cfgs.hidden_dim
                )
            else:
                self.model = GraphGCN(
                    dataset.num_node_features,
                    dataset.num_classes,
                    hidden_dims=self.cfgs.gnn_cfgs.hidden_dim
                )
        else:
            raise NotImplementedError

        # load model params
        model_path = osp.join(cfgs.model_path,
                              f"{self.cfgs.gnn_cfgs.paper}/{self.cfgs.data.value}/best_model")

        assert osp.exists(model_path)

        ckpts = torch.load(model_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(ckpts['model_state_dict'])
        self.model.to(cfgs.device)
        # apply mask and init mask
        self.add_mask()
        mask_list = self.soft_mask_init()
        self.model.eval()
        # set optimizer
        self.optim = torch.optim.Adam(mask_list)

    def add_mask(self):
        r"""
        Add network weights mask to every module of given model.
        """
        for module in self.model.modules():
            if isinstance(module, pyg.nn.MessagePassing):
                # initialize network mask and apply it to module
                mask1_train = nn.Parameter(torch.ones_like(module.lin.weight)
                                           .to(module.lin.weight.dtype))
                mask1_fixed = nn.Parameter(torch.ones_like(module.lin.weight)
                                           .to(module.lin.weight.dtype), requires_grad=False)
                AddTrainableMask.apply(module.lin, 'weight', mask1_train, mask1_fixed)
            elif isinstance(module, nn.Linear):
                # initialize network mask and apply it to module
                mask1_train = nn.Parameter(torch.ones_like(module.weight)
                                           .to(module.weight.dtype))
                mask1_fixed = nn.Parameter(torch.ones_like(module.weight)
                                           .to(module.weight.dtype), requires_grad=False)
                AddTrainableMask.apply(module, 'weight', mask1_train, mask1_fixed)

    def soft_mask_init(self) -> List[nn.Parameter]:
        r"""
        Random initialize the network mask.
        """
        mask_list: List[nn.Parameter] = []
        C = 1e-5
        for module in self.model.modules():
            if isinstance(module, pyg.nn.MessagePassing):
                # random initialize network mask
                module.lin.weight_mask_train.requires_grad = False
                rand1 = (2 * torch.rand(module.lin.weight_mask_train.shape)
                         .to(module.lin.weight_mask_train.device) - 1) * C
                rand1 *= module.lin.weight_mask_train
                module.lin.weight_mask_train.add_(rand1)
                module.lin.weight_mask_train.requires_grad = True
                mask_list.append(module.lin.weight_mask_train)
            elif isinstance(module, nn.Linear):
                # random initialize network mask
                module.weight_mask_train.requires_grad = False
                rand1 = (2 * torch.rand(module.weight_mask_train.shape)
                         .to(module.weight_mask_train.device) - 1) * C
                rand1 *= module.weight_mask_train
                module.weight_mask_train.add_(rand1)
                module.weight_mask_train.requires_grad = True
                mask_list.append(module.weight_mask_train)

        return mask_list

    def get_mask_distribution(self):
        r"""
        Get the network mask vector.
        """
        weight_mask_vector: List[Tensor] = []
        for module in self.model.modules():
            if isinstance(module, pyg.nn.MessagePassing):
                weight = module.lin.weight_mask_train.flatten()
                nonzero = weight.abs() > 1e-4
                weight_mask_vector.append(weight[nonzero])
            elif isinstance(module, nn.Linear):
                weight = module.weight_mask_train.flatten()
                nonzero = weight.abs() > 1e-4
                weight_mask_vector.append(weight[nonzero])

        return torch.cat(weight_mask_vector, dim=-1).detach().cpu()

    def binary_weight(self):
        r"""
        Set the mask of network as binary value.
        """
        weight_mask  = self.get_mask_distribution()
        # number of total weight
        weight_total = weight_mask.shape[0]
        # sort all weight
        weight_value, weight_indices = torch.sort(weight_mask)
        # get index of threshold
        weight_th_index = int(weight_total * self.cfgs.net_ratio_1)
        # get value of threshold
        weight_th       = weight_value[weight_th_index]
        # set weight under threshold
        for module in self.model.modules():
            if isinstance(module, pyg.nn.MessagePassing):
                weight        = module.lin.weight_mask_train
                binary_weight = nn.Parameter(torch.where(weight.abs() >= weight_th,
                                                         torch.ones_like(weight),
                                                         torch.zeros_like(weight)).float(), requires_grad=True)
                module.lin.weight_mask_train = binary_weight
                module.lin.weight_mask_fixed = binary_weight
            elif isinstance(module, nn.Linear):
                weight        = module.weight_mask_train
                binary_weight = nn.Parameter(torch.where(weight.abs() >= weight_th,
                                                         torch.ones_like(weight),
                                                         torch.zeros_like(weight)).float(), requires_grad=True)
                module.weight_mask_train = binary_weight
                module.weight_mask_fixed = binary_weight

    def train_step(self):
        r"""
        Training the network mask.
        """
        for epoch in range(self.cfgs.lth_epochs):
            logits = self.model(self.data.x, self.data.edge_index).softmax(dim=-1)
            loss   = -logits[0, self.data.y]
            loss.backward()
            self.optim.step()
        self.binary_weight()

class CGExplainer(object):
    r"""
    An implementation for `cooperative explanation of GNNs`,
    get into https://dl.acm.org/doi/abs/10.1145/3539597.3570378 for more details,
    code are based on official code https://github.com/MangoKiller/CGE_demo,
    thanks for their extraordinary work.
    """
    def __init__(
        self,
        model:           nn.Module,
        algorithm:       List[Optional[InstanceExplainAlgorithm]],
        dataset:         Union[Dataset, InMemoryDataset, List[Data]],
        cfgs:            Union[Munch, EasyDict],
        explain_type:    ExplainTask = ExplainTask.EDGE,
        use_edge_weight: bool = False
    ):
        self.model        = model
        self.dataset      = dataset
        self.cfgs         = cfgs
        self.explain_type = explain_type
        self.use_edge_weight = use_edge_weight
        self.algorithm, self.algorithm_cge = algorithm
        self.algorithm.to(cfgs.device)
        self.algorithm_cge.to(cfgs.device)

    def prepare_for_dataset(self, indices: List[int]) -> List[Data]:
        r"""
        Prepare dataset for cg_explainer.
        """
        if self.dataset.type == 'node':
            # get k hop subgraph
            dataset = [select_data(self.dataset,
                                   idx,
                                   num_hops=self.cfgs.num_hops,
                                   relabel_nodes=True,
                                   remove_self_loop=False)
                       for idx in indices]
        else:
            dataset = [select_data(self.dataset,
                                   idx,
                                   remove_self_loop=False) for idx in indices]

        return dataset

    def m_step(self, dataset: List[Data]) -> List[Data]:
        # M Step: train network mask
        # step 1: extract explanation subgraph under given top ratio
        explanation_subgraphs = [pack_explanatory_subgraph(data,
                                                           data.edge_mask.sigmoid(),
                                                           self.cfgs.graph_ratio_1)
                                 for data in dataset]
        # step 2: training network mask for every subgraph
        for subgraph in tqdm(explanation_subgraphs, desc="M-step Sub-network Training"):
            lth_model_trainer = NetworkMaskTrainer(self.cfgs, subgraph, self.dataset)
            # train lth model
            lth_model_trainer.train_step()
            # save to subgraph as a gnn predictor
            subgraph.predictor = lth_model_trainer.model
            # initialize edge mask
            edge_mask, feat_mask = self.algorithm_cge.initialize_mask(subgraph.x,
                                                                      subgraph.edge_index,
                                                                      subgraph.N)
            # initialize edge mask and feat mask
            params = []
            if edge_mask is not None:
                params.append(edge_mask)
            if feat_mask is not None:
                params.append(feat_mask)
            assert params is not None
            subgraph.optimizer = Optimizer(self.cfgs.cge_cfgs.optimizer_cfgs,
                                           self.cfgs.cge_cfgs.scheduler_cfgs,
                                           params)
            subgraph.pred_edge_mask = edge_mask
            subgraph.pred_feat_mask = feat_mask

        # step 3: restart explainer process
        pbar = tqdm(range(200), desc="M-step Explainer Training")
        for _ in pbar:
            for data in explanation_subgraphs:
                data.optimizer.zero_grad()
                loss_dict = self.algorithm_cge(data, data.predictor, self.use_edge_weight)
                data.optimizer.compute_gradients(loss_dict.values(), mode='sum')
                data.optimizer.step()

        # step 4: combine two explanation edge mask
        for explanation1, explanation2 in zip(dataset, explanation_subgraphs):
            edge_mask = combine_mask([explanation1.edge_mask.detach().sigmoid(),
                                      explanation2.pred_edge_mask.detach().sigmoid()],
                                     [self.cfgs.graph_ratio_1])
            explanation1.edge_mask = self.norm_imp(edge_mask)

        return dataset

    def prepare_for_gnne(self, indices: Union[Tensor, List[int]]) -> List[Data]:
        r"""
        For explainer like GNNExplainer, we get the explanation result straightly.
        Args:
            indices: Test indices.

        Returns: a list of data, which include explanation.
        """
        dataset = self.prepare_for_dataset(indices)
        # E step: train explainer and get explanation
        # step1: get explainer information
        for data in dataset:
            data.to(self.cfgs.device)
            if self.dataset.type == 'node':
                num_nodes = self.dataset[0].num_nodes
            else:
                num_nodes = data.num_nodes
            edge_mask, feat_mask = self.algorithm.initialize_mask(data.x, data.edge_index, num_nodes)
            # initialize edge mask and feat mask
            params = []
            if edge_mask is not None:
                params.append(edge_mask)
            if feat_mask is not None:
                params.append(feat_mask)
            assert params is not None
            data.optimizer      = Optimizer(self.cfgs.explainer_cfgs.optimizer_cfgs,
                                            self.cfgs.explainer_cfgs.scheduler_cfgs,
                                            params)
            data.pred_edge_mask = edge_mask
            data.pred_feat_mask = feat_mask

            data.target_label   = get_preds(
                self.model,
                data.x,
                data.edge_index,
                index=data.get('corn_node_id'),
                task_type=self.dataset.type,
                batch=data.batch,
                return_type='label'
            )
        # step 2: train explainer
        for _ in tqdm(range(self.cfgs.explainer_cfgs.epochs), desc="E-step Explainer Training"):
            for data in dataset:
                data.optimizer.zero_grad()
                loss_dict = self.algorithm(data, self.model, self.use_edge_weight)
                data.optimizer.compute_gradients(loss_dict.values(), mode='sum')
                data.optimizer.step()

        return dataset

    def prepare(
        self,
        indices: Union[Tensor, List[int]],
        pretrained: bool = False,
        save_params: bool = False,
        save_name  : str = "best_model",
        **kwargs
    ):
        r"""
        Training explainer like PGExplainer.
        """
        if pretrained:
            print(f"Trying to load the parameters which had be pretrained!")
            self.load_parameters(save_name)
            print(f"The parameters has been loaded!")
            return

        if not self.algorithm.pretrain:
            print(f"There is no need to pretrain explainer model!")
            return

        dataset         = self.prepare_for_dataset(indices)
        self.algorithm.train()
        explainer_optim = Optimizer(
            self.cfgs.explainer_cfgs.optimizer_cfgs,
            self.cfgs.explainer_cfgs.scheduler_cfgs,
            list(self.algorithm.parameters())
        )
        # E step: training explainer -- prepare explainer information
        if self.dataset.type == 'node':
            with torch.no_grad():
                all_embeddings = get_embeddings(
                    self.model,
                    use_hook     = self.cfgs.use_hook,
                    x            = self.dataset.x.to(self.cfgs.device),
                    edge_index   = self.dataset.edge_index.to(self.cfgs.device)
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

        # E step: training explainer -- explainer training step
        pbar = tqdm(range(self.cfgs.explainer_cfgs.epochs), desc="E-step Explainer Training")
        for epoch in pbar:
            # explainer train loop
            self.algorithm.train()
            explainer_optim.zero_grad()
            for data in dataset:
                explainer_loss_dict = self.algorithm.train_loop(data, self.model, epoch)

                explainer_optim.compute_gradients(explainer_loss_dict.values(), mode='sum')
            explainer_optim.step()

        # if save_params: self.save_parameters(save_name)

    @staticmethod
    def norm_imp(imp: Tensor) -> Tensor:
        imp[imp < 0] = 0
        imp += 1e-16
        return imp / imp.sum()

    def save_parameters(self, save_name: str = "best_model"):
        model_name = f"cge_{self.algorithm.name}_{save_name}.pkl"
        file_name  = osp.join(self.cfgs.model_path, "explainers", self.dataset.name, "cge")
        check_dir(file_name)
        save_path  = osp.join(file_name, model_name)
        model_state_dict = {
            "explainer": self.algorithm.state_dict()
                         if self.algorithm.pretrain else None,
        }
        torch.save(model_state_dict, save_path)
        print(f"model is saving at {save_path}")

    def load_parameters(self, save_name: str = "best_model"):
        model_name = f"cge_{self.algorithm.name}_{save_name}.pkl"
        file_name  = osp.join(self.cfgs.model_path, "explainers", self.dataset.name, "cge")
        save_path  = osp.join(file_name, model_name)

        assert osp.exists(save_path), Exception(f"{save_path} is not exist!")

        model_state_dict = torch.load(save_path)
        self.algorithm.load_state_dict(model_state_dict["explainer"])
        self.algorithm.to(self.cfgs.device)

    def __call__(self, idxs: Union[int, List[int], Tensor], sifting: bool = True) -> List[Explanation]:
        r"""
        Get the predicting explanation.
        """
        self.algorithm.eval()

        ##############################################
        ################ E step #####################
        #############################################

        explanations: List[Explanation] = []

        if isinstance(idxs, int): idxs = [idxs]

        if self.algorithm.name == 'gnnexplainer':
            # for gnnexplainer and other explainer like it
            # we don't need to pretrain
            dataset = self.prepare_for_gnne(idxs)
            for data in dataset:
                if sifting:
                    if (data.get('edge_mask') == 1).sum() == 0: continue
                edge_mask = data.get('pred_edge_mask').detach()
                explanations.append(
                    Explanation(
                        x                   = data.x,
                        y                   = data.target_label,
                        corn_node_id        = data.get('corn_node_id'),
                        edge_index          = data.edge_index,
                        edge_mask           = edge_mask,
                        ground_truth_mask   = data.edge_mask,
                        edge_attr           = data.edge_attr,
                        target_label        = data.target_label
                    )
                )
        else:
            # for pgexplainer or other explainer like this
            # we have to train explainer first
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
                    data = select_data(self.dataset, idx).to(self.cfgs.device)
                    embeds = get_embeddings(
                        self.model,
                        self.cfgs.use_hook,
                        data.x,
                        data.edge_index
                    )[-1]
                if sifting:
                    # only test explanation for data with ground truth
                    if (data.get('edge_mask') == 1).sum() == 0: return None
                data.target_label = get_preds(
                    self.model,
                    data.x,
                    data.edge_index,
                    task_type=self.dataset.type,
                    return_type='label'
                )
                data.ori_embeddings = embeds
                edge_mask, _ = self.algorithm(data)

                edge_mask = edge_mask[0].detach()

                return Explanation(
                    x                 = data.x,
                    y                 = data.target_label,
                    corn_node_id      = data.get('corn_node_id'),
                    edge_index        = data.edge_index,
                    ground_truth_mask = data.edge_mask,
                    edge_mask         = edge_mask,
                    edge_attr         = data.edge_attr,
                    target_label      = data.target_label
                )

            for idx in idxs:
                if (explanation := get_explanation(idx)) is not None:
                    explanations.append(explanation)

        explanations = self.m_step(explanations)

        return explanations

    def __repr__(self):
        return f"cge_{self.algorithm.name}"