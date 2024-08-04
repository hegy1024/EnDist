from math import sqrt

from explain.algorithm.base import InstanceExplainAlgorithm
from explain.explanation import Explanation
from utils.io_utils import check_dir
from utils.typing_utils import *
from utils.data_utils import select_data
from utils.optimizer_utils import Optimizer
from utils.model_utils import get_preds, get_embeddings
from utils.loss_utils import LossInfos
from explain.utils import set_masks, clear_masks
from .utils import mixup_graph


class MixUpExplainer(object):
    r"""
    An implementation for `MixupExplainer: Generalizing Explanations for Graph Neural Networks with Data Augmentation`
    (check https://arxiv.org/abs/2307.07832 for more details)
    Code are based on official implementation, I rewrite a part of code to get result under same setting.
    (check https://github.com/jz48/MixupExplainer/tree/main/MixupExplainer for more details)
    Thanks for their extraordinary work.
    """
    def __init__(
        self,
        model: nn.Module,
        algorithm: InstanceExplainAlgorithm,
        dataset: Union[Dataset, InMemoryDataset, List[Data]],
        cfgs: Union[Munch, EasyDict],
        explain_type: ExplainTask = ExplainTask.EDGE,
        use_edge_weight: bool = False
    ):
        self.model     = model
        self.algorithm = algorithm
        self.dataset   = dataset
        self.cfgs      = cfgs
        self.explain_type = explain_type
        self.use_edge_weight = use_edge_weight
        self.dropout      = nn.Dropout(0.1)

    def initialize_mask(self, data: Data, N: Optional[int] = None):
        E   = data.num_edges
        if N is None:
            N = data.num_nodes

        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        pred_edge_mask1     = torch.nn.Parameter(torch.randn(E).to(self.cfgs.device) * std)
        pred_edge_mask2     = torch.nn.Parameter(torch.randn(E).to(self.cfgs.device) * std)
        data.pred_edge_mask = [pred_edge_mask1, pred_edge_mask2]

    def prepare_for_dataset(self, indices: List[int], sifting: bool = True, sorted_edge: bool = True) -> List[Data]:
        if self.dataset.type == 'node':
            # get k hop subgraph for node classification
            dataset = [select_data(self.dataset, idx,
                                   num_hops=self.cfgs.num_hops,
                                   relabel_nodes=False)
                       for idx in indices]
        else:
            dataset = [select_data(self.dataset, idx) for idx in indices]
        if sifting:
            indices = [idx for i, idx in enumerate(indices) if (dataset[i].edge_mask == 1).sum() > 0]
            dataset = [data for data in dataset if (data.edge_mask == 1).sum() > 0]
        # mixup random data to target data for explainer
        candi_dataset = self.get_candidates_data(indices, candis=list(range(len(dataset))))
        # candi_dataset = self.get_candidates_data(indices, candis=indices)
        mixup_dataset = []

        for i in tqdm(range(len(dataset)), desc="Mixup Graph"):
            data1, data2 = dataset[i], candi_dataset[i]
            mixup_dataset.append(mixup_graph(data1, data2, self.cfgs.yita, sorted_edge=sorted_edge))
        return mixup_dataset

    def get_candidates_data(self, indices, candis: List) -> List[Data]:
        # candis = range(len(self.dataset))
        # candis = indices
        candis_dataset = []
        for idx1 in indices:
            # random sample index
            while (idx2 := random.choice(candis)) == idx1: continue
            if self.dataset.type == 'node':
                data = select_data(self.dataset, idx2, num_hops=self.cfgs.num_hops, relabel_nodes=False)
            else:
                data = select_data(self.dataset, idx2)
            candis_dataset.append(data)
        return candis_dataset

    def prepare_for_gnne(self, indices: Union[Tensor, List[int]]):
        r"""
        Use explainer model like gnne, which are doesn't have parameters to be trained.
        """
        def explain_forward_hook(edge_mask: List[Tensor], *args, **kwargs):
            """pred under mixup edge mask"""
            edge_mask1 = edge_mask[0].sigmoid() * data.mask1
            edge_mask2 = edge_mask[1].sigmoid() * data.mask2
            t1 = self.dropout(data.mask2 - edge_mask2)
            t2 = self.dropout(data.mask1 - edge_mask1)
            # t1 = F.dropout(data.mask2 - edge_mask2, p=0.1)
            # t2 = F.dropout(data.mask1 - edge_mask1, p=0.1)
            pred_edge_mask1 = edge_mask1 + t1
            pred_edge_mask2 = edge_mask2 + t2
            # get prediction of gnn
            # as same as the official implement, we use edge weights
            mask_pred1 = self.model(data.x, data.edge_index, edge_weights=pred_edge_mask1, batch=data.batch)
            mask_pred2 = self.model(data.x, data.edge_index, edge_weights=pred_edge_mask2, batch=data.batch)
            # different with official implement, we use set mask func
            # if using edge weight, we got wrong answer
            # set_masks(self.model, pred_edge_mask1, data.edge_index, apply_sigmoid=False)
            # mask_pred1 = self.model(data.x, data.edge_index, batch=data.batch)
            # set_masks(self.model, pred_edge_mask2, data.edge_index, apply_sigmoid=False)
            # mask_pred2 = self.model(data.x, data.edge_index, batch=data.batch)
            # clear_masks(self.model)
            if self.dataset.type == 'node':
                mask_pred1 = mask_pred1[data.corn_node_id[0]]
                mask_pred2 = mask_pred2[data.corn_node_id[1]]
            return [pred_edge_mask1, pred_edge_mask2], [mask_pred1, mask_pred2]

        def explain_backward_hook(*args, **kwargs):
            pass

        # dataset process
        dataset = self.prepare_for_dataset(indices)
        for data in dataset:
            data.to(self.cfgs.device)
            # initialize edge mask for mixed graph
            self.initialize_mask(data)
            # create optimizer
            data.optimizer = Optimizer(self.cfgs.explainer_cfgs.optimizer_cfgs,
                                       self.cfgs.explainer_cfgs.scheduler_cfgs,
                                       data.pred_edge_mask)
            # get target label
            target_label1 = get_preds(
                self.model,
                data.x1,
                data.edge_index1,
                index=data.get('corn_node_id')[0],
                task_type=self.dataset.type,
                return_type='label'
            )
            target_label2 = get_preds(
                self.model,
                data.x2,
                data.edge_index2,
                index=data.get('corn_node_id')[1],
                task_type=self.dataset.type,
                return_type='label'
            )
            data.target_label = [target_label1, target_label2]

        explainer_loss_infos = LossInfos(self.cfgs.explainer_cfgs.epochs, size=len(dataset))

        explain_forward_handle  = self.algorithm.register_explain_forward_hook(explain_forward_hook)
        explain_backward_handle = self.algorithm.register_explain_backward_hook(explain_backward_hook)

        for epoch in range(self.cfgs.explainer_cfgs.epochs):
            # training start
            for data in dataset:
                # training start
                data.optimizer.zero_grad()
                loss_dict = self.algorithm(data,
                                           self.model,
                                           self.use_edge_weight,
                                           apply_sigmoid=False)
                data.optimizer.compute_gradients(loss_dict.values(), mode='sum')
                explainer_loss_infos.update(epoch, loss_dict)
                data.optimizer.step()
            explainer_loss_infos.print(epoch=epoch)

        print(f"mixup+{self.algorithm.name} running over")
        explain_forward_handle.remove()
        explain_backward_handle.remove()
        return dataset

    def save_parameters(self, save_name: str = "best_model"):
        model_name = f"mixup_{self.algorithm.name}_{save_name}.pkl"
        file_name  = osp.join(self.cfgs.model_path, "explainers", self.dataset.name)
        check_dir(file_name)
        save_path  = osp.join(file_name, model_name)
        model_state_dict = {
            "explainer": self.algorithm.state_dict() if self.algorithm.pretrain else None,
        }
        torch.save(model_state_dict, save_path)
        print(f"model is saving at {save_path}")

    def load_parameters(self, save_name: str = "best_model"):
        model_name = f"mixup_{self.algorithm.name}_{save_name}.pkl"
        file_name  = osp.join(self.cfgs.model_path, "explainers", self.dataset.name)
        save_path  = osp.join(file_name, model_name)
        assert osp.exists(save_path), Exception(f"{save_path} is not exists!")
        model_state_dict = torch.load(save_path)
        if self.algorithm.pretrain:
            self.algorithm.load_state_dict(model_state_dict["explainer"])
            self.algorithm.to(self.cfgs.device)

    def prepare(
        self,
        indices: Union[Tensor, List[int]],
        pretrained: bool = False,
        save_params: bool = False,
        sifting    : bool = True,
        save_name  : str = "best_model",
    ):
        r"""
        training explainer model and generator model
        Args:
            indices: list index of train graph / node
            pretrained: whether has been pretrained
            save_params: whether to save params
            sifting: whether to split data
        """
        if pretrained:
            print(f"Trying to load the parameters which had be pretrained!")
            self.load_parameters(save_name)
            print(f"The parameters has been loaded!")
            return

        if not self.algorithm.pretrain:
            print(f"There is no need to pretrain explainer model!")
            return

        def explain_forward_hook(edge_masks: List[Tensor], *args, **kwargs):
            """pred under mixup edge mask"""
            edge_mask1 = edge_masks[0].sigmoid() * data.mask1
            edge_mask2 = edge_masks[1].sigmoid() * data.mask2
            t1 = self.dropout(data.mask2 - edge_mask2)
            t2 = self.dropout(data.mask1 - edge_mask1)
            # t1 = F.dropout(data.mask2 - edge_mask2, p=0.1)
            # t2 = F.dropout(data.mask1 - edge_mask1, p=0.1)
            pred_edge_mask1 = (edge_mask1 + t1).sigmoid()
            pred_edge_mask2 = (edge_mask2 + t2).sigmoid()
            # get prediction of gnn
            set_masks(self.model, pred_edge_mask1, data.edge_index, apply_sigmoid=False)
            mask_pred1 = self.model(data.x, data.edge_index, batch=data.batch)
            # mask_pred1 = self.model(data.x, data.edge_index, edge_weights=pred_edge_mask1, batch=data.batch)
            set_masks(self.model, pred_edge_mask2, data.edge_index, apply_sigmoid=False)
            mask_pred2 = self.model(data.x, data.edge_index, batch=data.batch)
            # mask_pred2 = self.model(data.x, data.edge_index, edge_weights=pred_edge_mask2, batch=data.batch)
            clear_masks(self.model)
            if self.dataset.type == 'node':
                mask_pred1 = mask_pred1[data.corn_node_id[0]]
                mask_pred2 = mask_pred2[data.corn_node_id[1]]
            return [pred_edge_mask1, pred_edge_mask2], [mask_pred1, mask_pred2]

        def explain_backward_hook(*args, **kwargs):
            pass

        dataset = self.prepare_for_dataset(indices, sifting=sifting)
        # step1. get embeddings for all data
        self.process_data(dataset)
        # initialize optimizer and hook function
        self.algorithm.to(self.cfgs.device)
        self.algorithm.train()
        explainer_optim = Optimizer(
            self.cfgs.explainer_cfgs.optimizer_cfgs,
            self.cfgs.explainer_cfgs.scheduler_cfgs,
            list(self.algorithm.parameters())
        )
        explainer_loss_infos = LossInfos(self.cfgs.explainer_cfgs.epochs, size=len(indices))

        explain_forward_handle  = self.algorithm.register_explain_forward_hook(explain_forward_hook)
        explain_backward_handle = self.algorithm.register_explain_backward_hook(explain_backward_hook)

        # step2. training explainer
        for epoch in range(self.cfgs.explainer_cfgs.epochs):
            explainer_optim.zero_grad()
            for data in dataset:
                ########################
                # explainer train step #
                ########################
                loss_dict = self.algorithm.train_loop(
                    data,
                    self.model,
                    epoch,
                    self.use_edge_weight,
                    apply_sigmoid=False
                )
                explainer_optim.compute_gradients(loss_dict.values(), mode='sum')
                explainer_loss_infos.update(epoch, loss_dict)

            explainer_loss_infos.print(epoch=epoch)
            explainer_optim.step()

        explain_forward_handle.remove()
        explain_backward_handle.remove()

        if save_params:
            self.save_parameters(save_name)

    def process_data(self, dataset: List[Data]) -> List[Data]:
        r"""
        Given a list of data, compute node 1) embeddings, 2) target label
        """
        if self.dataset.type == 'node':
            with torch.no_grad():
                all_embeddings = get_embeddings(
                    self.model,
                    use_hook=self.cfgs.use_hook,
                    x=self.dataset.x.to(self.cfgs.device),
                    edge_index=self.dataset.edge_index.to(self.cfgs.device)
                )[-1]
        for data in dataset:
            data.to(self.cfgs.device)
            if self.dataset.type == 'node':
                ori_embeddings1 = all_embeddings[data.ori_node_idx1]
                ori_embeddings2 = all_embeddings[data.ori_node_idx2]
            else:
                with torch.no_grad():
                    ori_embeddings1 = get_embeddings(
                        self.model,
                        use_hook=self.cfgs.use_hook,
                        x=data.x1,
                        edge_index=data.edge_index1
                    )[-1]
                    ori_embeddings2 = get_embeddings(
                        self.model,
                        use_hook=self.cfgs.use_hook,
                        x=data.x2,
                        edge_index=data.edge_index2
                    )[-1]
            ori_embeddings = torch.cat((ori_embeddings1, ori_embeddings2), dim=0)
            data.ori_embeddings = [ori_embeddings, ori_embeddings]
            data.hard_masks     = [data.mask1, data.mask2]
            target_label1  = get_preds(
                self.model,
                data.x1,
                data.edge_index1,
                index       = data.get('corn_node_id', [None])[0],
                task_type   = self.dataset.type,
                return_type = 'label'
            )
            target_label2 = get_preds(
                self.model,
                data.x2,
                data.edge_index2,
                index       = data.get('corn_node_id', [None])[-1],
                task_type   = self.dataset.type,
                return_type = 'label'
            )
            data.target_label = [target_label1, target_label2]

    def get_explanation(self, dataset: List[Data]):
        r"""
        Get explanation for explainer like pgexplainer.
        """
        self.process_data(dataset)
        # explanations: List[Data] = []
        for data in dataset:
            data.to(self.cfgs.device)
            edge_mask, _ = self.algorithm(data)
            data.pred_edge_mask = edge_mask

    @staticmethod
    def split_explanation(dataset: List[Data]) -> List[Explanation]:
        r"""
        Splitting explanation from given dataset.
        """
        explanations: List[Data] = []
        for data in dataset:
            edge_mask = data.get('pred_edge_mask')[0].detach().sigmoid()
            edge_mask = edge_mask[data.mask1.bool()]
            explanations.append(
                Explanation(
                    x                 = data.x1,
                    y                 = data.target_label[0],
                    corn_node_id      = data.get('corn_node_id', [None])[0],
                    edge_index        = data.edge_index1,
                    edge_mask         = edge_mask,
                    ground_truth_mask = data.edge_mask,
                )
            )
        return explanations

    def __call__(self, idxs: Union[int, List[int], Tensor], sifting: bool = True) -> List[Explanation]:
        self.algorithm.eval()
        if isinstance(idxs, int): idxs = [idxs]
        if self.algorithm.name == 'gnnexplainer':
            dataset = self.prepare_for_gnne(idxs)
        else:
            dataset = self.prepare_for_dataset(idxs, sifting=sifting, sorted_edge=False)
            self.get_explanation(dataset)

        return self.split_explanation(dataset)

    def __repr__(self):
        return f"mixup_{self.algorithm.name}"