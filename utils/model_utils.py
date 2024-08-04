import warnings
from numpy.random import RandomState
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import reset

from explain.utils import set_masks, clear_masks
from utils.typing_utils import *
from utils.optimizer_utils import Optimizer
from gnn.gcn import NodeGCN, GraphGCN


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_normalization(name: Optional[str] = None):
    r"""
    Args:
        name: the name of normalization method, include: 'lr': i.e. the layernorm; 'bn': i.e. the batchnorm
    """
    if name is None:
        # nothing to do
        return nn.Identity
    elif name == 'ln':
        # layer norm
        return nn.LayerNorm
    elif name == 'bn':
        # batch norm
        return nn.BatchNorm1d
    else:
        raise NotImplementedError(f"{name} is not implemented.")

def create_activation(name: Optional[str] = None):
    if name is None:
        # nothing to do
        return nn.Identity()
    elif name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name == "elu":
        return nn.ELU()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise NotImplementedError(f"{name} is not implemented.")
    
def get_embeddings(
    model: torch.nn.Module,
    use_hook: bool = False,
    *args,
    **kwargs,
) -> List[Tensor]:
    """Returns the output embeddings of all
    :class:`~torch_geometric.nn.conv.MessagePassing` layers in
    :obj:`model`.

    Internally, this method registers forward hooks on all
    :class:`~torch_geometric.nn.conv.MessagePassing` layers of a :obj:`model`,
    and runs the forward pass of the :obj:`model` by calling
    :obj:`model(*args, **kwargs)`.

    Args:
        model (torch.nn.Module): The message passing model.
        use_hook: Whether to use hook to get embeddings.
        *args: Arguments passed to the model.
        **kwargs (optional): Additional keyword arguments passed to the model.
    """
    embeddings: List[Tensor] = []

    if use_hook:
        def hook(model: nn.Module, inputs: Any, outputs: Any):
            # Clone output in case it will be later modified in-place:
            outputs = outputs[0] if isinstance(outputs, tuple) else outputs
            assert isinstance(outputs, Tensor)
            embeddings.append(outputs.clone())

        hook_handles = []
        for module in model.modules():  # Register forward hooks:
            if isinstance(module, MessagePassing):
                hook_handles.append(module.register_forward_hook(hook))

        if len(hook_handles) == 0:
            warnings.warn("The 'model' does not have any 'MessagePassing' layers")

        training = model.training
        model.eval()
        model(*args, **kwargs)
        model.train(training)

        for handle in hook_handles:  # Remove hooks:
            handle.remove()
    else:
        embeddings.append(model.embedding(*args, **kwargs)[0])

    return embeddings

def get_preds(
    model: nn.Module,
    x: Tensor,
    edge_index: Tensor,
    task_type: str = 'node',
    edge_weight: Optional[Tensor] = None,
    batch: Optional[Tensor] = None,
    index: Union[Tensor, int] = None,
    use_edge_weight: bool = True,
    return_type: str = 'prob'
) -> Tensor:
    """获取预测概率或者类别"""
    model.eval()

    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), dtype=torch.float, device=x.device)
    if use_edge_weight:
        logits = model(x, edge_index, edge_weights=edge_weight, batch=batch)
    else:
        set_masks(model, edge_weight, edge_index, apply_sigmoid=False)
        logits = model(x, edge_index, batch=batch)
        clear_masks(model)
    # logits = model(x=x, edge_index=edge_index, edge_weights=edge_weight)

    if task_type == 'node' and index is not None:
        logits = logits[index]
    if return_type == 'prob':
        return logits.softmax(dim=-1)
    elif return_type == 'raw':
        return logits
    elif return_type == 'label':
        return logits.argmax(dim=-1)
    else:
        raise ValueError(f"{return_type} does not support now!")
    

class GNNTrainer(object):
    """
    Trainer for gnn classification model(include node classification, graph classification and link prediction).
    """
    def __init__(
        self,
        cfgs: Union[Munch, EasyDict],
        dataset: Union[InMemoryDataset, List[Data], Dataset],
        criterion: Optional[Callable] = F.cross_entropy,
        model: nn.Module = None,
    ):
        self.cfgs = cfgs
        self.dataset = dataset
        self.criterion = criterion

        if model is None:
            self.__initialize__()
        else:
            self.model = model

    def __initialize__(self):
        """
        Initialize the GNN model.
        """
        if self.cfgs.gnn_cfgs.paper == "gcn":
            if self.dataset.type == 'node':
                self.model = NodeGCN(
                    self.dataset.num_node_features,
                    self.dataset.num_classes,
                    hidden_dims=self.cfgs.gnn_cfgs.hidden_dim
                )
            else:
                self.model = GraphGCN(
                    self.dataset.num_node_features,
                    self.dataset.num_classes,
                    hidden_dims=self.cfgs.gnn_cfgs.hidden_dim
                )
        else:
            raise NotImplementedError

    def _divide(self, N, shuffle=True):
        """
        Dividing dataset to train dataset, val dataset and test dataset.
        """
        idx = np.arange(0, N)
        if shuffle:
            prng = RandomState(42)
            idx = prng.permutation(idx)
        idx = torch.tensor(idx, dtype=torch.long, device=self.cfgs.device)
        tr_idx = idx[:int(N * self.cfgs.tr_ratio)]
        vl_idx = idx[int(N * self.cfgs.vl_ratio): int(N * (self.cfgs.tr_ratio + self.cfgs.vl_ratio))]
        ts_idx = idx[int(N * (1 - self.cfgs.ts_ratio)):]

        return tr_idx, vl_idx, ts_idx

    def _data_loader(self, tr_idx, ts_idx, vl_idx):
        """
        Load graph classification dataset.
        """
        train_loader = DataLoader(self.dataset[tr_idx],
                                  batch_size=self.cfgs.batch_size,
                                  shuffle=self.cfgs.shuffle)
        test_loader  = DataLoader(self.dataset[ts_idx.to(torch.long)],
                                  batch_size=self.cfgs.batch_size,
                                  shuffle=self.cfgs.shuffle)
        val_loader   = DataLoader(self.dataset[vl_idx.to(torch.long)],
                                  batch_size=self.cfgs.batch_size,
                                  shuffle=self.cfgs.shuffle)

        return train_loader, val_loader, test_loader

    def _test(self, ts_data):
        """
        Test the gnn model, figure out the accuracy.
        :param ts_data: node index for node classification
                        or dataloader for graph classification
        """
        if self.dataset.type == 'node':
            data = self.dataset[0].to(self.cfgs.device)
            out  = self.model(data.x, data.edge_index)
            correct = out[ts_data].argmax(dim=-1).eq(data.y[ts_data]).sum()
            if isinstance(ts_data, torch.BoolTensor):
                size = ts_data.sum()
            else:
                size = ts_data.size(0)
            acc = float(correct / size)
        else:
            total_correct = total = 0
            for data in ts_data:
                data.to(self.cfgs.device)
                out = self.model(data.x, data.edge_index, batch=data.batch)
                total_correct += out.argmax(dim=-1).eq(data.y).sum().item()
                total += data.y.size(-1)
            acc = total_correct / total
        return acc

    def _store_checkpoints(self, state_dict, tr_acc, vl_acc, ts_acc, best_epoch, path):
        """
        Store checkpoint information into file.
        """
        ckpt = Munch(model_state_dict=state_dict,
                     train_acc=tr_acc,
                     test_acc=ts_acc,
                     val_acc=vl_acc,
                     best_epoch=best_epoch)
        paths = path.split('/')
        index = paths.index('checkpoints')
        file_path = osp.join(*paths[index:-1])
        if not osp.exists(file_path):
            os.makedirs(file_path)
        torch.save(ckpt, path)

    def load_model_params(self, path: str, print_checkpoint: bool = False):
        assert osp.exists(path)

        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        state_dict = checkpoint['model_state_dict']
        self.model.load_state_dict(state_dict)
        self.model.to(self.cfgs.device)
        self.model.eval()

        if print_checkpoint:
            print(
                f"This model obtained: "
                f"Train Acc: {checkpoint['train_acc']:.4f} "
                f"Val Acc: {checkpoint['val_acc']:.4f} "
                f"Test Acc: {checkpoint['test_acc']:.4f}."
            )

    def load_dataset(self):
        try:
            tr_idx = self.dataset.train_idx
            vl_idx = self.dataset.val_idx
            ts_idx = self.dataset.test_idx
        except:
            if self.dataset.type == 'node':
                N = self.dataset[0].num_nodes
            else:
                N = len(self.dataset)
            tr_idx, vl_idx, ts_idx = self._divide(N)

        if self.dataset.type == 'graph':
            tr_idx, vl_idx, ts_idx = self._data_loader(tr_idx, vl_idx, ts_idx)

        return tr_idx, vl_idx, ts_idx

    def train_model_params(
            self,
            tr_data: Union[Tensor, DataLoader],
            vl_data: Union[Tensor, DataLoader],
            ts_data: Union[Tensor, DataLoader],
            save_path: str
    ):
        self.model.to(self.cfgs.device)
        reset(self.model)
        # self.model.reset_params()

        optim = Optimizer(
            self.cfgs.gnn_cfgs.optimizer_cfgs,
            self.cfgs.gnn_cfgs.scheduler_cfgs,
            list(self.model.parameters())
        )

        best_val_acc, best_epoch = 0.0, 0
        # Main training loop.
        for epoch in range(self.cfgs.gnn_cfgs.epochs):
            self.model.train()
            if self.dataset.type == 'node':
                # For node classification task.
                data = self.dataset[0].to(self.cfgs.device)
                out  = self.model(data.x, data.edge_index)
                loss = self.criterion(out[tr_data], data.y[tr_data])
                # optim.compute_gradients(loss, mode='sum')
            else:
                # For graph classification task.
                loss = torch.FloatTensor([0]).to(self.cfgs.device)
                for data in tr_data:
                    data.to(self.cfgs.device)
                    out = self.model(data.x, data.edge_index, batch=data.batch)
                    id_loss = self.criterion(out, data.y)
                    loss = torch.add(loss, id_loss)
                    # optim.compute_gradients(loss, mode='sum')
            optim.compute_gradients([loss])
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfgs.gnn_cfgs.clip_max)
            optim.step(epoch)
            self.model.eval()
            tr_acc = self._test(tr_data)
            vl_acc = self._test(vl_data)
            ts_acc = self._test(ts_data)
            print(f"Epoch: {epoch} Loss: {loss.item():.4f} Train Acc: {tr_acc:.4f} "
                  f"Val Acc: {vl_acc:.4f} Test Acc: {ts_acc:.4f} Lr: {optim.optim.param_groups[-1]['lr']}")

            if vl_acc > best_val_acc:
                best_val_acc = vl_acc
                best_epoch = epoch
                self._store_checkpoints(
                    self.model.state_dict(), tr_acc, vl_acc, ts_acc, best_epoch, save_path
                )
            # Early stopping.
            if epoch - best_epoch > self.cfgs.gnn_cfgs.early_stopping and best_val_acc > 0.99: break

    @torch.no_grad()
    def test_model(self, tr_data, vl_data, ts_data):
        tr_acc = self._test(tr_data)
        vl_acc = self._test(vl_data)
        ts_acc = self._test(ts_data)
        print(f"Final Train Acc: {tr_acc:.4f} Final Val Acc: {vl_acc:.4f} Final Test Acc: {ts_acc:.4f}")

    def __call__(self, root: str, save_name: Optional[str] = None, pretrained=True, print_checkpoint=False):
        """
        Load pre-trained parameters for GNN or train parameters for GNN.
        """
        if save_name is None:
            save_name = 'best_model'
        else:
            save_name += '_best_model'

        model_path = osp.join(root, f"{self.cfgs.gnn_cfgs.paper}/{self.cfgs.data.value}/{save_name}")

        tr_data, vl_data, ts_data = self.load_dataset()

        if not pretrained or not osp.exists(model_path):
            print(f"Need to train model, this may take some times...")
            # Training the given model.
            self.train_model_params(tr_data, vl_data, ts_data, model_path)

        self.load_model_params(model_path, print_checkpoint)
        self.test_model(tr_data, vl_data, ts_data)

        return self.model