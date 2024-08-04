from torch.utils.hooks import RemovableHandle

from utils.typing_utils import *

class BaseAlgorithm(nn.Module):
    def __init__(self):
        super(BaseAlgorithm, self).__init__()
        self._explain_forward_hook:    Dict[int, Callable] = OrderedDict()
        self._explain_backward_hook:   Dict[int, Callable] = OrderedDict()

    @property
    def pretrain(self) -> bool:
        r"""Whether to train explainer. """
        return True

    def register_explain_forward_hook(self, hook: Callable[..., None]) -> RemovableHandle:
        r"""External processing when prediction with gnn model. """
        handle = RemovableHandle(self._explain_forward_hook)
        self._explain_forward_hook[handle.id] = hook
        return handle

    def register_explain_backward_hook(self, hook: Callable[..., None]) -> RemovableHandle:
        r"""External processing for loss of explainer."""
        handle = RemovableHandle(self._explain_backward_hook)
        self._explain_backward_hook[handle.id] = hook
        return handle


class InstanceExplainAlgorithm(BaseAlgorithm):
    @abstractmethod
    def forward(self, *args, **kwargs) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        r"""main method of instance level explainer. """

    @abstractmethod
    def __loss__(self, *args, **kwargs) -> Munch:
        r"""compute loss of explainer. """

    @abstractmethod
    def train_loop(self, *args, **kwargs) -> Munch:
        r"""training explainer in one loop. """


class ModelLevelExplainAlgorithm(BaseAlgorithm):
    @abstractmethod
    def forward(self, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        r"""Main method of model level explainer. """

    @abstractmethod
    def train_loop(self, *args, **kwargs) -> Munch:
        r"""Training model level explainer. """

    @abstractmethod
    def __loss__(self, *args, **kwargs) -> Munch:
        r"""Computing loss of model level explainer. """