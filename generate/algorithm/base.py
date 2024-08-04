# from abc import abstractmethod

from utils.typing_utils import *
class GenerateAlgorithm(nn.Module):
    """
    Basic of generate model.
    """
    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        the sample process of generate model.
        """

    @abstractmethod
    def __loss__(self, *args, **kwargs):
        """
        compute loss when train generate model.
        """

    @abstractmethod
    def train_loop(self, *args, **kwargs):
        """
        train the generative model.
        """

class DiffusionAlgorithm(GenerateAlgorithm):
    """
    Basic of diffusion generator.
    """
    @abstractmethod
    def _q_pred(self, *args, **kwargs):
        r"""
        Get the A^t when given A^{t-1}.

        Returns: prediction of A^t
        """

    @abstractmethod
    def _p_pred(self, *args, **kwargs):
        r"""
        Get the A^{t-1} when given A^t.

        Returns: prediction of A^{t-1}

        """