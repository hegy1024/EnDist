import torch
from typing import *
from torch import Tensor
import torch.nn.functional as F

from utils.model_utils import create_activation

def norm(x: Tensor) -> Tensor:
    """
    对输入x进行规范化: x - x.min() / x.max() - x.min()
    """
    max_values = x.reshape(x.shape[0], -1).max(dim=-1, keepdim=True)[0]
    min_values = x.reshape(x.shape[0], -1).min(dim=-1, keepdim=True)[0]

    return ((x - min_values.unsqueeze(2).unsqueeze(3))
            / (max_values.unsqueeze(2).unsqueeze(3) - min_values.unsqueeze(2).unsqueeze(3)))

def manifold_mix_up(embed_a: Tensor, embed_b: Tensor, alpha: Union[Tensor, float] = 0.2) -> Tensor:
    """
    流形混合: 混合图a与图b的embedding
    Args:
        embed_a: 图a的embedding
        embed_b: 图b的embedding
        alpha:   beta分布的参数
    Returns:     混合后的embedding
    """
    if not isinstance(alpha, Tensor):
        alpha = torch.tensor([Tensor])
    beta = torch.distributions.beta.Beta(alpha, alpha)  # beta distribution

    mix_lambda = beta.rsample(sample_shape=(1, 1)).to(embed_a.device)
    mixup_embeds = mix_lambda * embed_a + (1 - mix_lambda) * embed_b

    return mixup_embeds

def extract(a: Tensor, index: Tensor, shape: Tensor, dim: int = 0) -> Tensor:
    r"""
    extracting the input with index and reshape it.
    """
    out = a.gather(index=index, dim=dim)
    return out.view([index.shape[0]] + [1] * (len(shape) - 1))

def reparameterize(logits: Tensor, temperature: float = 1., bias: float = 0.0001,
                   activation: Optional[str] = 'sigmoid', training: bool = False):
    r"""
    Implementation of the reparameterization trick to obtain a sample graph
    while maintaining the posibility to backprop.
    """
    if training:
        eps = (2 * bias - 1) * torch.rand(logits.size()) + 1 - bias
        logits = ((eps.log() - (1 - eps).log()).to(logits.device) + logits) / temperature

    return create_activation(activation)(logits)

###############################################################
########################Diffusion utils #######################
###############################################################

def get_beta_schedule(
    beta_schedule: str,
    num_diffusion_time_steps: int,
    *args
) -> Tensor:
    """
    get the noise schedule for every diffusion step.
    """
    if beta_schedule == "quad":
        beta_start, beta_end = args
        betas = (
            torch.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_time_steps,
                dtype=torch.float,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        beta_start, beta_end = args
        betas = torch.linspace(
            beta_start, beta_end, num_diffusion_time_steps, dtype=torch.float
        )
    elif beta_schedule == "const":
        beta_end = args
        betas = beta_end * torch.ones(num_diffusion_time_steps, dtype=torch.float)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / torch.linspace(
            1, num_diffusion_time_steps, num_diffusion_time_steps, dtype=torch.float
        )
    elif beta_schedule == "sigmoid":
        beta_start, beta_end = args
        betas = torch.linspace(-6, 6, num_diffusion_time_steps)
        betas = betas.sigmoid() * (beta_end - beta_start) + beta_start
    elif beta_schedule == 'cos':
        """
        cosine schedule
        as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
        """
        s = args
        steps = num_diffusion_time_steps + 1
        x = torch.linspace(0, steps, steps)
        alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = (alphas_cumprod[1:] / alphas_cumprod[:-1]).clip(min=0.001, max=1.).sqrt()
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_time_steps,)
    return betas

def index2log_onehot(x: Tensor, num_classes: Union[Tensor, int]) -> Tensor:
    r"""
    Translating the input x into one-hot tensor with num_classes dims at dim1.
    """
    assert x.max().item() < num_classes, \
        f'Error: {x.max().item()} >= {num_classes}'

    x_onehot = F.one_hot(x, num_classes)
    permute_order = (0, -1) + tuple(range(1, len(x.size())))
    x_onehot = x_onehot.permute(permute_order)  # rearange x-onehot under permute-order

    return x_onehot.float().clamp(min=1e-30).log()

def log_onehot2index(log_x: Tensor) -> Tensor:
    r"""
    Translating the onehot input log_x into indices tensor.
    """
    return log_x.argmax(dim=1)

def log_sample_categorical(logits, num_classes) -> Tensor:
    r"""
    add noise to the given logits.
    """
    uniform = torch.rand_like(logits)  # uniform distribution noise
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
    sample = (gumbel_noise + logits).argmax(dim=1)
    log_sample = index2log_onehot(sample, num_classes)
    return log_sample

def log_1_sub_a(a: Tensor) -> Tensor:
    r"""
    given input tensor a, compute log(1 - exp(a))
    """
    return torch.log(1 - a.exp() + 1e-40)

def log_add_exp(a: Tensor, b: Tensor) -> Tensor:
    r"""
    compute log sum exp of tensor a and tensor b.
    """
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))