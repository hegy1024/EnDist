import torch
from typing import Union, Tuple, Optional, List

import concurrent.futures
from functools import partial

import numpy as np
import pyemd
from scipy.linalg import toeplitz

def relabel_nodes(
    edge_index: torch.Tensor,
    index: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    node_set = edge_index.flatten().unique()
    index_dict = {int(x): i for i, x in enumerate(node_set)}
    edge_index_ = list(map(lambda x: list(map(lambda y: index_dict[y], x)),
                           edge_index.tolist()))
    if index is not None:
        index = [index_dict[int(idx)] for idx in index]
    return (torch.tensor(edge_index_, device=edge_index.device),
            torch.tensor(index, device=edge_index.device) if index is not None else None)

#####################################################################################
#################################dist helper#########################################
#####################################################################################

def process_tensor(x, y):
    """
    Helper function to pad tensors to the same size
    :param x: tensor
    :param y: tensor
    :return: padded tensors
    """
    support_size = max(len(x), len(y))
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))
    return x, y

def emd(x, y, distance_scaling=1.0):
    """
    Earth Mover's Distance (EMD) between two 1D pmf
    :param x: 1D pmf
    :param y: 1D pmf
    :param distance_scaling: scaling factor for distance matrix
    :return: EMD distance
    """
    x = x.astype(float)
    y = y.astype(float)
    support_size = max(len(x), len(y))
    d_mat = toeplitz(range(support_size)).astype(float)  # diagonal-constant matrix
    distance_mat = d_mat / distance_scaling
    x, y = process_tensor(x, y)

    emd_value = pyemd.emd(x, y, distance_mat)
    return np.abs(emd_value)


def l2(x, y):
    """
    L2 distance between two 1D pmf
    :param x: 1D pmf
    :param y: 1D pmf
    :return: L2 distance
    """
    dist = np.linalg.norm(x - y, 2)
    return dist

def gaussian_emd(x, y, sigma=1.0, distance_scaling=1.0):
    """
    Gaussian kernel with squared distance in exponential term replaced by EMD
    :param x: 1D pmf
    :param y: 1D pmf
    :param sigma: standard deviation
    :param distance_scaling: scaling factor for distance matrix
    :return: Gaussian kernel with EMD
    """
    emd_value = emd(x, y, distance_scaling)
    return np.exp(-emd_value * emd_value / (2 * sigma * sigma))


def gaussian(x, y, sigma=1.0):
    x = x.astype(float)
    y = y.astype(float)
    x, y = process_tensor(x, y)
    dist = np.linalg.norm(x - y, 2)
    return np.exp(-dist * dist / (2 * sigma * sigma))


def gaussian_tv(x, y, sigma=1.0):
    # convert histogram values x and y to float, and make them equal len
    x = x.astype(float)
    y = y.astype(float)
    x, y = process_tensor(x, y)

    dist = np.abs(x - y).sum() / 2.0
    return np.exp(-dist * dist / (2 * sigma * sigma))


def kernel_parallel_unpacked(x, samples2, kernel):
    d = 0
    for s2 in samples2:
        d += kernel(x, s2)
    return d


def kernel_parallel_worker(t):
    return kernel_parallel_unpacked(*t)


def disc(samples1, samples2, kernel, is_parallel=False, *args, **kwargs):
    """
    Discrepancy between 2 samples
    :param samples1: list of samples
    :param samples2: list of samples
    :param kernel: kernel function
    :param is_parallel: whether to use parallel computation
    :param args: args for kernel
    :param kwargs: kwargs for kernel
    """
    d = 0
    if not is_parallel:
        for s1 in samples1:
            for s2 in samples2:
                d += kernel(s1, s2, *args, **kwargs)
    else:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for dist in executor.map(
                kernel_parallel_worker,
                [(s1, samples2, partial(kernel, *args, **kwargs)) for s1 in samples1],
            ):
                d += dist
    if len(samples1) * len(samples2) > 0:
        d /= len(samples1) * len(samples2)
    else:
        d = 1e6
    return d


def compute_mmd(samples1, samples2, kernel, is_hist=True, *args, **kwargs):
    """
    MMD between two samples
    :param samples1: list of samples
    :param samples2: list of samples
    :param kernel: kernel function
    :param is_hist: whether the samples are histograms or pmf
    :param args: args for kernel
    :param kwargs: kwargs for kernel
    """
    # normalize histograms into pmf
    if is_hist:
        samples1 = [s1 / np.sum(s1) for s1 in samples1]
        samples2 = [s2 / np.sum(s2) for s2 in samples2]
    return (
        disc(samples1, samples1, kernel, *args, **kwargs)
        + disc(samples2, samples2, kernel, *args, **kwargs)
        - 2 * disc(samples1, samples2, kernel, *args, **kwargs)
    )


def compute_emd(samples1, samples2, kernel, is_hist=True, *args, **kwargs):
    """
    EMD between average of two samples
    :param samples1: list of samples
    :param samples2: list of samples
    :param kernel: kernel function
    :param is_hist: whether the samples are histograms or pmf
    :param args: args for kernel
    :param kwargs: kwargs for kernel
    """
    # normalize histograms into pmf
    if is_hist:
        samples1 = [np.mean(samples1)]
        samples2 = [np.mean(samples2)]
    return disc(samples1, samples2, kernel, *args, **kwargs), [samples1[0], samples2[0]]