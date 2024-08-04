import os
import sys
import math
import random
import json
import copy
import torch
import torch.nn as nn
import torch_geometric.nn as pygnn
import torch_geometric as pyg
import torch.nn.functional as F
import networkx as nx
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt

from typing import *
from collections import defaultdict
from tqdm import tqdm
from enum import Enum
from munch import Munch
from easydict import EasyDict
from abc import abstractmethod
from torch import Tensor
from torch.types import Device
from torch_geometric.data import Data, InMemoryDataset, Dataset
from torch_geometric.loader import DataLoader
from torch_scatter import scatter


class NodeClassifyDataset(Enum):
    BASHAPES    = 'syn1'
    BACOMMUNITY = 'syn2'
    TREECYCLES  = 'syn3'
    TREEGRIDS   = 'syn4'
    CORA        = 'cora'
    CITESEER    = 'citeseer'


class GraphClassifyDataset(Enum):
    BA2MOTIFS = 'ba2'
    BA3MOTIFS = 'ba3'
    MUTAG188  = 'mutag'
    MUTAG     = 'Mutagenicity'
    BENZENE   = 'benzene'
    ALKANE    = 'alkane_carbonyl'
    FLUORIDE  = 'fluoride_carbonyl'


class TaskLevel(Enum):
    NODECLASSIFY  = 'node'
    GRAPHCLASSIFY = 'graph'


class ExplainModel(Enum):
    GNNEXPLAINER   = 'gnnexplainer'
    PGEXPLAINER    = 'pgexplainer'
    KFACTEXPLAINER = 'kfactexplainer'


class VisualizeTool(Enum):
    NETWORKX = 'networkx'
    IGRAPH   = 'igraph'


class ExplainTask(Enum):
    EDGE    = 'edge'
    FEATURE = 'feat'
    BOTH    = 'both'