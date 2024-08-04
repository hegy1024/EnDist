import argparse
from experiments import Experiments
import time

from utils.load_utils import load_dataset, load_arguments, load_explainer, load_indices
from utils.typing_utils import *
from utils.model_utils import fix_seed, GNNTrainer
from evaluations import ExplanationEvaluator
from utils.io_utils import save_arguments, setup_logger, save_explanations

def arg_parser():
    parser = argparse.ArgumentParser()
    # Experiment details
    parser.add_argument("--mode", type=str, default="raw", choices=["ed", "mixup", "proxy", "cge", "raw"],
                        help="which model to run")
    parser.add_argument("--data", type=str, default="mutag",
                        help="name of dataset which want to use")
    parser.add_argument("--backbone", type=str, default="pge", choices=["gnne", "kfact", "pge"],
                        help="name of backbone explainer")
    parser.add_argument("--device",    type=int, default=0,
                        help="index of device")
    parser.add_argument("--pretrained", action="store_false",
                        help="whether to train GNN model")
    parser.add_argument("--save_params", action="store_true",
                        help="whether to save params of explainer")
    parser.add_argument("--with_distribution_shift", action="store_false",
                        help="whether to consider the distribution shift issue in the proces of training explainer")
    parser.add_argument("--read_configs", action="store_true",
                        help="load configs in files")
    parser.add_argument("--save_configs", action="store_true",
                        help="save configs to files")
    parser.add_argument("--use_edge_weight", action="store_true",
                        help="whether to use edge weight")
    parser.add_argument("--with_consistency_loss", action="store_false",
                        help="whether to use consistency_loss in explainer training process")
    parser.add_argument("--metric", type=str, default="acc",
                        help="the metric for explanations evaluation")
    parser.add_argument("--k", type=int, default=30,
                        help="the number of K-nearest-neighbour")
    parser.add_argument("--save_generator", action="store_true",
                        help="whether to save parameters of generator")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args       = arg_parser()
    experiment = Experiments(args)
    experiment.run()