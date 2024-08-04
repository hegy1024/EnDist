import argparse

from utils.typing_utils import *

class Configurations(object):
    def __init__(self):
        self.root_path = os.getcwd()

    @property
    def total_configs(self):
        return Munch(
            data       = 'syn1',           # name of dataset
            top_k      = 20,               # the rates of edge when compute fidelity
            device     = 'cpu',            # use cpu or gpu
            seed       = 2024,             # random seed
            batch_size = 64,               # batch size of dataset
            use_hook   = False,            # whether to use hook function when get preds or get embeddings
            scheduler  = None,             # option value: 'step', 'lambda', 'exponetial',
                                           #               'multistep', 'plateau'
            pretrained  = False,           # whether to retrain GNN model
            shuffle     = False,           # whether to shuffle dataset when loader dataset
            mode        = 'ed',            # candidates: 'ed'       -- vanilla explainer applied the proposed framework
                                           #             'raw'      -- vanilla explainer
                                           #             'mixup'    -- MixUpExplainer
                                           #             'proxy'    -- ProxyExplainer
                                           #             'cge'      -- CgExplainer
            hook_point            = "range(10, 100)",      # plugin point
            generator_train_point = "range(0, 50)",

            ood_explain   = True,              # whether to use proxy graph generator
            knn_loss      = True,              # whether to use consistency loss

            save_params               = False,             # whether to save model params
            save_generator            = False,             # whether to save proxy graph generator in each epoch

            plot_explanation          = False,             # draw explanation subgraph

            log_path         = osp.join(self.root_path, "log"),
            model_path       = osp.join(self.root_path, "checkpoints"),
            result_path      = osp.join(self.root_path, "result"),
            data_path        = osp.join(self.root_path, "data"),
            config_path      = osp.join(self.root_path, "configs"),
            explanation_path = osp.join(self.root_path, "result/explanations")
        )

    @property
    def gnn_configs(self):
        return Munch(
            paper          = 'gcn',
            epochs         = 300,
            shuffle        = False,
            clip_max       = 2.0,
            input_dim      = 10,
            output_dim     = 4,
            hidden_dim     = 20,
            early_stopping = 100,
        )

    @property
    def explainer_configs(self):
        return Munch(
            epochs           = 100,            # for gnnexplainer, setting epochs as 100,
                                               # while setting epochs as 30 for pgexplainer and kfactexplainer
            k                = 10,             # for kfactexplainer, setting k as 10
            coeffs     = Munch(
                reg_size       = 0.05,
                reg_ent        = 1.,
                EPS            = 1e-15,
                edge_reduction = 'sum',
                temp0          = 5.0,
                temp1          = 2.0,
                bias           = 0.0001
            )
        )

    @property
    def generator_configs(self):
        return Munch(
            K            = 30,              # for the K-nearest-neighbour, we set K as 30
            epochs       = 10,              # only train proxy graph generator 10 epochs per explainer epoch
            input_dim    = 60,
            output_dim   = 1,
            hidden_dim   = 128,
            coeffs       = Munch(
                loss_weight = 1.,
                bias        = 0.0001,
                temp0       = 5.,
                temp1       = 2.
            )
        )

    @property
    def scheduler_configs(self): 
        return Munch(
            name  = None, 
            gamma = 0.99
        )
    
    @property
    def optimizer_configs(self):
        return Munch(
            name         = 'adam',
            lr           = 0.01,        # for gnne, setting as 0.01; for pge, setting as 0.003
            weight_decay = 0.0,
            grad_fn      = None         # if set None, don't do anything, option: pc-grad
        )
    
    def parse_args(self):
        r"""
        Initialize the default configurations.
        """
        total_cfgs                    = self.total_configs
        gnn_cfgs                      = self.gnn_configs
        generator_cfgs                = self.generator_configs
        explainer_cfgs                = self.explainer_configs
        optimizer_cfgs                = self.optimizer_configs
        scheduler_cfgs                = self.scheduler_configs

        total_cfgs.explain_type       = ExplainTask.EDGE

        generator_cfgs.optimizer_cfgs = optimizer_cfgs.copy()
        generator_cfgs.scheduler_cfgs = scheduler_cfgs.copy()
        total_cfgs.generator_cfgs     = generator_cfgs

        gnn_cfgs.optimizer_cfgs       = optimizer_cfgs.copy()
        gnn_cfgs.scheduler_cfgs       = scheduler_cfgs.copy()
        total_cfgs.gnn_cfgs           = gnn_cfgs

        explainer_cfgs.optimizer_cfgs = optimizer_cfgs.copy()
        explainer_cfgs.scheduler_cfgs = scheduler_cfgs.copy()
        total_cfgs.explainer_cfgs     = explainer_cfgs

        return total_cfgs


class ConfigurationsOld(object):
    def __init__(self):
        self.root_path = os.getcwd()

    @property
    def total_configs(self):
        return Munch(
            data='syn1',  # name of dataset
            num_hops=3,  # number of hops for subgraph, as same as MixUpExplainer
            alpha=0.5,  # param of manifold mixup
            top_k=20,  # the rates of edge when compute fidelity
            epochs=20,  # number of epochs for gnn model training
            device='cpu',  # use cpu or gpu
            seed=2024,  # random seed
            batch_size=64,  # batch size of dataset
            use_hook=False,  # whether to use hook function when get preds or get embeddings
            scheduler=None,  # option: 'step', 'lambda', 'exponetial',
            #         'multistep', 'plateau'
            pretrained=False,  # whether to pretrain
            save_params=False,  # whether to save model params
            shuffle=False,  # whether to shuffle dataset when loader dataset
            mode='ood',
            hook_point="range(10, 100)",
            generator_train_point="range(0, 50)",
            use_edge_weight=False,
            remove_strategy='soft',
            ood_explain=True,
            knn_loss=False,
            log_path=osp.join(self.root_path, "log"),
            model_path=osp.join(self.root_path, "checkpoints"),
            result_path=osp.join(self.root_path, "result"),
            data_path=osp.join(self.root_path, "data"),
            config_path=osp.join(self.root_path, "configs"),
            explanation_path=osp.join(self.root_path, "result/explanations")
        )

    @property
    def gnn_configs(self):
        return Munch(
            paper='gcn',
            epochs=300,
            shuffle=False,
            clip_max=2.0,
            input_dim=10,
            output_dim=4,
            hidden_dim=20,
            early_stopping=100,
        )

    @property
    def explainer_configs(self):
        return Munch(
            epochs=100,
            k=10,
            coeffs=Munch(
                reg_size=0.05,
                reg_ent=1.,
                EPS=1e-15,
                edge_reduction='sum',
                temp0=5.0,
                temp1=2.0,
                bias=0.0001
            )
        )

    @property
    def generate_configs(self):
        return Munch(
            K=30,
            epochs=10,
            denoise_fn='mlp',
            input_dim=60,
            output_dim=1,
            hidden_dim=128,
            coeffs=Munch(
                knn_loss=1.,
                loss2=1.,
                bias=0.0001,
                temp0=5.,
                temp1=2.
            )
        )

    @property
    def scheduler_configs(self):
        return Munch(
            name=None,
            gamma=0.99
        )

    @property
    def optimizer_configs(self):
        return Munch(
            name='adam',
            lr=0.01,
            weight_decay=0.0,
            grad_fn=None  # option: pc-grad
        )

    def parse_args(self):
        r"""
        get relevance configs.
        """
        total_cfgs = self.total_configs
        gnn_cfgs = self.gnn_configs
        generator_cfgs = self.generate_configs
        explainer_cfgs = self.explainer_configs
        optimizer_cfgs = self.optimizer_configs
        scheduler_cfgs = self.scheduler_configs

        total_cfgs.explain_type = ExplainTask.EDGE

        generator_cfgs.optimizer_cfgs = optimizer_cfgs.copy()
        generator_cfgs.scheduler_cfgs = scheduler_cfgs.copy()
        total_cfgs.generator_cfgs = generator_cfgs

        gnn_cfgs.optimizer_cfgs = optimizer_cfgs.copy()
        gnn_cfgs.scheduler_cfgs = scheduler_cfgs.copy()
        total_cfgs.gnn_cfgs = gnn_cfgs

        explainer_cfgs.optimizer_cfgs = optimizer_cfgs.copy()
        explainer_cfgs.scheduler_cfgs = scheduler_cfgs.copy()
        total_cfgs.explainer_cfgs = explainer_cfgs

        return total_cfgs


def arg_parser():
    parser = argparse.ArgumentParser()
    # Experiment details
    parser.add_argument("--mode", type=str, default="raw", choices=["ed", "mixup", "proxy", "cge", "raw"],
                        help="which model to run")
    parser.add_argument("--data", type=str, default="mutag",
                        help="name of dataset which want to use")
    parser.add_argument("--backbone", type=str, default="pge", choices=["gnne", "kfact", "pge"],
                        help="name of backbone explainer")
    parser.add_argument("--device",    type=int, default=2,
                        help="index of device")
    parser.add_argument("--pretrained", action="store_true",
                        help="whether to train GNN model")
    parser.add_argument("--save_params", action="store_true",
                        help="whether to save params of explainer")
    parser.add_argument("--with_ood_loss", action="store_true",
                        help="whether to consider the distribution shift issue in the proces of training explainer")
    parser.add_argument("--read_configs", action="store_true",
                        help="load configs in files")
    parser.add_argument("--save_configs", action="store_true",
                        help="save configs to files")
    parser.add_argument("--use_edge_weight", action="store_true",
                        help="whether to use edge weight")
    parser.add_argument("--with_consistency_loss", action="store_true",
                        help="whether to use consistency_loss in explainer training process")
    parser.add_argument("--metric", type=str, default="acc",
                        help="the metric for explanations evaluation")
    parser.add_argument("--k", type=int, default=30,
                        help="the number of K-nearest-neighbour")
    parser.add_argument("--save_generator", action="store_true",
                        help="whether to save parameters of generator")

    args = parser.parse_args()
    return args