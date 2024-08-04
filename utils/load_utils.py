import torch.cuda
from torch_geometric.datasets import TUDataset

from explain import (
    EnDistExplainer,
    GNNExplainer,
    PGExplainer,
    KFactExplainer,
    MixUpExplainer,
    ProxyExplainer,
    CGExplainer
)
from generate import DenoiseNet, ExplanationGenerator
from datasets import (
    Mutagenicity,
    Benzene,
    AlkaneCarbonyl,
    FluorideCarbonyl
)
from datasets.transform import (
    pre_process_mutag,
    pre_transform_mutag,
    normalize,
    padding_nodes
)
from utils.argparse_utils import *
from utils.io_utils import read_arguments

def load_dataset(data_name, root='./data', *args) -> Dataset:
    if GraphClassifyDataset[data_name].value == 'Mutagenicity':
        dataset = Mutagenicity(root, pre_transform=pre_transform_mutag)
        # dataset = Mutagenicity(root)
    elif GraphClassifyDataset[data_name].value == 'mutag':
        dataset = TUDataset(root, name='MUTAG', pre_transform=pre_process_mutag)
        dataset.type = 'graph'
        dataset.name = 'MUTAG188'
    elif GraphClassifyDataset[data_name].value == 'benzene':
        dataset = Benzene(root)
    elif GraphClassifyDataset[data_name].value == 'alkane_carbonyl':
        dataset = AlkaneCarbonyl(root, pre_transform=padding_nodes)
    elif GraphClassifyDataset[data_name].value == 'fluoride_carbonyl':
        dataset = FluorideCarbonyl(root)
    else:
        raise NotImplementedError(f"{GraphClassifyDataset[data_name].value} is not Implemented")
    return dataset

def load_explainer(cfgs: Munch, model: nn.Module, dataset: InMemoryDataset):
    if cfgs.mode in ['ed', 'raw']:
        denoise_model = DenoiseNet(cfgs.gnn_cfgs.hidden_dim, 64, 64)

        if cfgs.explainer == ExplainModel.PGEXPLAINER:
            return EnDistExplainer(
                model           = model,
                algorithm       = PGExplainer(cfgs.device,
                                              cfgs.explainer_cfgs.epochs,
                                              dataset.type,
                                              **cfgs.explainer_cfgs.coeffs),
                generator       = ExplanationGenerator(denoise_model,
                                                       cfgs.device,
                                                       cfgs.generator_cfgs.epochs,
                                                       **cfgs.generator_cfgs.coeffs),
                dataset         = dataset,
                cfgs            = cfgs,
                explain_type    = cfgs.explain_type,
                use_edge_weight = cfgs.use_edge_weight
            )
        elif cfgs.explainer == ExplainModel.GNNEXPLAINER:
            return EnDistExplainer(
                model           = model,
                algorithm       = GNNExplainer(cfgs.device,
                                               dataset.type,
                                               **cfgs.explainer_cfgs.coeffs),
                generator       = ExplanationGenerator(denoise_model,
                                                       cfgs.device,
                                                        cfgs.generator_cfgs.epochs,
                                                       **cfgs.generator_cfgs.coeffs),
                dataset         = dataset,
                cfgs            = cfgs,
                explain_type    = cfgs.explain_type,
                use_edge_weight = cfgs.use_edge_weight
            )
        elif cfgs.explainer == ExplainModel.KFACTEXPLAINER:
            return EnDistExplainer(
                model           = model,
                algorithm       = KFactExplainer(cfgs.device,
                                                 cfgs.explainer_cfgs.epochs,
                                                 dataset.type,
                                                 cfgs.explainer_cfgs.k,
                                                 **cfgs.explainer_cfgs.coeffs),

                generator       = ExplanationGenerator(denoise_model,
                                                       cfgs.device,
                                                       cfgs.generator_cfgs.epochs,
                                                       **cfgs.generator_cfgs.coeffs),
                dataset         = dataset,
                cfgs            = cfgs,
                explain_type    = cfgs.explain_type,
                use_edge_weight = cfgs.use_edge_weight
            )
        else:
            raise NotImplementedError
    elif cfgs.mode == 'mixup':
        if cfgs.explainer == ExplainModel.PGEXPLAINER:
            return MixUpExplainer(
                model           = model,
                algorithm       = PGExplainer(cfgs.device,
                                              cfgs.explainer_cfgs.epochs,
                                              dataset.type,
                                              **cfgs.explainer_cfgs.coeffs),
                dataset         = dataset,
                cfgs            = cfgs,
                explain_type    = cfgs.explain_type,
                use_edge_weight = cfgs.use_edge_weight
            )
        elif cfgs.explainer == ExplainModel.GNNEXPLAINER:
            return MixUpExplainer(
                model           = model,
                algorithm       = GNNExplainer(cfgs.device,
                                               dataset.type,
                                               **cfgs.explainer_cfgs.coeffs),
                dataset         = dataset,
                cfgs            = cfgs,
                explain_type    = cfgs.explain_type,
                use_edge_weight = cfgs.use_edge_weight
            )
        elif cfgs.explainer == ExplainModel.KFACTEXPLAINER:
            return MixUpExplainer(
                model           = model,
                algorithm       = KFactExplainer(cfgs.device,
                                                 cfgs.explainer_cfgs.epochs,
                                                 dataset.type,
                                                 cfgs.explainer_cfgs.k,
                                                 **cfgs.explainer_cfgs.coeffs),
                dataset         = dataset,
                cfgs            = cfgs,
                explain_type    = cfgs.explain_type,
                use_edge_weight = cfgs.use_edge_weight
            )
        else:
            raise NotImplementedError
    elif cfgs.mode == 'proxy':
        if cfgs.explainer == ExplainModel.PGEXPLAINER:
            return ProxyExplainer(
                model           = model,
                algorithm       = PGExplainer(cfgs.device,
                                              cfgs.explainer_cfgs.epochs,
                                              dataset.type,
                                              **cfgs.explainer_cfgs.coeffs),
                dataset         = dataset,
                cfgs            = cfgs,
                use_edge_weight = cfgs.use_edge_weight
            )
        elif cfgs.explainer == ExplainModel.GNNEXPLAINER:
            return ProxyExplainer(
                model           = model,
                algorithm       = GNNExplainer(cfgs.device,
                                               dataset.type,
                                               **cfgs.explainer_cfgs.coeffs),
                dataset         = dataset,
                cfgs            = cfgs,
                use_edge_weight = cfgs.use_edge_weight
            )
        elif cfgs.explainer == ExplainModel.KFACTEXPLAINER:
            return ProxyExplainer(
                model           = model,
                algorithm       = KFactExplainer(cfgs.device,
                                                 cfgs.explainer_cfgs.epochs,
                                                 dataset.type,
                                                 cfgs.explainer_cfgs.k,
                                                 **cfgs.explainer_cfgs.coeffs),
                dataset         = dataset,
                cfgs            = cfgs,
                use_edge_weight = cfgs.use_edge_weight
            )
        else:
            raise NotImplementedError
    elif cfgs.mode == 'cge':
        cge_explainer = GNNExplainer(
            cfgs.device, dataset.type, **cfgs.cge_cfgs.coeffs
        )
        if cfgs.explainer == ExplainModel.PGEXPLAINER:
            return CGExplainer(
                model           = model,
                algorithm       = [PGExplainer(cfgs.device,
                                               cfgs.explainer_cfgs.epochs,
                                               dataset.type,
                                               **cfgs.explainer_cfgs.coeffs),
                                   cge_explainer],
                dataset         = dataset,
                cfgs            = cfgs)
        elif cfgs.explainer == ExplainModel.GNNEXPLAINER:
            return CGExplainer(
                model           = model,
                algorithm       = [GNNExplainer(cfgs.device,
                                                dataset.type,
                                                **cfgs.explainer_cfgs.coeffs),
                                   cge_explainer],
                dataset         = dataset,
                cfgs            = cfgs)
        elif cfgs.explainer == ExplainModel.KFACTEXPLAINER:
            return CGExplainer(
                model           = model,
                algorithm       = [KFactExplainer(cfgs.device,
                                                  cfgs.explainer_cfgs.epochs,
                                                  dataset.type,
                                                  cfgs.explainer_cfgs.k,
                                                  **cfgs.explainer_cfgs.coeffs),
                                   cge_explainer],
                dataset         = dataset,
                cfgs            = cfgs)
    else:
        raise NotImplementedError

def load_data_name(name: str):
    if name == "mutag":
        return GraphClassifyDataset.MUTAG
    elif name == "mutag188":
        return GraphClassifyDataset.MUTAG188
    elif name == "benz":
        return GraphClassifyDataset.BENZENE
    elif name == 'car1':
        return GraphClassifyDataset.ALKANE
    elif name == 'car2':
        return GraphClassifyDataset.FLUORIDE
    else:
        raise NotImplementedError

def load_explainer_name(name: str):
    if name == "pge":
        return ExplainModel.PGEXPLAINER
    elif name == "gnne":
        return ExplainModel.GNNEXPLAINER
    elif name == "kfact":
        return ExplainModel.KFACTEXPLAINER
    else:
        raise NotImplementedError

def load_indices(dataset):
    if dataset == GraphClassifyDataset.MUTAG:
        return list(range(1000))
    elif dataset == GraphClassifyDataset.BENZENE:
        return list(range(1000))
    elif dataset == GraphClassifyDataset.ALKANE:
        return list(range(1000))
    elif dataset == GraphClassifyDataset.FLUORIDE:
        return list(range(1000))
    else:
        raise NotImplementedError

def load_device(device):
    return f'cuda:{device}' if device != -1 and torch.cuda.is_available() else 'cpu'

def load_arguments(args):
    r"""
    load arguments from argparse_utils and file.    
    """
    cfgs       = Configurations()
    total_cfgs = cfgs.parse_args()
    total_cfgs.data      = load_data_name(args.data)
    total_cfgs.explainer = load_explainer_name(args.backbone)
    if args.read_configs:
        total_cfgs       = read_arguments(total_cfgs)
    total_cfgs.device    = load_device(args.device)

    total_cfgs.mode                     = args.mode
    total_cfgs.ood_explain              = args.with_distribution_shift
    total_cfgs.knn_loss                 = args.with_consistency_loss
    total_cfgs.pretrained               = args.pretrained
    total_cfgs.save_params              = args.save_params
    total_cfgs.metric                   = args.metric
    total_cfgs.generator_cfgs.K         = args.k
    total_cfgs.save_generator           = args.save_generator

    return total_cfgs