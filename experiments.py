import time

from utils.typing_utils import *
from utils.load_utils import load_dataset, load_arguments, load_explainer, load_indices
from utils.model_utils import fix_seed, GNNTrainer
from evaluations import ExplanationEvaluator
from utils.io_utils import setup_logger, save_explanations

def load_save_name(cfgs):
    if cfgs.mode == 'ed':
        save_name = [f"EnDist_{cfgs.explainer.value}_without_consistency",
                     f"EnDist_{cfgs.explainer.value}_within_consistency"][int(cfgs.knn_loss)]
        save_name += "_within_generator" if cfgs.ood_explain else "_without_generator"
    else:
        save_name = f"{cfgs.mode}_{cfgs.explainer.value}"

    return save_name

class Experiments(object):
    r"""
    Experiment for 2 part for both proposed framework and baseline:
    1. Accuracy:        compute auc-roc metric
    2. Robust Fidelity: compute robust fidelity value
    """
    def __init__(self, args):
        self.cfgs = load_arguments(args)

        # initialize default parameters
        if args.mode == 'ed':
            # for the proposed framework
            self.initialize4ed()
        else:
            # for the baseline
            self.initialize4other()

        self.seeds = range(6, 11)

        self.results         : Dict[str, List] = defaultdict(list)
        self.all_explanations: Dict[int] = dict()
        self.all_explainers  : Dict[int] = dict()

    def initialize4ed(self):
        """
        Initialize parameters for the proposed EnDist model.
        """
        self.cfgs.generator_train_point = eval(self.cfgs.generator_train_point)
        self.cfgs.hook_point            = eval(self.cfgs.hook_point)

        self.logger = setup_logger(
            "EnDist Result: " + load_save_name(self.cfgs),
            osp.join(self.cfgs.result_path, "metrics", self.cfgs.data.value),
            filename=f"EnDist_{self.cfgs.explainer.value}.txt"
        )

    def initialize4other(self):
        """
        Initialize parameters for other model, include mixupexplainer, proxy explainer and cg_explainer.
        """
        if self.cfgs.mode == 'cge':
            # for cge explainer, we need to load hyperparameter of gnnexplainer for the M step
            apply_cfgs_path = osp.join(self.cfgs.config_path, f"gnnexplainer/{self.cfgs.data.value}.json")
            with open(apply_cfgs_path, 'r', encoding='utf-8') as f:
                apply_infos = Munch.fromDict(json.load(f))
            self.cfgs.cge_cfgs = apply_infos.explainer_cfgs

        if self.cfgs.mode == 'raw':
            self.cfgs.ood_explain = False
            self.cfgs.knn_loss    = False

        # default setting for cge explainer
        self.cfgs.graph_ratio_1 = 0.5
        self.cfgs.net_ratio_1   = 0.0001
        self.cfgs.lth_epochs    = 200

        # default setting for mixup explainer
        self.cfgs.yita = 5

        # default setting for proxy explainer
        self.cfgs.generator_cfgs.hidden_channels   = 64
        self.cfgs.generator_cfgs.out_channels      = 16
        self.cfgs.generator_cfgs.epochs            = 10
        self.cfgs.generator_cfgs.optimizer_cfgs.lr = 0.0003

        self.logger = setup_logger(
            "Baseline Result: " + load_save_name(self.cfgs),
            osp.join(self.cfgs.result_path, "metrics", self.cfgs.data.value),
            filename=f"{self.cfgs.mode}_{self.cfgs.explainer.value}.txt"
        )

    def pipeline(self):
        dataset = load_dataset(self.cfgs.data.name, self.cfgs.data_path)
        trainer = GNNTrainer(self.cfgs, dataset)
        model   = trainer(self.cfgs.model_path, print_checkpoint=True, pretrained=self.cfgs.pretrained)
        # set to eval
        model.to(self.cfgs.device)
        model.eval()

        for seed in self.seeds:
            fix_seed(seed)
            # load evaluator
            evaluator = ExplanationEvaluator(model)

            # load explainer and indices
            explainer = load_explainer(self.cfgs, model, dataset)
            indices   = load_indices(self.cfgs.data)

            # train explainer
            evaluator.eval_start('train')
            explainer.prepare(indices, seed=seed)
            evaluator.eval_finish('train')
            self.results['train_time'].append(evaluator.time_consume('train') / len(indices))

            # test explainer
            evaluator.eval_start('test')
            explanations = explainer(indices, sifting=False)
            evaluator.eval_finish('test')
            self.results['test_time'].append(evaluator.time_consume('test') / len(explanations))

            # calculate metric
            pbar = tqdm(explanations, desc="Evaluation Explanations", leave=False)
            for idx, explanation in enumerate(pbar):
                kwargs = {}
                if self.cfgs.metric == 'fid':
                    if self.cfgs.data == GraphClassifyDataset.MUTAG:
                        max_edges           = int(explanation.num_edges * self.cfgs.topk / 100)
                        edge_indices        = explanation.edge_mask.topk(max_edges)[-1]
                        kwargs['edge_list'] = explanation.edge_index[:, edge_indices].T.tolist()
                    else:
                        kwargs['max_nodes'] = (explanation.edge_index[:, explanation.edge_mask == 1]
                                               .flatten()
                                               .unique()
                                               .size(0))
                        if (explanation.edge_mask == 1).sum(-1) == 0:
                            # don't compute for data without ground truth
                            kwargs['compute_fid'] = False
                evaluator.collect(explanation, self.cfgs.metric, **kwargs)

                if self.cfgs.plot_explanation and (explanation.ground_truth_mask == 1).sum(-1) != 0:
                    num_edges = (explanation.ground_truth_mask == 1).sum(-1)
                    # plot explanation
                    explanation.plot_explanation_subgraph(topk_edges=num_edges,
                                                          save_name=f"{idx}_final_expl.jpg",
                                                          save_path=osp.join(self.cfgs.explanation_path,
                                                                             "plot",
                                                                             self.cfgs.data.value,
                                                                             f"{explainer}",
                                                                             f"seed_{seed}"))
                    # plot ground truth
                    explanation.plot_ground_truth_subgraph(save_name=f"{idx}_ground_truth_expl.jpg",
                                                           save_path=osp.join(self.cfgs.explanation_path,
                                                                              "plot",
                                                                              self.cfgs.data.value,
                                                                              f"{explainer}",
                                                                              f"seed_{seed}"))

            evaluator.get_summarized_results()

            # accuracy of explanation
            self.results['acc'].append(evaluator.accuracy)

            # other results: robust fidelity
            for name, value in evaluator.results_dict.items():
                self.results[name].append(value[0])

            self.all_explanations[seed] = {"Accuracy": f"{evaluator.accuracy:.6f}",
                                           "Explanation": explanations}
            self.all_explainers[seed]   = explainer

            if len(self.results['best_seed']) == 0 or self.results['best_seed'][1] < evaluator.accuracy:
                self.results['best_seed'] = [seed, evaluator.accuracy]

    def logging(self):
        self.logger.info(f"Result at {time.ctime()}"
                         f"\n================================================================================\n"
                         f"-------------------------------- Params -------------------------------------")
        self.logger.info(f"Dataset: {self.cfgs.data.value:<10} Backbone: {self.cfgs.explainer.value:<10} "
                         f"Seeds: {str(self.seeds):<10}")
        self.logger.info(f"Hook Point: {str(self.cfgs.hook_point):<10} "
                         f"Generator Train Point: {str(self.cfgs.generator_train_point):<10}")
        self.logger.info(f"Explainer lr: {self.cfgs.explainer_cfgs.optimizer_cfgs.lr:<10}"
                         f"Generator lr: {self.cfgs.generator_cfgs.optimizer_cfgs.lr:<10}")
        self.logger.info(f"Explainer Epoch: {self.cfgs.explainer_cfgs.epochs:<10}"
                         f"Generator Epoch: {self.cfgs.generator_cfgs.epochs:<10}")
        self.logger.info(f"Reg ent: {self.cfgs.explainer_cfgs.coeffs.reg_ent:<10}"
                         f"Reg size: {self.cfgs.explainer_cfgs.coeffs.reg_size:<10}")
        self.logger.info(f"Explainer Temp0: {self.cfgs.explainer_cfgs.coeffs.temp0:<10}"
                         f"Explainer Temp1: {self.cfgs.explainer_cfgs.coeffs.temp1:<10}")
        self.logger.info(f"Generator Temp0: {self.cfgs.generator_cfgs.coeffs.temp0:<10}"
                         f"Generator Temp1: {self.cfgs.generator_cfgs.coeffs.temp1:<10}\n"
                         f"-------------------------------- Result -------------------------------------")

        # compute average and standard deviation value, then output finial result to the given file
        for i, (name, values) in enumerate(self.results.items()):
            self.logger.info(f"{i}. {name:<10}: (Mean){np.mean(values):<10.6f}  (Var){np.std(values):<10.6f}")
        for seed, accuracy in zip(self.seeds, self.results['acc']):
            self.logger.info(f"Seed: {seed} Result: {accuracy:.6f}")

        self.logger.info(f"Best Seed: {self.results['best_seed'][0]} "
                         f"Value: {self.results['best_seed'][1]:.6f}"
                         f"\n================================================================================\n")

    def save(self):
        best_seed = self.results['best_seed'][0]
        save_name = load_save_name(self.cfgs)

        # save best explanation
        save_explanations(self.all_explanations[best_seed],
                          self.all_explainers[best_seed],
                          self.cfgs.explanation_path,
                          self.cfgs.data.value,
                          f"{save_name}_best_explanations")

        # save best explainer
        if self.cfgs.save_params:
            self.all_explainers[best_seed].save_parameters(save_name=f"{save_name}_best_explainer")

        # save all explanation and explainer
        for seed in self.seeds:
            save_explanations(self.all_explanations[seed],
                              self.all_explainers[seed],
                              self.cfgs.explanation_path,
                              self.cfgs.data.value,
                              save_name=f"{save_name}_explanations_{seed}")
            if self.cfgs.save_params:
                self.all_explainers[seed].save_parameters(save_name=f"{save_name}_{seed}")

    def run(self):
        # experiment process
        self.pipeline()

        # logging result and config
        self.logging()

        # save explainer and explanations
        self.save()
