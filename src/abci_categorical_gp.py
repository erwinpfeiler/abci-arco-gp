from typing import Callable, List, Dict, Any

import networkx as nx
import torch.optim

from src.abci_base import ABCIBase
from src.config import ABCICategoricalGPConfig
from src.environments.environment import Environment
from src.environments.experiment import Experiment
from src.experimental_design.exp_designer_abci_categorical_gp import ExpDesignerABCICategoricalGP
from src.graph_models.categorical_model import CategoricalModel
from src.mechanism_models.gp_model import GaussianProcessModel
from src.utils.graphs import dag_to_cpdag
from src.utils.metrics import aid, compute_structure_metrics


class ABCICategoricalGP(ABCIBase):
    cfg: ABCICategoricalGPConfig

    def __init__(self, env: Environment = None, cfg: ABCICategoricalGPConfig = None,
                 param_dict: Dict[str, Any] = None):
        assert env is not None or param_dict is not None
        if param_dict is not None:
            num_workers = param_dict['cfg_param_dict']['num_workers']
            super().__init__(num_workers)
            self.load_param_dict(param_dict)
        else:
            # load (default) config
            self.cfg = ABCICategoricalGPConfig() if cfg is None else cfg
            super().__init__(self.cfg.num_workers, env)

            # init models
            self.graph_prior = CategoricalModel(self.env.node_labels)
            self.graph_posterior = CategoricalModel(self.env.node_labels)
            self.mechanism_model = GaussianProcessModel(env.node_labels)

            # init mechanisms for all graphs
            for graph in self.graph_prior.graphs:
                self.mechanism_model.init_mechanisms(graph)

    def experiment_designer_factory(self):
        distributed = self.cfg.num_workers > 1
        return ExpDesignerABCICategoricalGP(self.env.intervention_bounds, opt_strategy='gp-ucb',
                                            distributed=distributed)

    def run(self):
        for epoch in range(self.cfg.num_experiments):
            # pick intervention/data according to policy
            if self.cfg.policy == 'static-obs-dataset':
                # stop after one epoch on static dataset
                if epoch > 0:
                    break
                print(f'Load static observational training data from environment...', flush=True)
                assert self.env.observational_train_data is not None
                self.experiments.extend(self.env.observational_train_data)
            elif self.cfg.policy == 'static-intr-dataset':
                # stop after one epoch on static dataset
                if epoch > 0:
                    break
                print(f'Load static interventional training data from environment...', flush=True)
                # if available, load observational training data as well
                if self.env.observational_train_data is not None:
                    self.experiments.extend(self.env.observational_train_data)
                # load interventional training data
                assert self.env.interventional_train_data is not None
                self.experiments.extend(self.env.interventional_train_data)
            else:
                print(f'Starting experiment cycle {epoch + 1}/{self.cfg.num_experiments}...')
                print(f'Design and perform experiment...', flush=True)
                if self.cfg.policy == 'observational' or len(
                        self.experiments) == 0 and self.cfg.num_initial_obs_samples > 0:
                    interventions = {}
                elif self.cfg.policy == 'random':
                    interventions = self.get_random_intervention()
                elif self.cfg.policy == 'random-fixed-value':
                    interventions = self.get_random_intervention(0.)
                elif self.cfg.policy == 'oracle':
                    interventions = self.get_oracle_intervention(self.cfg.batch_size)
                else:
                    if self.cfg.policy == 'graph-info-gain' or self.cfg.policy == 'scm-info-gain':
                        args = {'mechanism_model': self.mechanism_model,
                                'graph_posterior': self.graph_posterior,
                                'batch_size': self.cfg.batch_size,
                                'num_exp_per_graph': self.cfg.num_exp_per_graph,
                                'policy': self.cfg.policy,
                                'mode': 'full'}
                    elif self.cfg.policy == 'intervention-info-gain':
                        outer_mc_graphs, outer_log_weights = self.graph_posterior.get_mc_graphs(self.cfg.outer_mc_mode,
                                                                                                self.cfg.num_outer_mc_graphs)
                        args = {'mechanism_model': self.mechanism_model,
                                'graph_posterior': self.graph_posterior,
                                'experiments': self.experiments,
                                'interventional_queries': self.env.cfg.interventional_queries,
                                'outer_mc_graphs': outer_mc_graphs,
                                'outer_log_weights': outer_log_weights,
                                'num_mc_queries': self.cfg.num_mc_queries,
                                'num_batches_per_query': self.cfg.num_batches_per_query,
                                'batch_size': self.cfg.batch_size,
                                'num_exp_per_graph': self.cfg.num_exp_per_graph,
                                'policy': self.cfg.policy,
                                'mode': 'full'}
                    else:
                        assert False, print(f'Invalid policy {self.cfg.policy}!')

                    if self.cfg.num_workers > 1:
                        interventions = self.design_experiment_distributed(args)
                    else:
                        designer = self.experiment_designer_factory()
                        designer.init_design_process(args)
                        interventions = designer.get_best_experiment(self.env.intervenable_nodes)

                # perform experiment
                num_experiments_conducted = len(self.experiments)
                batch_size = self.cfg.batch_size
                if num_experiments_conducted == 0 and self.cfg.num_initial_obs_samples > 0:
                    batch_size = self.cfg.num_initial_obs_samples
                self.experiments.append(self.env.sample(interventions, batch_size))

            # clear caches
            self.mechanism_model.clear_prior_mll_cache()
            self.mechanism_model.clear_posterior_mll_cache()
            self.mechanism_model.clear_rmse_cache()

            # update model
            print(f'Updating model...', flush=True)
            self.mechanism_model.update_gp_hyperparameters(self.experiments)
            self.graph_posterior = self.compute_graph_posterior(self.experiments)

            # save model checkpoints
            if (epoch + 1) % self.cfg.checkpoint_interval == 0 or epoch == self.cfg.num_experiments - 1:
                print('Creating model checkpoint and logging stats...')
                self.save_model()
                self.compute_stats()
                self.export_stats()
            elif self.cfg.log_interval > 0 and epoch % self.cfg.log_interval == 0:
                # print log output
                print('Logging stats...')
                self.compute_stats()
                self.export_stats()
                print(f'Experiment {epoch + 1}/{self.cfg.num_experiments},'
                      f'ESHD is {self.stats["eshd"][-1].item():.2f}, '
                      f'A-AID is {self.stats["aaid"][-1].item():.2f}', flush=True)

    def get_oracle_intervention(self, num_samples: int, num_candidates_per_node: int = 10):
        current_entropy = self.graph_posterior.entropy()

        self.experiments.append(self.env.sample({}, num_samples))
        posterior = self.compute_graph_posterior(self.experiments)
        best_intervention = {}
        best_info_gain = current_entropy - posterior.entropy()
        best_posterior = posterior

        for node in self.env.node_labels:
            bounds = self.env.intervention_bounds[node]
            candidates = torch.linspace(bounds[0], bounds[1], num_candidates_per_node)
            for i in range(num_candidates_per_node):
                self.experiments[-1] = self.env.sample({node: candidates[i]}, num_samples)
                posterior = self.compute_graph_posterior(self.experiments)
                info_gain = current_entropy - posterior.entropy()
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_intervention = {node: candidates[i]}
                    best_posterior = posterior

        self.graph_posterior = best_posterior
        return best_intervention

    def compute_graph_posterior(self, experiments: List[Experiment]) -> CategoricalModel:
        posterior = CategoricalModel(self.env.node_labels)
        self.mechanism_model.clear_prior_mll_cache()
        with torch.no_grad():
            for graph in posterior.graphs:
                mll = self.mechanism_model.mll(experiments, graph, prior_mode=True, use_cache=True).squeeze()
                log_prob = self.mechanism_model.log_hp_prior(graph)
                log_prob += mll + self.graph_prior.log_prob(graph)

                posterior.set_log_prob(log_prob, graph)

            posterior.normalize()
        return posterior

    def graph_posterior_expectation(self, func: Callable[[nx.DiGraph], torch.Tensor], logspace=False):
        with torch.no_grad():
            # compute function values
            func_values = [func(graph) for graph in self.graph_posterior.graphs]
            func_output_shape = func_values[0].shape
            func_output_dim = len(func_output_shape)
            func_values = torch.stack(func_values).view(self.graph_posterior.num_graphs, *func_output_shape)

            # compute expectation
            log_graph_weights = torch.tensor([self.graph_posterior.log_prob(graph) for graph in
                                              self.graph_posterior.graphs])
            log_graph_weights = log_graph_weights.view(self.graph_posterior.num_graphs, *([1] * func_output_dim))

            if logspace:
                expected_value = (log_graph_weights + func_values).logsumexp(dim=0)
                return expected_value

            expected_value = (log_graph_weights.exp() * func_values).sum(dim=0)
            return expected_value

    def compute_stats(self):
        # compute posterior edge probs
        with torch.no_grad():
            posterior_edge_probs = self.graph_posterior.edge_probs()

        print('Computing structure metrics (AUROC, AUPRC, SHD,...)', flush=True)
        # record expected number of edges
        enum_edges = posterior_edge_probs.sum()
        self.record_stat('enum_edges', enum_edges)

        # DAG metrics
        true_adj_mat = self.env.get_adj_mat()
        stats = compute_structure_metrics(true_adj_mat, posterior_edge_probs, dag_to_cpdag=False)
        for stat_name, value in stats.items():
            self.record_stat(stat_name, value)

        # CPDAG metrics
        true_cpdag = self.env.get_cpdag()
        stats = compute_structure_metrics(true_cpdag, posterior_edge_probs, dag_to_cpdag=True)
        for stat_name, value in stats.items():
            self.record_stat(stat_name + '_cpdag', value)

        # record graph posterior entropy
        self.record_stat('graph_entropy', self.graph_posterior.entropy().detach())

        # record env graph LL
        self.record_stat('graph_ll', self.graph_posterior.log_prob(self.env.graph))

        print('Computing AID metrics...', flush=True)

        # record AID metrics to true DAG
        def aid_wrapper(g: nx.DiGraph, mode: str):
            adj_mat = self.graph_prior.graph_to_adj_mat(g)
            return aid(true_adj_mat, adj_mat, mode=mode)

        aaid = self.graph_posterior_expectation(lambda g: aid_wrapper(g, mode='ancestor'))
        self.record_stat('aaid', aaid)

        paid = self.graph_posterior_expectation(lambda g: aid_wrapper(g, mode='parent'))
        self.record_stat('paid', paid)

        oset_aid = self.graph_posterior_expectation(lambda g: aid_wrapper(g, mode='oset'))
        self.record_stat('oset_aid', oset_aid)

        # record AID metrics to true CPDAG
        def aid_wrapper_cpdag(g: nx.DiGraph, mode: str):
            return aid(true_cpdag, dag_to_cpdag(g, self.env.node_labels), mode=mode)

        aaid = self.graph_posterior_expectation(lambda g: aid_wrapper_cpdag(g, mode='ancestor'))
        self.record_stat('aaid_cpdag', aaid)

        paid = self.graph_posterior_expectation(lambda g: aid_wrapper_cpdag(g, mode='parent'))
        self.record_stat('paid_cpdag', paid)

        oset_aid = self.graph_posterior_expectation(lambda g: aid_wrapper_cpdag(g, mode='oset'))
        self.record_stat('oset_aid_cpdag', oset_aid)

    def param_dict(self) -> Dict[str, Any]:
        params = super().param_dict()
        params.update({'mechanism_model_params': self.mechanism_model.param_dict(),
                       'graph_prior_params': self.graph_prior.param_dict(),
                       'graph_posterior_params': self.graph_posterior.param_dict(),
                       'cfg_param_dict': self.cfg.param_dict()})
        return params

    def load_param_dict(self, param_dict):
        super().load_param_dict(param_dict)
        self.cfg = ABCICategoricalGPConfig(param_dict['cfg_param_dict'])
        self.graph_prior = CategoricalModel(self.env.node_labels)
        self.graph_prior.load_param_dict(param_dict['graph_prior_params'])
        self.graph_posterior = CategoricalModel(self.env.node_labels)
        self.graph_posterior.load_param_dict(param_dict['graph_posterior_params'])
        self.mechanism_model = GaussianProcessModel(param_dict=param_dict['mechanism_model_params'])
        self.mechanism_model.set_data(self.experiments)

    @classmethod
    def load(cls, path):
        param_dict = torch.load(path, weights_only=False)
        return ABCICategoricalGP(param_dict=param_dict)
