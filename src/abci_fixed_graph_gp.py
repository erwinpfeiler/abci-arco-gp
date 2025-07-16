from typing import Dict, Any

import networkx as nx
import torch.optim

from src.abci_base import ABCIBase
from src.config import ABCIFixedGraphGPConfig
from src.environments.environment import Environment
from src.mechanism_models.gp_model import GaussianProcessModel
from src.utils.graphs import dag_to_cpdag, graph_to_adj_mat
from src.utils.metrics import aid, compute_structure_metrics, mmd


class ABCIFixedGraphGP(ABCIBase):
    cfg: ABCIFixedGraphGPConfig

    def __init__(self, env: Environment = None, cfg: ABCIFixedGraphGPConfig = None,
                 param_dict: Dict[str, Any] = None):
        assert env is not None or param_dict is not None
        if param_dict is not None:
            num_workers = param_dict['cfg_param_dict']['num_workers']
            super().__init__(num_workers)
            self.load_param_dict(param_dict)
        else:
            # load (default) config
            self.cfg = ABCIFixedGraphGPConfig() if cfg is None else cfg
            super().__init__(self.cfg.num_workers, env)

            # init mechanism model
            self.mechanism_model = GaussianProcessModel(env.node_labels)
            self.graph: nx.DiGraph = None
            self.adj_mat: torch.Tensor = None

    def set_graph(self, graph: nx.DiGraph):
        self.graph = graph
        self.adj_mat = graph_to_adj_mat(self.graph, self.mechanism_model.node_labels)
        self.mechanism_model.init_mechanisms(graph)

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
                else:
                    assert False, print(f'Invalid policy {self.cfg.policy}!')

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

    def sample(self, interventions: dict, num_samples_per_graph: int):
        samples = {}
        with torch.no_grad():
            exp = self.mechanism_model.sample(interventions, 1, num_samples_per_graph, self.graph)
            for node in self.mechanism_model.node_labels:
                samples[node] = exp.data[node].reshape(-1)

            weights = torch.ones(num_samples_per_graph) / num_samples_per_graph
        return samples, weights

    def compute_stats(self):
        # compute posterior edge probs
        with torch.no_grad():
            posterior_edge_probs = self.adj_mat

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

        print('Computing AID metrics...', flush=True)

        # record AID metrics to true DAG
        aaid = aid(true_adj_mat, self.adj_mat, mode='ancestor')
        self.record_stat('aaid', aaid)

        paid = aid(true_adj_mat, self.adj_mat, mode='parent')
        self.record_stat('paid', paid)

        oset_aid = aid(true_adj_mat, self.adj_mat, mode='oset')
        self.record_stat('oset_aid', oset_aid)

        # record AID metrics to true CPDAG
        def aid_wrapper_cpdag(mode: str):
            return aid(true_cpdag, dag_to_cpdag(self.adj_mat, self.env.node_labels), mode=mode)

        aaid = aid_wrapper_cpdag(mode='ancestor')
        self.record_stat('aaid_cpdag', aaid)

        paid = aid_wrapper_cpdag(mode='parent')
        self.record_stat('paid_cpdag', paid)

        oset_aid = aid_wrapper_cpdag(mode='oset')
        self.record_stat('oset_aid_cpdag', oset_aid)

        if self.cfg.compute_distributional_stats:
            if self.env.interventional_test_data is not None:
                print(f'Computing distributional metrics on interventional test data...', flush=True)
                mean_errors = []
                mmds = []
                with torch.no_grad():
                    for eidx, exp in enumerate(self.env.interventional_test_data):
                        print(f'Computing metrics for intervention {eidx}/{len(self.env.interventional_test_data)}')
                        env_samples = torch.stack([exp.data[node].squeeze() for node in self.env.node_labels], dim=-1)
                        env_mean = env_samples.mean(dim=0)

                        # sample from learned posterior distribution
                        samples, weights = self.sample(exp.interventions, self.cfg.num_samples_per_graph)
                        samples = torch.stack([samples[node].squeeze() for node in self.env.node_labels], dim=-1)
                        posterior_mean = weights @ samples

                        mean_errors.append(posterior_mean - env_mean)
                        mmds.append(mmd(samples, env_samples, weights, bandwidth=0.2))

                mmds = torch.stack(mmds, dim=0)
                self.record_stat('mmd', mmds.mean())
                print(f'Average MMD is {mmds.mean()}')

                mean_errors = torch.stack(mean_errors, dim=0)
                dmae = mean_errors.abs().sum(dim=-1).mean()
                dmse = mean_errors.pow(2).sum(dim=-1).sqrt().mean()
                self.record_stat('dmae', dmae)
                self.record_stat('dmse', dmse)
                print(f'Average distribution mean L1 distance is {dmae}')
                print(f'Average distribution mean L2 distance is {dmse}')

    def param_dict(self) -> Dict[str, Any]:
        params = super().param_dict()
        params.update({'mechanism_model_params': self.mechanism_model.param_dict(),
                       'graph': self.graph,
                       'cfg_param_dict': self.cfg.param_dict()})
        return params

    def load_param_dict(self, param_dict):
        super().load_param_dict(param_dict)
        self.cfg = ABCIFixedGraphGPConfig(param_dict['cfg_param_dict'])
        self.graph = param_dict['graph']
        self.adj_mat = graph_to_adj_mat(self.graph, self.env.node_labels)
        self.mechanism_model = GaussianProcessModel(param_dict=param_dict['mechanism_model_params'])
        self.mechanism_model.set_data(self.experiments)

    @classmethod
    def load(cls, path):
        param_dict = torch.load(path, weights_only=False)
        return ABCIFixedGraphGP(param_dict=param_dict)
