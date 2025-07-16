import math
from typing import Callable, List, Dict, Any

import networkx as nx
import torch
import torch.optim
from torch.nn.functional import log_softmax

from src.abci_base import ABCIBase
from src.config import ABCIDiBSGPConfig, DiBSConfig
from src.environments.environment import Environment
from src.environments.experiment import Experiment
from src.graph_models.dibs_model import DiBSModel
from src.mechanism_models.gp_model import get_unique_mechanisms
from src.mechanism_models.shared_data_gp_model import SharedDataGaussianProcessModel
from src.utils.graphs import dag_to_cpdag
from src.utils.metrics import compute_structure_metrics, aid, mmd
from src.utils.utils import inf_tensor


class ABCIDiBSGP(ABCIBase):
    cfg: ABCIDiBSGPConfig

    def __init__(self, env: Environment = None, cfg: ABCIDiBSGPConfig = None,
                 param_dict: Dict[str, Any] = None):
        assert env is not None or param_dict is not None
        if param_dict is not None:
            num_workers = param_dict['cfg_param_dict']['num_workers']
            super().__init__(num_workers)
            self.load_param_dict(param_dict)
        else:
            # load (default) config
            self.cfg = ABCIDiBSGPConfig() if cfg is None else cfg
            super().__init__(self.cfg.num_workers, env)

            # init models
            dibs_cfg = DiBSConfig()
            dibs_cfg.num_particles = self.cfg.num_particles
            self.graph_model = DiBSModel(env.node_labels, dibs_cfg)
            self.mechanism_model = SharedDataGaussianProcessModel(env.node_labels)
            self.sample_time = 0

    def experiment_designer_factory(self):
        raise NotImplementedError

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

                # resample particles
                num_experiments_conducted = len(self.experiments)
                if num_experiments_conducted > 0:
                    self.resample_particles(use_cache=True)

                # perform experiment
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
            self.update_graph_model()

            print(f'There are currently {self.mechanism_model.get_num_gps()} unique GPs and ...')
            print(f'... {len(self.mechanism_model.topological_orders)} topological orders in our model...')

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

    def resample_particles(self, use_cache: bool = False):
        mc_graphs, mc_adj_mats = self.sample_mc_graphs()
        num_particles, num_mc_graphs = mc_adj_mats.shape[0:2]

        max_particles_to_keep = math.ceil(num_particles * self.cfg.resampling_frac)
        with torch.no_grad():
            # compute particle_weights
            graph_mlls = self.compute_graph_mlls(mc_graphs, use_cache=use_cache)
            particle_mlls = graph_mlls.logsumexp(dim=1) - math.log(num_mc_graphs)
            log_particle_prior = self.graph_model.unnormalized_log_prior(self.cfg.beta_eval, mc_adj_mats)
            particle_weights = log_softmax(log_particle_prior + particle_mlls, dim=0).exp()

            particle_idc = particle_weights.argsort(descending=True).cpu().numpy()
            num_kept = 0
            resampled_particles = []
            for i in particle_idc:
                if num_kept >= max_particles_to_keep or particle_weights[i] < self.cfg.resampling_threshold:
                    self.graph_model.particles[i] = self.graph_model.sample_initial_particles(1).squeeze(0)
                    resampled_particles.append(i)
                else:
                    num_kept += 1

        print(f'Resampling particles {resampled_particles} according to weights {particle_weights.squeeze()} (kept '
              f'{num_kept}/{max_particles_to_keep}')

    def update_graph_model(self):
        optimizer = torch.optim.Adam([self.graph_model.particles], lr=self.cfg.lr)
        alphas = self.cfg.alpha_offset + self.cfg.alpha_slope * torch.arange(self.cfg.num_svgd_steps).cpu().numpy()
        betas = self.cfg.beta_offset + self.cfg.beta_slope * torch.arange(self.cfg.num_svgd_steps).cpu().numpy()

        for i in range(self.cfg.num_svgd_steps):
            mc_graphs, mc_adj_mats = self.sample_mc_graphs(alphas[i])
            num_particles, num_mc_graphs = mc_adj_mats.shape[0:2]

            optimizer.zero_grad()

            # compute log prior p(Z)
            ec_adj_mats = self.graph_model.sample_soft_graphs(self.cfg.num_graphs_for_ec, alphas[i])
            log_prior = self.graph_model.unnormalized_log_prior(betas[i], ec_adj_mats).sum()

            # compute graph weights with baseline for variance reduction (p(D|G) - b) / p(D|Z)
            with torch.no_grad():
                graph_mlls = self.compute_graph_mlls(mc_graphs, use_cache=True)
                log_normalization = graph_mlls.logsumexp(dim=1)
                graph_weights = (graph_mlls - log_normalization.unsqueeze(1)).exp()

            baseline = torch.ones(num_particles, 1) / num_mc_graphs

            # compute score function estimator
            log_generative_probs = self.graph_model.log_generative_prob(mc_adj_mats, alphas[i])
            loss = log_prior + ((graph_weights - baseline) * log_generative_probs).sum()
            log_posterior_grads = torch.autograd.grad(loss, self.graph_model.particles)[0]
            # avoid exploding gradients by clipping!
            torch.clamp_(log_posterior_grads, -1., 1.)

            # compute SVGD repulsion terms
            bandwidth = self.graph_model.cfg.embedding_size * self.graph_model.num_nodes * 2.
            particle_similarities = self.graph_model.particle_similarities(bandwidth=bandwidth)
            sim_grads = torch.autograd.grad(particle_similarities.sum(), self.graph_model.particles)[0]
            particle_grads = torch.einsum('ab,bdef->adef', particle_similarities.detach(),
                                          log_posterior_grads) - sim_grads

            self.graph_model.particles.grad = -particle_grads / self.graph_model.cfg.num_particles

            optimizer.step()

            self.record_stat('svgd_loss', loss.item())
            if i > self.cfg.es_min_steps:
                change = (torch.tensor(self.stats['svgd_loss'][-1 - self.cfg.es_win_size:-1]).mean() -
                          torch.tensor(self.stats['svgd_loss'][-2 - self.cfg.es_win_size:-2]).mean()).abs()
                if change < self.cfg.es_threshold:
                    print(f'Stopping particle optimization after {i + 1} steps ...', flush=True)
                    break

            if self.cfg.svgd_log_interval > 0 and i % self.cfg.svgd_log_interval == 0:
                print(f'Step {i + 1} of {self.cfg.num_svgd_steps}, loss is {loss.item()}...', flush=True)

    def sample_mc_graphs(self, alpha: float = None, set_data=False, num_graphs: int = None, only_dags=False):
        num_graphs = self.cfg.num_mc_graphs if num_graphs is None else num_graphs
        alpha = self.cfg.alpha_eval if alpha is None else alpha

        with ((torch.no_grad())):
            mc_graphs, mc_adj_mats = self.graph_model.sample_graphs(num_graphs, alpha)

        # reduce mc graphs to get a reasonable amount of GPs to update
        if self.cfg.max_num_mc_mechanisms is not None:
            while len(get_unique_mechanisms(mc_graphs)) > self.cfg.max_num_mc_mechanisms and mc_adj_mats.shape[1] > 1:
                mc_graphs = [graphs[1:] for graphs in mc_graphs]
                mc_adj_mats = mc_adj_mats[:, 1:]

        if only_dags:
            self.graph_model.dagify_graphs(mc_graphs, mc_adj_mats, alpha)

        # initialize mechanisms
        self.sample_time += 1
        mechanism_keys = set()
        num_particles, num_graphs = mc_adj_mats.shape[0:2]
        num_cyclic = 0
        for pidx in range(num_particles):
            for gidx in range(num_graphs):
                if not nx.is_directed_acyclic_graph(mc_graphs[pidx][gidx]):
                    num_cyclic += 1
                else:
                    self.mechanism_model.init_topological_order(mc_graphs[pidx][gidx], self.sample_time)
                keys = self.mechanism_model.init_graph_mechanisms(mc_graphs[pidx][gidx], self.sample_time)
                mechanism_keys.update(keys)
        # print(f'Sampled {num_cyclic}/{num_particles * num_graphs} cyclic graphs!')

        # discard older gps/topolocial orders
        self.mechanism_model.discard_gps()
        self.mechanism_model.discard_topo_orders()

        # update mechanism hyperparameters
        if self.cfg.inference_mode == 'joint':
            self.mechanism_model.update_gp_hyperparameters(self.experiments, list(mechanism_keys))
        elif set_data:
            self.mechanism_model.set_data(self.experiments)

        return mc_graphs, mc_adj_mats

    def compute_graph_mlls(self, graphs: List[List[nx.DiGraph]], experiments: List[Experiment] = None, prior_mode=True,
                           use_cache=True):
        num_particles, num_mc_graphs = len(graphs), len(graphs[0])
        experiments = self.experiments if experiments is None else experiments
        graph_mlls = []
        for pidx in range(num_particles):
            for graph in graphs[pidx]:
                mll = self.mechanism_model.mll(experiments, graph, prior_mode=prior_mode, use_cache=use_cache)
                if self.cfg.inference_mode == 'joint':
                    mll += self.mechanism_model.log_hp_prior(graph)
                graph_mlls.append(mll)

        graph_mlls = torch.stack(graph_mlls).view(num_particles, num_mc_graphs)
        return graph_mlls

    def graph_posterior_expectation(self, func: Callable[[nx.DiGraph], torch.Tensor],
                                    mc_graphs: List[List[nx.DiGraph]],
                                    mc_adj_mats: torch.Tensor,
                                    use_cache=True,
                                    logspace=False):
        num_particles, num_mc_graphs = mc_adj_mats.shape[0:2]

        # compute function values
        func_values = [func(graph) for i in range(num_particles) for graph in mc_graphs[i]]
        func_output_shape = func_values[0].shape
        func_output_dim = len(func_output_shape)
        func_values = torch.stack(func_values).view(num_particles, num_mc_graphs, *func_output_shape)

        # compute expectation
        if logspace:
            log_graph_weights, log_particle_weights = self.compute_mc_weights(mc_graphs,
                                                                              mc_adj_mats,
                                                                              use_cache=use_cache,
                                                                              log_weights=True)
            log_graph_weights = log_graph_weights.view(num_particles, num_mc_graphs, *([1] * func_output_dim))
            log_particle_weights = log_particle_weights.view(num_particles, *([1] * func_output_dim))
            expected_value = (log_graph_weights + func_values).logsumexp(dim=1)
            expected_value = (log_particle_weights + expected_value).logsumexp(dim=0)
            return expected_value

        graph_weights, particle_weights = self.compute_mc_weights(mc_graphs,
                                                                  mc_adj_mats,
                                                                  use_cache=use_cache)
        graph_weights = graph_weights.view(num_particles, num_mc_graphs, *([1] * func_output_dim))
        particle_weights = particle_weights.view(num_particles, *([1] * func_output_dim))

        expected_value = (particle_weights * (graph_weights * func_values).sum(dim=1)).sum(dim=0)
        return expected_value

    def compute_posterior_edge_probs(self, mc_graphs: List[List[nx.DiGraph]], mc_adj_mats: torch.Tensor,
                                     use_cache: bool = True):

        # compute expectation
        graph_weights, particle_weights = self.compute_mc_weights(mc_graphs, mc_adj_mats, use_cache=use_cache)
        edge_probs = mc_adj_mats * graph_weights.unsqueeze(-1).unsqueeze(-1)
        edge_probs = edge_probs.sum(dim=1) * particle_weights.view(-1, 1, 1)
        return edge_probs.sum(dim=0)

    def compute_mc_weights(self, mc_graphs, mc_adj_mats, use_cache=False, log_weights: bool = False,
                           exclude_cyclic: bool = True):
        num_particles, num_mc_graphs = mc_adj_mats.shape[0:2]
        graph_mlls = self.compute_graph_mlls(mc_graphs, use_cache=use_cache)

        # assign cyclic graphs zero weight -> only acyclic graphs hold weight
        if exclude_cyclic:
            cyclic = []
            for pidx in range(num_particles):
                cyclic.append(torch.tensor([not nx.is_directed_acyclic_graph(g) for g in mc_graphs[pidx]]))

            cyclic = torch.stack(cyclic, dim=0)
            graph_mlls = torch.where(cyclic, -inf_tensor(), graph_mlls)

        log_normalization = graph_mlls.logsumexp(dim=1)
        log_normalization = torch.where(torch.isnan(log_normalization), -inf_tensor(), log_normalization)

        log_graph_weights = (graph_mlls - log_normalization.unsqueeze(1))
        log_graph_weights = torch.where(torch.isnan(log_graph_weights), -inf_tensor(), log_graph_weights)
        if self.cfg.dibs_plus:
            log_particle_prior = self.graph_model.unnormalized_log_prior(self.cfg.beta_eval, mc_adj_mats)
            particle_mlls = log_normalization - math.log(num_mc_graphs)
            log_particle_weights = log_softmax(log_particle_prior + particle_mlls, dim=0)
            log_particle_weights = torch.where(torch.isnan(log_particle_weights), -inf_tensor(), log_particle_weights)
        else:
            log_particle_weights = -torch.tensor(num_particles).log() * torch.ones(num_particles)

        if log_weights:
            return log_graph_weights, log_particle_weights
        return log_graph_weights.exp(), log_particle_weights.exp()

    def sample(self, interventions: dict, num_samples_per_graph: int, mc_graphs: List[List[nx.DiGraph]] = None,
               mc_adj_mats: torch.Tensor = None):

        if mc_graphs is None or mc_adj_mats is None:
            mc_graphs, mc_adj_mats = self.sample_mc_graphs(set_data=True)

        num_particles, num_graphs = mc_adj_mats.shape[0:2]
        samples = {node: torch.zeros(num_particles, num_graphs, num_samples_per_graph) for node in
                   self.mechanism_model.node_labels}

        with torch.no_grad():
            for pidx in range(num_particles):
                for gidx, graph in enumerate(mc_graphs[pidx]):
                    if nx.is_directed_acyclic_graph(graph):
                        exp = self.mechanism_model.sample(interventions, 1, num_samples_per_graph, graph)
                        for node in samples:
                            samples[node][pidx, gidx] = exp.data[node].squeeze()

            for node in samples:
                samples[node] = samples[node].reshape(-1)

            # compute sample weights
            graph_weights, particle_weights = self.compute_mc_weights(mc_graphs, mc_adj_mats)
            weights = graph_weights * particle_weights.unsqueeze(1)
            weights = weights.unsqueeze(-1).expand(-1, -1, num_samples_per_graph).reshape(-1) / num_samples_per_graph
        return samples, weights

    def estimate_ace(self, target: str, interventions: dict, num_samples: int, mc_graphs: List[List[nx.DiGraph]] = None,
                     mc_adj_mats: torch.Tensor = None) -> torch.Tensor:

        if mc_graphs is None or mc_adj_mats is None:
            mc_graphs, mc_adj_mats = self.sample_mc_graphs(set_data=True)

        def ace_wrapper(graph: nx.DiGraph):
            if nx.is_directed_acyclic_graph(graph):
                return self.mechanism_model.sample_ace(target, interventions, num_samples, graph).mean()
            else:
                return torch.tensor(1e6)

        with torch.no_grad():
            ace = self.graph_posterior_expectation(ace_wrapper, mc_graphs, mc_adj_mats)
        return ace

    def estimate_aces(self, interventions: dict, num_samples: int, mc_graphs: List[List[nx.DiGraph]] = None,
                      mc_adj_mats: torch.Tensor = None) -> torch.Tensor:

        if mc_graphs is None or mc_adj_mats is None:
            mc_graphs, mc_adj_mats = self.sample_mc_graphs(set_data=True)

        def ace_wrapper(graph: nx.DiGraph):
            if nx.is_directed_acyclic_graph(graph):
                return self.mechanism_model.sample_aces(interventions, num_samples, graph).mean(dim=1)
            else:
                return torch.ones(self.graph_model.num_nodes) * 1e6

        with torch.no_grad():
            aces = self.graph_posterior_expectation(ace_wrapper, mc_graphs, mc_adj_mats)
        return aces

    def compute_stats(self):
        mc_graphs, mc_adj_mats = self.sample_mc_graphs(set_data=True)

        # compute posterior edge probs
        with torch.no_grad():
            posterior_edge_probs = self.compute_posterior_edge_probs(mc_graphs, mc_adj_mats, use_cache=True)

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

        is_cyclic = [all([not nx.is_directed_acyclic_graph(g) for g in pgraphs]) for pgraphs in mc_graphs]
        if all(is_cyclic):
            # for the exceptional case that all sampled graphs are cyclic, the AIDs are set to 1.
            print('All sampled graphs are cyclic!')
            self.record_stat('aaid', torch.tensor(1.))
            self.record_stat('paid', torch.tensor(1.))
            self.record_stat('oset_aid', torch.tensor(1.))
            self.record_stat('aaid_cpdag', torch.tensor(1.))
            self.record_stat('paid_cpdag', torch.tensor(1.))
            self.record_stat('oset_aid_cpdag', torch.tensor(1.))
        else:
            # record AID metrics to true DAG
            def aid_wrapper(g: nx.DiGraph, mode: str):
                if nx.is_directed_acyclic_graph(g):
                    adj_mat = self.graph_model.graph_to_adj_mat(g)
                    return aid(true_adj_mat, adj_mat, mode=mode)
                else:
                    return torch.tensor(1.)  # gets zero weight in the expectation

            with torch.no_grad():
                aaid = self.graph_posterior_expectation(lambda g: aid_wrapper(g, mode='ancestor'), mc_graphs,
                                                        mc_adj_mats)
                self.record_stat('aaid', aaid)

                paid = self.graph_posterior_expectation(lambda g: aid_wrapper(g, mode='parent'), mc_graphs, mc_adj_mats)
                self.record_stat('paid', paid)

                oset_aid = self.graph_posterior_expectation(lambda g: aid_wrapper(g, mode='oset'), mc_graphs,
                                                            mc_adj_mats)
                self.record_stat('oset_aid', oset_aid)

            # record AID metrics to true CPDAG
            def aid_wrapper_cpdag(g: nx.DiGraph, mode: str):
                if nx.is_directed_acyclic_graph(g):
                    return aid(true_cpdag, dag_to_cpdag(g, self.mechanism_model.node_labels), mode=mode)
                else:
                    return torch.tensor(1.)  # gets zero weight in the expectation

            with torch.no_grad():
                aaid = self.graph_posterior_expectation(lambda g: aid_wrapper_cpdag(g, mode='ancestor'), mc_graphs,
                                                        mc_adj_mats)
                self.record_stat('aaid_cpdag', aaid)

                paid = self.graph_posterior_expectation(lambda g: aid_wrapper_cpdag(g, mode='parent'), mc_graphs,
                                                        mc_adj_mats)
                self.record_stat('paid_cpdag', paid)

                oset_aid = self.graph_posterior_expectation(lambda g: aid_wrapper_cpdag(g, mode='oset'), mc_graphs,
                                                            mc_adj_mats)
                self.record_stat('oset_aid_cpdag', oset_aid)

        if self.cfg.compute_distributional_stats:
            if self.env.interventional_test_data is not None:
                print(f'Computing distributional metrics on interventional test data...')
                mean_errors = []
                mmds = []
                with torch.no_grad():
                    for eidx, exp in enumerate(self.env.interventional_test_data):
                        print(f'Computing metrics for intervention {eidx}/{len(self.env.interventional_test_data)}')
                        env_samples = torch.stack([exp.data[node].squeeze() for node in self.env.node_labels],
                                                  dim=-1)
                        env_mean = env_samples.mean(dim=0)

                        # sample from learned posterior distribution
                        samples, weights = self.sample(exp.interventions, self.cfg.num_samples_per_graph, mc_graphs,
                                                       mc_adj_mats)
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
                       'graph_model_params': self.graph_model.param_dict(),
                       'cfg_param_dict': self.cfg.param_dict()})
        return params

    def load_param_dict(self, param_dict):
        super().load_param_dict(param_dict)
        self.cfg = ABCIDiBSGPConfig(param_dict['cfg_param_dict'])
        self.graph_model = DiBSModel(param_dict=param_dict['graph_model_params'])
        self.mechanism_model = SharedDataGaussianProcessModel(param_dict=param_dict['mechanism_model_params'])
        self.mechanism_model.set_data(self.experiments)
        sample_times = self.mechanism_model.gp_sample_times.values()
        self.sample_time = max(sample_times) if len(sample_times) else 0

    @classmethod
    def load(cls, path):
        param_dict = torch.load(path, weights_only=False)
        return ABCIDiBSGP(param_dict=param_dict)
