from typing import Callable, List, Dict, Any, Optional, Union

import networkx as nx
import torch.optim
from torch.distributions import Categorical

from src.abci_base import ABCIBase
from src.config import ABCIArCOGPConfig
from src.environments.environment import Environment
from src.graph_models.arco import ArCO
from src.mechanism_models.mechanisms import get_mechanism_key
from src.mechanism_models.shared_data_gp_model import SharedDataGaussianProcessModel
from src.utils.causal_orders import CausalOrder, generate_all_mechanisms, generate_all_parent_sets
from src.utils.graphs import dag_to_cpdag, adj_mat_to_graph
from src.utils.metrics import aid, compute_structure_metrics, mmd
from src.utils.utils import inf_tensor


class ABCIArCOGP(ABCIBase):
    cfg: ABCIArCOGPConfig

    def __init__(self, env: Environment = None, cfg: ABCIArCOGPConfig = None,
                 param_dict: Dict[str, Any] = None):
        assert env is not None or param_dict is not None
        if param_dict is not None:
            num_workers = param_dict['cfg_param_dict']['num_workers']
            super().__init__(num_workers)
            self.load_param_dict(param_dict)
        else:
            # load (default) config
            self.cfg = ABCIArCOGPConfig() if cfg is None else cfg
            super().__init__(self.cfg.num_workers, env)

            # init models
            self.co_model = ArCO(env.node_labels)
            self.mechanism_model = SharedDataGaussianProcessModel(env.node_labels)
            self.sample_time = 0

        # init caches
        self.node_label_to_id_dict = dict(zip(self.env.node_labels, list(range(self.env.num_nodes))))
        self.ps_weight_cache: Dict[str, torch.Tensor] = {}  # holds log p(X_i, \psi_i | pa_i) * p(pa_i | L)
        self.co_weight_cache: Dict[str, torch.Tensor] = {}  # holds log p(X_i, \psi_i | L) for i=1..d
        self.co_weights: Optional[torch.Tensor] = None  # shape (num_mc_cos, num_nodes)

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
            self.co_weight_cache.clear()
            self.ps_weight_cache.clear()

            # update model
            self.update_co_model()
            print(f'There are currently {self.mechanism_model.get_num_gps()} unique GPs in our model.')

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

    def update_co_model(self):
        print(f'Optimising causal order model...', flush=True)
        arco_optimizer = torch.optim.Adam(self.co_model.parameters(), lr=self.cfg.arco_lr)
        co_baselines = [torch.tensor(0.)]
        for step in range(self.cfg.num_arco_steps):
            mc_cos, mc_adj_masks = self.sample_mc_cos(num_cos=self.cfg.num_cos_arco_opt)
            co_weights = self.co_weights.sum(dim=1)
            co_weights = (co_weights - co_weights.logsumexp(dim=0)).exp()

            # compute log prior p(\theta)
            log_arco_prior = self.co_model.log_param_prior().sum()

            # compute arco loss
            bl = co_weights.mean() * self.cfg.tau + co_baselines[-1] * (1. - self.cfg.tau)
            co_baselines.append(bl)

            co_log_prior = self.co_model.log_prob(mc_cos).squeeze(0)
            arco_loss = -((co_weights - co_baselines[-1]) * co_log_prior).sum() - log_arco_prior

            # co model updates
            arco_optimizer.zero_grad(set_to_none=True)
            arco_loss.backward()
            arco_optimizer.step()

            self.record_stat('arco_loss', arco_loss.item())
            if step > self.cfg.arco_es_min_steps:
                arco_change = (torch.tensor(self.stats['arco_loss'][-1 - self.cfg.es_win_size:-1]).mean() -
                               torch.tensor(self.stats['arco_loss'][-2 - self.cfg.es_win_size:-2]).mean()).abs()
                if arco_change < self.cfg.es_threshold:
                    print(f'Stopping com optimization after {step + 1} steps ...', flush=True)
                    break

            if self.cfg.opt_log_interval > 0 and step % self.cfg.opt_log_interval == 0:
                print(f'Step {step + 1} of {self.cfg.num_arco_steps}, com loss is {arco_loss.item()}', flush=True)

    def sample_mc_cos(self, set_data=False, num_cos: int = None):
        num_cos = self.cfg.num_mc_cos if num_cos is None else num_cos

        with torch.no_grad():
            # sample causal orders
            mc_cos, mc_adj_masks = self.co_model.sample(num_cos)

            mechanism_keys = set()
            for cidx in range(num_cos):
                mechs = generate_all_mechanisms(self.env.node_labels, self.cfg.max_ps_size, mc_adj_masks[cidx])
                mechanism_keys.update(set(mechs))

            mechanism_keys = list(mechanism_keys)

        # initialize mechanisms
        self.sample_time += 1
        self.mechanism_model.init_mechanisms_from_keys(mechanism_keys, self.sample_time)

        # discard older gps/topolocial orders
        self.mechanism_model.discard_gps()

        # update mechanism hyperparameters
        if self.cfg.inference_mode == 'joint':
            self.mechanism_model.update_gp_hyperparameters(self.experiments, mechanism_keys)
        elif set_data:
            self.mechanism_model.set_data(self.experiments)

        # pre-compute mc weights
        self.co_weights = torch.zeros(num_cos, self.env.num_nodes)
        with torch.no_grad():
            for cidx, co in enumerate(mc_cos):
                # if co weights already pre-computed, take cached value
                if co.__repr__() in self.co_weight_cache:
                    self.co_weights[cidx] = self.co_weight_cache[co.__repr__()]
                    continue

                # else compute co and ps weights
                parent_sets = self.generate_co_parent_sets(mc_adj_masks[cidx])
                for nidx, node in enumerate(self.env.node_labels):
                    node_weights = torch.zeros(len(parent_sets[node]))
                    for pidx, parents in enumerate(parent_sets[node]):
                        key = get_mechanism_key(node, parents)
                        if key in self.ps_weight_cache:
                            weight = self.ps_weight_cache[key].clone()
                        else:
                            with torch.no_grad():
                                weight = self.mechanism_model.node_mll(self.experiments, node, parents, prior_mode=True)
                                weight -= torch.tensor(len(parent_sets[node])).log()  # uniform prior over parent sets
                                if self.cfg.inference_mode == 'joint':
                                    weight += self.mechanism_model.mechanism_log_hp_priors([key])

                            self.ps_weight_cache[key] = weight

                        node_weights[pidx] = weight
                    self.co_weights[cidx, nidx] = node_weights.logsumexp(dim=0)

                self.co_weight_cache[co.__repr__()] = self.co_weights[cidx]

        return mc_cos, mc_adj_masks

    def sample_mc_graphs(self, mc_cos: List[CausalOrder], mc_adj_masks: torch.Tensor, num_mc_graphs: int = None):
        num_mc_graphs = self.cfg.num_mc_graphs if num_mc_graphs is None else num_mc_graphs

        mc_adj_mats = torch.zeros(len(mc_cos), num_mc_graphs, self.env.num_nodes, self.env.num_nodes)
        with torch.no_grad():
            for cidx, co in enumerate(mc_cos):
                parent_sets = self.generate_co_parent_sets(mc_adj_masks[cidx])

                for nidx, node in enumerate(self.env.node_labels):
                    ps_logits = torch.tensor(
                        [self.ps_weight_cache[get_mechanism_key(node, ps)] for ps in parent_sets[node]])

                    mc_ps_idc = Categorical(logits=ps_logits).sample(torch.Size((num_mc_graphs,)))
                    mc_ps = [parent_sets[node][ps_idx] for ps_idx in mc_ps_idc]
                    for ps_idx, ps in enumerate(mc_ps):
                        parent_ids = [self.node_label_to_id_dict[pa] for pa in ps]
                        mc_adj_mats[cidx, ps_idx, parent_ids, nidx] = 1

        return mc_adj_mats

    def graph_posterior_expectation_additive(self, func: Callable[[str, List[str]], torch.Tensor],
                                             mc_cos: List[CausalOrder],
                                             logspace=False):
        num_cos = len(mc_cos)
        co_values = torch.zeros(num_cos, self.env.num_nodes)
        for cidx, co in enumerate(mc_cos):
            parent_sets = self.generate_co_parent_sets(co)
            for nidx, node in enumerate(self.env.node_labels):
                weighted_values = []
                for parents in parent_sets[node]:
                    key = get_mechanism_key(node, parents)
                    weight = self.ps_weight_cache[key]
                    func_value = func(node, parents)
                    if logspace:
                        weighted_values.append(weight + func_value)
                    else:
                        weighted_values.append(weight.exp() * func_value)

                tmp = self.co_weights[cidx].sum() - self.co_weights[cidx, nidx]
                if logspace:
                    co_values[cidx, nidx] = tmp + torch.stack(weighted_values, dim=0).logsumexp(dim=0)
                else:
                    co_values[cidx, nidx] = tmp.exp() * torch.stack(weighted_values, dim=0).sum(dim=0)

        normalisation = self.co_weights.sum(dim=1).logsumexp(dim=0)
        if logspace:
            return co_values.logsumexp(dim=1).logsumexp(dim=0) - normalisation
        else:
            return co_values.sum() / normalisation.exp()

    def graph_posterior_expectation_factorising(self, func: Callable[[str, List[str]], torch.Tensor],
                                                mc_cos: List[CausalOrder],
                                                logspace=False):
        num_cos = len(mc_cos)
        if logspace:
            co_values = torch.zeros(num_cos)
        else:
            co_values = torch.ones(num_cos)

        for cidx, co in enumerate(mc_cos):
            parent_sets = self.generate_co_parent_sets(co)
            for nidx, node in enumerate(self.env.node_labels):
                weighted_values = []
                for parents in parent_sets[node]:
                    key = get_mechanism_key(node, parents)
                    weight = self.ps_weight_cache[key]
                    func_value = func(node, parents)
                    if logspace:
                        weighted_values.append(weight + func_value)
                    else:
                        weighted_values.append(weight.exp() * func_value)

                if logspace:
                    co_values[cidx] += torch.stack(weighted_values, dim=0).logsumexp(dim=0)
                else:
                    co_values[cidx] *= torch.stack(weighted_values, dim=0).sum(dim=0)

        normalisation = self.co_weights.sum(dim=1).logsumexp(dim=0)
        if logspace:
            return co_values.logsumexp(dim=0) - normalisation
        else:
            return co_values.sum() / normalisation.exp()

    def graph_posterior_expectation_mc(self, func: Callable[[torch.Tensor], torch.Tensor],
                                       mc_cos: List[CausalOrder] = None, mc_adj_mats: torch.Tensor = None,
                                       logspace=False):

        if mc_cos is None and mc_adj_mats is None:
            mc_cos, _ = self.sample_mc_cos(set_data=True)

        if mc_adj_mats is None:
            mc_adj_masks = torch.stack([co.get_adjacency_mask() for co in mc_cos])
            mc_adj_mats = self.sample_mc_graphs(mc_cos, mc_adj_masks)

        num_cos, num_graphs = mc_adj_mats.shape[0:2]
        if logspace:
            co_values = torch.zeros(num_cos)
        else:
            co_values = torch.ones(num_cos)

        # compute function values
        func_values = [func(mc_adj_mats[cidx, gidx]) for cidx in range(num_cos) for gidx in range(num_graphs)]
        func_output_shape = func_values[0].shape
        func_dims = func_values[0].dim()
        func_values = torch.stack(func_values).view(num_cos, num_graphs, *func_output_shape)

        # compute expectation
        log_co_weights = self.co_weights.sum(dim=1)
        log_co_weights -= log_co_weights.logsumexp(dim=0)
        log_co_weights = log_co_weights.view(num_cos, *([1] * func_dims))
        if logspace:
            func_values = func_values.logsumexp(dim=1) - torch.tensor(num_graphs).log()
            expected_value = (log_co_weights + func_values).logsumexp(dim=0)
            return expected_value

        func_values = func_values.mean(dim=1)
        expected_value = (func_values * log_co_weights.exp()).sum(dim=0)
        return expected_value

    def co_posterior_expectation(self, func: Callable[[CausalOrder], torch.Tensor],
                                 mc_cos: List[CausalOrder]):
        num_cos = len(mc_cos)

        # compute function values
        func_values = torch.stack([func(co) for co in mc_cos])
        func_output_dim = func_values.dim() - 1

        # compute expectation
        log_co_weights = self.co_weights.sum(dim=1)
        log_co_weights = (log_co_weights - log_co_weights.logsumexp(dim=0)).view(num_cos, *([1] * func_output_dim))
        expected_value = (log_co_weights.exp() * func_values).sum()
        return expected_value

    def compute_posterior_edge_probs(self, mc_cos: List[CausalOrder]):
        log_probs = -torch.ones(len(mc_cos), self.env.num_nodes, self.env.num_nodes) * inf_tensor()
        for cidx, co in enumerate(mc_cos):
            parent_sets = self.generate_co_parent_sets(co)
            for sidx, source in enumerate(self.env.node_labels):
                for tidx, target in enumerate(self.env.node_labels):
                    if sidx == tidx:
                        continue

                    keys = [get_mechanism_key(target, parents) for parents in parent_sets[target] if source in parents]
                    if len(keys) == 0:
                        continue

                    log_prob = self.co_weights[cidx].sum() - self.co_weights[cidx, tidx]
                    weights = torch.tensor([self.ps_weight_cache[key] for key in keys])
                    log_probs[cidx, sidx, tidx] = log_prob + weights.logsumexp(dim=0)

        normalisation = self.co_weights.sum(dim=1).logsumexp(dim=0)
        posterior_edge_probs = log_probs.logsumexp(dim=0) - normalisation
        return posterior_edge_probs.exp()

    def estimate_ace(self, target: str, interventions: dict, num_samples: int, mc_cos: List[CausalOrder] = None,
                     mc_adj_mats: torch.Tensor = None) -> torch.Tensor:
        def ate_wrapper(adj_mat: torch.Tensor):
            graph = adj_mat_to_graph(adj_mat, self.mechanism_model.node_labels)
            self.mechanism_model.init_topological_order(graph, self.sample_time)
            return self.mechanism_model.sample_ace(target, interventions, num_samples, graph).mean()

        ate = self.graph_posterior_expectation_mc(ate_wrapper, mc_cos, mc_adj_mats)
        return ate

    def estimate_aces(self, interventions: dict, num_samples: int, mc_cos: List[CausalOrder] = None,
                      mc_adj_mats: torch.Tensor = None) -> torch.Tensor:
        def ace_wrapper(adj_mat: torch.Tensor):
            graph = adj_mat_to_graph(adj_mat, self.mechanism_model.node_labels)
            self.mechanism_model.init_topological_order(graph, self.sample_time)
            return self.mechanism_model.sample_aces(interventions, num_samples, graph).mean(dim=1)

        aces = self.graph_posterior_expectation_mc(ace_wrapper, mc_cos, mc_adj_mats)
        return aces

    def sample(self, interventions: dict, num_samples_per_graph: int, adj_mats: torch.Tensor = None):
        if adj_mats is None:
            mc_cos, mc_adj_masks = self.sample_mc_cos(set_data=True, num_cos=self.cfg.num_mc_cos)
            adj_mats = self.sample_mc_graphs(mc_cos, mc_adj_masks, num_mc_graphs=self.cfg.num_mc_graphs)

        num_cos, num_graphs = adj_mats.shape[0:2]
        samples = {node: torch.zeros(num_cos, num_graphs, num_samples_per_graph) for node in
                   self.mechanism_model.node_labels}

        with torch.no_grad():
            for cidx in range(num_cos):
                for gidx in range(num_graphs):
                    graph = adj_mat_to_graph(adj_mats[cidx, gidx], self.mechanism_model.node_labels)
                    self.mechanism_model.init_topological_order(graph, self.sample_time)
                    exp = self.mechanism_model.sample(interventions, 1, num_samples_per_graph, graph)
                    for node in samples:
                        samples[node][cidx, gidx] = exp.data[node].squeeze()

            for node in samples:
                samples[node] = samples[node].reshape(-1)

            # compute sample weights
            log_co_weights = self.co_weights.sum(dim=1)
            log_co_weights -= log_co_weights.logsumexp(dim=0)
            weights = log_co_weights.exp() / (num_graphs * num_samples_per_graph)
            weights = weights.unsqueeze(1).expand(-1, num_graphs * num_samples_per_graph).reshape(-1)
        return samples, weights

    def sample_ace(self, target: str, interventions: dict, num_samples: int, adj_mats: torch.Tensor):
        num_cos, num_graphs = adj_mats.shape[0:2]
        ates = torch.zeros(num_cos, num_graphs, num_samples)

        log_co_weights = self.co_weights.sum(dim=1)
        log_co_weights -= log_co_weights.logsumexp(dim=0)
        co_weights = log_co_weights.exp()
        for cidx in range(num_cos):
            for gidx in range(num_graphs):
                graph = adj_mat_to_graph(adj_mats[cidx, gidx], self.mechanism_model.node_labels)
                self.mechanism_model.init_topological_order(graph, self.sample_time)
                ates[cidx, gidx] = self.mechanism_model.sample_ace(target, interventions, num_samples, graph)

        ates = ates.view(num_cos, -1)
        weights = co_weights.unsqueeze(-1).expand_as(ates).reshape(-1) / (num_graphs * num_samples)
        return ates.view(-1), weights

    def generate_co_parent_sets(self, co: Union[CausalOrder, torch.Tensor]):
        adj_mask = co.get_adjacency_mask() if isinstance(co, CausalOrder) else co
        return generate_all_parent_sets(self.env.node_labels, self.cfg.max_ps_size, adj_mask)

    def compute_stats(self):
        mc_cos, mc_adj_masks = self.sample_mc_cos(set_data=True)

        # compute posterior edge probs
        with torch.no_grad():
            posterior_edge_probs = self.compute_posterior_edge_probs(mc_cos)

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

        # for illustration: ESHD could also computed this way...
        # true_parent_sets = {node: set(self.env.graph.predecessors(node)) for node in self.env.node_labels}
        #
        # def ps_shd(target_node: str, parents: List[str]):
        #     return torch.tensor(len(true_parent_sets[target_node].symmetric_difference(set(parents)))).log()
        #
        # with torch.no_grad():
        #     eshd = self.graph_posterior_expectation_additive(ps_shd, mc_cos, logspace=True).exp()
        # self.record_stat('eshd', eshd)

        print('Computing AID metrics...', flush=True)
        # sample mc graphs
        mc_adj_masks = torch.stack([co.get_adjacency_mask() for co in mc_cos])
        mc_adj_mats = self.sample_mc_graphs(mc_cos, mc_adj_masks)

        # record AID metrics to true DAG
        aaid = self.graph_posterior_expectation_mc(lambda g: aid(true_adj_mat, g, mode='ancestor'),
                                                   mc_adj_mats=mc_adj_mats)
        self.record_stat('aaid', aaid)

        paid = self.graph_posterior_expectation_mc(lambda g: aid(true_adj_mat, g, mode='parent'),
                                                   mc_adj_mats=mc_adj_mats)
        self.record_stat('paid', paid)

        oset_aid = self.graph_posterior_expectation_mc(lambda g: aid(true_adj_mat, g, mode='oset'),
                                                       mc_adj_mats=mc_adj_mats)
        self.record_stat('oset_aid', oset_aid)

        # record AID metrics to true CPDAG
        def aid_wrapper(g: nx.DiGraph, mode: str):
            return aid(true_cpdag, dag_to_cpdag(g, self.env.node_labels), mode=mode)

        aaid = self.graph_posterior_expectation_mc(lambda g: aid_wrapper(g, mode='ancestor'), mc_adj_mats=mc_adj_mats)
        self.record_stat('aaid_cpdag', aaid)

        paid = self.graph_posterior_expectation_mc(lambda g: aid_wrapper(g, mode='parent'), mc_adj_mats=mc_adj_mats)
        self.record_stat('paid_cpdag', paid)

        oset_aid = self.graph_posterior_expectation_mc(lambda g: aid_wrapper(g, mode='oset'), mc_adj_mats=mc_adj_mats)
        self.record_stat('oset_aid_cpdag', oset_aid)

        # record AID metric for the causal orders
        def aid_wrapper_co(co: CausalOrder):
            return aid(true_adj_mat, co.adjacency_mask, mode='ancestor')

        order_aid = self.co_posterior_expectation(aid_wrapper_co, mc_cos)
        self.record_stat('order_aid', order_aid)

        if self.cfg.compute_distributional_stats:
            if self.env.interventional_test_data is not None:
                print(f'Computing distributional metrics on interventional test data...')
                mean_errors = []
                mmds = []
                with torch.no_grad():
                    for eidx, exp in enumerate(self.env.interventional_test_data):
                        print(f'Computing metrics for intervention {eidx}/{len(self.env.interventional_test_data)}')
                        env_samples = torch.stack([exp.data[node].squeeze() for node in self.env.node_labels], dim=-1)
                        env_mean = env_samples.mean(dim=0)

                        # sample from learned posterior distribution
                        samples, weights = self.sample(exp.interventions, self.cfg.num_samples_per_graph, mc_adj_mats)
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
                       'co_model_params': self.co_model.param_dict(),
                       'cfg_param_dict': self.cfg.param_dict()})
        return params

    def load_param_dict(self, param_dict):
        super().load_param_dict(param_dict)
        self.cfg = ABCIArCOGPConfig(param_dict['cfg_param_dict'])
        self.co_model = ArCO(param_dict=param_dict['co_model_params'])
        self.mechanism_model = SharedDataGaussianProcessModel(param_dict=param_dict['mechanism_model_params'])
        self.mechanism_model.set_data(self.experiments)
        sample_times = self.mechanism_model.gp_sample_times.values()
        self.sample_time = max(sample_times) if len(sample_times) else 0

    @classmethod
    def load(cls, path):
        param_dict = torch.load(path)
        return ABCIArCOGP(param_dict=param_dict)
