import itertools
import math
from typing import Optional, List, Dict, Any

import networkx as nx
import torch
from torch.linalg import vector_norm

from src.config import GPModelConfig
from src.environments.environment import Experiment, gather_data
from src.environments.experiment import InterventionalDistributionsQuery
from src.mechanism_models.mechanisms import Mechanism, GaussianProcess, GaussianRootNode, get_mechanism_key, \
    resolve_mechanism_key
from src.utils.graphs import get_graph_key, get_parents


def get_unique_mechanisms(graphs: List[List[nx.DiGraph]]):
    batch_size, num_graphs = len(graphs), len(graphs[0])
    keys = set()
    for bidx in range(batch_size):
        for gidx in range(num_graphs):
            graph = graphs[bidx][gidx]
            for node in graph.nodes:
                keys.add(get_mechanism_key(node, get_parents(node, graph)))

    return keys


class GaussianProcessModel:
    def __init__(self, node_labels: List[str] = None, cfg: GPModelConfig = None, param_dict: Dict[str, Any] = None):
        assert node_labels is not None or param_dict is not None
        if param_dict is not None:
            self.load_param_dict(param_dict)
        else:
            # load config
            self.cfg = GPModelConfig() if cfg is None else cfg

            self.node_labels = sorted(list(set(node_labels)))
            self.mechanisms = dict()
            self.mechanism_init_times = dict()
            self.mechanism_update_times = dict()
            self.topological_orders = dict()

        # init caches
        self.topological_orders_init_times = dict()
        self.prior_mll_cache = dict()
        self.posterior_mll_cache = dict()
        self.entropy_cache = dict()
        self.rmse_cache = dict()

    def get_parameters(self, keys: List[str] = None):
        keys = self.mechanisms.keys() if keys is None else keys
        param_lists = [list(self.mechanisms[key].parameters()) for key in keys]
        return list(itertools.chain(*param_lists))

    def init_mechanisms(self, graph: nx.DiGraph, init_time: int = 0):
        graph_key = get_graph_key(graph)
        if graph_key not in self.topological_orders and nx.is_directed_acyclic_graph(graph):
            self.topological_orders[graph_key] = list(nx.topological_sort(graph))
        if graph_key in self.topological_orders:
            self.topological_orders_init_times[graph_key] = init_time

        initialized_mechanisms = []
        for node in graph:
            parents = get_parents(node, graph)
            key = get_mechanism_key(node, parents)
            initialized_mechanisms.append(key)
            self.mechanism_init_times[key] = init_time
            if key not in self.mechanisms:
                self.mechanism_update_times[key] = 0
                self.mechanisms[key] = self.create_mechanism(len(parents))

        self.eval(initialized_mechanisms)
        return initialized_mechanisms

    def discard_mechanisms(self, current_time: int, max_age: int):
        keys = [key for key, time in self.mechanism_init_times.items() if current_time - time > max_age]
        print(f'Discarding {len(keys)} old mechanisms...')
        for key in keys:
            del self.mechanisms[key]
            del self.mechanism_init_times[key]
            del self.mechanism_update_times[key]

        keys = [key for key, time in self.topological_orders_init_times.items() if current_time - time > max_age]
        print(f'Discarding {len(keys)} old topological orders...')
        for key in keys:
            del self.topological_orders[key]
            del self.topological_orders_init_times[key]

    def eval(self, keys: List[str] = None):
        keys = self.mechanisms.keys() if keys is None else keys
        for key in keys:
            self.mechanisms[key].eval()

    def train(self, keys: List[str] = None):
        keys = self.mechanisms.keys() if keys is None else keys
        for key in keys:
            self.mechanisms[key].train()

    def clear_prior_mll_cache(self, keys: List[str] = None):
        if keys is None:
            self.prior_mll_cache.clear()
        else:
            for key in keys:
                if key in self.prior_mll_cache:
                    del self.prior_mll_cache[key]

    def clear_posterior_mll_cache(self, keys: List[str] = None):
        if keys is None:
            self.posterior_mll_cache.clear()
        else:
            for key in keys:
                if key in self.posterior_mll_cache:
                    del self.posterior_mll_cache[key]

    def clear_rmse_cache(self, keys: List[str] = None):
        if keys is None:
            self.rmse_cache.clear()
        else:
            for key in keys:
                if key in self.rmse_cache:
                    del self.rmse_cache[key]

    def get_mechanism(self, node, graph: nx.DiGraph = None, parents: List[str] = None) -> Mechanism:
        assert graph is not None or parents is not None

        # get unique mechanism key
        parents = get_parents(node, graph) if parents is None else list(set(parents))
        key = get_mechanism_key(node, parents)

        # return mechanism if it already exists
        if key in self.mechanisms:
            return self.mechanisms[key]

        # if mechanism does not yet exist in the model, create a new mechanism
        num_parents = len(parents)
        self.mechanisms[key] = self.create_mechanism(num_parents)
        return self.mechanisms[key]

    def mechanism_rmse(self, experiments: List[Experiment], node: str, parents: List[str],
                       use_cache=False) -> torch.Tensor:
        # return cache value if available
        key = get_mechanism_key(node, parents)
        if use_cache and key in self.rmse_cache:
            return self.rmse_cache[key]

        # compute rmse value otherwise
        inputs, targets = gather_data(experiments, node, parents=parents, mode='independent_samples')
        if targets is None:
            return torch.tensor(0.)

        if len(parents) == 0:
            prediction = self.mechanisms[key](targets.unsqueeze(-1)).squeeze()
            rmse = vector_norm(targets.squeeze() - prediction) / vector_norm(targets)
        else:
            prediction = self.mechanisms[key](inputs).squeeze()
            rmse = vector_norm(targets.squeeze() - prediction) / vector_norm(targets)

        # cache rmse value
        if use_cache:
            self.rmse_cache[key] = rmse

        return rmse

    def rmse(self, experiments: List[Experiment], graph: nx.DiGraph, use_cache=False) -> torch.Tensor:
        rmse = torch.tensor(0.)
        for node in self.node_labels:
            parents = get_parents(node, graph)
            rmse += self.mechanism_rmse(experiments, node, parents, use_cache)

        return rmse / len(self.node_labels)

    def node_mll(self, experiments: List[Experiment], node: str, parents: List[str], prior_mode=False,
                 use_cache=False, mode='joint', reduce=True) -> torch.Tensor:
        cache = self.prior_mll_cache if prior_mode else self.posterior_mll_cache
        key = get_mechanism_key(node, parents)
        mll = torch.tensor(0.)
        if not use_cache or key not in cache:
            # gather data from the experiments
            inputs, targets = gather_data(experiments, node, parents=parents, mode=mode)
            # check if we have any data for this node
            if targets is not None:
                # compute log-likelihood
                mechanism = self.get_mechanism(node, parents=parents)
                try:
                    mll = mechanism.mll(inputs, targets, prior_mode, reduce=reduce)
                except Exception as e:
                    print(
                        f'Exception occured in GaussianProcessModel.mll() when computing MLL for mechanism {key} '
                        f'with prior mode {prior_mode} and use cache {use_cache}:')
                    print(e)
            # cache mll
            cache[key] = mll
        else:
            mll = cache[key]

        return mll

    def mll(self, experiments: List[Experiment], graph: nx.DiGraph, prior_mode=False, use_cache=False,
            mode='joint', reduce=True) -> torch.Tensor:
        mll = torch.tensor(0.)
        for node in self.node_labels:
            parents = get_parents(node, graph)
            mll = mll + self.node_mll(experiments, node, parents, prior_mode, use_cache, mode=mode, reduce=reduce)
        return mll

    def log_hp_prior(self, graph: nx.DiGraph) -> torch.Tensor:
        keys = []
        for node in self.node_labels:
            parents = get_parents(node, graph)
            keys.append(get_mechanism_key(node, parents))

        return self.mechanism_log_hp_priors(keys)

    def select_gp_hyperparams(self, posterior_hp: bool):
        gps = [mech for mech in self.mechanisms.values() if isinstance(mech, GaussianProcess)]
        for gp in gps:
            gp.gp.select_hyperparameters(posterior_hp)

    def expected_noise_entropy(self, interventions, graph: nx.DiGraph, use_cache=False) -> torch.Tensor:
        entropy = torch.tensor(0.)
        for node in self.node_labels:
            if node in interventions:
                continue

            parents = get_parents(node, graph)
            key = get_mechanism_key(node, parents)
            if not use_cache or key not in self.entropy_cache:
                # compute and cache entropy
                mechanism = self.get_mechanism(node, parents=parents)
                mechanism_entropy = mechanism.expected_noise_entropy()
                self.entropy_cache[key] = mechanism_entropy
            else:
                # take entropy from cache
                mechanism_entropy = self.entropy_cache[key]

            entropy += mechanism_entropy
        return entropy

    def get_num_mechanisms(self):
        return len(self.mechanisms)

    def gp_mlls(self, experiments: List[Experiment], keys: List[str] = None, prior_mode=False) -> torch.Tensor:
        # if no keys are given compute mlls for all mechanisms
        keys = self.mechanisms.keys() if keys is None else keys

        mlls = torch.tensor(0.)
        for key in keys:
            node, parents = resolve_mechanism_key(key)

            # gather data from the experiments
            inputs, targets = gather_data(experiments, node, parents=parents, mode='joint')

            # check if we have any data for this node
            if targets is None:
                continue

            # compute log-likelihood
            mechanism = self.mechanisms[key]
            try:
                mlls += mechanism.mll(inputs, targets, prior_mode) / targets.numel()
            except Exception as e:
                print(
                    f'Exception occured in GaussianProcessModel.gp_mlls() when computing MLL for mechanism '
                    f'{key} with prior  mode {prior_mode}:')
                print(e)
                if isinstance(mechanism, GaussianProcess):
                    print('Resampling GP hyperparameters...')
                    mechanism.gp.init_hyperparams()
        return mlls

    def mechanism_log_hp_priors(self, keys: List[str] = None) -> torch.Tensor:
        # if no keys are given compute mlls for all mechanisms
        keys = self.mechanisms.keys() if keys is None else keys

        gps = [self.mechanisms[key] for key in keys if isinstance(self.mechanisms[key], GaussianProcess)]
        if not gps:
            return torch.tensor(0.)

        log_priors = torch.stack([gp.gp.hyperparam_log_prior() for gp in gps])
        return log_priors.sum()

    def create_mechanism(self, num_parents: int, param_dict: Dict[str, Any] = None) -> Mechanism:
        if num_parents > 0:
            return GaussianProcess(num_parents, param_dict=param_dict)
        else:
            return GaussianRootNode(param_dict=param_dict)

    def set_data(self, experiments: List[Experiment], keys: List[str] = None):
        # if no keys are given set data for all mechanisms
        keys = self.mechanisms.keys() if keys is None else keys

        for key in keys:
            node, parents = resolve_mechanism_key(key)

            # gather data from the experiments
            inputs, targets = gather_data(experiments, node, parents=parents, mode='joint')

            # check if we have any data for this node
            if targets is None:
                continue

            # set GP data
            self.mechanisms[key].set_data(inputs, targets)

    def sample(self, interventions: dict, batch_size: int, num_batches: int, graph: nx.DiGraph) -> Experiment:
        data = dict()
        for node in self.topological_orders[get_graph_key(graph)]:
            # check if node is intervened upon
            if node in interventions:
                node_samples = torch.ones(num_batches, batch_size, 1) * interventions[node]
            else:
                # sample from mechanism
                parents = get_parents(node, graph)
                mechanism = self.get_mechanism(node, parents=parents)
                if not parents:
                    node_samples = mechanism.sample(torch.empty(num_batches, batch_size, 1))
                else:
                    x = torch.cat([data[parent] for parent in parents], dim=-1)
                    assert x.shape == (num_batches, batch_size, mechanism.in_size)
                    node_samples = mechanism.sample(x)

            # store samples
            data[node] = node_samples

        return Experiment(interventions, data)

    def interventional_mll(self, targets, node: str, interventions: dict, graph: nx.DiGraph, reduce=True):
        assert targets.dim() == 2, print(f'Invalid shape {targets.shape}')
        num_batches, batch_size = targets.shape
        num_mc_samples = self.cfg.imll_mc_samples
        parents = get_parents(node, graph)
        mechanism = self.get_mechanism(node, parents=parents)

        if len(parents) == 0:
            # if we have a root note imll is simple
            mll = mechanism.mll(None, targets, prior_mode=False, reduce=False)
            assert mll.shape == (num_batches,), print(f'Invalid shape {mll.shape}!')
            return mll.sum() if reduce else mll

        # otherwise, do MC estimate via ancestral sampling
        samples = self.sample(interventions, batch_size, num_mc_samples, graph)
        # assemble inputs and targets
        inputs, _ = gather_data([samples], node, parents=parents, mode='independent_batches')
        inputs = inputs.unsqueeze(0).expand(num_batches, -1, -1, -1)
        assert inputs.shape == (num_batches, num_mc_samples, batch_size, len(parents))
        targets = targets.unsqueeze(1).expand(-1, num_mc_samples, batch_size)
        assert targets.shape == (num_batches, num_mc_samples, batch_size)
        # compute interventional mll
        mll = mechanism.mll(inputs, targets, prior_mode=False, reduce=False)
        assert mll.shape == (num_batches, num_mc_samples), print(f'Invalid shape {mll.shape}!')
        mll = mll.logsumexp(dim=1) - math.log(num_mc_samples)
        return mll.sum() if reduce else mll

    def sample_queries(self, queries: List[InterventionalDistributionsQuery], num_mc_queries: int,
                       num_batches_per_query: int, graph: nx.DiGraph):

        interventional_queries = [query.clone() for query in queries]
        with torch.no_grad():
            for query in interventional_queries:
                experiments = []
                for i in range(num_mc_queries):
                    interventions = query.sample_intervention()
                    experiments.append(self.sample(interventions, 1, num_batches_per_query, graph))

                query.set_sample_queries(experiments)

        return interventional_queries

    def query_log_probs(self, queries: List[InterventionalDistributionsQuery], graph: nx.DiGraph):
        num_queries = len(queries)
        num_mc_queries = len(queries[0].sample_queries)
        num_batches_per_query = queries[0].sample_queries[0].num_batches

        query_lls = torch.zeros(num_mc_queries, num_queries, num_batches_per_query)
        for i in range(num_mc_queries):
            for query_idx, query in enumerate(queries):
                query_node = query.query_nodes[0]  # ToDo: supports only single query node!!!
                targets = query.sample_queries[i].data[query_node].squeeze(-1)
                imll = self.interventional_mll(targets, query_node, query.sample_queries[i].interventions, graph,
                                               reduce=False)
                query_lls[i, query_idx] = imll

        return query_lls.sum(dim=1)

    def update_gp_hyperparameters(self, experiments: List[Experiment], keys: Optional[List[str]] = None):
        # if no keys are given update all gps' hyperparams
        if keys is None:
            keys = list(self.mechanisms.keys()) if keys is None else keys

        # gather keys with older update times
        update_time = len(experiments)
        keys = [key for key in keys if self.mechanism_update_times[key] != update_time]
        if not keys:
            return

        self.set_data(experiments)

        # keep only GP keys
        keys = [key for key in keys if len(resolve_mechanism_key(key)[1])]
        if not keys:
            return

        print(f'Updating {len(keys)} GP\'s hyperparams on {len(experiments)} experiments...', flush=True)

        # batch all mechanisms to avoid out of mem
        batch_size = self.cfg.opt_batch_size
        num_full_batches = len(keys) // batch_size
        key_batches = [keys[i * batch_size:(i + 1) * batch_size] for i in range(num_full_batches)]
        if len(keys) % batch_size > 0:
            key_batches.append(keys[batch_size * num_full_batches:])

        # update GP hyperparams
        for bidx, batch in enumerate(key_batches):
            print(f'Updating {len(batch)} GPs in batch {bidx + 1}/{len(key_batches)}', flush=True)
            optimizer = torch.optim.RMSprop(self.get_parameters(batch), lr=self.cfg.lr)
            losses = []
            self.train(batch)
            for i in range(self.cfg.num_steps):
                optimizer.zero_grad()
                loss = -self.gp_mlls(experiments, batch, prior_mode=True) - self.mechanism_log_hp_priors(batch)
                loss /= len(batch)

                loss.backward()
                optimizer.step()
                losses.append(loss.item())

                if i > self.cfg.es_min_steps:
                    change = (torch.tensor(losses[-1 - self.cfg.es_win_size:-1]).mean() -
                              torch.tensor(losses[-2 - self.cfg.es_win_size:-2]).mean()).abs()
                    if change < self.cfg.es_threshold:
                        print(f'Stopping GP parameter optimization after {i + 1} steps ...', flush=True)
                        break

                if self.cfg.log_interval > 0 and i % self.cfg.log_interval == 0:
                    print(f'Step {i + 1} of {self.cfg.num_steps}, GP loss is {loss.item()}...', flush=True)

            self.clear_prior_mll_cache(batch)
            self.clear_posterior_mll_cache(batch)

        # put mechanisms into eval mode
        self.eval(keys)

        for key in keys:
            self.mechanism_update_times[key] = update_time

    def submodel(self, graphs):
        mechanisms_keys = {get_mechanism_key(node, get_parents(node, graph)) for graph in graphs for node in graph}
        submodel = self.__class__(self.node_labels)
        submodel.mechanisms = {key: self.mechanisms[key] for key in mechanisms_keys}
        submodel.topological_orders = self.topological_orders
        return submodel

    def param_dict(self) -> Dict[str, Any]:
        mechanism_param_dict = {key: m.param_dict() for key, m in self.mechanisms.items()}
        params = {'node_labels': self.node_labels,
                  'mechanism_init_times': self.mechanism_init_times,
                  'mechanism_update_times': self.mechanism_update_times,
                  'topological_orders': self.topological_orders,
                  'mechanism_param_dict': mechanism_param_dict,
                  'cfg_param_dict': self.cfg.param_dict()}
        return params

    def load_param_dict(self, param_dict):
        self.cfg = GPModelConfig()
        self.cfg.load_param_dict(param_dict['cfg_param_dict'])

        self.node_labels = param_dict['node_labels']
        self.mechanism_init_times = param_dict['mechanism_init_times']
        self.mechanism_update_times = param_dict['mechanism_update_times']
        self.topological_orders = param_dict['topological_orders']
        self.mechanisms = dict()
        for key, d in param_dict['mechanism_param_dict'].items():
            self.mechanisms[key] = self.create_mechanism(d['in_size'], param_dict=d)
