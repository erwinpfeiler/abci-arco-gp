import itertools
import math
from typing import Optional, List, Dict, Any

import networkx as nx
import torch
from torch.linalg import vector_norm

from src.config import GPModelConfig
from src.environments.environment import Experiment, gather_data
from src.environments.experiment import InterventionalDistributionsQuery
from src.mechanism_models.mechanisms import SharedDataGaussianProcess, GaussianRootNode, get_mechanism_key, \
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


class SharedDataGaussianProcessModel:
    gps: Dict[str, SharedDataGaussianProcess]
    root_mechs: Dict[str, GaussianRootNode]

    def __init__(self, node_labels: List[str] = None, cfg: GPModelConfig = None, param_dict: Dict[str, Any] = None):
        assert node_labels is not None or param_dict is not None
        if param_dict is not None:
            self.load_param_dict(param_dict)
        else:
            # load config
            self.cfg = GPModelConfig() if cfg is None else cfg

            self.node_labels = sorted(list(set(node_labels)))
            num_nodes = len(self.node_labels)
            self.node_to_dim_map = {node: idx for idx, node in enumerate(self.node_labels)}
            self.gps = {n: SharedDataGaussianProcess(num_nodes, self.node_to_dim_map, self.cfg.linear) for n in
                        self.node_labels}
            self.root_mechs = {n: GaussianRootNode() for n in self.node_labels}
            self.mechanism_update_times = {get_mechanism_key(node, []): 0 for node in self.node_labels}
            self.gp_sample_times = dict()
            self.topological_orders = dict()
            self.topological_order_sample_times = dict()

        # init caches
        self.prior_mll_cache = dict()
        self.posterior_mll_cache = dict()
        self.rmse_cache = dict()

    def get_parameters(self, keys: List[str] = None):
        params = [gp.get_parameters(keys) for gp in self.gps.values()]
        return itertools.chain(*params)

    def init_topological_order(self, graph: nx.DiGraph, init_time: int = 0):
        graph_key = get_graph_key(graph)
        if nx.is_directed_acyclic_graph(graph):
            self.topological_order_sample_times[graph_key] = init_time
            if graph_key not in self.topological_orders:
                self.topological_orders[graph_key] = list(nx.topological_sort(graph))

    def init_graph_mechanisms(self, graph: nx.DiGraph, init_time: int = 0):
        initialized_mechanisms = []
        for node in self.node_labels:
            parents = get_parents(node, graph)
            key = get_mechanism_key(node, parents)
            initialized_mechanisms.append(key)
            if len(parents):
                self.gp_sample_times[key] = init_time
                if not self.gps[node].exists(key):
                    self.gps[node].init_kernel(key)
                    self.mechanism_update_times[key] = 0

        return initialized_mechanisms

    def init_mechanisms_from_keys(self, keys: List[str], init_time: int = 0):
        for key in keys:
            node, parents = resolve_mechanism_key(key)
            if len(parents):
                self.gp_sample_times[key] = init_time
                if not self.gps[node].exists(key):
                    self.gps[node].init_kernel(key)
                    self.mechanism_update_times[key] = 0

        return keys

    def discard_gps(self):
        # if number of mechanisms is less than the cfg threshold we don't delete anything
        current_num_gps = self.get_num_gps()
        if current_num_gps < self.cfg.discard_threshold_gps:
            return

        # determine discard time
        num_to_discard = 0
        discard_time = 0
        for sample_time in sorted(set(self.gp_sample_times.values())):
            num_to_discard += self.get_num_gps(sample_time)
            discard_time = sample_time
            if current_num_gps - num_to_discard <= int(0.8 * self.cfg.discard_threshold_gps):
                break

        # keep latest sampled gps
        latest_sampling_time = max(self.gp_sample_times.values())
        discard_time = discard_time if discard_time < latest_sampling_time else latest_sampling_time - 1

        # determine gp keys to be discarded
        keys = [key for key, time in self.gp_sample_times.items() if time <= discard_time]
        num_to_discard = len(keys)
        if num_to_discard == 0:
            return

        print(f'Discarding {num_to_discard} gps older than {discard_time}/{latest_sampling_time} draws, '
              f'remaining {current_num_gps - num_to_discard}...')

        # discard gps
        for gp in self.gps.values():
            gp.delete_kernels(keys)

        keys = set(keys)
        self.gp_sample_times = {k: v for k, v in self.gp_sample_times.items() if k not in keys}
        self.mechanism_update_times = {k: v for k, v in self.mechanism_update_times.items() if k not in keys}
        self.prior_mll_cache = {k: v for k, v in self.prior_mll_cache.items() if k not in keys}
        self.posterior_mll_cache = {k: v for k, v in self.posterior_mll_cache.items() if k not in keys}
        self.rmse_cache = {k: v for k, v in self.rmse_cache.items() if k not in keys}

    def discard_topo_orders(self):
        # if number of topo orders is less than the cfg threshold we don't delete anything
        current_num_orders = self.get_num_topo_orders()
        if current_num_orders < self.cfg.discard_threshold_topo_orders:
            return

        # determine discard time
        num_to_discard = 0
        discard_time = 0
        for sample_time in sorted(set(self.topological_order_sample_times.values())):
            num_to_discard += self.get_num_topo_orders(sample_time)
            discard_time = sample_time
            if current_num_orders - num_to_discard <= int(0.8 * self.cfg.discard_threshold_topo_orders):
                break

        # keep latest sampled gps
        latest_sampling_time = max(self.topological_order_sample_times.values())
        discard_time = discard_time if discard_time < latest_sampling_time else latest_sampling_time - 1

        # determine orders to be discarded
        keys = [key for key, time in self.topological_order_sample_times.items() if time <= discard_time]
        num_to_discard = len(keys)
        if num_to_discard == 0:
            return

        print(f'Discarding {num_to_discard} topological orders older than {discard_time}/{latest_sampling_time} draws, '
              f'remaining {current_num_orders - num_to_discard}...')

        self.topological_orders = {k: v for k, v in self.topological_orders.items() if k not in keys}
        self.topological_order_sample_times = {k: v for k, v in self.topological_order_sample_times.items() if
                                               k not in keys}

    def eval(self):
        for gp in self.gps.values():
            gp.eval()

    def train(self):
        for gp in self.gps.values():
            gp.train()

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

    def apply_mechanism(self, inputs: torch.Tensor, key: str) -> torch.Tensor:
        node, parents = resolve_mechanism_key(key)
        if len(parents) == 0:
            return self.root_mechs[node](inputs)
        else:
            return self.gps[node](inputs, key)

    def mechanism_rmse(self, experiments: List[Experiment], node: str, parents: List[str],
                       use_cache=False) -> torch.Tensor:
        # return cache value if available
        key = get_mechanism_key(node, parents)
        if use_cache and key in self.rmse_cache:
            return self.rmse_cache[key]

        # compute rmse value otherwise
        rmse = torch.tensor(0.)
        if len(parents) == 0:
            inputs, targets = gather_data(experiments, node, parents=parents, mode='independent_samples')
            if targets is not None:
                prediction = self.root_mechs[node](targets.unsqueeze(-1)).squeeze()
                rmse = vector_norm(targets.squeeze() - prediction) / vector_norm(targets)
        else:
            inputs, targets = gather_data(experiments, node, parents=self.node_labels, mode='independent_samples')
            if targets is not None:
                prediction = self.gps[node](inputs, key).squeeze()
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
        if not use_cache or key not in cache:
            # gather data from the experiments
            inputs, targets = gather_data(experiments, node, parents=self.node_labels, mode=mode)

            # check if we have any data for this node
            mll = torch.tensor(0.)
            if targets is not None:
                # compute log-likelihood
                if len(parents):
                    try:
                        mll = self.gps[node].mll(inputs, targets, key, prior_mode, reduce=reduce)
                    except Exception as e:
                        print(
                            f'Exception occured in GaussianProcessModel.mll() when computing MLL for mechanism {key} '
                            f'with prior mode {prior_mode} and use cache {use_cache}:')
                        print(e)
                else:
                    mll = self.root_mechs[node].mll(None, targets, prior_mode, reduce=reduce)

            # cache mll
            if use_cache:
                cache[key] = mll
        else:
            mll = cache[key].clone()

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

    def mechanism_log_hp_priors(self, keys: List[str]) -> torch.Tensor:
        log_priors = torch.tensor(0.)
        for key in keys:
            node, parents = resolve_mechanism_key(key)
            if self.gps[node].exists(key):
                log_priors += self.gps[node].hyperparam_log_prior(key)

        return log_priors

    def get_num_gps(self, sample_time: int = None):
        if sample_time is None:
            return len(self.gp_sample_times.keys())
        else:
            tmp = [None for time in self.gp_sample_times.values() if time == sample_time]
            return len(tmp)

    def get_num_topo_orders(self, sample_time: int = None):
        if sample_time is None:
            return len(self.topological_order_sample_times.keys())
        else:
            tmp = [None for time in self.topological_order_sample_times.values() if time == sample_time]
            return len(tmp)

    def set_data(self, experiments: List[Experiment]):
        for node in self.node_labels:
            # gather data from the experiments
            inputs, targets = gather_data(experiments, node, parents=self.node_labels, mode='joint')

            # check if we have any data for this node
            if targets is None:
                continue

            # set mechanism data
            self.gps[node].set_data(inputs, targets)
            self.root_mechs[node].set_data(inputs, targets)

        self.eval()

    def sample(self, interventions: dict, batch_size: int, num_batches: int, graph: nx.DiGraph) -> Experiment:
        data = dict()
        x = torch.zeros(num_batches, batch_size, len(self.node_labels))
        for node in self.topological_orders[get_graph_key(graph)]:
            # check if node is intervened upon
            if node in interventions:
                node_samples = torch.ones(num_batches, batch_size, 1) * interventions[node]
            else:
                # sample from mechanism
                parents = get_parents(node, graph)
                if not parents:
                    node_samples = self.root_mechs[node].sample(torch.empty(num_batches, batch_size, 1))
                else:
                    node_samples = self.gps[node].sample(x, get_mechanism_key(node, parents))

            # store samples
            x[:, :, self.node_to_dim_map[node]] = node_samples.squeeze(-1)
            data[node] = node_samples

        return Experiment(interventions, data)

    def sample_ace(self, target: str, interventions: dict, num_samples: int, graph: nx.DiGraph) -> torch.Tensor:
        # check if target is intervened upon (although it makes little sense)
        if target in interventions:
            return torch.ones(num_samples) * interventions[target]

        # if the target is a root node in the graph, the ATE is trivial
        parents_target = get_parents(target, graph)
        if not parents_target:
            return self.root_mechs[target](torch.empty(num_samples, 1, 1)).squeeze()

        # otherwise, perform ancestral sampling to estimate the ATE
        x = torch.zeros(num_samples, 1, len(self.node_labels))
        for node in self.topological_orders[get_graph_key(graph)]:
            if node == target:
                # if we sampled all ancestors we can compute the ATE for each ancestral sample
                return self.gps[node](x, get_mechanism_key(target, parents_target)).squeeze()
            elif node in interventions:
                # check if node is intervened upon
                node_samples = torch.ones(num_samples, 1, 1) * interventions[node]
            else:
                # sample from mechanism
                parents = get_parents(node, graph)
                if not parents:
                    node_samples = self.root_mechs[node].sample(torch.empty(num_samples, 1, 1))
                else:
                    node_samples = self.gps[node].sample(x, get_mechanism_key(node, parents))

            # store samples
            x[:, :, self.node_to_dim_map[node]] = node_samples.squeeze(-1)

        assert False, print(f'Node {target} not in graph!.')

    def sample_aces(self, interventions: dict, num_samples: int, graph: nx.DiGraph) -> torch.Tensor:
        # perform ancestral sampling to estimate the ACEs
        x = torch.zeros(num_samples, 1, len(self.node_labels))
        aces = torch.ones(len(self.node_labels), num_samples)
        for node in self.topological_orders[get_graph_key(graph)]:
            if node in interventions:
                # check if node is intervened upon
                node_samples = torch.ones(num_samples, 1, 1) * interventions[node]
                ace_samples = torch.ones(num_samples) * interventions[node]
            else:
                # sample from mechanism
                parents = get_parents(node, graph)
                if not parents:
                    node_samples = self.root_mechs[node].sample(torch.empty(num_samples, 1, 1))
                    ace_samples = self.root_mechs[node](torch.empty(num_samples, 1, 1)).squeeze()
                else:
                    key = get_mechanism_key(node, parents)
                    node_samples = self.gps[node].sample(x, key)
                    ace_samples = self.gps[node](x, key).squeeze()

            # store samples
            x[:, :, self.node_to_dim_map[node]] = node_samples.squeeze(-1)
            aces[self.node_to_dim_map[node]] = ace_samples

        return aces

    def interventional_mll(self, targets, node: str, interventions: dict, graph: nx.DiGraph, reduce=True):
        assert targets.dim() == 2, print(f'Invalid shape {targets.shape}')
        num_batches, batch_size = targets.shape
        num_mc_samples = self.cfg.imll_mc_samples
        parents = get_parents(node, graph)

        if len(parents) == 0:
            # if we have a root note imll is simple
            mll = self.root_mechs[node].mll(None, targets, prior_mode=False, reduce=False)
            assert mll.shape == (num_batches,), print(f'Invalid shape {mll.shape}!')
            return mll.sum() if reduce else mll

        # otherwise, do MC estimate via ancestral sampling
        samples = self.sample(interventions, batch_size, num_mc_samples, graph)
        # assemble inputs and targets
        inputs, _ = gather_data([samples], node, parents=self.node_labels, mode='independent_batches')
        inputs = inputs.unsqueeze(0).expand(num_batches, -1, -1, -1)
        assert inputs.shape == (num_batches, num_mc_samples, batch_size, len(self.node_labels)), print(inputs.shape)
        targets = targets.unsqueeze(1).expand(-1, num_mc_samples, -1)
        assert targets.shape == (num_batches, num_mc_samples, batch_size)
        # compute interventional mll
        key = get_mechanism_key(node, parents)
        mll = self.gps[node].mll(inputs, targets, key, prior_mode=False, reduce=False)
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
            keys = [self.gps[node].get_keys() for node in self.node_labels]
            keys = list(itertools.chain(*keys))

        # gather keys with older update times
        update_time = len(experiments)
        keys = [key for key in keys if self.mechanism_update_times[key] != update_time]
        if not keys:
            return

        with torch.no_grad():
            self.set_data(experiments)

        # keep only GP keys and sort by node
        keys_by_node = {node: [] for node in self.node_labels}
        for key in keys:
            node, parents = resolve_mechanism_key(key)
            if len(parents) == 0:
                self.mechanism_update_times[key] = update_time
            else:
                keys_by_node[node].append(key)

        num_total_gps = sum([len(keys_by_node[node]) for node in self.node_labels])
        print(f'Updating {num_total_gps} GP\'s hyperparams on {len(experiments)} experiments...', flush=True)

        # update gps per target node
        self.train()
        num_finished_gps = 0
        for node in self.node_labels:
            if len(keys_by_node[node]) == 0:
                continue

            inputs, targets = gather_data(experiments, node, parents=self.node_labels, mode='joint')
            if targets is None:
                continue

            # batch all mechanisms to avoid out of mem
            batch_size = self.cfg.opt_batch_size
            num_full_batches = len(keys_by_node[node]) // batch_size
            key_batches = [keys_by_node[node][i * batch_size:(i + 1) * batch_size] for i in range(num_full_batches)]
            if len(keys_by_node[node]) % batch_size > 0:
                key_batches.append(keys_by_node[node][batch_size * num_full_batches:])

            # update GP hyperparams
            for bidx, batch in enumerate(key_batches):
                optimizer = torch.optim.RMSprop(self.get_parameters(batch), lr=self.cfg.lr)
                losses = []
                for i in range(self.cfg.num_steps):

                    mlls = torch.tensor(0.)
                    for key in batch:
                        node, parents = resolve_mechanism_key(key)
                        # compute marginal log-likelihood
                        try:
                            mlls += self.gps[node].mll(inputs, targets, key, True) / targets.numel()
                        except Exception as e:
                            print('Exception occured in SharedDataGaussianProcessModel.update_gp_hyperparameters() '
                                  f'when computing MLL for mechanism {key} in prior mode:')
                            print(e)
                            print('Resampling GP hyperparameters...')
                            self.gps[node].init_hyperparams(key)

                    loss = -(mlls + self.mechanism_log_hp_priors(batch)) / len(batch)

                    optimizer.zero_grad()
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
                        print(f'Step {i + 1} of {self.cfg.num_steps}, GP loss is {losses[-1]}...', flush=True)

                num_finished_gps += len(batch)
                print(f'Updated {num_finished_gps}/{num_total_gps} GPs... ', flush=True)

        # put mechanisms back into eval mode
        self.eval()

        for key in keys:
            self.mechanism_update_times[key] = update_time

    def submodel(self, graphs):
        return self

    def param_dict(self) -> Dict[str, Any]:
        gp_param_dict = {node: gp.param_dict() for node, gp in self.gps.items()}
        root_mech_param_dict = {node: m.param_dict() for node, m in self.root_mechs.items()}
        params = {'node_labels': self.node_labels,
                  'mechanism_update_times': self.mechanism_update_times,
                  'gp_sample_times': self.gp_sample_times,
                  'topological_orders': self.topological_orders,
                  'topological_order_sample_times': self.topological_order_sample_times,
                  'gp_param_dict': gp_param_dict,
                  'root_mech_param_dict': root_mech_param_dict,
                  'cfg_param_dict': self.cfg.param_dict()}
        return params

    def load_param_dict(self, param_dict):
        self.node_labels = param_dict['node_labels']
        self.node_to_dim_map = {node: idx for idx, node in enumerate(self.node_labels)}
        self.mechanism_update_times = param_dict['mechanism_update_times']
        self.gp_sample_times = param_dict['gp_sample_times']
        self.topological_orders = param_dict['topological_orders']
        self.topological_order_sample_times = param_dict['topological_order_sample_times']
        self.cfg = GPModelConfig()
        self.cfg.load_param_dict(param_dict['cfg_param_dict'])

        self.gps = {}
        for node, d in param_dict['gp_param_dict'].items():
            self.gps[node] = SharedDataGaussianProcess(len(self.node_labels), param_dict=d)
            self.gps[node].load_param_dict(d)
        self.root_mechs = {}
        for node, d in param_dict['root_mech_param_dict'].items():
            self.root_mechs[node] = GaussianRootNode(param_dict=d)
