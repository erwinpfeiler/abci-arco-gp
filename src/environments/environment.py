import math
import os
import random
import string
from typing import List, Dict, Any, Optional

import networkx as nx
import pandas as pd
import torch

from src.config import EnvironmentConfig
from src.environments.experiment import Experiment, gather_data, get_exp_param_dicts
from src.mechanism_models.mechanisms import GaussianProcess, GaussianRootNode, AdditiveSigmoids, resolve_mechanism_key
from src.utils.graphs import get_parents, graph_to_adj_mat, dag_to_cpdag


class Environment:
    def __init__(self, num_nodes: int = None, cfg: EnvironmentConfig = None, init_mechanisms: bool = True,
                 graph: nx.DiGraph = None, param_dict: Dict[str, Any] = None):
        assert num_nodes is not None or param_dict is not None
        if param_dict is not None:
            self.load_param_dict(param_dict)
        else:
            self.cfg = EnvironmentConfig() if cfg is None else cfg
            self.observational_train_data: Optional[List[Experiment]] = None
            self.interventional_train_data: Optional[List[Experiment]] = None
            self.observational_test_data: Optional[List[Experiment]] = None
            self.interventional_test_data: Optional[List[Experiment]] = None
            self.normalisation_means: Optional[Dict[str, torch.Tensor]] = None
            self.normalisation_stds: Optional[Dict[str, torch.Tensor]] = None

            # generate unique env name
            seed = ''.join([random.choice(string.ascii_letters + string.digits) for _ in range(self.cfg.id_length)])
            self.name = self.__class__.__name__ + f'-{num_nodes}-{seed}'

            # construct graph
            self.num_nodes = num_nodes
            self.graph = self.construct_graph(num_nodes) if graph is None else graph
            self.topological_order = list(nx.topological_sort(self.graph))
            self.node_labels = sorted(list(set(self.graph.nodes)))

            # optional: restrict intervenable nodes
            if self.cfg.non_intervenable_nodes is not None:
                self.non_intervenable_nodes = self.cfg.non_intervenable_nodes
            elif self.cfg.frac_non_intervenable_nodes is not None:
                num_non_intervenable_nodes = int(num_nodes * self.cfg.frac_non_intervenable_nodes)
                node_idc = torch.randperm(num_nodes)[:num_non_intervenable_nodes]
                self.non_intervenable_nodes = set(self.node_labels[i] for i in node_idc)
            else:
                self.non_intervenable_nodes = set()

            self.intervenable_nodes = set(self.node_labels) - self.non_intervenable_nodes

            # set intervention bounds for experiment design
            self.intervention_bounds = dict(
                zip(self.node_labels, [self.cfg.intervention_bounds for _ in range(self.num_nodes)]))

            # generate mechanisms & datasets
            if init_mechanisms:
                mechanisms = []
                for node in self.node_labels:
                    parents = get_parents(node, self.graph)
                    mechanisms.append(self.create_mechanism(len(parents)))
                self.mechanisms = dict(zip(self.node_labels, mechanisms))

                if self.cfg.generate_static_obs_dataset:
                    self.generate_obs_dataset()
                elif self.cfg.normalise_data:
                    # if no train data was generated we need to init the normalisation constants here
                    if self.cfg.normalise_data and self.normalisation_means is None:
                        ref_data = self.sample({}, 500, 1)
                        self.normalisation_means = {}
                        self.normalisation_stds = {}
                        for node, values in ref_data.data.items():
                            self.normalisation_means[node] = values.mean()
                            self.normalisation_stds[node] = values.std()

                if self.cfg.generate_static_intr_dataset:
                    self.generate_intr_dataset()

                if self.cfg.generate_test_queries:
                    self.generate_test_queries()

            else:
                self.mechanisms = None

    def generate_obs_dataset(self):
        # generate observational training data
        if self.cfg.num_observational_train_samples > 0:
            num_samples = self.cfg.num_observational_train_samples
            self.observational_train_data = [self.sample({}, 1, num_samples)]

            # init normalisation constants
            if self.cfg.normalise_data:
                self.normalisation_means = {}
                self.normalisation_stds = {}
                for node, values in self.observational_train_data[0].data.items():
                    self.normalisation_means[node] = values.mean()
                    self.normalisation_stds[node] = values.std()

                self.observational_train_data[0].normalise(self.normalisation_means, self.normalisation_stds)

        # generate observational test data
        if self.cfg.num_observational_test_samples > 0:
            num_samples = self.cfg.num_observational_test_samples
            self.observational_test_data = [self.sample({}, 1, num_samples)]

    def generate_intr_dataset(self):
        # generate interventional training data
        if self.cfg.num_train_interventions > 0 and self.cfg.num_interventional_train_samples > 0:
            self.interventional_train_data = []
            for sidx in range(self.cfg.num_train_interventions):
                target_node = random.choice(list(self.intervenable_nodes))
                bounds = self.intervention_bounds[target_node]
                intr_value = torch.rand(1) * (bounds[1] - bounds[0]) + bounds[0]
                exp = self.sample({target_node: intr_value}, batch_size=self.cfg.num_interventional_train_samples)
                self.interventional_train_data.append(exp)

        # generate interventional test data
        if self.cfg.num_test_interventions > 0 and self.cfg.num_interventional_test_samples > 0:
            self.interventional_test_data = []
            for node in self.node_labels:
                bounds = self.intervention_bounds[node]
                for bidx in range(self.cfg.num_test_interventions):
                    intr_value = torch.rand(1) * (bounds[1] - bounds[0]) + bounds[0]
                    exp = self.sample({node: intr_value}, batch_size=self.cfg.num_interventional_test_samples)
                    self.interventional_test_data.append(exp)

    def generate_test_queries(self):
        if self.cfg.num_test_queries > 0 and self.cfg.interventional_queries is not None:
            with torch.no_grad():
                for query in self.cfg.interventional_queries:
                    experiments = []
                    for i in range(self.cfg.num_test_queries):
                        interventions = query.sample_intervention()
                        experiments.append(self.sample(interventions, 1))

                    query.set_sample_queries(experiments)

    def create_mechanism(self, num_parents: int):
        if self.cfg.mechanism_model == 'gp-model':
            if num_parents > 0:
                return GaussianProcess(num_parents, static=True, linear=self.cfg.linear)
            else:
                return GaussianRootNode(static=True)
        elif self.cfg.mechanism_model == 'additive-sigmoids':
            if num_parents > 0:
                return AdditiveSigmoids(num_parents)
            else:
                return GaussianRootNode(static=True)

        assert False, print(f'Invalid mechanism model {self.cfg.mechanism_model}!')

    def apply_mechanism(self, inputs: torch.Tensor, key: str, normalised: bool = True) -> torch.Tensor:
        node, parents = resolve_mechanism_key(key)
        if get_parents(node, self.graph) != parents:
            print(f'No such mechanism in the environment: {key}.')
            return torch.tensor(0.)

        # if data is normalised we need to un-normalise before we can evaluate the mechanisms
        if normalised and self.normalisation_means is not None and self.normalisation_stds is not None:
            means = torch.tensor([self.normalisation_means[p] for p in parents])
            stds = torch.tensor([self.normalisation_stds[p] for p in parents])
            scaled_inputs = inputs * stds + means

            outputs = self.mechanisms[node](scaled_inputs)
            outputs = (outputs - self.normalisation_means[node]) / self.normalisation_stds[node]
        else:
            outputs = self.mechanisms[node](inputs)

        return outputs

    def sample(self, interventions: dict, batch_size: int, num_batches: int = 1) -> Experiment:
        # if data is normalised, we assume the intervention values are normalised as well -> we need to un-normalise
        # them to perform ancestral sampling with the true mechanisms
        scaled_interventions = interventions.copy()
        if self.cfg.normalise_data:
            scaled_interventions = {node: value * self.normalisation_stds[node] + self.normalisation_means[node] for
                                    node, value in scaled_interventions.items()}

        # perform ancestral sampling
        data = dict()
        for node in self.topological_order:
            # check if node is intervened upon
            if node in scaled_interventions:
                samples = torch.ones(num_batches, batch_size, 1) * scaled_interventions[node]
            else:
                with torch.no_grad():
                    # sample from mechanism
                    mech = self.mechanisms[node]
                    parents = get_parents(node, self.graph)
                    if not parents:
                        samples = mech.sample(torch.empty(num_batches, batch_size, 1))
                    else:
                        x = torch.cat([data[parent] for parent in parents], dim=-1)
                        assert x.shape == (num_batches, batch_size, mech.in_size), print(f'Invalid shape {x.shape}!')
                        samples = mech.sample(x)

            # store samples
            data[node] = samples

        exp = Experiment(scaled_interventions, data)
        if self.cfg.normalise_data and self.normalisation_means is not None and self.normalisation_stds is not None:
            exp.normalise(self.normalisation_means, self.normalisation_stds)

        return exp

    def sample_ace(self, target: str, interventions: dict, num_samples: int) -> torch.Tensor:
        # if data is normalised, we assume the intervention values are normalised as well -> we need to un-normalise
        # them to perform ancestral sampling with the true mechanisms
        scaled_interventions = interventions.copy()
        if self.cfg.normalise_data:
            scaled_interventions = {node: value * self.normalisation_stds[node] + self.normalisation_means[node] for
                                    node, value in scaled_interventions.items()}

        # perform ancestral sampling
        aces = torch.ones(num_samples)
        with torch.no_grad():
            # check if target is intervened upon (although it makes little sense)
            if target in interventions:
                return aces * interventions[target]

            parents_target = get_parents(target, self.graph)
            target_mechanism = self.mechanisms[target]
            # if the target is a root node in the graph, the ACE is trivial
            if not parents_target:
                aces *= target_mechanism(torch.empty(num_samples, 1, 1)).mean()
            else:
                # otherwise, perform ancestral sampling to estimate the ACE
                data = dict()
                for node in self.topological_order:
                    if node == target:
                        # if we sampled all ancestors we can compute the ACE for each ancestral sample
                        x = torch.cat([data[parent] for parent in parents_target], dim=-1)
                        aces = target_mechanism(x).squeeze()
                        break
                    elif node in scaled_interventions:
                        # check if node is intervened upon
                        node_samples = torch.ones(num_samples, 1, 1) * scaled_interventions[node]
                    else:
                        # sample from mechanism
                        parents = get_parents(node, self.graph)
                        mechanism = self.mechanisms[node]
                        if not parents:
                            node_samples = mechanism.sample(torch.empty(num_samples, 1, 1))
                        else:
                            x = torch.cat([data[parent] for parent in parents], dim=-1)
                            node_samples = mechanism.sample(x)

                    # store samples
                    data[node] = node_samples

        if self.cfg.normalise_data:
            aces = (aces - self.normalisation_means[target]) / self.normalisation_stds[target]

        return aces

    def estimate_ace(self, target: str, interventions: dict, num_samples: int) -> torch.Tensor:
        return self.sample_ace(target, interventions, num_samples).mean()

    def sample_aces(self, interventions: dict, num_samples: int) -> torch.Tensor:
        # if data is normalised, we assume the intervention values are normalised as well -> we need to un-normalise
        # them to perform ancestral sampling with the true mechanisms
        scaled_interventions = interventions.copy()
        if self.cfg.normalise_data:
            scaled_interventions = {node: value * self.normalisation_stds[node] + self.normalisation_means[node] for
                                    node, value in scaled_interventions.items()}

        # perform ancestral sampling to estimate the ACEs
        aces = dict()
        data = dict()
        for node in self.topological_order:
            if node in interventions:
                # check if node is intervened upon
                node_samples = torch.ones(num_samples, 1, 1) * scaled_interventions[node]
                ace_samples = torch.ones(num_samples) * scaled_interventions[node]
            else:
                # sample from mechanism
                parents = get_parents(node, self.graph)
                with torch.no_grad():
                    mechanism = self.mechanisms[node]
                    if not parents:
                        node_samples = mechanism.sample(torch.empty(num_samples, 1, 1))
                        ace_samples = mechanism(torch.empty(num_samples, 1, 1)).squeeze()
                    else:
                        x = torch.cat([data[parent] for parent in parents], dim=-1)
                        node_samples = mechanism.sample(x)
                        ace_samples = mechanism(x).squeeze()

            # store samples
            data[node] = node_samples
            if self.cfg.normalise_data:
                aces[node] = (ace_samples - self.normalisation_means[node]) / self.normalisation_stds[node]
            else:
                aces[node] = ace_samples

        aces = torch.stack([aces[node] for node in self.node_labels], dim=0)
        return aces

    def log_likelihood(self, experiments: List[Experiment], reduce: bool = True) -> torch.Tensor:
        # if data is normalised, we need to un-normalise the data before computing the log-likelihood
        scaled_experiments = experiments
        if self.cfg.normalise_data:
            scaled_experiments = [exp.unnormalise(self.normalisation_means, self.normalisation_stds) for exp in
                                  experiments]

        ll = torch.tensor(0.)
        for node in self.node_labels:
            # gather data from the experiments
            parents = get_parents(node, self.graph)
            inputs, targets = gather_data(scaled_experiments, node, parents=parents, mode='independent_samples')

            # check if we have any data for this node and compute log-likelihood
            mechanism_ll = torch.tensor(0.)
            if targets is not None:
                try:
                    with torch.no_grad():
                        mechanism_ll = self.mechanisms[node].mll(inputs, targets, prior_mode=False, reduce=reduce)
                except Exception as e:
                    print(f'Exception occured in Environment.log_likelihood() when computing LL for mechanism {node}:')
                    print(e)

            ll = ll + mechanism_ll
        return ll

    def interventional_mll(self, targets, node: str, interventions: dict, reduce=True):
        assert targets.dim() == 2, print(f'Invalid shape {targets.shape}')
        num_batches, batch_size = targets.shape
        num_mc_samples = self.cfg.imll_mc_samples

        parents = get_parents(node, self.graph)
        mechanism = self.mechanisms[node]

        if len(parents) == 0:
            # if we have a root note imll is simple
            with torch.no_grad():
                ll = mechanism.mll(None, targets, prior_mode=False, reduce=False)
            assert ll.shape == (num_batches,), print(f'Invalid shape {ll.shape}!')
            return ll.sum() if reduce else ll

        # otherwise, do MC estimate via ancestral sampling
        samples = self.sample(interventions, batch_size, num_mc_samples)
        # assemble inputs and targets
        inputs, _ = gather_data([samples], node, parents=parents, mode='independent_batches')
        inputs = inputs.unsqueeze(0).expand(num_batches, -1, -1, -1)
        assert inputs.shape == (num_batches, num_mc_samples, batch_size, len(parents))
        targets = targets.unsqueeze(1).expand(-1, num_mc_samples, batch_size)
        assert targets.shape == (num_batches, num_mc_samples, batch_size)
        # compute interventional ll
        with torch.no_grad():
            ll = mechanism.mll(inputs, targets, prior_mode=False, reduce=False).squeeze(-1)
        assert ll.shape == (num_batches, num_mc_samples), print(f'Invalid shape: {ll.shape}')
        ll = ll.logsumexp(dim=1) - math.log(num_mc_samples)
        return ll.sum() if reduce else ll

    def construct_graph(self, num_nodes: int) -> nx.DiGraph:
        raise NotImplementedError

    def get_adj_mat(self):
        return graph_to_adj_mat(self.graph, self.node_labels)

    def get_cpdag(self):
        return dag_to_cpdag(self.graph, self.node_labels)

    def param_dict(self) -> Dict[str, Any]:
        mechanism_param_dict = None
        if self.mechanisms is not None:
            mechanism_param_dict = {key: m.param_dict() for key, m in self.mechanisms.items()}

        params = {'name': self.name,
                  'num_nodes': self.num_nodes,
                  'graph': self.graph,
                  'mechanism_param_dict': mechanism_param_dict,
                  'non_intervenable_nodes': self.non_intervenable_nodes,
                  'intervention_bounds': self.intervention_bounds,
                  'observational_train_data': get_exp_param_dicts(self.observational_train_data),
                  'interventional_train_data': get_exp_param_dicts(self.interventional_train_data),
                  'observational_test_data': get_exp_param_dicts(self.observational_test_data),
                  'interventional_test_data': get_exp_param_dicts(self.interventional_test_data),
                  'normalisation_means': self.normalisation_means,
                  'normalisation_stds': self.normalisation_stds,
                  'cfg_param_dict': self.cfg.param_dict()}
        return params

    def load_param_dict(self, param_dict):
        self.cfg = EnvironmentConfig()
        self.cfg.load_param_dict(param_dict['cfg_param_dict'])

        self.name = param_dict['name']
        self.num_nodes = param_dict['num_nodes']
        self.graph = param_dict['graph']
        self.topological_order = list(nx.topological_sort(self.graph))
        self.node_labels = sorted(list(set(self.graph.nodes)))
        self.non_intervenable_nodes = param_dict['non_intervenable_nodes']
        self.intervenable_nodes = set(self.node_labels) - self.non_intervenable_nodes
        self.intervention_bounds = param_dict['intervention_bounds']
        self.observational_train_data = Experiment.load_param_dict(param_dict['observational_train_data'])
        self.interventional_train_data = Experiment.load_param_dict(param_dict['interventional_train_data'])
        self.observational_test_data = Experiment.load_param_dict(param_dict['observational_test_data'])
        self.interventional_test_data = Experiment.load_param_dict(param_dict['interventional_test_data'])

        self.normalisation_means = param_dict['normalisation_means']
        self.normalisation_stds = param_dict['normalisation_stds']
        self.mechanisms = dict()
        if param_dict['mechanism_param_dict'] is not None:
            for key, d in param_dict['mechanism_param_dict'].items():
                self.mechanisms[key] = self.create_mechanism(d['in_size'])
                self.mechanisms[key].load_param_dict(d)
        else:
            self.mechanisms = None

    def save(self, path):
        torch.save(self.param_dict(), path)

    @classmethod
    def load(cls, path):
        param_dict = torch.load(path)
        return Environment(param_dict=param_dict)

    def export_to_csv(self, outdir: str = './'):
        # export graph as adjacency matrix
        adj_mat = self.get_adj_mat()
        dag = pd.DataFrame(adj_mat.numpy(), columns=self.node_labels)
        dag.to_csv(os.path.join(outdir, self.name + '-adj-mat.csv'), index=False)

        # export data
        if self.observational_train_data is not None:
            obs_train = self.observational_train_data[0].to_pandas_df(self.node_labels)
            obs_train.to_csv(os.path.join(outdir, self.name + '-obs-train.csv'), index=False)

        if self.observational_test_data is not None:
            obs_test = self.observational_test_data[0].to_pandas_df(self.node_labels)
            obs_test.to_csv(os.path.join(outdir, self.name + '-obs-test.csv'), index=False)

        if self.interventional_train_data is not None:
            for eidx, exp in enumerate(self.interventional_train_data):
                df = exp.to_pandas_df(self.node_labels, add_interventions=True)
                filename = self.name + f'-intr-train-{eidx + 1}.csv'
                df.to_csv(os.path.join(outdir, filename), index=False)

        if self.interventional_test_data is not None:
            for eidx, exp in enumerate(self.interventional_test_data):
                df = exp.to_pandas_df(self.node_labels, add_interventions=True)
                filename = self.name + f'-intr-test-{eidx + 1}.csv'
                df.to_csv(os.path.join(outdir, filename), index=False)

    @classmethod
    def export_csv_dataset(cls, env_file: str, outdir: str):
        env = Environment.load(env_file)
        env.export_to_csv(outdir)

    @classmethod
    def load_static_dataset(cls, graph: nx.DiGraph,
                            obs_train_data: pd.DataFrame = None,
                            obs_test_data: pd.DataFrame = None,
                            intr_train_data: List[pd.DataFrame] = None,
                            intr_test_data: List[pd.DataFrame] = None,
                            normalise: bool = True):

        # create env with given graph
        num_nodes = len(graph.nodes)
        env = Environment(num_nodes, cfg=None, init_mechanisms=False, graph=graph)

        normalisation_means = None
        normalisation_stds = None
        # load observational training data and optionally compute normalisation parameters
        if obs_train_data is not None:
            exp = Experiment.from_pandas_df(obs_train_data)
            if normalise:
                normalisation_means = {}
                normalisation_stds = {}
                for node, values in exp.data.items():
                    normalisation_means[node] = values.mean()
                    normalisation_stds[node] = values.std()

                env.normalisation_means = normalisation_means
                env.normalisation_stds = normalisation_stds
                env.cfg.normalise_data = True
                exp.normalise(normalisation_means, normalisation_stds)

            env.observational_train_data = [exp]

        # load observational test data
        if obs_test_data is not None:
            exp = Experiment.from_pandas_df(obs_test_data)
            if normalisation_means is not None:
                exp.normalise(normalisation_means, normalisation_stds)
            env.observational_test_data = [exp]

        # load interventional training data
        if intr_train_data is not None:
            exps = []
            for df in intr_train_data:
                exp = Experiment.from_pandas_df(df, includes_interventions=True)
                if normalisation_means is not None:
                    exp.normalise(normalisation_means, normalisation_stds)
                exps.append(exp)

            env.interventional_train_data = exps

        # load interventional test data
        if intr_test_data is not None:
            exps = []
            for df in intr_test_data:
                exp = Experiment.from_pandas_df(df, includes_interventions=True)
                if normalisation_means is not None:
                    exp.normalise(normalisation_means, normalisation_stds)
                exps.append(exp)

            env.interventional_test_data = exps

        return env
