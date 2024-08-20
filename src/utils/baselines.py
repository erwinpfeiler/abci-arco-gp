import os
from typing import Union, Callable, List, Tuple

import gpytorch
import numpy as np
import pandas as pd
import sumu
import torch
from castle.algorithms import GES, DAG_GNN, GOLEM, GraNDAG, ANMNonlinear, PC, GAE
from causallearn.graph.GraphClass import CausalGraph
from causallearn.graph.GraphClass import GeneralGraph
from causallearn.search.PermutationBased.GRaSP import grasp
from lingam import RESIT
from scipy.stats import multivariate_t as mvt

from src.config import GPModelConfig
from src.environments.environment import Environment
from src.mechanism_models.mechanisms import GaussianProcess
from src.utils.graphs import dag_to_cpdag
from src.utils.metrics import compute_structure_metrics, aid
from src.utils.utils import export_stats


def causal_graph_to_cpdag(cg: Union[CausalGraph, GeneralGraph]) -> torch.Tensor:
    adj_mat = cg.graph if isinstance(cg, GeneralGraph) else cg.G.graph
    num_nodes = adj_mat.shape[0]
    cpdag = torch.zeros(num_nodes, num_nodes)
    for i in range(num_nodes):
        for j in range(i, num_nodes):
            if adj_mat[i, j] == 0:
                continue
            elif adj_mat[i, j] == adj_mat[j, i]:
                cpdag[i, j] = cpdag[j, i] = 1
            elif adj_mat[i, j] == 1 and adj_mat[j, i] == -1:
                cpdag[i, j] = 0
                cpdag[j, i] = 1
            elif adj_mat[i, j] == -1 and adj_mat[j, i] == 1:
                cpdag[i, j] = 1
                cpdag[j, i] = 0
            else:
                assert False

    return cpdag


def graph_expectation(graphs: torch.Tensor, func: Callable) -> torch.Tensor:
    assert graphs.dim() == 3
    num_graphs = graphs.shape[0]
    func_values = torch.tensor([func(graphs[i]) for i in range(num_graphs)])
    return func_values.mean()


class GPRegressor:
    def __init__(self, linear: bool = False):
        self.linear = linear
        self.gp = None

    def fit(self, inputs: torch.Tensor, targets: torch.Tensor):
        inputs = torch.tensor(inputs) if not isinstance(inputs, torch.Tensor) else inputs
        targets = torch.tensor(targets) if not isinstance(targets, torch.Tensor) else targets
        assert inputs.dim() == 2 and targets.dim() == 1 and inputs.shape[0] == targets.numel()
        num_samples, num_parents = inputs.shape

        cfg = GPModelConfig()
        self.gp = GaussianProcess(num_parents, linear=self.linear)
        self.gp.set_data(inputs, targets)
        self.gp.train()

        # fit GP hyperparameters
        optimizer = torch.optim.RMSprop(self.gp.parameters(), lr=cfg.lr)
        losses = []
        for i in range(cfg.num_steps):
            optimizer.zero_grad()
            try:
                mll = self.gp.mll(inputs, targets, prior_mode=True) / targets.numel()
            except Exception as e:
                print(
                    f'Exception occured in GaussianProcessModel.gp_mlls() when computing MLL:')
                print(e)
                print('Resampling GP hyperparameters...')
                self.gp.gp.init_hyperparams()
                continue

            loss = -mll - self.gp.gp.hyperparam_log_prior()

            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            if i > cfg.es_min_steps:
                change = (torch.tensor(losses[-1 - cfg.es_win_size:-1]).mean() -
                          torch.tensor(losses[-2 - cfg.es_win_size:-2]).mean()).abs()
                if change < cfg.es_threshold:
                    break

    def predict(self, inputs: torch.Tensor):
        assert self.gp is not None
        inputs = torch.tensor(inputs) if not isinstance(inputs, torch.Tensor) else inputs
        with torch.no_grad():
            with gpytorch.settings.debug(state=False):  # avoid annoying warning that train data matches test data
                self.gp.eval()
                predictions = self.gp(inputs).view(-1).numpy()

        return predictions


class Baseline:

    def __init__(self, env: Union[Environment, str], method: str, output_dir: str = None, run_id: str = '',
                 policy: str = 'static-obs-dataset'):
        self.env = Environment.load(env) if isinstance(env, str) else env
        self.method = method
        self.output_dir = output_dir
        self.run_id = run_id
        self.policy = policy

    def run(self):
        # load data
        if self.policy == 'static-obs-dataset':
            # data shape will be (num_samples, num_nodes)
            data = self.env.observational_train_data[0].to_pandas_df(self.env.node_labels).to_numpy()
        else:
            raise NotImplementedError

        # run baseline
        cpdag_prediction = False  # indicates whether the baseline returns DAG(s) or CPDAG(s)
        if self.method == 'anm':
            model = ANMNonlinear()
            model.learn(data)
            graphs = torch.tensor(model.causal_matrix).unsqueeze(0)
        elif self.method == 'ges':
            model = GES()
            model.learn(data)
            graphs = torch.tensor(model.causal_matrix).unsqueeze(0)
            cpdag_prediction = True
        elif self.method == 'daggnn':
            model = DAG_GNN()  # graph_threshold=0.5 for linear networks
            model.learn(data)
            graphs = torch.tensor(model.causal_matrix).unsqueeze(0)
        elif self.method in {'gadget', 'beeps'}:
            model = Gadget(data)
            graphs = model.sample()
        elif self.method == 'gae':
            model = GAE()
            model.learn(data)
            graphs = torch.tensor(model.causal_matrix).unsqueeze(0)
        elif self.method == 'golem':
            model = GOLEM()
            model.learn(data)
            graphs = torch.tensor(model.causal_matrix).float().unsqueeze(0)
        elif self.method == 'grandag':
            model = GraNDAG(input_dim=data.shape[1])
            model.learn(data)
            graphs = torch.tensor(model.causal_matrix).unsqueeze(0)
        elif self.method == 'grasp':
            graphs = causal_graph_to_cpdag(grasp(data)).unsqueeze(0)
            cpdag_prediction = True
        elif self.method == 'pc':
            model = PC(variant='stable')
            model.learn(data)
            graphs = torch.tensor(model.causal_matrix).unsqueeze(0)
            cpdag_prediction = True
        elif self.method == 'resit':
            if self.env.cfg.linear:
                print('RESIT: using linear GP regressor.')
            torch.set_default_dtype(torch.float32)
            model = RESIT(regressor=GPRegressor(linear=self.env.cfg.linear))
            model.fit(data)
            graphs = torch.tensor(model.adjacency_matrix_).unsqueeze(0)
        else:
            raise NotImplementedError

        print('Computing structure stats...')
        num_graphs = graphs.shape[0]
        true_adj_mat = self.env.get_adj_mat()
        true_cpdag = self.env.get_cpdag()
        if cpdag_prediction:
            assert num_graphs == 1, print('Handling multiple CPDAG predictions not implemented yet.')
            cpdag = graphs[0]

            # compute expected number of edges
            tmp = cpdag.triu() + cpdag.tril().T
            enum_edges = torch.where(tmp.int() == 2, torch.tensor(1), tmp).sum()
            stats = {'enum_edges': [enum_edges]}

            # compute structure metrics
            for stat_name, value in compute_structure_metrics(true_adj_mat, cpdag).items():
                stats[stat_name] = [value]

            for stat_name, value in compute_structure_metrics(true_cpdag, cpdag).items():
                stats[stat_name + '_cpdag'] = [value]

            # compute aid variants
            try:
                stats['aaid'] = [aid(true_adj_mat, cpdag, mode='ancestor')]
                stats['paid'] = [aid(true_adj_mat, cpdag, mode='parent')]
                stats['oset_aid'] = [aid(true_adj_mat, cpdag, mode='oset')]

                stats['aaid_cpdag'] = [aid(true_cpdag, cpdag, mode='ancestor')]
                stats['paid_cpdag'] = [aid(true_cpdag, cpdag, mode='parent')]
                stats['oset_aid_cpdag'] = [aid(true_cpdag, cpdag, mode='oset')]
            except Exception as e:
                # the PC alg may yield potentially cyclic CPDAGs for which the AID is not computable
                print(e)
                print()
                print('Not recording AID stats...')
                stats['aaid'] = [torch.tensor(-1.)]
                stats['paid'] = [torch.tensor(-1.)]
                stats['oset_aid'] = [torch.tensor(-1.)]

                stats['aaid_cpdag'] = [torch.tensor(-1.)]
                stats['paid_cpdag'] = [torch.tensor(-1.)]
                stats['oset_aid_cpdag'] = [torch.tensor(-1.)]

            # order aid n/a
            stats['order_aid'] = [torch.tensor(-1.)]
        else:
            edge_probs = graphs.mean(dim=0)
            stats = {'enum_edges': [edge_probs.sum()]}

            # compute structure metrics
            for stat_name, value in compute_structure_metrics(true_adj_mat, edge_probs).items():
                stats[stat_name] = [value]

            for stat_name, value in compute_structure_metrics(true_cpdag, edge_probs, dag_to_cpdag=True).items():
                stats[stat_name + '_cpdag'] = [value]

            # compute aid variants
            stats['aaid'] = [graph_expectation(graphs, lambda g: aid(true_adj_mat, g, mode='ancestor'))]
            stats['paid'] = [graph_expectation(graphs, lambda g: aid(true_adj_mat, g, mode='parent'))]
            stats['oset_aid'] = [graph_expectation(graphs, lambda g: aid(true_adj_mat, g, mode='oset'))]

            def aid_wrapper(g, mode: str):
                return aid(true_cpdag, dag_to_cpdag(g, self.env.node_labels), mode=mode)

            stats['aaid_cpdag'] = [graph_expectation(graphs, lambda g: aid_wrapper(g, mode='ancestor'))]
            stats['paid_cpdag'] = [graph_expectation(graphs, lambda g: aid_wrapper(g, mode='parent'))]
            stats['oset_aid_cpdag'] = [graph_expectation(graphs, lambda g: aid_wrapper(g, mode='oset'))]

            # order aid n/a
            stats['order_aid'] = [torch.tensor(-1.)]

        if self.output_dir is not None:
            num_experiments_conducted = 1
            outpath = os.path.join(self.output_dir,
                                   f'stats-{self.method}-{self.policy}-{self.env.name}'
                                   f'-{self.run_id}-exp-{num_experiments_conducted}.csv')
            export_stats(stats, outpath)

        if self.method == 'beeps':
            model = Beeps(graphs, data)
            num_env_samples = 10000
            num_nodes = len(self.env.node_labels)
            node_label_to_id_dict = {node: i for i, node in enumerate(self.env.node_labels)}
            maes = torch.zeros(num_nodes, num_nodes)
            for i, node in enumerate(self.env.node_labels):
                print(f'Computing MAEs for node {node}', flush=True)
                interventions = {node: torch.tensor(1.)}
                aces = model.estimate_aces(interventions, node_label_to_id_dict)
                env_aces = self.env.sample_aces(interventions, num_env_samples).mean(dim=1)
                maes[:, i] = (aces - env_aces).abs()

            if self.output_dir is not None:
                df = pd.DataFrame(maes)
                df.columns = [f'do({node})' for node in self.env.node_labels]
                outpath = os.path.join(self.output_dir,
                                       f'stats-{self.method}-{self.policy}-{self.env.name}'
                                       f'-{self.run_id}-mae.csv')
                df.to_csv(outpath, index=False)

            return stats, maes

        return stats


class Gadget:
    # A wrapper around GADGET from the sumu package.
    def __init__(self, data: np.ndarray):
        _, self.num_nodes = data.shape  # data shape is (num training data, num nodes)

        # structure learning
        params = {"data": sumu.Data(data, discrete=False),
                  "scoref": 'bge',
                  "max_id": -1,
                  "K": min(self.num_nodes - 1, 16),
                  "d": 2,
                  "cp_algo": 'greedy-lite',
                  "mc3_chains": 16,
                  "burn_in": 1000,
                  "iterations": 1000,
                  "thinning": 10,
                  "logging": {"silent": False, "period": 1},
                  }

        self.model = sumu.Gadget(**params)

    def sample(self):
        dags, _ = self.model.sample()
        graphs = torch.stack([self.adj_list_to_adj_mat(adj_list, self.num_nodes) for adj_list in dags], dim=0)
        return graphs

    @classmethod
    def adj_list_to_adj_mat(cls, adj_list: List[Tuple[int, Tuple[int]]], num_nodes: int):
        adj_mat = torch.zeros(num_nodes, num_nodes)
        for entry in adj_list:
            # check if descendants
            if len(entry) > 1:
                sink, source = entry
                adj_mat[source, sink] = 1.

        return adj_mat


class Beeps:
    '''
    2020, Sumu developers.
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions
    are met:

    1. Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above
    copyright notice, this list of conditions and the following
    disclaimer in the documentation and/or other materials provided
    with the distribution.

    3. Neither the name of the copyright holder nor the names of its
    contributors may be used to endorse or promote products derived
    from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    '''

    def __init__(self, adj_mats: torch.Tensor, data: np.ndarray):
        num_graphs = adj_mats.shape[0]
        self.dags = [adj_mats[i].T.numpy() for i in range(num_graphs)]
        self.Bs = None

        self.N, self.n = data.shape  # data shape is (num training data, num nodes)
        # Prior parameters
        self.nu = np.zeros(self.n)
        self.am = 1
        aw = self.n + self.am + 1
        Tmat = np.identity(self.n) * (aw - self.n - 1) / (self.am + 1)

        # Sufficient statistics
        self.xN = np.mean(data, axis=0)
        SN = (data - self.xN).T @ (data - self.xN)

        # Parameters for the posterior
        self.awN = aw + self.N
        self.R = (
                Tmat + SN + ((self.am * self.N) / (self.am + self.N)) * np.outer((self.nu - self.xN),
                                                                                 (self.nu - self.xN))
        )

    def sample_pairwise(self):
        As = np.ones((len(self.dags), self.n, self.n))
        Bs = self.sample_weights()
        for i in range(Bs.shape[0]):
            As[i] = np.linalg.inv(np.eye(self.n) - Bs[i])
        return As

    def sample_direct(self):
        return self.sample_weights() + np.eye(self.n)

    def sample_joint(self, *, y, x, resample=False):
        As = np.ones((len(self.dags), len(y), len(x)))
        Bs = self.sample_weights(resample)
        n = self.n
        for i in range(Bs.shape[0]):
            Umat = np.eye(n)
            Umat[x, :] = 0
            A = np.linalg.inv(np.eye(n) - Umat @ Bs[i])
            A = A[y, :][:, x]
            As[i] = A
        return As

    def sample_weights(self, resample=True):
        if resample is False and self.Bs is not None:
            return self.Bs
        n = self.n
        R = self.R
        awN = self.awN
        Bs = np.zeros((len(self.dags), n, n))
        for i, dag in enumerate(self.dags):
            for node in range(n):
                pa = np.where(dag[node])[0]
                if len(pa) == 0:
                    continue
                l = len(pa) + 1

                R11 = R[pa[:, None], pa]
                R12 = R[pa, node]
                R21 = R[node, pa]
                R22 = R[node, node]
                R11inv = np.linalg.inv(R11)
                df = awN - n + l
                mb = R11inv @ R12
                divisor = R22 - R21 @ R11inv @ R12
                covb = divisor / df * R11inv
                b = mvt.rvs(loc=mb, shape=covb, df=df)
                Bs[i, node, pa] = b
        self.Bs = Bs
        return Bs

    def get_posterior_mean(self):
        return (self.am * self.nu + self.N * self.xN) / (self.am + self.N)

    def estimate_aces(self, interventions: dict, node_label_to_id_dict):
        weights = self.sample_weights()
        posterior_means = self.get_posterior_mean()
        aces = np.zeros((len(self.dags), self.n))
        for gidx, graph in enumerate(self.dags):
            adj_mat = np.copy(graph)
            mu = np.copy(posterior_means)
            for node, value in interventions.items():
                adj_mat[node_label_to_id_dict[node], :] = 0
                mu[node_label_to_id_dict[node]] = value

            intervened_weights = (adj_mat * weights[gidx])
            aces[gidx] = np.linalg.inv(np.eye(self.n) - intervened_weights) @ mu

        return torch.tensor(aces).mean(dim=0).float()
