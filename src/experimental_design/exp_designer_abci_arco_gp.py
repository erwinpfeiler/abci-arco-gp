
from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Callable

import networkx as nx
import torch

from src.environments.environment import Experiment
from src.experimental_design.exp_designer_base import ExpDesignerBase
from src.mechanism_models.shared_data_gp_model import (
    SharedDataGaussianProcessModel,
)
from src.utils.graphs import adj_mat_to_graph
from src.mechanism_models.mechanisms import get_mechanism_key
from src.utils.causal_orders import CausalOrder, generate_all_mechanisms, generate_all_parent_sets
import time


class ExpDesignerABCIArCOGP(ExpDesignerBase):

    def __init__(
        self,
        intervention_bounds: Dict[str, Tuple[float, float]],
        opt_strategy: str = "gp-ucb",
        distributed: bool = False,
    ) -> None:
        super().__init__(intervention_bounds, opt_strategy, distributed)
        self.mech_model: Optional[SharedDataGaussianProcessModel] = None

    def init_design_process(self, args: dict):
        
        self.mech_model = args['mechanism_model']
        self.batch_size = args['batch_size']
        self.num_exp_batches_per_graph = args['num_exp_batches_per_graph']

        self.mc_cos = args['mc_cos']
        self.adj_mats = args['adj_mats']
        self.co_weights = args['co_weights']
        self.sample_time = args['sample_time']
        self.max_ps_size = args['max_ps_size']
        self.ps_weight_cache = args['ps_weight_cache']
        self.env_node_labels = args['env_node_labels']
        self.mc_adj_masks = args['mc_adj_masks']#.contiguous()

        def _utility(interventions: Dict[str, float]):
            return self._graph_info_gain(interventions)
        self.utility = _utility

    def _simulate_experiment(
        self,
        interventions: Dict[str, float],
        graph,
    ) -> Experiment:
        """Draw synthetic outcomes X_t ~ p(·|interventions, D_E) (batch omitted)."""
        #self.mech_model.eval()
        return self.mech_model.sample(interventions, self.batch_size, self.num_exp_batches_per_graph, graph=graph)
    

    def _graph_info_gain(
        self,
        interventions: Dict[str, float],
    ) -> torch.Tensor:

        # 1) sample causal orders under p(L | D_E)
        co_weights = self.co_weights
        adj_mats = self.adj_mats
        mc_cos = self.mc_cos
        sample_time = self.sample_time

        num_cos, num_graphs = adj_mats.shape[0:2]

        log_w_co = co_weights.sum(dim=1)
        log_Z = log_w_co.logsumexp(dim=0)
        w_co = (log_w_co - log_Z).exp()

        with torch.inference_mode():

            expected_info_gain = None

            for cidx in range(num_cos):

                per_order_avg = None

                for gidx in range(num_graphs):

                    # ---- graph build ----
                    graph = adj_mat_to_graph(
                        adj_mats[cidx, gidx],
                        self.mech_model.node_labels
                    )

                    self.mech_model.init_topological_order(graph, sample_time)

                    exp = self._simulate_experiment(interventions, graph)

                    # ---- clear caches ----
                    self.mech_model.clear_posterior_mll_cache()
                    #self.mech_model.clear_prior_mll_cache()

                    def pred_log_node(node: str, parents: list[str], _exp=exp) -> torch.Tensor:
                        return self.mech_model.node_mll(
                            [_exp],
                            node,
                            parents,
                            prior_mode=False,
                            use_cache=True,
                            reduce=True
                        )


                    inner_exp_log = self.graph_posterior_expectation_factorising(
                        func=pred_log_node,
                        mc_cos=mc_cos,
                        co_weights=co_weights,
                        env_node_labels=self.env_node_labels,
                        max_ps_size=self.max_ps_size,
                        adj_mask=self.mc_adj_masks,
                        ps_weight_cache=self.ps_weight_cache,
                        logspace=True,
                    )

                    outer_log = self.mech_model.mll(
                        [exp],
                        graph,
                        prior_mode=False,
                        use_cache=True,
                        mode='independent_batches',
                        reduce=True
                    )

                    contrib = outer_log - inner_exp_log # this has correct signs!

                    if per_order_avg is None:
                        per_order_avg = contrib / num_graphs
                    else:
                        per_order_avg = per_order_avg + (contrib / num_graphs)

                weight = w_co[cidx]
                if expected_info_gain is None:
                    expected_info_gain = weight * per_order_avg
                else:
                    expected_info_gain = expected_info_gain + weight * per_order_avg

        
        return expected_info_gain
    
    def _mechanism_expected_noise_entropy(
        self,
        node: str,
        parents: list[str],
    ) -> torch.Tensor:
        key = get_mechanism_key(node, parents)
        if key in self.entropy_cache:
            return self.entropy_cache[key]

        if len(parents) == 0:
            entropy = self.mech_model.root_mechs[node].expected_noise_entropy(prior_mode=False)
        else:
            entropy = self.mech_model.gps[node].expected_noise_entropy(key)

        self.entropy_cache[key] = entropy
        return entropy


    def _model_self_expected_log_likelihood(
        self,
        interventions: Dict[str, float],
        graph,
    ) -> torch.Tensor:
        total_num_samples = self.batch_size * self.num_exp_batches_per_graph
        log_prob = torch.tensor(0.)

        for node in self.env_node_labels:
            if node in interventions:
                continue

            parents = list(graph.predecessors(node))
            entropy = self._mechanism_expected_noise_entropy(node, parents)
            log_prob = log_prob - total_num_samples * entropy

        return log_prob


    def _model_info_gain(
        self,
        interventions: Dict[str, float],
    ) -> torch.Tensor:

        co_weights = self.co_weights
        adj_mats = self.adj_mats
        mc_cos = self.mc_cos
        sample_time = self.sample_time

        num_cos, num_graphs = adj_mats.shape[0:2]

        log_w_co = co_weights.sum(dim=1)
        log_Z = log_w_co.logsumexp(dim=0)
        w_co = (log_w_co - log_Z).exp()

        total_num_samples = self.batch_size * self.num_exp_batches_per_graph

        with torch.inference_mode():
            expected_info_gain = None

            for cidx in range(num_cos):
                per_order_avg = None

                for gidx in range(num_graphs):
                    graph = adj_mat_to_graph(
                        adj_mats[cidx, gidx],
                        self.mech_model.node_labels
                    )

                    self.mech_model.init_topological_order(graph, sample_time)

                    # needed for the posterior-predictive mixture term
                    exp = self._simulate_experiment(interventions, graph)

                    # clear only predictive MLL cache for the hypothetical dataset
                    self.mech_model.clear_posterior_mll_cache()
                    self.mech_model.clear_entropy_cache()

                    def pred_log_node(node: str, parents: list[str], _exp=exp) -> torch.Tensor:
                        return self.mech_model.node_mll(
                            [_exp],
                            node,
                            parents,
                            prior_mode=False,
                            use_cache=True,
                            reduce=True
                        )

                    # log E_{L',G'|D}[ p(X_t | G', D) ]
                    inner_exp_log = self.graph_posterior_expectation_factorising(
                        func=pred_log_node,
                        mc_cos=mc_cos,
                        co_weights=co_weights,
                        env_node_labels=self.env_node_labels,
                        max_ps_size=self.max_ps_size,
                        adj_mask=self.mc_adj_masks,
                        ps_weight_cache=self.ps_weight_cache,
                        logspace=True,
                    )

                    # E_{X_t|M}[ log p(X_t | M) ] = - N_t * expected_noise_entropy(M)
                    entropy = self.mech_model.expected_noise_entropy(
                        interventions,
                        graph,
                        use_cache=True
                    )
                    outer_log = -total_num_samples * entropy

                    # correct sign: self term minus posterior-predictive mixture term
                    contrib = outer_log - inner_exp_log

                    if per_order_avg is None:
                        per_order_avg = contrib / num_graphs
                    else:
                        per_order_avg = per_order_avg + (contrib / num_graphs)

                weight = w_co[cidx]
                if expected_info_gain is None:
                    expected_info_gain = weight * per_order_avg
                else:
                    expected_info_gain = expected_info_gain + weight * per_order_avg

        return expected_info_gain


    def graph_posterior_expectation_factorising(
        self,
        func: Callable[[str, List[str]], torch.Tensor],
        mc_cos: List[CausalOrder],
        co_weights,
        env_node_labels,
        max_ps_size,
        adj_mask,
        ps_weight_cache,
        logspace=False
    ):


        num_cos = len(mc_cos)

        if logspace:
            co_values = torch.zeros(num_cos)
        else:
            co_values = torch.ones(num_cos)

        for cidx, co in enumerate(mc_cos):

            parent_sets = generate_all_parent_sets(
                env_node_labels,
                max_ps_size,
                adj_mask[cidx]
            )

            for nidx, node in enumerate(env_node_labels):

                weighted_values = []

                for parents in parent_sets[node]:

                    key = get_mechanism_key(node, parents)
                    weight = ps_weight_cache[key]

                    func_value = func(node, parents)

                    if logspace:
                        weighted_values.append(weight + func_value)
                    else:
                        weighted_values.append(weight.exp() * func_value)

                stacked = torch.stack(weighted_values, dim=0)

                if logspace:
                    co_values[cidx] += stacked.logsumexp(dim=0)
                else:
                    co_values[cidx] *= stacked.sum(dim=0)

        normalisation = co_weights.sum(dim=1).logsumexp(dim=0)

        if logspace:
            result = co_values.logsumexp(dim=0) - normalisation
        else:
            result = co_values.sum() / normalisation.exp()

        return result
