
from __future__ import annotations

from typing import Dict, List, Tuple, Optional

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
        
        assert args["policy"] == "graph-info-gain"
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
        self.mc_adj_masks = args['mc_adj_masks'].contiguous()

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
        """Exact U_CD using closed-form factorising and additive expectations.
        Equation from my handwritten notes:
        theta = MAP estimator, psi = MAP estimator
        U_CD = E(L|theta) [w_L * E(G|L,psi,D)[E(Xt|G,psi)[
          log(E(L'|theta) [w_L'* E(G'|L',psi,D)[p(Xt|M')]])/(p(Xt|G',D))]
          ]]
          
        So first I will sample causal orders called mc_cos in code and L in equation above
        The same set mc_cos is also used for L'
        
        Then I will iterate over L/mc_cos
        
        Then I will calculate the order weights w_L (this will be recyceled later for w_L')
        
        Then for each L I sample a set of G called graphs (this will be only used here
        not for G' which we will circumvent by calculating the posterior in closed form)
        """
        
        # 1) sample causal orders under p(L | D_E)
        co_weights = self.co_weights
        adj_mats = self.adj_mats
        mc_cos = self.mc_cos
        sample_time = self.sample_time
    
        num_cos, num_graphs = adj_mats.shape[0:2]
        # 3) normalize order weights: w_L = p(L|D) in linear space
        #    co_weights are log-weights per node; sum over nodes, then logsumexp over orders for Z
        log_w_co = co_weights.sum(dim=1)                 # (num_cos,)
        log_Z = log_w_co.logsumexp(dim=0)                # scalar
        w_co = (log_w_co - log_Z).exp()                  # normalized weights over orders, linear space

        # 4) accumulate expected info gain
        #    U ≈ Σ_L w_L * (1/|G|) Σ_G E_{Xt|G}[ log E_{L',G'} p(Xt|M')  –  log p(Xt|G,D) ]
        with torch.inference_mode():
            expected_info_gain = None  # lazy init on correct device/dtype
            for cidx in range(num_cos):
                per_order_avg = None
                for gidx in range(num_graphs):
                    # build graph and simulate Xt ~ p(Xt | G)
                    graph = adj_mat_to_graph(adj_mats[cidx, gidx], self.mech_model.node_labels)
                    self.mech_model.init_topological_order(graph, sample_time)
                    exp = self._simulate_experiment(interventions, graph)
                    # define per-node log predictive under GP for the *sampled* Xt
                    # first clear MLL cache:
                    self.mech_model.clear_posterior_mll_cache()
                    self.mech_model.clear_prior_mll_cache()
                    def pred_log_node(node: str, parents: list[str], _exp=exp) -> torch.Tensor:
                        return self.mech_model.node_mll([_exp], node, parents, prior_mode=False,
                                                        use_cache=True, reduce=True)

                    # inner expectation over (L', G' | D) in closed form (returns log E[...] )
                    inner_exp_log = self.graph_posterior_expectation_factorising(
                        func=pred_log_node,
                        mc_cos=mc_cos,
                        co_weights=co_weights,
                        env_node_labels=self.env_node_labels,
                        max_ps_size=self.max_ps_size,
                        adj_mask=self.mc_adj_masks.contiguous(),
                        ps_weight_cache=self.ps_weight_cache,
                        logspace=True,
                    )
                    # outer denominator: log p(Xt | G, D)
                    outer_log = self.mech_model.mll(
                        [exp], graph, prior_mode=False,
                        use_cache=True, mode='independent_batches', reduce=True
                    )

                    # contribution for this graph
                    contrib = inner_exp_log - outer_log  # both log-space scalars

                    # accumulate over graphs (mean)
                    if per_order_avg is None:
                        per_order_avg = contrib / num_graphs
                    else: # numeric stable mean accumulation:
                        per_order_avg = per_order_avg + (contrib / num_graphs)

                # weight by normalized order posterior w_L and add
                weight = w_co[cidx]
                if expected_info_gain is None:
                    expected_info_gain = weight * per_order_avg
                else:
                    expected_info_gain = expected_info_gain + weight * per_order_avg

        return expected_info_gain
    
    def graph_posterior_expectation_factorising(self, func: Callable[[str, List[str]], torch.Tensor],
                                                mc_cos: List[CausalOrder],
                                                co_weights,
                                                env_node_labels,
                                                max_ps_size,
                                                adj_mask,
                                                ps_weight_cache,
                                                logspace=False):
        """
        Compute E[ ∏_i f(i, Pa_i) ] under the posterior over (orders, parent‐sets)
        in closed form (factorising queries).

        Args:
            func: Maps (node_label, parent_list) → tensor value for that node.
            mc_cos: Sampled causal orders.
            logspace: If True, perform computations in log‐space.

        Returns:
            Tensor: Scalar expectation of the factorising query.
        """

        num_cos = len(mc_cos)
        if logspace:
            co_values = torch.zeros(num_cos)
        else:
            co_values = torch.ones(num_cos)

        for cidx, co in enumerate(mc_cos):
            parent_sets = generate_all_parent_sets(env_node_labels, max_ps_size, adj_mask[cidx])
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

                if logspace:
                    co_values[cidx] += torch.stack(weighted_values, dim=0).logsumexp(dim=0)
                else:
                    co_values[cidx] *= torch.stack(weighted_values, dim=0).sum(dim=0)

        normalisation = co_weights.sum(dim=1).logsumexp(dim=0)
        if logspace:
            return co_values.logsumexp(dim=0) - normalisation
        else:
            return co_values.sum() / normalisation.exp()


