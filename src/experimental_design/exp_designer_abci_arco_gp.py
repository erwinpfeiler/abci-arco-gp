
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

# remove
from tqdm import tqdm, trange

class ExpDesignerABCIArCOGP(ExpDesignerBase):
    def __init__(
        self,
        intervention_bounds: Dict[str, Tuple[float, float]],
        opt_strategy: str = "gp-ucb",
        distributed: bool = False,
    ) -> None:
        super().__init__(intervention_bounds, opt_strategy, distributed)
        self.agent = None
        self.mech_model: Optional[SharedDataGaussianProcessModel] = None

    def init_design_process(self, args: dict):
        """Called once before the first query. Expects:
           - 'agent': ABCIArCOGP instance
           - 'mechanism_model': SharedDataGaussianProcessModel
           - 'policy': 'graph-info-gain'
           - optional: 'batch_size', 'num_exp_per_graph'
        """
        assert args["policy"] == "graph-info-gain"
        print(f"args = {args}")
        self.agent = args['agent']
        self.mech_model = args['mechanism_model']
        self.batch_size = args.get("batch_size", 1)
        self.num_exp_per_graph = args.get("num_exp_per_graph", 1)

        def _utility(interventions: Dict[str, float]):
            return self._graph_info_gain(interventions)
        self.utility = _utility

    def _simulate_experiment(
        self,
        interventions: Dict[str, float],
        graph,
    ) -> Experiment:
        """Draw synthetic outcomes X_t ~ p(·|interventions, D_E) (batch omitted)."""
        return self.mech_model.sample(interventions, self.batch_size, self.num_exp_per_graph, graph=graph)

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
        num_mc_graphs = 5 # TODO: get from args dict
        
        # 1) sample causal orders under p(L | D_E)
        mc_cos, mc_adj_masks = self.agent.sample_mc_cos(set_data=True)
        co_weights = self.agent.co_weights
        adj_mats = self.agent.sample_mc_graphs(mc_cos, mc_adj_masks, num_mc_graphs)
        """
        mc_adj_mats : torch.Tensor
        Adjacency matrices of shape (num_cos, num_mc_graphs, num_nodes, num_nodes)
        """
    
        num_samples_per_graph = self.batch_size
        num_cos, num_graphs = adj_mats.shape[0:2]
        # 3) normalize order weights: w_L = p(L|D) in linear space
        #    co_weights are log-weights per node; sum over nodes, then logsumexp over orders for Z
        log_w_co = co_weights.sum(dim=1)                 # (num_cos,)
        log_Z = log_w_co.logsumexp(dim=0)                # scalar
        w_co = (log_w_co - log_Z).exp()                  # normalized weights over orders, linear space

        
        """
        with torch.no_grad():
            for cidx in range(num_cos):
                for gidx in range(num_graphs):
                    graph = adj_mat_to_graph(adj_mats[cidx, gidx], self.mechanism_model.node_labels)
                    self.mechanism_model.init_topological_order(graph, self.sample_time)
                    exp = self.mechanism_model.sample(interventions, 1, num_samples_per_graph, graph)
                    for node in samples: # maybe remove this, I don't know what it is doing
                        samples[node][cidx, gidx] = exp.data[node].squeeze()
        """
        # 4) accumulate expected info gain
        #    U ≈ Σ_L w_L * (1/|G|) Σ_G E_{Xt|G}[ log E_{L',G'} p(Xt|M')  –  log p(Xt|G,D) ]
        with torch.inference_mode():
            expected_info_gain = None  # lazy init on correct device/dtype
            for cidx in range(num_cos):
                per_order_avg = None
                for gidx in range(num_graphs):
                    # build graph and simulate Xt ~ p(Xt | G)
                    graph = adj_mat_to_graph(adj_mats[cidx, gidx], self.mech_model.node_labels)
                    self.mech_model.init_topological_order(graph, self.agent.sample_time)
                    exp = self._simulate_experiment(interventions, graph)
                    # define per-node log predictive under GP for the *sampled* Xt
                    def pred_log_node(node: str, parents: list[str], _exp=exp) -> torch.Tensor:
                        return self.mech_model.node_mll([_exp], node, parents, prior_mode=False)

                    # inner expectation over (L', G' | D) in closed form (returns log E[...] )
                    inner_exp_log = self.agent.graph_posterior_expectation_factorising(
                        func=pred_log_node,
                        mc_cos=mc_cos,
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

        for cidx in trange(num_cos):
            for gidx in trange(num_graphs):
                # now we have to sample experiments Xt|G, for each cidx for each gidx
                graph = adj_mat_to_graph(adj_mats[cidx, gidx], self.mech_model.node_labels)
                self.mech_model.init_topological_order(graph, self.agent.sample_time) # TODO: what is sample_time?
                print(f"graph = {graph}")
                exp = self._simulate_experiment(interventions, graph) # exp is the et of Xt
                #for node in samples:
                #    samples[node][cidx, gidx] = exp.data[node].squeeze()
                def pred_log_node(node: str, parents: list[str], _exp=exp) -> torch.Tensor:
                  return self.mech_model.node_mll([_exp], node, parents, prior_mode=False)
                """
                now calculate in closed form
                log(E(L'|theta) [w_L'* E(G'|L',psi,D)[p(Xt|M')]])/(p(Xt|G',D))]
                inner_exp := E(G'|L',psi,D)[p(Xt|M')]]
                denominator = p(Xt|G',D)
                """
                parent_sets = self.agent.generate_co_parent_sets(mc_adj_masks[cidx])
                for nidx, node in enumerate(self.agent.env.node_labels):
                    for pidx, parents in enumerate(parent_sets[node]):  
                        #denominator = pred_log_node(node, parents) # this is wrong
                        outer_log = self.mech_model.mll(
                            [exp], graph, prior_mode=False,
                            use_cache=True, mode='independent_batches', reduce=True
                        )
                        inner_exp_log = self.agent.graph_posterior_expectation_factorising(
                          func=pred_log_node,
                          mc_cos=mc_cos,
                          logspace=True
                        )
                        # actually I am of course calculating log-probabilities so I don't need to
                        # divide them but take the difference: inner_exp-denominator
        quit(0)
        return torch.Tensor(0.0)
