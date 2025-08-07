
from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import networkx as nx
import torch

from src.environments.environment import Experiment
from src.experimental_design.exp_designer_base import ExpDesignerBase
from src.mechanism_models.shared_data_gp_model import (
    SharedDataGaussianProcessModel,
)
from src.graph_models.arco import ArCO  # auto‑regressive causal‑order model
from src.abci_arco_gp import ABCIArCOGP

class ExpDesignerABCIArCOGP(ExpDesignerBase):
    def __init__(
        self,
        intervention_bounds: Dict[str, Tuple[float, float]],
        opt_strategy: str = "gp-ucb",
        distributed: bool = False,
    ) -> None:
        super().__init__(intervention_bounds, opt_strategy, distributed)
        self.agent: Optional[ABCIArCOGP] = None
        self.mech_model: Optional[SharedDataGaussianProcessModel] = None

    def init_design_process(self, args: dict):
        """Called once before the first query. Expects:
           - 'agent': ABCIArCOGP instance
           - 'mechanism_model': SharedDataGaussianProcessModel
           - 'policy': 'graph-info-gain'
           - optional: 'batch_size', 'num_exp_per_graph'
        """
        assert args["policy"] == "graph-info-gain"
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
        graph,  # not used in exact version
    ) -> Experiment:
        """Draw synthetic outcomes X_t ~ p(·|interventions, D_E) (batch omitted)."""
        # no specific graph needed for exact predictive
        return self.mech_model.sample(interventions, self.batch_size, self.num_exp_per_graph, graph=None)

    def _graph_info_gain(
        self,
        interventions: Dict[str, float],
    ) -> torch.Tensor:
        """Exact U_CD using closed-form factorising and additive expectations."""
        # 1) sample causal orders under p(L | D_E)
        mc_cos, _ = self.agent.sample_mc_cos(set_data=True)

        # 2) predictive log-density: log p(X_t | D_E)
        def pred_log_node(node: str, parents: list[str]) -> torch.Tensor:
            # log p(x_{t,i} | parents, D_E)
            # here simulate one batch and compute node-wise log-likelihood
            exp = self._simulate_experiment(interventions, graph=None)
            return self.mech_model.node_mll([exp], node, parents, prior_mode=False)

        log_p_xt = self.agent.graph_posterior_expectation_factorising(
            func=pred_log_node,
            mc_cos=mc_cos,
            logspace=True
        )

        # 3) expected log-likelihood: E_{G|D_E}[log p(X_t | G, D)]
        def loglik_node(node: str, parents: list[str]) -> torch.Tensor:
            exp = self._simulate_experiment(interventions, graph=None)
            return self.mech_model.node_mll([exp], node, parents, prior_mode=False)

        E_logp = self.agent.graph_posterior_expectation_additive(
            func=loglik_node,
            mc_cos=mc_cos,
            logspace=True
        )

        # 4) U_CD is difference between expected log-likelihood and predictive log-density
        return E_logp - log_p_xt
