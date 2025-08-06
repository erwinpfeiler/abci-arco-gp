"""exp_designer_abci_arco_gp.py
================================================
Experiment‑designer for **ARCO‑GP** (Active Bayesian Causal Inference)
--------------------------------------------------------------------
Implements *causal‑discovery* information gain (Eq. 4.13, ABCI paper)
using the same class interface and coding style as
``exp_designer_abci_dibs_gp.py`` so it can be dropped into the existing
pipeline without touching client code.

Key references used in comments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ABCI paper Eq. 4.13  –  causal‑discovery utility.
* ARCO‑GP paper Eq. (4) –  importance weight *w_L*.
* ARCO‑GP paper §4.4   –  closed‑form GP marginal likelihood & predictive.
* Previous derivations  –  fully expanded U_CD (message ★).
"""

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

###############################################################################
# Helper classes                                                              #
###############################################################################

class GraphPosteriorArCO:
    """Adapter that turns an **ARCO** order posterior into a *graph* posterior.

    The ABCI designer only needs a list of graphs and their log posterior
    probabilities.  We obtain those by

    1. sampling causal orders L ~ p(L|θ),
    2. enumerating *all* DAGs G respecting L with ≤K parents each,
    3. evaluating the (closed‑form) marginal likelihood p(D|G,ψ) and
       normalising across all sampled graphs to approximate p(G|D).
    """

    def __init__(
        self,
        arco: ArCO,
        mech_model: SharedDataGaussianProcessModel,
        max_parent_set: int = 2,
        num_orders: int = 20,
    ) -> None:
        self.arco = arco
        self.mech_model = mech_model
        self.K = max_parent_set
        self.M = num_orders

    # ------------------------------------------------------------------
    # Public API expected by ExpDesignerBase subclasses
    # ------------------------------------------------------------------
    def mc_graphs(self) -> Tuple[List[nx.DiGraph], torch.Tensor]:
        """Return `(graphs, log_posterior)` where the second entry is a 1‑D
        tensor of *log* probabilities that sum to 0 in log‑space."""
        graphs: List[nx.DiGraph] = []
        logps: List[torch.Tensor] = []

        # 1. sample a small batch of orders – same strategy as DiBS designer
        orders, masks = self.arco.sample(self.M)

        # 2. enumerate DAGs for each order and compute log p(D|G,ψ)
        for mask in masks:
            dags = self.arco.enumerate_graphs_from_order(mask, self.K)
            for G in dags:
                graphs.append(G)
                # GP‑marginal likelihood, Eq. 11 (closed form; cached)
                logps.append(
                    self.mech_model.mll(
                        [], G, prior_mode=False, use_cache=True
                    )
                )

        logp_tensor = torch.tensor(logps, dtype=torch.float32)
        # 3. normalise – log posterior over graphs up to sampled support
        logp_tensor = logp_tensor - torch.logsumexp(logp_tensor, dim=0)
        return graphs, logp_tensor

###############################################################################
# Designer class                                                              #
###############################################################################

class ExpDesignerABCIArCOGP(ExpDesignerBase):
    """Active‑learning designer for ARCO‑GP *(causal discovery only)*.

    Interface and control‑flow mirror ``ExpDesignerABCIDiBSGP`` so that
    ``active_bayesian_causal_inference.py`` can instantiate either
    designer without branching logic.
    """

    def __init__(
        self,
        intervention_bounds: Dict[str, Tuple[float, float]],
        opt_strategy: str = "gp-ucb",
        distributed: bool = False,
    ) -> None:
        super().__init__(intervention_bounds, opt_strategy, distributed)
        # The following members are populated in `init_design_process()`
        self.mech_model: Optional[SharedDataGaussianProcessModel] = None
        self.graph_posterior: Optional[GraphPosteriorArCO] = None

    # ------------------------------------------------------------------
    # Initialisation hook (same name/signature as DiBS designer)         
    # ------------------------------------------------------------------
    def init_design_process(self, args: dict):
        """Called once by the ABCI main loop right before the first query.

        Expected ``args`` keys (identical to DiBS designer):
            * 'mechanism_model' : SharedDataGaussianProcessModel
            * 'order_model'     : ArCO
            * 'policy'          : str – must be 'graph-info-gain'
            * 'batch_size', 'num_exp_per_graph', 'mode'  – optional
        """
        self.mech_model = args["mechanism_model"]
        arco_model: ArCO = args["order_model"]
        self.graph_posterior = GraphPosteriorArCO(arco_model, self.mech_model)

        # chooser identical to DiBS designer – here only graph IG supported
        assert args["policy"] == "graph-info-gain", (
            "ARCO designer currently supports only 'graph-info-gain'."
        )

        def _utility(intv: Dict[str, float]):
            return self._graph_info_gain(
                intv,
                batch_size=args.get("batch_size", 1),
                num_exp_per_graph=args.get("num_exp_per_graph", 1),
            )

        self.utility = _utility

    # ------------------------------------------------------------------
    # Internal helpers (keep same names as DiBS designer where applicable)
    # ------------------------------------------------------------------
    def _simulate_experiment(
        self,
        interventions: Dict[str, float],
        batch_size: int,
        num_exp_per_graph: int,
        graph: nx.DiGraph,
    ) -> Experiment:
        """Draw *synthetic* outcome(s) X_t ~ p(·|G,do(a),D) (one batch)."""
        return self.mech_model.sample(
            interventions, batch_size, num_exp_per_graph, graph
        )

    def _log_pred_density(
        self, exp: Experiment, graph: nx.DiGraph
    ) -> torch.Tensor:
        """Closed‑form GP predictive log‑density log p(exp | G, D)."""
        return self.mech_model.mll(
            [exp], graph, prior_mode=False, use_cache=True, reduce=True
        )

    # ------------------------------------------------------------------
    # Causal‑discovery information gain                                  
    # ------------------------------------------------------------------
    def _graph_info_gain(
        self,
        interventions: Dict[str, float],
        batch_size: int = 1,
        num_exp_per_graph: int = 1,
        num_mc_graphs: int = 20,
    ) -> torch.Tensor:
        """Monte‑Carlo estimate of U_CD (Eq. 4.13)."""
        assert self.graph_posterior is not None and self.mech_model is not None

        # (1) MC outer graphs & log weights (p(G|D) over sampled support)
        outer_graphs, log_post = self.graph_posterior.mc_graphs()
        post_probs = log_post.exp()

        # storage for individual IG estimates per outer graph
        ig_estimates = torch.zeros(len(outer_graphs))

        for idx, G in enumerate(outer_graphs):
            # (2) simulate fictitious experiment under G
            synthetic_data = self._simulate_experiment(
                interventions, batch_size, num_exp_per_graph, G
            )

            # IMPORTANT: new synthetic data ⇒ clear GP cache
            self.mech_model.clear_posterior_mll_cache()

            # (3) log p(x̃ | G, D)
            ll_G = self._log_pred_density(synthetic_data, G)

            # (4) mixture predictive – reuse the SAME synthetic data and graphs
            ll_all = torch.tensor([
                self._log_pred_density(synthetic_data, Gp) for Gp in outer_graphs
            ])
            log_mix = torch.logsumexp(ll_all + log_post, dim=0)

            #  (ll_G - log_mix) averaged over synthetic draws (already scalar)
            ig_estimates[idx] = ll_G - log_mix

        # (5) final expectation over graph posterior samples
        return (post_probs * ig_estimates).sum()
