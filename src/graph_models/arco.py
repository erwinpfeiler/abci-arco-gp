from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
from torch.nn.utils import parameters_to_vector

from src.config import ArCOConfig
from src.utils.causal_orders import CausalOrder, co_from_co_mat
from src.utils.utils import inf_tensor


class ArCO:
    def __init__(self, node_labels: List[str] = None, cfg: ArCOConfig = None, param_dict=None):
        if param_dict is not None:
            self.load_param_dict(param_dict)
        else:
            # load (default) config
            self.cfg = ArCOConfig() if cfg is None else cfg
            self.node_labels = sorted(set(node_labels))
            self.num_nodes = len(self.node_labels)

            # init logit map
            if self.cfg.map_mode == 'mlp':
                self.logit_map = MLPLogitMap(self.num_nodes, self.cfg)
            elif self.cfg.map_mode == 'simple':
                self.logit_map = SimpleLogitMap(self.num_nodes, self.cfg)
            else:
                assert False, print(f'Invalid map mode {self.cfg.map_mode}.')

    def sample(self, num_cos: int = 1) -> Tuple[List[CausalOrder], torch.Tensor]:
        co_list = []
        for _ in range(num_cos):
            unassigned = torch.ones(self.num_nodes, dtype=torch.long)
            co_mat = torch.zeros(self.num_nodes, self.num_nodes)
            for pidx in range(self.num_nodes - 1):
                # sample next element of the causal order
                with torch.no_grad():
                    logits = self.logit_map(co_mat.unsqueeze(0), torch.LongTensor([pidx])).squeeze()
                    logits = torch.where(unassigned.bool(), logits, -inf_tensor())
                    elem_idx = Categorical(logits=logits).sample()

                co_mat[pidx, elem_idx] = 1.
                unassigned[elem_idx] = 0

            # prob of the last unassigned element is 1
            assert unassigned.sum() == 1
            elem_idx = unassigned.nonzero().squeeze()
            co_mat[-1, elem_idx] = 1.

            co_list.append(co_from_co_mat(co_mat, self.node_labels))

        # assemble adjacency masks
        masks = []
        for cidx in range(num_cos):
            masks.append(co_list[cidx].get_adjacency_mask())

        masks = torch.stack(masks)

        return co_list, masks

    def log_prob(self, cos: List[CausalOrder]) -> torch.Tensor:
        num_cos = len(cos)
        co_log_probs = torch.zeros(num_cos)
        for co_idx in range(num_cos):
            if cos[co_idx].num_layers != self.num_nodes:
                # not a total order / permutation of nodes -> prob is zero
                co_log_probs[co_idx] = -inf_tensor()
                continue

            co_mat = cos[co_idx].get_co_mat().unsqueeze(0)
            co_cum_mat = cos[co_idx].get_co_cum_mat().unsqueeze(0)
            layer_idc = torch.arange(cos[co_idx].num_layers, dtype=torch.long)

            logits = self.logit_map(co_mat, layer_idc)
            logits = torch.where(co_cum_mat[:, layer_idc].bool(), -inf_tensor(), logits)
            logprobs = torch.log_softmax(logits, dim=-1)
            logprobs = torch.where(co_mat[:, layer_idc].bool(), logprobs, -inf_tensor())
            co_log_probs[co_idx] = logprobs.logsumexp(dim=-1).sum(dim=-1)

        return co_log_probs

    def parameters(self):
        return list(self.logit_map.parameters())

    def log_param_prior(self):
        return self.logit_map.log_param_prior()

    def param_dict(self) -> Dict[str, Any]:
        """
        Returns the current parameters of an instance of this class as a dictionary.

        Returns
        ----------
        Dict[str, Any]
            Parameter dictionary.
        """
        params = {'node_labels': self.node_labels,
                  'num_nodes': self.num_nodes,
                  'cfg_param_dict': self.cfg.param_dict(),
                  'logit_map_param_dict': self.logit_map.param_dict()}
        return params

    def load_param_dict(self, param_dict: Dict[str, Any]):
        """
        Sets the parameters of this class instance with the parameter values given in `param_dict`.

        Parameters
        ----------
        param_dict : Dict[str, Any]
            Parameter dictionary.
        """
        self.node_labels = param_dict['node_labels']
        self.num_nodes = param_dict['num_nodes']
        self.cfg = ArCOConfig()
        if 'cfg_param_dict' in param_dict:
            cfg_param_dict = param_dict['cfg_param_dict']
        else:
            # for backwards compatibility with existing stored models
            cfg_param_dict = param_dict['logit_map_param_dict']['cfg_param_dict']
        self.cfg.load_param_dict(cfg_param_dict)

        if self.cfg.map_mode == 'mlp':
            self.logit_map = MLPLogitMap(self.num_nodes, self.cfg)
        elif self.cfg.map_mode == 'simple':
            self.logit_map = SimpleLogitMap(self.num_nodes, self.cfg)
        else:
            assert False, print(f'Invalid map mode {self.cfg.map_mode}.')

        self.logit_map.load_param_dict(param_dict['logit_map_param_dict'])


class LogitMapBase(nn.Module):
    def __init__(self, num_nodes: int, cfg: ArCOConfig = None):
        """
        Parameters
        ----------
        num_nodes : int
            Number of inputs.
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.cfg = ArCOConfig() if cfg is None else cfg

    def forward(self, co_mats: torch.Tensor, layer_idc: torch.LongTensor) -> torch.Tensor:
        raise NotImplementedError

    def log_param_prior(self) -> torch.Tensor:
        raise NotImplementedError

    def param_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    def load_param_dict(self, param_dict: Dict[str, Any]):
        raise NotImplementedError


class SimpleLogitMap(LogitMapBase):
    def __init__(self, num_nodes: int, cfg: ArCOConfig = None):
        """
        Parameters
        ----------
        num_nodes : int
            Number of inputs.
        """
        super().__init__(num_nodes, cfg)

        self.prior = Normal(self.cfg.prior_loc, self.cfg.prior_scale)

        self.theta = nn.Parameter(torch.zeros(self.num_nodes, ))
        # self.theta = nn.Parameter(self.prior.sample((self.num_nodes,)))

    def forward(self, co_mats: torch.Tensor, layer_idc: torch.LongTensor) -> torch.Tensor:
        assert co_mats.dim() == 3 and co_mats.shape[-2:] == (self.num_nodes, self.num_nodes), \
            print(f'Invalid parameter shape {co_mats.shape}.')
        assert layer_idc.dim() == 1

        batch_size = co_mats.shape[0]
        num_layer_idc = layer_idc.numel()
        return self.theta.expand(batch_size, num_layer_idc, -1)

    def log_param_prior(self):
        return self.prior.log_prob(self.theta).mean()

    def param_dict(self) -> Dict[str, Any]:
        """
        Returns the current parameters of this class instance as a dictionary.

        Returns
        ----------
        Dict[str, Any]
            Parameter dictionary.
        """
        params = {'theta': self.theta}
        return params

    def load_param_dict(self, param_dict: Dict[str, Any]):
        """
        Sets the parameters of this class instance with the parameter values given in `param_dict`.

        Parameters
        ----------
        param_dict : Dict[str, Any]
            Parameter dictionary.
        """
        self.theta = param_dict['theta']


class MLPLogitMap(LogitMapBase):
    def __init__(self, num_nodes: int, cfg: ArCOConfig = None):
        """
        Parameters
        ----------
        num_nodes : int
            Number of inputs.
        """
        super().__init__(num_nodes, cfg)
        self.prior = Normal(self.cfg.prior_loc, self.cfg.prior_scale)

        out_size = self.num_nodes
        self.mlp = nn.Sequential(
            nn.Linear(self.num_nodes ** 2, self.cfg.num_hidden),
            nn.ReLU(),
            nn.Linear(self.cfg.num_hidden, out_size)
        )

        # init layer mask
        self.layer_mask = torch.zeros(1, self.num_nodes, self.num_nodes, self.num_nodes)
        for i in range(self.num_nodes):
            self.layer_mask[0, i, :i] = 1.

    def forward(self, co_mats: torch.Tensor, layer_idc: torch.LongTensor) -> torch.Tensor:
        assert co_mats.dim() == 3 and co_mats.shape[-2:] == (self.num_nodes, self.num_nodes), print(co_mats.shape)
        assert layer_idc.dim() == 1

        batch_size = co_mats.shape[0]
        num_layer_idc = layer_idc.numel()

        z = co_mats.unsqueeze(1) * self.layer_mask[:, layer_idc, :, :]
        z = z.view(batch_size, num_layer_idc, -1)
        logits = self.mlp(z)
        return logits

    def log_param_prior(self):
        param_vec = parameters_to_vector(self.parameters())
        return self.prior.log_prob(param_vec).mean()

    def param_dict(self) -> Dict[str, Any]:
        """
        Returns the current parameters of this class instance as a dictionary.

        Returns
        ----------
        Dict[str, Any]
            Parameter dictionary.
        """
        params = {'mlp_state_dict': self.mlp.state_dict()}
        return params

    def load_param_dict(self, param_dict: Dict[str, Any]):
        """
        Sets the parameters of this class instance with the parameter values given in `param_dict`.

        Parameters
        ----------
        param_dict : Dict[str, Any]
            Parameter dictionary.
        """
        self.mlp.load_state_dict(param_dict['mlp_state_dict'])
