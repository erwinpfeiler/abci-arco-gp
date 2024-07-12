import functools
import math
from typing import List, Tuple, Dict, Any

import networkx as nx
import torch

from src.utils.graphs import generate_all_dgs, get_graph_key, graph_to_adj_mat


class CategoricalModel:
    """
    Class that represents a categorical distribution over graphs.

    Attributes
    ----------
    graphs : List[nx.DiGraph]
        List of possible DAGs for this model.
    node_labels : List[str]
        List of node labels.
    num_nodes : int
        The number of nodes in this model.
    num_graphs : int
        The number of possible DAGs for this model.
    log_probs : Dict[str, torch.Tensor]
        Dictionary of graph identifiers and corresponding log probabilities.
    """

    def __init__(self, node_labels: List[str] = None, param_dict: Dict[str, Any] = None):
        """
        Parameters
        ----------
        node_labels : List[str]
            List of node labels.
        """
        assert node_labels is not None or param_dict is not None
        if param_dict is not None:
            self.load_param_dict(param_dict)
        else:
            self.node_labels = sorted(list(set(node_labels)))
            self.num_nodes = len(node_labels)
            self.graphs = generate_all_dgs(node_labels=node_labels)
            self.num_graphs = len(self.graphs)
            graph_keys = [get_graph_key(graphs) for graphs in self.graphs]
            self.log_probs = dict(zip(graph_keys, -torch.log(self.num_graphs * torch.ones(self.num_graphs))))

    def log_prob(self, graph: nx.DiGraph) -> torch.Tensor:
        """
        Returns the log probability for a given graph.

        Parameters
        ----------
        graph : nx.DiGraph
            The query graph.

        Returns
        ----------
        torch.Tensor
            The log probability.
        """
        assert get_graph_key(graph) in self.log_probs
        return self.log_probs[get_graph_key(graph)]

    def prob(self, graph: nx.DiGraph) -> torch.Tensor:
        """
        Returns the probability for a given graph.

        Parameters
        ----------
        graph : nx.DiGraph
            The query graph.

        Returns
        ----------
        torch.Tensor
            The probability.
        """
        assert get_graph_key(graph) in self.log_probs
        return self.log_probs[get_graph_key(graph)].exp()

    def set_log_prob(self, log_prob: torch.tensor, graph: nx.DiGraph):
        """
        Sets the log probability of a given graph.

        Parameters
        ----------
        log_prob: torch.Tensor
            The new log prabability.
        graph : nx.DiGraph
            The target graph.
        """
        assert log_prob.numel() == 1
        assert get_graph_key(graph) in self.log_probs

        self.log_probs[get_graph_key(graph)] = log_prob.squeeze()

    def normalize(self):
        """
        Normalizes the distribution over graphs such that the sum over all graphs probabilities equals 1.
        """
        logits = torch.stack([p for p in self.log_probs.values()])
        log_evidence = logits.logsumexp(dim=0)
        log_probs = logits - log_evidence
        self.log_probs = dict(zip(self.log_probs.keys(), log_probs))

    def entropy(self):
        """
        Returns the entropy of the categorical distribution over graphs.

        Returns
        ----------
        torch.Tensor
            The entropy.
        """
        tmp = torch.stack(list(self.log_probs.values()))
        return -(tmp.exp() * tmp).sum()

    def sample(self, num_graphs: int) -> List[nx.DiGraph]:
        """
        Samples a graph.

        Parameters
        ----------
        num_graphs: int
            Number of graphs to sample.

        Returns
        ----------
        List[nx.DiGraph]
            List of sampled graph objects.
        """
        probs = torch.stack([self.prob(graph) for graph in self.graphs])
        graph_idc = torch.multinomial(probs, num_graphs, replacement=True)
        return [self.graphs[idx] for idx in graph_idc]

    def sort_by_prob(self, descending: bool = True) -> List[nx.DiGraph]:
        """
        Returns a list of all graphs sorted by their probabilities.

        Parameters
        ----------
        descending: bool
            If true, sort in descending order, ascending otherwise.

        Returns
        ----------
        List[nx.DiGraph]
            List of sorted graph objects.
        """

        def compare(graph1, graph2):
            return (self.log_prob(graph1) - self.log_prob(graph2)).item()

        return sorted(self.graphs, key=functools.cmp_to_key(compare), reverse=descending)

    def edge_probs(self) -> torch.Tensor:
        """
        Returns the matrix of edge probabilities.

        Returns
        ----------
        torch.Tensor
            Matrix of edge probabilities.
        """
        edge_probs = torch.zeros(self.num_nodes, self.num_nodes)
        for graph in self.graphs:
            adj_mat = self.graph_to_adj_mat(graph)
            edge_probs += self.prob(graph) * adj_mat

        return edge_probs

    def graph_to_adj_mat(self, graph: nx.DiGraph) -> torch.Tensor:
        return graph_to_adj_mat(graph, self.node_labels)

    def get_mc_graphs(self, mode: str, num_mc_graphs: int = 20) -> Tuple[List[nx.DiGraph], torch.Tensor]:
        """
        Returns a set of graphs and corresponding log-weights for Monte Carlo estimation.

        Parameters
        ----------
        mode: str
            There are three strategies for generating the MC graph set:
              'full': Returns all graphs and their log-probabilities as log-weights.
              'sampling': Returns `num_mc_graphs` samples from the distribution over graphs with uniform weights.
              'n-best': Returns the `num_mc_graphs` graphs with the highest probabilities weighted according to their
                        re-normalized probabilities.
        num_mc_graphs: int
            The size of the returned set of graphs (see description above). Has no effect when mode is 'full'.

        Returns
        ----------
        List[nx.DiGraph], torch.Tensor
            The set of MC graphs and their corresponding log-weights.
        """
        if mode not in {'full', 'sampling', 'n-best'}:
            print('Invalid sampling mode >' + mode + '<. Doing <full> instead.')
            mode = 'full'

        if mode == 'sampling':
            graphs = self.sample(num_mc_graphs)
            log_weights = -torch.ones(num_mc_graphs) * math.log(num_mc_graphs)
        elif mode == 'n-best':
            graphs = self.sort_by_prob()[:num_mc_graphs]
            log_weights = [self.log_prob(graph) for graph in graphs]
            log_weights = torch.log_softmax(torch.stack(log_weights), dim=0)
        else:
            graphs = self.graphs
            log_weights = [self.log_prob(graph) for graph in graphs]
            log_weights = torch.stack(log_weights)

        return graphs, log_weights

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
                  'graphs': self.graphs,
                  'num_graphs': self.num_graphs,
                  'log_probs': self.log_probs}
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
        self.graphs = param_dict['graphs']
        self.num_graphs = param_dict['num_graphs']
        self.log_probs = param_dict['log_probs']
