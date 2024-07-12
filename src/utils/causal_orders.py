from itertools import chain, combinations, product, permutations
from typing import List, Set, Optional

import networkx as nx
import torch

from src.mechanism_models.mechanisms import get_mechanism_key
from src.utils.graphs import adj_mat_to_graph


class CausalOrder:
    num_elements: int
    num_layers: int
    layers: List[Set[str]]
    co_mat: torch.Tensor
    adjacency_mask: torch.Tensor

    def __init__(self, layers: List[Set[str]]):
        # check if layers are disjoint and non-empty
        elements = set()
        for layer_idx, layer in enumerate(layers):
            assert len(layer) > 0, print('Invalid causal order contains empty layers!')
            assert elements.isdisjoint(layer), print('Invalid causal order ', layers)
            elements = elements.union(layer)

        self.node_labels = sorted(elements)
        self.node_label_to_id_dict = dict(zip(self.node_labels, list(range(len(self.node_labels)))))
        self.num_elements = len(elements)
        self.num_layers = len(layers)
        self.layers = layers

        # generate adjacency mask
        self.adjacency_mask = torch.zeros((self.num_elements, self.num_elements))
        succeeding_nodes = elements
        for layer in self.layers:
            succeeding_nodes = succeeding_nodes.difference(layer)
            node_ids = [self.node_label_to_id_dict[node] for node in layer]
            succ_node_ids = [self.node_label_to_id_dict[node] for node in succeeding_nodes]
            admitted_edges = product(node_ids, succ_node_ids)
            for e in admitted_edges:
                self.adjacency_mask[e] = 1.

        # generate co matrix representation
        self.co_mat = torch.zeros((self.num_elements, self.num_elements), dtype=torch.long)
        for layer_idx, layer in enumerate(self.layers):
            node_ids = [self.node_label_to_id_dict[node] for node in layer]
            self.co_mat[layer_idx, node_ids] = 1

        # create node:layer membership dict
        self.memberships = {}
        for layer_idx, layer in enumerate(self.layers):
            for member in layer:
                self.memberships[member] = layer_idx

    def get_num_layers(self):
        return len(self.layers)

    def __repr__(self):
        return '|'.join([','.join(sorted(layer)) for layer in self.layers])

    def __eq__(self, other):
        if not isinstance(other, CausalOrder) or \
                self.node_labels != other.node_labels or \
                self.num_layers != other.num_layers:
            return False

        for idx, layer in enumerate(self.layers):
            if self.layers[idx] != other.layers[idx]:
                return False

        return True

    def get_layer_idx(self, element: str):
        return self.memberships[element]

    def get_adjacency_mask(self):
        return self.adjacency_mask

    def get_adjacency_graph(self):
        return adj_mat_to_graph(self.adjacency_mask, self.node_labels)

    def get_co_mat(self):
        return self.co_mat

    def get_co_cum_mat(self):
        """Returns a matrix representation of the causal order as a (d+1) x d matrix: entry (i,j) is 1 if
        element j belongs to the (i-1)-th or any of it's preceeding layers, and is 0 otherwise. In words, the i-th row
        contains ones at all indices of elements that belong to layers (i-1) and earlier. Note: the first
        row is always just zeros.

        Returns
        ------
        torch.LongTensor
            The adjacency matrix of the graph.
        """
        co_mat = torch.zeros((self.num_elements + 1, self.num_elements))

        assigned = set()
        for layer_idx, layer in enumerate(self.layers):
            assigned = assigned.union(layer)
            node_ids = [self.node_label_to_id_dict[node] for node in assigned]
            co_mat[layer_idx + 1, node_ids] = 1

        co_mat[len(self.layers):] = 1
        return co_mat


def resolve_co_key(key: str) -> CausalOrder:
    """Return a CausalOrder object according to the given key.

    Parameters
    ----------
    key : str
        The string representation of the causal order to be generated.

    Returns
    ------
    CausalOrder
        A CausalOrder object corresponding to the given key.
    """
    layer_strings = key.split('|')
    layers = []
    for lstr in layer_strings:
        layers.append(set(lstr.split(',')))

    return CausalOrder(layers)


def co_from_co_mat(co_mat: torch.Tensor, node_labels: List[str]) -> CausalOrder:
    """Return a CausalOrder object according to the matrix representation.

    Parameters
    ----------
    co_mat: torch.Tensor
        The matrix representation of the causal order to be generated.
    node_labels: List[str]
        List of node labels.

    Returns
    ------
    CausalOrder
        A CausalOrder object corresponding to the given matrix representation.
    """
    assert co_mat.dim() == 2 and co_mat.shape[0] == co_mat.shape[1], print(co_mat.shape)
    num_nodes = co_mat.shape[0]

    layers = []
    num_assigned = 0
    for layer_idx in range(num_nodes):
        layer = co_mat[layer_idx].nonzero().squeeze(-1).tolist()
        if len(layer) == 0:
            break
        else:
            layer = {node_labels[node] for node in layer}
            layers.append(set(layer))
            num_assigned += len(layer)

    assert num_assigned == num_nodes, print(f'{num_assigned} elements assigned to {num_nodes} layers!', co_mat)
    return CausalOrder(layers)


def co_from_graph(dag: nx.DiGraph) -> Optional[CausalOrder]:
    if not nx.is_directed_acyclic_graph(dag):
        return None

    topological_order = nx.topological_sort(dag)
    return CausalOrder([{node} for node in topological_order])


def generate_all_permutations(node_labels: List[str]) -> List[CausalOrder]:
    perm = permutations([{node} for node in node_labels])
    return [CausalOrder(list(p)) for p in perm]


def generate_all_parent_sets(node_labels: List[str], max_parent_set_size: int, adj_mask: torch.Tensor):
    num_nodes = len(node_labels)

    # generate for each node the possible parent sets given the adjeceny mask and the size constraint
    parent_sets_per_node = dict()
    for nidx in range(num_nodes):
        possible_parents = adj_mask[:, nidx].nonzero().view(-1).tolist()
        possible_parents = [node_labels[i] for i in possible_parents]
        possible_parent_sets = [combinations(possible_parents, size) for size in range(max_parent_set_size + 1)]
        possible_parent_sets = chain(*possible_parent_sets)
        parent_sets_per_node[node_labels[nidx]] = [list(ps) for ps in possible_parent_sets]

    return parent_sets_per_node


def generate_all_mechanisms(node_labels: List[str], max_parent_set_size: int, adj_mask: torch.Tensor):
    mechanisms = []
    for nidx, node in enumerate(node_labels):
        possible_parents = adj_mask[:, nidx].nonzero().view(-1).tolist()
        possible_parents = [node_labels[i] for i in possible_parents]
        possible_parent_sets = [combinations(possible_parents, size) for size in range(max_parent_set_size + 1)]
        possible_parent_sets = chain(*possible_parent_sets)
        mechanisms.extend([get_mechanism_key(node, list(ps)) for ps in possible_parent_sets])

    return mechanisms
