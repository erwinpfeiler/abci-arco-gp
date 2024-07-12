from typing import List, Union

import networkx as nx
import pandas as pd
import torch


def generate_all_dgs(num_nodes: int = 3, node_labels: List[str] = None, only_acyclic: bool = True) -> List[nx.DiGraph]:
    """Generates all directed acyclic graphs with a given number of nodes or node labels.

    Parameters
    ----------
    num_nodes : int
        The number of nodes. If `node_labels` is given this has no effect.
    node_labels : List[str]
        List of node labels. If not None, the number of nodes is inferred automatically.
    only_acyclic : bool
        If true, only directed acyclic graphs are returned, otherwise cyclic graphs will be included as well.

    Returns
    ------
    List[nx.DiGraph]
        A list of DAGs as Networkx DiGraph objects..
    """
    # check if node labels given and create enumerative mapping
    if node_labels is not None:
        node_labels = sorted(list(set(node_labels)))
        num_nodes = len(node_labels)
        node_map = dict(zip(list(range(num_nodes)), node_labels))
    else:
        node_map = dict(zip(list(range(num_nodes)), list(range(num_nodes))))

    # check dag size feasibility
    assert num_nodes > 0, f'There is no such thing as a graph with {num_nodes} nodes.'
    assert num_nodes < 5, f'There are a lot of DAGs with {num_nodes} nodes...'

    # generate adjecency lists of all possible, simple graphs with num_nodes nodes
    adj_lists = [[]]
    for src in range(num_nodes):
        for dest in range(num_nodes):
            if src != dest:
                adj_lists = adj_lists + [[[node_map[src], node_map[dest]]] + adj_list for adj_list in adj_lists]

    # create graphs
    graphs = []
    for adj_list in adj_lists:
        graph = nx.DiGraph()
        graph.add_nodes_from(list(node_map.values()))
        graph.add_edges_from(adj_list)
        if not only_acyclic or nx.is_directed_acyclic_graph(graph):
            # and keep only DAGs
            graphs.append(graph)

    return graphs


def get_graph_key(graph: nx.DiGraph) -> str:
    """Generates a unique string representation of a directed graph. Can be used as a dictionary key.

    Parameters
    ----------
    graph : nx.DiGraph
        The graph for which to generate the string representation.

    Returns
    ------
    str
        A unique string representation of `graph`.
    """
    graph_str = ''
    for i, node in enumerate(sorted(graph)):
        if i > 0:
            graph_str += '|'
        graph_str += str(node) + '<-' + ','.join([str(parent) for parent in get_parents(node, graph)])

    return graph_str


def resolve_graph_key(key: str) -> nx.DiGraph:
    """Return a NetworkX DiGraph object according to the given graph key.

    Parameters
    ----------
    key : str
        The string representation of the graph to be generated.

    Returns
    ------
    nx.DiGraph
        A graph object corresponding to the given graph key.
    """
    graph = nx.DiGraph()
    mech_strings = key.split('|')
    for mstr in mech_strings:
        idx = mstr.find('<-')
        node = mstr[:idx]
        parents = mstr[idx + 2:].split(',') if len(mstr) > idx + 2 else []

        graph.add_node(node)
        for parent in parents:
            graph.add_edge(parent, node)

    return graph


def get_parents(node: str, graph: nx.DiGraph) -> List[str]:
    """Returns a list of parents for a given node in a given graph.

    Parameters
    ----------
    node : str
        The child node.
    graph : nx.DiGraph
        The graph inducing the parent set.

    Returns
    ------
    List[str]
        The list of parents.
    """
    return sorted(list(graph.predecessors(node)))


def graph_to_adj_mat(graph: nx.DiGraph, node_labels: List[str]) -> torch.Tensor:
    """Returns the adjecency matrix of the given graph as tensor.

    Parameters
    ----------
    graph : nx.DiGraph
        The graph.
    node_labels : List[str]
        The list of node labels determining the order of the adjacency matrix.

    Returns
    ------
    torch.Tensor
        The adjacency matrix of the graph.
    """
    return torch.tensor(nx.to_numpy_array(graph, nodelist=node_labels)).float()


def adj_mat_to_graph(adj_mat: torch.Tensor, node_labels: List[str]) -> nx.DiGraph:
    assert adj_mat.dim() == 2 and adj_mat.shape[0] == adj_mat.shape[1], print(adj_mat.shape)
    graph = nx.from_numpy_array(adj_mat.int().cpu().numpy(), create_using=nx.DiGraph)
    node_labels = dict(zip(list(range(len(node_labels))), node_labels))
    graph = nx.relabel_nodes(graph, node_labels)
    return graph


def dag_to_cpdag(dag: Union[nx.DiGraph, torch.Tensor], node_labels: List[str]):
    """Returns the adjecency matrix of the CPDAG representing the MEC of the input DAG. This implements the algorithm
       presented by Chickering, D. M. (1995), "A transformational characterization of Bayesian network structures".

        Parameters
        ----------
        dag : Union[nx.DiGraph, torch.Tensor]
            A directed acyclic graph.
        node_labels: List[str]
            List of node labels determining the order of nodes in the adjacency matrix.

        Returns
        ------
        torch.Tensor
            The adjacency matrix of the CPDAG representing the MEC of the input DAG.
    """
    # convert adjacency matrix to dag if necessary
    dag = adj_mat_to_graph(dag, node_labels) if isinstance(dag, torch.Tensor) else dag

    # order edges
    ordered_edges = []
    topo_order = list(nx.topological_sort(dag))
    for sink in reversed(list(topo_order)):
        ordered_parents = [node for node in topo_order if node in dag.predecessors(sink)]
        ordered_edges.extend([(parent, sink) for parent in ordered_parents])

    # label edges
    unknown_edges = set(ordered_edges)
    compelled_edges = set()
    reversible_edges = set()
    for x, y in reversed(ordered_edges):
        if not (x, y) in unknown_edges:
            continue

        # check cycle constraint
        cycle_constraint = False
        compelled_into_x = {edge for edge in compelled_edges if edge[1] == x}
        for edge in compelled_into_x:
            w = edge[0]
            if w not in dag.predecessors(y):
                edges = {(parent, y) for parent in dag.predecessors(y)}
                compelled_edges.update(edges)
                unknown_edges.difference_update(edges)
                cycle_constraint = True
                break
            else:
                compelled_edges.update({(w, y)})
                unknown_edges.discard((w, y))

        if cycle_constraint:
            continue

        # check v-structures
        edges = {edge for edge in unknown_edges if edge[1] == y}
        unknown_edges.difference_update(edges)
        tmp = {(z, y) for z in dag.predecessors(y) if z != x and z not in dag.predecessors(x)}
        if len(tmp) > 0:
            compelled_edges.update(edges)
        else:
            reversible_edges.update(edges)

    # return cpdag
    cpdag = nx.DiGraph()
    cpdag.add_nodes_from(dag.nodes)
    cpdag.add_edges_from(compelled_edges)
    cpdag.add_edges_from(reversible_edges)
    cpdag.add_edges_from([(y, x) for x, y in reversible_edges])
    adj_mat = graph_to_adj_mat(cpdag, node_labels)

    return adj_mat


def graph_from_csv(file: str):
    df = pd.read_csv(file)
    df = df.set_index(df.columns)
    return nx.from_pandas_adjacency(df, create_using=nx.DiGraph)
