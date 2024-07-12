import networkx as nx
import torch

from src.config import EnvironmentConfig
from src.environments.environment import Environment


class ErdosRenyi(Environment):
    def __init__(self, num_nodes: int, cfg: EnvironmentConfig = None, p: float = None):
        assert num_nodes >= 2
        self.edge_prob = p if p is not None else 4. / (num_nodes - 1) if num_nodes > 5 else 0.5
        super().__init__(num_nodes, cfg)

    def construct_graph(self, num_nodes: int) -> nx.DiGraph:
        adj_mat = torch.bernoulli(torch.tensor(self.edge_prob).expand(num_nodes, num_nodes))
        adj_mat = torch.triu(adj_mat, diagonal=1).bool()
        order = torch.randperm(num_nodes)

        graph = nx.DiGraph()
        graph.add_nodes_from([f'X{i}' for i in range(num_nodes)])
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if adj_mat[i, j]:
                    graph.add_edge(f'X{order[i]}', f'X{order[j]}')

        return graph


class BarabasiAlbert(Environment):
    def __init__(self, num_nodes: int, cfg: EnvironmentConfig = None, num_parents_per_node: int = 2):
        assert num_nodes >= 2
        self.num_parents_per_node = num_parents_per_node
        super().__init__(num_nodes, cfg)

    def construct_graph(self, num_nodes: int) -> nx.DiGraph:
        graph = nx.generators.barabasi_albert_graph(num_nodes, self.num_parents_per_node)
        adj_mat = torch.tensor(nx.to_numpy_array(graph))
        adj_mat = torch.triu(adj_mat, diagonal=1).bool()

        graph = nx.DiGraph()
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if adj_mat[i, j]:
                    graph.add_edge(f'X{i}', f'X{j}')

        return graph


class SachsGraphGeneric(Environment):
    def __init__(self, cfg: EnvironmentConfig = None):
        super().__init__(num_nodes=11, cfg=cfg)

    def construct_graph(self, num_nodes: int) -> nx.DiGraph:
        graph = nx.DiGraph()
        graph.add_edge('X0', 'X1')
        graph.add_edge('X0', 'X2')
        graph.add_edge('X0', 'X3')
        graph.add_edge('X0', 'X4')
        graph.add_edge('X0', 'X5')
        graph.add_edge('X1', 'X2')
        graph.add_edge('X1', 'X3')
        graph.add_edge('X1', 'X7')
        graph.add_edge('X1', 'X6')
        graph.add_edge('X1', 'X5')
        graph.add_edge('X1', 'X4')
        graph.add_edge('X4', 'X5')
        graph.add_edge('X5', 'X6')
        graph.add_edge('X6', 'X7')
        graph.add_edge('X8', 'X9')
        graph.add_edge('X8', 'X10')
        graph.add_edge('X9', 'X10')
        return graph


class CRGraph(Environment):
    def __init__(self, cfg: EnvironmentConfig = None):
        super().__init__(num_nodes=5, cfg=cfg)

    def construct_graph(self, num_nodes: int) -> nx.DiGraph:
        graph = nx.DiGraph()
        graph.add_edge('X0', 'X1')
        graph.add_edge('X0', 'X2')
        graph.add_edge('X0', 'X3')
        graph.add_edge('X1', 'X2')
        graph.add_edge('X1', 'X3')
        graph.add_edge('X2', 'X4')
        graph.add_edge('X3', 'X4')
        return graph


class BiDiag(Environment):
    def __init__(self, num_nodes: int, cfg: EnvironmentConfig = None):
        assert num_nodes >= 2
        super().__init__(num_nodes, cfg)

    def construct_graph(self, num_nodes: int) -> nx.DiGraph:
        graph = nx.DiGraph()
        graph.add_edge('X0', 'X1')
        for i in range(2, num_nodes):
            graph.add_edge(f'X{i - 2}', f'X{i}')
            graph.add_edge(f'X{i - 1}', f'X{i}')
        return graph


class Chain(Environment):
    def __init__(self, num_nodes: int, cfg: EnvironmentConfig = None):
        assert num_nodes >= 2
        super().__init__(num_nodes, cfg)

    def construct_graph(self, num_nodes: int) -> nx.DiGraph:
        graph = nx.DiGraph()
        for i in range(1, num_nodes):
            graph.add_edge(f'X{i - 1}', f'X{i}')
        return graph


class Collider(Environment):
    def __init__(self, num_nodes: int, cfg: EnvironmentConfig = None):
        assert num_nodes >= 2
        super().__init__(num_nodes, cfg)

    def construct_graph(self, num_nodes: int) -> nx.DiGraph:
        graph = nx.DiGraph()
        for i in range(1, num_nodes):
            graph.add_edge(f'X{i}', 'X0')
        return graph


class Full(Environment):
    def __init__(self, num_nodes: int, cfg: EnvironmentConfig = None):
        assert num_nodes >= 2
        super().__init__(num_nodes, cfg)

    def construct_graph(self, num_nodes: int) -> nx.DiGraph:
        graph = nx.DiGraph()
        for i in range(num_nodes):
            root_node = [f'X{i}'] * (num_nodes - i)
            target_nodes = [f'X{j}' for j in range(i + 1, num_nodes)]
            graph.add_edges_from(zip(root_node, target_nodes))

        return graph


class Independent(Environment):
    def __init__(self, num_nodes: int, cfg: EnvironmentConfig = None):
        assert num_nodes >= 2
        super().__init__(num_nodes, cfg)

    def construct_graph(self, num_nodes: int) -> nx.DiGraph:
        graph = nx.DiGraph()
        graph.add_nodes_from([f'X{i}' for i in range(num_nodes)])
        return graph


class Jungle(Environment):
    def __init__(self, num_nodes: int, cfg: EnvironmentConfig = None):
        assert num_nodes >= 2
        super().__init__(num_nodes, cfg)

    def construct_graph(self, num_nodes: int) -> nx.DiGraph:
        graph = nx.DiGraph()
        graph.add_edge('X0', 'X1')
        graph.add_edge('X0', 'X2')
        for i in range(3, num_nodes):
            parent = int((i + 1) / 2) - 1
            grandparent = int((parent + 1) / 2) - 1
            graph.add_edge(f'X{parent}', f'X{i}')
            graph.add_edge(f'X{grandparent}', f'X{i}')

        return graph
