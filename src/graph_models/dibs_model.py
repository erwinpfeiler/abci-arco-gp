from typing import List, Tuple, Dict, Any

import networkx as nx
import torch
import torch.distributions as dist
from torch.autograd import Function
from torch.nn.functional import logsigmoid

from src.config import DiBSConfig
from src.utils.graphs import graph_to_adj_mat, adj_mat_to_graph


class DiBSModel:
    def __init__(self, node_labels: List[str] = None, cfg: DiBSConfig = None, param_dict: Dict[str, Any] = None):
        assert node_labels is not None or param_dict is not None
        if param_dict is not None:
            self.load_param_dict(param_dict)
        else:
            # load config
            self.cfg = DiBSConfig() if cfg is None else cfg

            self.node_labels = sorted(list(set(node_labels)))
            self.num_nodes = len(self.node_labels)
            self.node_id_to_label_dict = dict(zip(list(range(self.num_nodes)), node_labels))
            self.node_label_to_id_dict = dict(zip(node_labels, list(range(self.num_nodes))))

            if self.cfg.embedding_size is None:
                self.cfg.embedding_size = self.num_nodes
            self.particles = self.sample_initial_particles(self.cfg.num_particles)

    def _check_particle_shape(self, z: torch.Tensor):
        assert z.dim() == 4 and z.shape[1:] == (self.cfg.embedding_size, self.num_nodes, 2), print(z.shape)

    def sample_initial_particles(self, num_particles: int) -> torch.Tensor:
        # sample particles from normal prior
        normal = torch.distributions.Normal(0., self.cfg.prior_scale)
        particles = normal.sample(torch.Size((num_particles, self.cfg.embedding_size, self.num_nodes, 2)))
        # move the means of the U,V components apart s.t. we do not get highly cyclic graphs initially
        particles[:, :, :, 0] -= self.cfg.prior_scale
        particles[:, :, :, 1] += self.cfg.prior_scale
        particles.requires_grad_(True)
        self._check_particle_shape(particles)
        return particles

    def edge_logits(self, alpha: float):
        return alpha * torch.einsum('ikj,ikl->ijl', self.particles[..., 0], self.particles[..., 1])

    def edge_probs(self, alpha: float):
        # compute edge probs
        edge_probs = torch.sigmoid(self.edge_logits(alpha))

        # set probs of self loops to 0
        mask = torch.eye(self.num_nodes).repeat(self.cfg.num_particles, 1, 1).bool()
        edge_probs[mask] = 0.
        return edge_probs

    def edge_log_probs(self, alpha: float):
        return logsigmoid(self.edge_logits(alpha))

    def log_generative_prob(self, adj_mats: torch.Tensor, alpha: float, batch_mode=True):
        assert adj_mats.dim() == 4 and adj_mats.shape[2:] == (self.num_nodes, self.num_nodes)
        assert adj_mats.shape[0] == self.particles.shape[0] or not batch_mode
        logits = self.edge_logits(alpha)
        log_edge_probs = logsigmoid(logits)
        log_not_edge_probs = log_edge_probs - logits  # =logsigmoid(-logits) = log(1-sigmoid(logits))

        # set probs of self loops to 0
        mask = torch.eye(self.num_nodes).repeat(self.cfg.num_particles, 1, 1).bool()
        log_edge_probs = torch.where(mask, torch.tensor(0.), log_edge_probs)
        log_not_edge_probs = torch.where(mask, torch.tensor(0.), log_not_edge_probs)

        if batch_mode:
            graph_log_probs = torch.einsum('hijk,hjk->hi', adj_mats, log_edge_probs) + \
                              torch.einsum('hijk,hjk->hi', (1. - adj_mats), log_not_edge_probs)
        else:
            graph_log_probs = torch.einsum('hijk,ljk->lhi', adj_mats, log_edge_probs) + \
                              torch.einsum('hijk,ljk->lhi', (1. - adj_mats), log_not_edge_probs)

        return graph_log_probs

    def unnormalized_log_prior(self, beta: float, adj_mats: torch.Tensor) -> torch.Tensor:
        assert adj_mats.dim() == 4 and adj_mats.shape[-1] == adj_mats.shape[-2], print(adj_mats.shape)

        # compute expected cyclicity
        num_mc_graphs = adj_mats.shape[1]
        ec = AcyclicityScore.apply(adj_mats).sum(dim=-1) / num_mc_graphs

        # compute the unnormalized log prior
        prior = torch.distributions.Normal(0., self.cfg.prior_scale)
        log_prior = prior.log_prob(self.particles).sum(dim=(1, 2, 3)) / self.particles[0].numel() - beta * ec
        return log_prior.float()

    def sample_soft_graphs(self, num_samples: int, alpha: float):
        edge_logits = self.edge_logits(alpha)

        transforms = [dist.SigmoidTransform().inv, dist.AffineTransform(loc=0., scale=1.)]
        logistic = torch.distributions.TransformedDistribution(dist.Uniform(0, 1), transforms)
        sample_shape = torch.Size((num_samples, *edge_logits.shape))
        reparam_logits = logistic.rsample(sample_shape) + edge_logits.unsqueeze(0)
        soft_adj_mats = torch.sigmoid(reparam_logits).permute(1, 0, 2, 3)

        # eliminate self loops
        mask = torch.eye(self.num_nodes).repeat(self.cfg.num_particles, num_samples, 1, 1).bool()
        soft_adj_mats = torch.where(mask, torch.tensor(0.), soft_adj_mats)
        return soft_adj_mats

    def sample_graphs(self, num_samples: int, alpha: float, fixed_edges: List[Tuple[int, int]] = None) \
            -> Tuple[List[List[nx.DiGraph]], torch.Tensor]:
        # compute bernoulli probs from latent particles
        with torch.no_grad():
            edge_probs = self.edge_probs(alpha)

        # modify probs
        if fixed_edges is not None:
            for i, j in fixed_edges:
                edge_probs[:, i, j] = 1.

        # sample adjacency matrices and generate graph objects
        adj_mats = torch.bernoulli(edge_probs.unsqueeze(1).expand(-1, num_samples, -1, -1))
        graphs = [[self.adj_mat_to_graph(adj_mats[pidx, sidx]) for sidx in range(num_samples)] for pidx in range(
            self.cfg.num_particles)]
        return graphs, adj_mats

    def dagify_graphs(self, graphs: List[List[nx.DiGraph]], adj_mats: torch.Tensor, alpha: float):
        """Uses a simple heuristic to 'dagify' cyclic graphs in-place. Note: this can be handy for testing and
        debugging during developement, but should not be necessary when the DiBS model is trained properly (as it
        should then almost always return DAGs when sampling).

        Parameters
        ----------
        graphs : List[List[nx.DiGraph]]
            Nested lists of graph objects.
        adj_mats : torch.Tensor
            Tensor of adjacency matrices corresponding to the graph objects in `graphs`.
        """
        edge_probs = self.edge_probs(alpha)
        for particle_idx in range(self.cfg.num_particles):
            num_dagified = 0
            for graph_idx, graph in enumerate(graphs[particle_idx]):
                # check if the graph is cyclic
                if not nx.is_directed_acyclic_graph(graphs[particle_idx][graph_idx]):
                    edges, _ = self.sort_edges(adj_mats[particle_idx, graph_idx], edge_probs[particle_idx])

                    graph = nx.DiGraph()
                    graph.add_nodes_from(self.node_labels)
                    adj_mats[particle_idx, graph_idx] = torch.zeros(self.num_nodes, self.num_nodes)
                    for edge_idx, edge in enumerate(edges):
                        source_node = self.node_id_to_label_dict[edge[0]]
                        sink_node = self.node_id_to_label_dict[edge[1]]
                        if not nx.has_path(graph, sink_node, source_node):
                            # if there is no path from the target to the source node, we can safely add the edge to
                            # the graph without creating a cycle
                            graph.add_edge(source_node, sink_node)
                            adj_mats[particle_idx, graph_idx, edge[0], edge[1]] = 1

                    graphs[particle_idx][graph_idx] = graph
                    num_dagified += 1

            if num_dagified > 0:
                print(f'Dagified {num_dagified} graphs of the {particle_idx + 1}-th particle!')

    def sort_edges(self, adj_mat: torch.Tensor, edge_weights: torch.Tensor, descending=True):
        edges = [(i, j) for i in range(self.num_nodes) for j in range(self.num_nodes) if adj_mat.bool()[i, j]]
        weights = edge_weights[adj_mat.bool()]
        weights, idc = torch.sort(weights, descending=descending)
        edges = [edges[idx] for idx in idc]
        return edges, weights

    def adj_mat_to_graph(self, adj_mat: torch.Tensor) -> nx.DiGraph:
        return adj_mat_to_graph(adj_mat, self.node_labels)

    def graph_to_adj_mat(self, graph: nx.DiGraph) -> torch.Tensor:
        return graph_to_adj_mat(graph, self.node_labels)

    def get_limit_graphs(self):
        # round edge probs to get limit graphs
        edge_probs = self.edge_probs(alpha=1.)
        adj_mats = edge_probs.round().unsqueeze(1)
        graphs = [[self.adj_mat_to_graph(adj_mats[i, 0])] for i in range(self.cfg.num_particles)]
        return graphs, adj_mats

    def particle_similarities(self, bandwidth=1.):
        distances = [(self.particles - self.particles[i:i + 1].detach()) ** 2 for i in range(self.cfg.num_particles)]
        similarities = [(-d.sum(dim=(1, 2, 3)) / bandwidth).exp() for d in distances]
        kernel_mat = torch.stack(similarities, dim=1)
        return kernel_mat

    def param_dict(self) -> Dict[str, Any]:
        params = {'node_id_to_label_dict': self.node_id_to_label_dict,
                  'node_label_to_id_dict': self.node_label_to_id_dict,
                  'particles': self.particles,
                  'cfg_param_dict': self.cfg.param_dict()}
        return params

    def load_param_dict(self, param_dict):
        self.node_id_to_label_dict = param_dict['node_id_to_label_dict']
        self.node_label_to_id_dict = param_dict['node_label_to_id_dict']
        self.node_labels = sorted(list(self.node_label_to_id_dict.keys()))
        self.num_nodes = len(self.node_labels)
        self.particles = param_dict['particles']
        self.cfg = DiBSConfig()
        self.cfg.load_param_dict(param_dict['cfg_param_dict'])


class AcyclicityScore(Function):
    @staticmethod
    def forward(ctx, adj_mat: torch.Tensor, round_edge_weights=False):
        assert adj_mat.dim() >= 3 and adj_mat.shape[-1] == adj_mat.shape[-2], print(
            f'Ill-shaped input: {adj_mat.shape}')
        num_nodes = adj_mat.shape[-1]
        eyes = torch.eye(num_nodes).double().expand_as(adj_mat)
        tmp = eyes + (adj_mat.round().double() if round_edge_weights else adj_mat) / num_nodes

        tmp_pow = tmp.matrix_power(num_nodes - 1)
        ctx.grad = tmp_pow.transpose(-1, -2)
        score = (tmp_pow @ tmp).diagonal(dim1=-2, dim2=-1).sum(dim=-1) - num_nodes
        return score

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        return ctx.grad * grad_output[..., None, None], None
