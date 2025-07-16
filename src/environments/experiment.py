from typing import List, Dict, Any, Union, Optional

import networkx as nx
import pandas as pd
import torch
from torch.distributions import Distribution

from src.utils.graphs import get_parents


class Experiment:
    def __init__(self, interventions: dict, data: Dict[str, torch.tensor]):
        num_batches, batch_size = list(data.values())[0].shape[0:2]
        assert all([node_data.shape == (num_batches, batch_size, 1) for node_data in data.values()])
        self.interventions = interventions
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.data = data

    def normalise(self, means: Dict[str, torch.Tensor], stds: Dict[str, torch.Tensor]):
        assert len(means) == len(self.data), print(means)
        assert len(stds) == len(self.data), print(stds)

        # normalise data
        for node in self.data:
            self.data[node] = (self.data[node] - means[node]) / stds[node]

        # normalise intervention values
        for node, value in self.interventions.items():
            self.interventions[node] = (value - means[node]) / stds[node]

    def unnormalise(self, means: Dict[str, torch.Tensor], stds: Dict[str, torch.Tensor]):
        assert len(means) == len(self.data), print(means)
        assert len(stds) == len(self.data), print(stds)

        new_data = {}
        # unnormalise data
        for node in self.data:
            new_data[node] = self.data[node] * stds[node] + means[node]

        # normalise intervention values
        new_interventions = {}
        for node, value in self.interventions.items():
            new_interventions[node] = value * stds[node] + means[node]

        return Experiment(new_interventions, new_data)

    def cuda(self):
        for key in self.data:
            self.data[key] = self.data[key].cuda()

    def cpu(self):
        for key in self.data:
            self.data[key] = self.data[key].cpu()

    def param_dict(self) -> Dict[str, Any]:
        params = {'interventions': self.interventions,
                  'data': self.data}
        return params

    def to_pandas_df(self, node_order: List[str], add_interventions: bool = False) -> pd.DataFrame:
        """
            Casts a (sub)set of the data as pandas dataframe.

            Parameters
            ----------
            node_order: List[str]
                List of node labels for which to include the data in the data frame in the given order.
            add_interventions: bool
                If true, add intervention values of intervened nodes and `n/i` for non-intervened nodes as the first
                line (after the header) to the dataframe.

            Returns
            -------
            pd.DataFrame
                The data as pandas DataFrame.
        """
        assert set(node_order).issubset(set(self.data.keys()))

        data = torch.cat([self.data[node] for node in node_order], dim=-1).view(-1, len(node_order))
        df = pd.DataFrame(data, columns=node_order)

        if add_interventions:
            tmp = []
            for node in node_order:
                tmp.append(self.interventions[node].item() if node in self.interventions else 'n/i')
            intr_df = pd.DataFrame([tmp], columns=node_order)

            df = pd.concat([intr_df, df], ignore_index=True)

        return df

    @classmethod
    def from_pandas_df(cls, df: pd.DataFrame, includes_interventions: bool = False):
        """
            Creates an Experiment object containing the data given in the given data frame.

            Parameters
            ----------
            df: pd.DataFrame
                The dataframe containing the data.
            includes_interventions: bool
                If true, extracts intervention targets and values from the first line (after the header) of the
                dataframe. Non-intervened nodes must have `n/i` as an entry.

            Returns
            -------
            Experiment
                The experiment object containing the data.
        """

        node_labels = sorted(df.columns)
        data_df = df
        interventions = {}

        if includes_interventions:
            intr_df = df.loc[[0]]
            interventions = {target: intr_df[target][0] for target in node_labels if intr_df[target][0] != 'n/i'}
            data_df = df.drop([0])

        data = {}
        for node in node_labels:
            data[node] = torch.tensor(data_df[node].to_numpy('float32')).view(1, -1, 1)

        return Experiment(interventions, data)

    @classmethod
    def load_param_dict(cls, param_dict: Union[None, dict, List[dict]]):
        experiments = None
        if isinstance(param_dict, dict):
            experiments = Experiment(param_dict['interventions'], param_dict['data'])
        elif isinstance(param_dict, list):
            experiments = [Experiment(d['interventions'], d['data']) for d in param_dict]

        return experiments


def get_exp_param_dicts(experiments: Optional[List[Experiment]]):
    return [exp.param_dict() for exp in experiments] if experiments else None


def gather_data(experiments: List[Experiment], node: str, graph: nx.DiGraph = None, parents: List[str] = None,
                mode: str = 'joint'):
    assert graph is not None or parents is not None
    assert mode in {'joint', 'independent_batches', 'independent_samples'}, print('Invalid gather mode: ', mode)

    # gather targets
    if mode == 'independent_batches':
        batch_size = experiments[0].batch_size
        assert all([experiment.batch_size == batch_size for experiment in experiments]), print('Batch size mismatch!')
        targets = [exp.data[node].squeeze(-1) for exp in experiments if node not in exp.interventions]
    elif mode == 'independent_samples':
        targets = [exp.data[node].view(-1, 1) for exp in experiments if node not in exp.interventions]
    else:  # mode == 'joint'
        targets = [exp.data[node].reshape(-1) for exp in experiments if node not in exp.interventions]

    # check if we have data for this node
    if not targets:
        return None, None
    targets = torch.cat(targets, dim=0)

    # check if we have parents
    parents = sorted(parents) if graph is None else get_parents(node, graph)
    if not parents:
        return None, targets

    # gather parent data
    num_parents = len(parents)
    if mode == 'independent_batches':
        inputs = [torch.cat([experiment.data[parent] for parent in parents], dim=-1) for experiment in experiments if
                  node not in experiment.interventions]
    elif mode == 'independent_samples':
        inputs = [torch.cat([experiment.data[parent] for parent in parents], dim=-1).view(-1, 1, num_parents) for
                  experiment in experiments if node not in experiment.interventions]
    else:  # mode == 'joint'
        inputs = [torch.cat([experiment.data[parent] for parent in parents], dim=-1).view(-1, num_parents) for
                  experiment in experiments if node not in experiment.interventions]

    inputs = torch.cat(inputs, dim=0)

    return inputs, targets


class InterventionalDistributionsQuery:
    def __init__(self, query_nodes: List[str], intervention_targets: Dict[str, Distribution]):
        self.query_nodes = query_nodes
        self.intervention_targets = intervention_targets
        self.sample_queries = None

    def sample_intervention(self):
        return {target: distr.sample() for target, distr in self.intervention_targets.items()}

    def set_sample_queries(self, sample_queries: List[Experiment]):
        self.sample_queries = sample_queries

    def clone(self):
        return InterventionalDistributionsQuery(self.query_nodes, self.intervention_targets)

    def param_dict(self) -> Dict[str, Any]:
        params = {'query_nodes': self.query_nodes,
                  'intervention_targets': self.intervention_targets,
                  'sample_queries': self.sample_queries}
        return params

    @classmethod
    def load_param_dict(cls, param_dict):
        query = InterventionalDistributionsQuery(param_dict['query_nodes'], param_dict['intervention_targets'])
        query.set_sample_queries(param_dict['sample_queries'])
        return query
