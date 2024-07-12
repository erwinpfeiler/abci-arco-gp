from typing import Tuple

import networkx as nx
import pandas as pd
from sklearn.model_selection import train_test_split

from src.environments.environment import Environment


class Sachs(Environment):
    def __init__(self, split: Tuple[int, int] = None, normalise=True, data_file: str = '../data/sachs.csv',
                 seed: int = None):
        self.split = split
        self.seed = seed
        super().__init__(num_nodes=11, cfg=None, init_mechanisms=False)

        # load/create data sets
        df = pd.read_csv(data_file)
        label_map = {'praf': 'Raf',
                     'pmek': 'Mek',
                     'plcg': 'Plcg',
                     'PIP2': 'PIP2',
                     'PIP3': 'PIP3',
                     'p44/42': 'Erk',
                     'pakts473': 'Akt',
                     'PKA': 'PKA',
                     'PKC': 'PKC',
                     'P38': 'P38',
                     'pjnk': 'Jnk'}
        df.columns = [label_map[df_label] for df_label in df.columns]

        if self.split is None:
            df_train = df
            df_test = None
        else:
            df_train, df_test = train_test_split(df, train_size=self.split[0], test_size=self.split[1],
                                                 random_state=self.seed, shuffle=True)

        env = Environment.load_static_dataset(self.graph, df_train, df_test, normalise=normalise)
        self.cfg.normalise_data = normalise
        self.normalisation_means = env.normalisation_means
        self.normalisation_stds = env.normalisation_stds
        self.observational_train_data = env.observational_train_data
        self.observational_test_data = env.observational_test_data

    def construct_graph(self, num_nodes: int) -> nx.DiGraph:
        graph = nx.DiGraph()
        node_labels = ['PKC', 'PKA', 'Jnk', 'P38', 'Raf', 'Mek', 'Erk', 'Akt', 'Plcg', 'PIP3', 'PIP2']
        graph.add_nodes_from(node_labels)

        graph.add_edge('PKC', 'PKA')
        graph.add_edge('PKC', 'Jnk')
        graph.add_edge('PKC', 'P38')
        graph.add_edge('PKC', 'Raf')
        graph.add_edge('PKC', 'Mek')
        graph.add_edge('PKA', 'Jnk')
        graph.add_edge('PKA', 'P38')
        graph.add_edge('PKA', 'Akt')
        graph.add_edge('PKA', 'Erk')
        graph.add_edge('PKA', 'Mek')
        graph.add_edge('PKA', 'Raf')
        graph.add_edge('Raf', 'Mek')
        graph.add_edge('Mek', 'Erk')
        graph.add_edge('Erk', 'Akt')
        graph.add_edge('Plcg', 'PIP3')
        graph.add_edge('Plcg', 'PIP2')
        graph.add_edge('PIP3', 'PIP2')

        return graph
