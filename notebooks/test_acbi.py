import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from src.abci_arco_gp import ABCIArCOGP as ABCI
from src.config import ABCIArCOGPConfig
from src.environments.experiment import gather_data
from src.environments.generic_environments import *
from src.mechanism_models.mechanisms import get_mechanism_key


# init environment

# specify the number of nodes 
num_nodes = 5
env_cfg = EnvironmentConfig()
env_cfg.num_observational_train_samples = 30
env_cfg.num_observational_test_samples = 20
env_cfg.generate_static_intr_dataset = True
env_cfg.num_interventional_train_samples = 10
env_cfg.num_train_interventions = 5
env_cfg.num_interventional_test_samples = 0#7
env_cfg.num_test_interventions = 0#3
env_cfg.linear = False
env_cfg.normalise_data = True

env_cfg.interventional_queries = None
env_cfg.num_test_queries = 30 #but not used

env = BarabasiAlbert(num_nodes, env_cfg)

# plot true graph
#nx.draw(env.graph, nx.circular_layout(env.graph), labels=dict(zip(env.graph.nodes, env.graph.nodes)))

cfg = ABCIArCOGPConfig()
#cfg.policy = 'static-obs-dataset'
cfg.policy = 'graph-info-gain'
#cfg.policy = 'random'
cfg.num_experiments = 3 #to get some iterative behaviour
cfg.max_ps_size = 2
cfg.num_workers = 1
cfg.num_arco_steps = 100
abci = ABCI(env, cfg)


abci.run()
