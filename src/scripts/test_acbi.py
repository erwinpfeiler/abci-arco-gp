import sys
import os

import os
import torch
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
from torch.distributed.rpc import TensorPipeRpcBackendOptions

from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
sys.path.append(os.path.abspath(os.getcwd()))

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from src.abci_arco_gp import ABCIArCOGP as ABCI
from src.config import ABCIArCOGPConfig
from src.environments.experiment import gather_data
from src.environments.generic_environments import *
from src.mechanism_models.mechanisms import get_mechanism_key


if __name__=="__main__":

    # specify the number of nodes 
    num_nodes = 5
    env_cfg = EnvironmentConfig()
    env_cfg.num_observational_train_samples = 30
    env_cfg.num_observational_test_samples = 20
    env_cfg.generate_static_intr_dataset = True
    env_cfg.num_interventional_train_samples = 10
    env_cfg.num_train_interventions = 5
    env_cfg.num_interventional_test_samples = 7
    env_cfg.num_test_interventions = 3
    env_cfg.linear = False
    env_cfg.normalise_data = True

    env_cfg.interventional_queries = None
    env_cfg.num_test_queries = 30 #but not used

    env = BarabasiAlbert(num_nodes, env_cfg)

    for i in range(3):
        continue
        print(f"\n\nSTARTING NEW EXPERIMENT RANDOM\n\n")
        cfg_random = ABCIArCOGPConfig()
        #cfg.policy = 'static-obs-dataset'
        cfg_random.policy = 'random'
        cfg_random.num_experiments = 4 #to get some iterative behaviour
        cfg_random.max_ps_size = 2
        cfg_random.num_workers = 1
        cfg_random.num_arco_steps = 100
        abci_random = ABCI(env, cfg_random)
        abci_random.run()

    for i in range(1):
        print(f"\n\nSTARTING NEW EXPERIMENT ACTIVE LEARNING\n\n")
        cfg = ABCIArCOGPConfig()
        #cfg.policy = 'static-obs-dataset'
        cfg.policy = 'graph-info-gain'
        #cfg.policy = 'random'
        cfg.num_experiments = 2 #to get some iterative behaviour
        cfg.max_ps_size = 2
        cfg.num_workers = 1
        cfg.num_arco_steps = 100
        abci = ABCI(env, cfg)
        abci.run()

"""
Traceback (most recent call last):
  File "/home/erwin/abci-arco-gp/bci-some-identifier-20251125_195806/src/scripts/run_single_env.py", line 205, in <module>
    run_single_env(args['env_file'], args['output_dir'], args['model'], args['num_workers'])
  File "/home/erwin/abci-arco-gp/bci-some-identifier-20251125_195806/src/scripts/run_single_env.py", line 184, in run_single_env
    abci.run()
  File "/home/erwin/abci-arco-gp/bci-some-identifier-20251125_195806/src/abci_arco_gp.py", line 150, in run
    interventions = designer.get_best_experiment(self.env.intervenable_nodes)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/erwin/abci-arco-gp/bci-some-identifier-20251125_195806/src/experimental_design/exp_designer_base.py", line 68, in get_best_experiment
    design = self.design_experiment(target_node)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/erwin/abci-arco-gp/bci-some-identifier-20251125_195806/src/experimental_design/exp_designer_base.py", line 58, in design_experiment
    target_value, score = gp_ucb(lambda x: self.utility({target_node: x}), bounds)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/erwin/abci-arco-gp/bci-some-identifier-20251125_195806/src/experimental_design/optimization.py", line 43, in gp_ucb
    utilities = torch.tensor([utility(candidates[i, 0]).squeeze() for i in range(num_initial_candidates)])
                              ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/erwin/abci-arco-gp/bci-some-identifier-20251125_195806/src/experimental_design/exp_designer_base.py", line 58, in <lambda>
    target_value, score = gp_ucb(lambda x: self.utility({target_node: x}), bounds)
                                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/erwin/abci-arco-gp/bci-some-identifier-20251125_195806/src/experimental_design/exp_designer_abci_arco_gp.py", line 46, in _utility
    return self._graph_info_gain(interventions)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/erwin/abci-arco-gp/bci-some-identifier-20251125_195806/src/experimental_design/exp_designer_abci_arco_gp.py", line 106, in _graph_info_gain
    exp = self._simulate_experiment(interventions, graph)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/erwin/abci-arco-gp/bci-some-identifier-20251125_195806/src/experimental_design/exp_designer_abci_arco_gp.py", line 55, in _simulate_experiment
    return self.mech_model.sample(interventions, self.batch_size, self.num_exp_batches_per_graph, graph=graph)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/erwin/abci-arco-gp/bci-some-identifier-20251125_195806/src/mechanism_models/shared_data_gp_model.py", line 335, in sample
    node_samples = self.gps[node].sample(x, get_mechanism_key(node, parents))
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/erwin/abci-arco-gp/bci-some-identifier-20251125_195806/src/mechanism_models/mechanisms.py", line 617, in sample
    y_dist = self.get_ydist(inputs, key, prior_mode)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/erwin/abci-arco-gp/bci-some-identifier-20251125_195806/src/mechanism_models/mechanisms.py", line 602, in get_ydist
    f_dist = self.get_fdist(inputs, key, prior_mode)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/erwin/abci-arco-gp/bci-some-identifier-20251125_195806/src/mechanism_models/mechanisms.py", line 596, in get_fdist
    f_dist = self.gp(inputs, key=key)
             ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/erwin/miniconda3/envs/abci-arco-gp/lib/python3.12/site-packages/gpytorch/models/exact_gp.py", line 279, in __call__
    raise RuntimeError("You must train on the training inputs!")
RuntimeError: You must train on the training inputs!
"""
