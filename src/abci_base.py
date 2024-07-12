import os
import random
import time
from typing import Dict, Any, List

import torch
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef, remote

from src.config import ABCIBaseConfig
from src.environments.environment import Environment
from src.environments.experiment import Experiment, get_exp_param_dicts
from src.experimental_design.exp_designer_base import Design
from src.utils.utils import export_stats


class ABCIBase:
    cfg: ABCIBaseConfig

    def __init__(self, num_workers: int = 1, env: Environment = None):
        # load config
        self.env = env
        self.experiments: List[Experiment] = []
        self.stats: Dict[str, list] = dict()  # dict for training or evaluation stats, logs, etc..

        # init distributed experiment design
        self.designed_experiments = {}
        self.open_targets = set()
        if num_workers > 1:
            self.worker_id = rpc.get_worker_info().id
            if self.worker_id == 0:
                self.experimenter_rref = RRef(self)
                self.designer_rrefs = []
                for worker_id in range(1, num_workers):
                    info = rpc.get_worker_info(f'ExperimentDesigner{worker_id}')
                    self.designer_rrefs.append(remote(info, self.experiment_designer_factory))

    def experiment_designer_factory(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def get_random_intervention(self, fixed_value: float = None):
        target_node = random.choice(list(self.env.intervenable_nodes) + ['OBSERVATIONAL'])
        if target_node == 'OBSERVATIONAL':
            return {}
        if fixed_value is None:
            bounds = self.env.intervention_bounds[target_node]
            target_value = torch.rand(1) * (bounds[1] - bounds[0]) + bounds[0]
        else:
            target_value = torch.tensor(fixed_value)
        return {target_node: target_value}

    def report_design(self, worker_id: int, design_key: str, design: Design):
        print(f'Worker {worker_id} designed {design.interventions} with info gain {design.info_gain}', flush=True)
        self.designed_experiments[design_key] = design

    def report_status(self, worker_id: int, message: str):
        print(f'Worker {worker_id} reports at {time.strftime("%H:%M:%S")}: {message}', flush=True)

    def get_target(self, worker_id: int):
        target = self.open_targets.pop() if self.open_targets else None
        print(f'Worker {worker_id} asks for new target at {time.strftime("%H:%M:%S")}. Assigning new target {target}.',
              flush=True)
        return target

    def design_experiment_distributed(self, args):
        self.open_targets = self.env.intervenable_nodes | {'OBSERVATIONAL'}
        self.designed_experiments.clear()

        # start workers
        futs = []
        for designer_rref in self.designer_rrefs:
            futs.append(designer_rref.rpc_async().run_distributed(self.experimenter_rref, args))

        # init and run designer of master process
        designer = self.experiment_designer_factory()
        designer.init_design_process(args)

        target_node = self.get_target(self.worker_id)
        while target_node:
            design = designer.design_experiment(target_node)
            self.report_design(self.worker_id, target_node, design)
            target_node = self.get_target(self.worker_id)

        # wait until all workers have finished
        for fut in futs:
            fut.wait()

        # pick most promising experiment
        print('Experiment design process has finished:')
        best_intervention = {}
        best_info_gain = self.designed_experiments['OBSERVATIONAL'].info_gain
        for _, design in self.designed_experiments.items():
            print(f'Interventions {design.interventions} expect info gain {design.info_gain}', flush=True)
            if design.info_gain > best_info_gain:
                best_info_gain = design.info_gain
                best_intervention = design.interventions

        return best_intervention

    def record_stat(self, stat_name: str, value: torch.Tensor):
        if stat_name not in self.stats:
            self.stats[stat_name] = []

        self.stats[stat_name].append(value)

    def export_stats(self, output_dir: str = None):
        output_dir = self.cfg.output_dir if output_dir is None else output_dir
        if output_dir is not None:
            num_experiments_conducted = len(self.experiments)
            export_dict = {k: v for k, v in self.stats.items() if k.count('loss') == 0}
            outpath = os.path.join(self.cfg.output_dir,
                                   f'stats-{self.cfg.model_name}-{self.cfg.policy}-{self.env.name}'
                                   f'-{self.cfg.run_id}-exp-{num_experiments_conducted}.csv')
            export_stats(export_dict, outpath)

    def save_model(self, output_dir: str = None):
        output_dir = self.cfg.output_dir if output_dir is None else output_dir
        if output_dir is not None:
            num_experiments_conducted = len(self.experiments)
            outpath = os.path.join(self.cfg.output_dir,
                                   f'{self.cfg.model_name}-{self.cfg.policy}-{self.env.name}'
                                   f'-{self.cfg.run_id}-exp-{num_experiments_conducted}.pth')

            torch.save(self.param_dict(), outpath)

    def param_dict(self) -> Dict[str, Any]:
        env_param_dict = self.env.param_dict()
        params = {'env_param_dict': env_param_dict,
                  'experiments': get_exp_param_dicts(self.experiments),
                  'stats': self.stats}
        return params

    def load_param_dict(self, param_dict):
        self.env = Environment(param_dict=param_dict['env_param_dict'])
        self.experiments = Experiment.load_param_dict(param_dict['experiments'])
        self.stats = param_dict['stats']
