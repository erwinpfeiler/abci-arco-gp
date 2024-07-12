import math
import os
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as sst
import torch


def init_plot_style():
    """Initialize the plot style for pyplot.
    """
    plt.rcParams.update({'figure.figsize': (12, 9)})
    plt.rcParams.update({'lines.linewidth': 5})
    plt.rcParams.update({'lines.markersize': 25})
    plt.rcParams.update({'lines.markeredgewidth': 2})
    plt.rcParams.update({'axes.labelpad': 10})
    plt.rcParams.update({'xtick.major.width': 2.5})
    plt.rcParams.update({'xtick.major.size': 15})
    plt.rcParams.update({'xtick.minor.size': 10})
    plt.rcParams.update({'ytick.major.width': 2.5})
    plt.rcParams.update({'ytick.minor.width': 2.5})
    plt.rcParams.update({'ytick.major.size': 15})
    plt.rcParams.update({'ytick.minor.size': 15})

    # for font settings see also https://stackoverflow.com/questions/2537868/sans-serif-math-with-latex-in-matplotlib
    plt.rcParams.update({'font.size': 50})
    plt.rcParams.update({'font.family': 'sans-serif'})
    plt.rcParams.update({'text.usetex': True})
    plt.rcParams['text.latex.preamble'] = '\n'.join([
        r'\usepackage{amsmath,amssymb,amsfonts,amsthm}',
        r'\usepackage[T1]{fontenc}',
        r'\usepackage{siunitx}',  # i need upright \micro symbols, but you need...
        r'\sisetup{detect-all}',  # ...this to force siunitx to actually use your fonts
        r'\usepackage{helvet}',  # set the normal font here
        r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
        r'\sansmath'  # <- tricky! -- gotta actually tell tex to use!
    ])


def parse_file_name(filename: str):
    tokens = filename.split('-')
    result_type = 'default'
    if tokens[-2] == 'exp':
        run_id = tokens[-3]
        env_id = tokens[-4]
        exp_num = int(tokens[-1][:-4])
    elif tokens[-1] == 'cos.csv':
        run_id = tokens[-3]
        env_id = tokens[-4]
        exp_num = int(tokens[-2])
        result_type = 'cos_variance'
    else:
        run_id = tokens[-1][:-4]
        env_id = tokens[-2]
        exp_num = 1
    return env_id, run_id, exp_num, result_type


class Simulation:
    def __init__(self, results_dir: str, num_experiments: int, file_type: str = '.csv',
                 plot_kwargs: Optional[dict] = None):
        self.results_dir = results_dir if results_dir[-1] == '/' else results_dir + '/'
        self.num_experiments = num_experiments
        if file_type not in {'.pth', '.csv'}:
            raise NotImplementedError
        self.file_type = file_type
        self.plot_kwargs = dict() if plot_kwargs is None else plot_kwargs
        self.stats = None

    def get_result_files(self, result_type: str = 'default'):
        results_dir = self.results_dir
        files = [entry for entry in os.scandir(results_dir) if
                 entry.is_file() and os.path.basename(entry)[-4:] == self.file_type]

        result_files = dict()
        for f in files:
            env_id, run_id, exp_num, res_type = parse_file_name(os.path.basename(f))
            if exp_num != self.num_experiments or res_type != result_type:
                continue

            if env_id in result_files:
                result_files[env_id].append(os.path.abspath(f))
            else:
                result_files[env_id] = [os.path.abspath(f)]

        return result_files

    def load_results(self, stats_names: List[str], result_type: str = 'default'):

        result_files = self.get_result_files(result_type)
        num_environments = len(result_files)
        print(f'Loading results for {num_environments} environments from {self.results_dir}.')

        # collect env-wise results
        aggregated_stats = dict()
        for env_id, env_files in result_files.items():
            # print(f'Got {len(env_files)} runs for environment {env_id}.')

            per_env_stats = dict()
            for file in env_files:
                # load stats from file
                if self.file_type == '.csv':
                    df = pd.read_csv(file)
                    stats = {}
                    for key in df.columns:
                        stats[key] = torch.tensor(df[key])
                else:
                    param_dict = torch.load(file)
                    stats = param_dict['stats']
                    for key in stats:
                        stats[key] = torch.tensor(stats[key])

                # collect stats per env
                for stat_name in stats_names:
                    if stat_name == 'interventional_test_ll':
                        keys = [key for key in stats if key.startswith('interventional_test_ll')]
                        if len(keys):
                            data = torch.stack([stats[key] for key in keys]).sum(dim=0)
                        else:
                            print(f'No results for {stat_name} available!')
                            continue
                    elif stat_name == 'avg_interventional_kld':
                        keys = [key for key in stats if key.startswith('avg_interventional_kld')]
                        if len(keys):
                            data = torch.stack([stats[key] for key in keys]).sum(dim=0)
                        else:
                            # print(f'No results for {stat_name} available!')
                            continue
                    elif stat_name not in stats:
                        # print(f'No results for {stat_name} available!')
                        continue
                    else:
                        data = stats[stat_name]

                    if stat_name in per_env_stats:
                        per_env_stats[stat_name].append(data)
                    else:
                        per_env_stats[stat_name] = [data]

            # aggregate env-wise results
            for stat_name in per_env_stats:
                with torch.no_grad():
                    data = torch.stack(per_env_stats[stat_name], dim=0)

                reduce = lambda x: x
                # reduce = lambda x: x.mean(dim=0, keepdims=True)

                if stat_name in aggregated_stats:
                    aggregated_stats[stat_name].append(reduce(data))
                else:
                    aggregated_stats[stat_name] = [reduce(data)]

        self.stats = {stat_name: torch.cat(stat_list, dim=0) for stat_name, stat_list in aggregated_stats.items()}
        return self.stats

    def plot_simulation_data(self, ax, stat_name: str):
        if self.stats is None:
            print('Nothing to plot...')
            return

        data = self.stats[stat_name]
        num_envs, num_exps = data.shape
        exp_numbers = torch.arange(1, num_exps + 1)

        # compute 95% CIs
        mean = data.mean(dim=0)
        std_err = data.std(unbiased=True, dim=0) / math.sqrt(num_envs) + 1e-8
        lower, upper = sst.t.interval(.95, df=num_envs - 1, loc=mean, scale=std_err)

        ax.plot(exp_numbers, mean.detach(), **self.plot_kwargs)
        ax.fill_between(exp_numbers, upper, lower, alpha=0.2, color=self.plot_kwargs['c'])
