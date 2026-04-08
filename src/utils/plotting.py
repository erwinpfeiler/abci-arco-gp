import math
import os
from collections import Counter
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as sst
import torch


# Stats that are expected to track outer experiment progress and therefore
# should usually have one entry per experiment/checkpoint.
_PROGRESS_STAT_PRIORITY = (
    'eshd',
    'aaid',
    'paid',
    'order_aid',
    'oset_aid',
    'enum_edges',
    'precision',
    'recall',
    'f1',
    'tpr',
    'fpr',
)


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


def _infer_progress_length_from_stats(stats: dict):
    """Infer the outer experiment progress length from a checkpoint stats dict.

    Many checkpoint files contain both per-experiment statistics (length should
    match the experiment number in the filename) and inner-optimization traces
    such as `arco_loss` that can be much longer. To avoid false positives, we
    first look for well-known per-experiment statistics; if none are present, we
    fall back to the most common 1D length across stats entries.
    """
    for key in _PROGRESS_STAT_PRIORITY:
        if key in stats:
            try:
                return len(stats[key]), key
            except TypeError:
                pass

    length_counter = Counter()
    example_key_for_length = {}
    for key, value in stats.items():
        try:
            length = len(value)
        except TypeError:
            continue
        length_counter[length] += 1
        example_key_for_length.setdefault(length, key)

    if not length_counter:
        return None, None

    inferred_length, _ = length_counter.most_common(1)[0]
    return inferred_length, example_key_for_length[inferred_length]


def _validate_file_length_matches_experiment(file_path: str, exp_num: int):
    """Check that a result/checkpoint file is internally consistent with its filename.

    For CSV files, this checks that the file has exactly `exp_num` rows.
    For PTH files, this checks that the inferred per-experiment stats length
    matches `exp_num`.

    Returns:
        Optional[str]: Warning message if a mismatch is found, else None.
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == '.csv':
        try:
            num_rows = len(pd.read_csv(file_path))
        except Exception as exc:
            return f'Could not read CSV {file_path} for length check: {exc}'

        if num_rows != exp_num:
            return (f'Length mismatch for CSV {os.path.basename(file_path)}: '
                    f'filename says exp={exp_num}, but file has {num_rows} rows.')
        return None

    if ext == '.pth':
        try:
            param_dict = torch.load(file_path, map_location='cpu')
        except Exception as exc:
            return f'Could not read checkpoint {file_path} for length check: {exc}'

        if not isinstance(param_dict, dict) or 'stats' not in param_dict:
            return f'Checkpoint {os.path.basename(file_path)} does not contain a stats dict.'

        inferred_length, basis_key = _infer_progress_length_from_stats(param_dict['stats'])
        if inferred_length is None:
            return f'Could not infer experiment length from checkpoint {os.path.basename(file_path)}.'

        if inferred_length != exp_num:
            return (f'Length mismatch for checkpoint {os.path.basename(file_path)}: '
                    f'filename says exp={exp_num}, but inferred stats length is {inferred_length} '
                    f'(based on key {basis_key!r}).')
        return None

    return None


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
        files = [entry.path for entry in os.scandir(results_dir)
                 if entry.is_file() and os.path.basename(entry.path)[-4:] == self.file_type]

        result_files = dict()
        available_exp_nums = set()
        available_result_types = set()
        validation_messages = []

        for file_path in files:
            file_name = os.path.basename(file_path)
            try:
                env_id, run_id, exp_num, res_type = parse_file_name(file_name)
            except Exception as exc:
                print(f'Could not parse file name {file_name}: {exc}')
                continue

            if res_type == result_type:
                available_exp_nums.add(exp_num)
            else:
                available_result_types.add(res_type)
                continue

            if exp_num != self.num_experiments:
                # Intermediate checkpoint files are expected when one file is
                # written per experiment. Silently ignore them as long as the
                # requested experiment exists.
                continue

            validation_message = _validate_file_length_matches_experiment(file_path, exp_num)
            if validation_message is not None:
                validation_messages.append(validation_message)

            # If we are plotting from CSVs, also validate the matching checkpoint
            # file, if it exists, because the user asked for an explicit check of
            # the corresponding `-n.pth` file.
            if self.file_type == '.csv':
                checkpoint_path = os.path.splitext(file_path)[0] + '.pth'
                if os.path.exists(checkpoint_path):
                    validation_message = _validate_file_length_matches_experiment(checkpoint_path, exp_num)
                    if validation_message is not None:
                        validation_messages.append(validation_message)

            if env_id in result_files:
                result_files[env_id].append(os.path.abspath(file_path))
            else:
                result_files[env_id] = [os.path.abspath(file_path)]

        if not result_files:
            available_exp_nums_str = sorted(available_exp_nums)
            print(f'No matching {self.file_type} files found for exp={self.num_experiments} '
                  f'and result_type={result_type} in {self.results_dir}')
            if available_exp_nums_str:
                print(f'Available experiment numbers for result_type={result_type}: {available_exp_nums_str}')
            if available_result_types:
                print(f'Available other result types: {sorted(available_result_types)}')

        for message in validation_messages:
            print(message)

        return result_files

    def load_results(self, result_type: str = 'default'):

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
                for stat_name, data in stats.items():
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