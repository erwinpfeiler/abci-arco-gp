import argparse
import os
import time

import pandas as pd

from src.abci_arco_gp import ABCIArCOGP
from src.abci_categorical_gp import ABCICategoricalGP
from src.abci_dibs_gp import ABCIDiBSGP
from src.abci_fixed_graph_gp import ABCIFixedGraphGP
from src.environments.generic_environments import *
from src.utils.metrics import mmd
from src.utils.utils import export_stats

MODELS = {'abci-categorical-gp': ABCICategoricalGP,
          'abci-dibs-gp': ABCIDiBSGP,
          'abci-arco-gp': ABCIArCOGP,
          'abci-resit-gp': ABCIFixedGraphGP,
          'abci-true-graph-gp': ABCIFixedGraphGP}

EVAL_SCRIPTS = {'aces-eval', 'mmd-eval', 'co-eval', 'structure-stats'}


def aces_eval(abci, output_dir: str):
    num_env_samples = 10000
    num_model_samples = 10
    num_nodes = len(abci.env.node_labels)

    maes = torch.zeros(num_nodes, num_nodes)
    for i, node in enumerate(abci.env.node_labels):
        print(f'Computing MAEs for node {node}')
        interventions = {node: torch.tensor(1.)}
        aces = abci.estimate_aces(interventions, num_samples=num_model_samples)
        env_aces = abci.env.sample_aces(interventions, num_env_samples).mean(dim=1)
        maes[:, i] = (aces - env_aces).abs()

    df = pd.DataFrame(maes)
    df.columns = [f'do({node})' for node in abci.env.node_labels]
    outpath = os.path.join(output_dir,
                           f'stats-{abci.cfg.model_name}-{abci.cfg.policy}-{abci.env.name}'
                           f'-{abci.cfg.run_id}-mae.csv')
    df.to_csv(outpath, index=False)


def mmd_eval(abci, output_dir: str):
    num_env_samples = 10000
    num_nodes = len(abci.env.node_labels)

    # set mc params s.t. we have 10000 samples
    num_samples_per_graph = 10000
    if isinstance(abci, ABCIArCOGP):
        abci.cfg.num_mc_cos = 100
        abci.cfg.num_mc_graphs = 10
        num_samples_per_graph = 10
    elif isinstance(abci, ABCIDiBSGP):
        abci.cfg.num_mc_graphs = 100
        num_samples_per_graph = 10

    mmds = torch.zeros(num_nodes)
    maes = torch.zeros(num_nodes)
    mses = torch.zeros(num_nodes)
    # compute MMDs between the true vs. inferred interventional distribution for single-node interventions
    for nidx, node in enumerate(abci.env.node_labels):
        print(f'Computing metrics for p(X | {node} = 1)')
        interventions = {node: torch.tensor(1.)}
        env_samples = abci.env.sample(interventions, num_env_samples, 1).data
        env_samples = torch.stack([env_samples[node].squeeze() for node in abci.env.node_labels], dim=-1)

        samples, weights = abci.sample(interventions, num_samples_per_graph)
        samples = torch.stack([samples[node].squeeze() for node in abci.env.node_labels], dim=-1)
        mmds[nidx] = mmd(samples, env_samples, weights, unbiased=True)
        print(f'MMD is {mmds[nidx].item()}')

        env_mean = env_samples.mean(dim=0)
        pred_mean = weights @ samples
        mean_error = pred_mean - env_mean
        maes[nidx] = mean_error.abs().sum()
        mses[nidx] = mean_error.pow(2).sum().sqrt()
        print(f'Expected absolute error norm of the distribution mean is {maes[nidx].item()}')
        print(f'Expected squared error norm of the distribution mean is {mses[nidx].item()}')

    for metric, values in zip(['mmd', 'dmae', 'dmse'], [mmds, maes, mses]):
        df = pd.DataFrame(values.view(1, -1))
        df.columns = [f'do({node})' for node in abci.env.node_labels]
        outpath = os.path.join(output_dir,
                               f'stats-{abci.cfg.model_name}-{abci.cfg.policy}-{abci.env.name}'
                               f'-{abci.cfg.run_id}-{metric}.csv')
        df.to_csv(outpath, index=False)


def co_eval(abci, output_dir: str):
    # co weights
    num_draws = 10
    num_mc_cos = 100
    print(f'Computing CO weights from {num_draws} random draws of {num_mc_cos} cos each.')
    co_log_weights = torch.zeros(num_draws, num_mc_cos)
    num_unique_cos = []
    for i in range(num_draws):
        mc_cos, _ = abci.sample_mc_cos(num_cos=num_mc_cos)

        # count number of unique cos
        unique = [cos.__repr__() for cos in mc_cos]
        num_unique_cos.append(len(set(unique)))

        # compute log co weights
        with torch.no_grad():
            weights = abci.co_weights.sum(dim=1)
            weights -= weights.logsumexp(dim=0)
            co_log_weights[i] = weights

    df = pd.DataFrame(co_log_weights)
    df.columns = [f'sweight{i}' for i in range(num_mc_cos)]
    df['num_unique'] = num_unique_cos
    outpath = (output_dir + 'stats-' + abci.cfg.model_name + '-' + abci.cfg.policy + '-' + abci.env.name
               + f'-{abci.cfg.run_id}-coweights.csv')
    df.to_csv(outpath, index=False)

    # compute stats
    num_draws = 10
    num_mc_cos_list = [1, 5, 10, 20, 50, 100]
    for num_mc_cos in num_mc_cos_list:
        abci.stats.clear()
        print(f'Computing stats from {num_draws} random draws of {num_mc_cos} cos each.')
        for i in range(num_draws):
            abci.cfg.num_mc_cos = num_mc_cos
            abci.compute_stats()

        outpath = os.path.join(output_dir,
                               f'stats-{abci.cfg.model_name}-{abci.cfg.policy}-{abci.env.name}'
                               f'-{abci.cfg.run_id}-{num_mc_cos}-cos.csv')
        export_stats(abci.stats, outpath)


def run_evaluation(abci_file: str, output_dir: str, abci_model: str, eval_type: str):
    # run post-processing
    print('\n-------------------------------------------------------------------------------------------')
    print(f'--------- Running evaluation script for {os.path.basename(abci_file)}')
    print(f'--------- Starting time: {time.strftime("%d.%m.%Y %H:%M:%S")}')
    print('-------------------------------------------------------------------------------------------\n')

    if torch.cuda.is_available():
        print('GPUs are available:')
        for i in range(torch.cuda.device_count()):
            print(f'CUDA{i}: {torch.cuda.get_device_name(i)}')
            print()

    # load trained abci model
    assert abci_model in MODELS, print(f'Invalid ABCI model {abci_model}!')
    print(f'Loading model ...', end='')
    abci = MODELS[abci_model].load(abci_file)
    print(' complete!', flush=True)

    if eval_type == 'aces-eval':
        aces_eval(abci, output_dir)
    elif eval_type == 'mmd-eval':
        mmd_eval(abci, output_dir)
    elif eval_type == 'co-eval':
        co_eval(abci, output_dir)
    elif eval_type == 'structure-stats':
        abci.stats.clear()
        abci.compute_stats()

        num_experiments_conducted = len(abci.experiments)
        export_dict = {k: v for k, v in abci.stats.items() if k.count('loss') == 0}
        outpath = os.path.join(output_dir,
                               f'stats-{abci.cfg.model_name}-{abci.cfg.policy}-{abci.env.name}'
                               f'-{abci.cfg.run_id}-exp-{num_experiments_conducted}.csv')
        export_stats(export_dict, outpath)
    else:
        assert False, print(eval_type)

    print('\n-------------------------------------------------------------------------------------------')
    print(f'--------- Finish time: {time.strftime("%d.%m.%Y %H:%M:%S")}')
    print('-------------------------------------------------------------------------------------------\n')


# parse arguments when run from shell
if __name__ == "__main__":
    parser = argparse.ArgumentParser('Usage:')
    parser.add_argument('abci_file', type=str, help=f'Path to environment file.')
    parser.add_argument('model', type=str, choices=MODELS, help=f'Available models: {MODELS}')
    parser.add_argument('output_dir', type=str, help='Output directory.')
    parser.add_argument('eval_type', type=str, choices=EVAL_SCRIPTS, help=f'Available models: {EVAL_SCRIPTS}')

    args = vars(parser.parse_args())
    run_evaluation(args['abci_file'], args['output_dir'], args['model'], args['eval_type'])
