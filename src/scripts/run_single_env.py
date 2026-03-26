import argparse
import os
import random
import socketserver
import string
import time

import torch.distributed.rpc as rpc
import torch.multiprocessing as mp

import sys
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
sys.path.append(os.path.abspath(os.getcwd()))
sys.path.append('/home/erwin/abci-arco-gp/')

from src.abci_arco_gp import ABCIArCOGP
from src.abci_categorical_gp import ABCICategoricalGP
from src.abci_dibs_gp import ABCIDiBSGP
from src.abci_fixed_graph_gp import ABCIFixedGraphGP
from src.config import ABCICategoricalGPConfig, ABCIDiBSGPConfig, ABCIArCOGPConfig, ABCIFixedGraphGPConfig
from src.environments.generic_environments import *
from src.utils.baselines import Baseline
from src.utils.graphs import adj_mat_to_graph, get_graph_key

MODELS = {'abci-categorical-gp', 'abci-dibs-gp', 'abci-arco-gp',
          'anm', 'ges', 'daggnn', 'gadget', 'gae', 'golem', 'grandag', 'grasp', 'pc', 'resit', 'beeps',
          'abci-resit-gp', 'abci-true-graph-gp',
          'abci-arco-gp-random', 'abci-arco-gp-graph-info', 'abci-arco-gp-random-fixed-value'}


def spawn_model(model: str, env: Environment, num_workers: int, output_dir: str, run_id: str):
    if model == 'abci-categorical-gp':
        cfg = ABCICategoricalGPConfig()
        cfg.num_workers = num_workers
        cfg.output_dir = output_dir
        cfg.run_id = run_id
        return ABCICategoricalGP(env, cfg)

    elif model == 'abci-dibs-gp':
        cfg = ABCIDiBSGPConfig()
        cfg.num_workers = num_workers
        cfg.output_dir = output_dir
        cfg.run_id = run_id
        return ABCIDiBSGP(env, cfg)

    elif model == 'abci-arco-gp':
        cfg = ABCIArCOGPConfig()
        cfg.num_workers = num_workers
        cfg.output_dir = output_dir
        cfg.run_id = run_id
        return ABCIArCOGP(env, cfg)
    
    elif model == 'abci-arco-gp-random':
        cfg = ABCIArCOGPConfig()
        cfg.num_workers = num_workers
        cfg.output_dir = output_dir
        cfg.run_id = run_id
        cfg.policy = "random"
        return ABCIArCOGP(env, cfg)
    
    elif model == 'abci-arco-gp-random-fixed-value':
        cfg = ABCIArCOGPConfig()
        cfg.num_workers = num_workers
        cfg.output_dir = output_dir
        cfg.run_id = run_id
        cfg.policy = 'random-fixed-value'
        return ABCIArCOGP(env, cfg)
    
    elif model == 'abci-arco-gp-graph-info':
        cfg = ABCIArCOGPConfig()
        cfg.num_workers = num_workers
        cfg.output_dir = output_dir
        cfg.run_id = run_id
        cfg.policy = 'graph-info-gain'
        return ABCIArCOGP(env, cfg)

    elif model == 'abci-resit-gp':
        cfg = ABCIFixedGraphGPConfig()
        cfg.num_workers = num_workers
        cfg.output_dir = output_dir
        cfg.run_id = run_id
        cfg.model_name = model

        print(f'Running RESIT to find graph for ABCI-RESIT-GP...')
        resit = Baseline(env, 'resit', policy=cfg.policy)
        resit.run(compute_stats=False)
        graph = adj_mat_to_graph(resit.graphs.squeeze(), env.node_labels)
        print(f'RESIT graph is {get_graph_key(graph)}')

        abci = ABCIFixedGraphGP(env, cfg)
        abci.set_graph(graph)
        return abci

    elif model == 'abci-true-graph-gp':
        cfg = ABCIFixedGraphGPConfig()
        cfg.num_workers = num_workers
        cfg.output_dir = output_dir
        cfg.run_id = run_id
        cfg.model_name = model

        abci = ABCIFixedGraphGP(env, cfg)
        abci.set_graph(env.graph)
        return abci

    else:
        if num_workers != 1:
            raise NotImplementedError

        return Baseline(env, model, output_dir, run_id)


def get_free_port():
    with socketserver.TCPServer(("localhost", 0), None) as s:
        free_port = s.server_address[1]
    return free_port


def generate_run_id():
    random.seed(random.SystemRandom().random())
    return ''.join(random.choices(string.ascii_letters + string.digits, k=6))


def run_worker(rank: int, env: Environment, master_port: str, num_workers: int, output_dir: str, run_id: str,
               abci_model: str):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = master_port
    torch.set_num_interop_threads(1)
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)  # or float64 if you prefer

    if rank == 0:
        opts = rpc.TensorPipeRpcBackendOptions(
            num_worker_threads=32,
            rpc_timeout=5*3600.0,          # seconds
            _transports=["uv"],          # <- key fix: disable ibv/shm selection
            _channels=["basic"],         # stable CPU channel
            init_method="env://",
        )
        rpc.init_rpc('Experimenter',
                     rank=rank,
                     world_size=num_workers,
                     rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                        num_worker_threads=num_workers,
                        rpc_timeout=5*3600,
                        _transports=["uv"],          # <- key fix: disable ibv/shm selection
                        _channels=["basic"],         # stable CPU channel
                        init_method="env://",)
                    )
        try: #TODO: return to try except block once all bugs are fixed
            abci = spawn_model(abci_model, env, num_workers, output_dir, run_id)
            abci.run()
        except Exception as e:
            print(f"EXCEPTION: ")
            print(e)
    else:
        rpc.init_rpc(f'ExperimentDesigner{rank}',
                     rank=rank,
                     world_size=num_workers,
                     rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                        num_worker_threads=num_workers,
                        rpc_timeout=5*3600,
                        _transports=["uv"],          # <- key fix: disable ibv/shm selection
                        _channels=["basic"],         # stable CPU channel
                        init_method="env://",)
                    )

    rpc.shutdown()


def run_single_env(env_file: str, output_dir: str, model: str, num_workers: int):
    torch.set_default_dtype(torch.float32)
    # load env
    env = Environment.load(env_file)

    assert model in MODELS, print(f'Invalid model {model}!')
    run_id = generate_run_id()

    # run benchmark env
    print('\n-------------------------------------------------------------------------------------------')
    print(f'--------- Running {model.upper()} on Environment {env.name}')
    print(f'--------- Job ID: {run_id}')
    print(f'--------- Starting time: {time.strftime("%d.%m.%Y %H:%M:%S")}')
    print('-------------------------------------------------------------------------------------------\n', flush=True)

    if torch.cuda.is_available():
        print('GPUs are available:')
        for i in range(torch.cuda.device_count()):
            print(f'CUDA{i}: {torch.cuda.get_device_name(i)}')
            print()

    if num_workers > 1:
        print(torch.__config__.parallel_info())
        mp.set_sharing_strategy('file_system')

        master_port = str(get_free_port())
        print(f'Starting {num_workers} workers on port ' + master_port)
        try:
            mp.spawn(
                run_worker,
                args=(env, master_port, num_workers, output_dir, run_id, model),
                nprocs=num_workers,
                join=True
            )
        except Exception as e:
            print(e)
    else:
        try:
            abci = spawn_model(model, env, 1, output_dir, run_id)
            abci.run()
        except Exception:
            import traceback
            traceback.print_exc()
            raise

    print('\n-------------------------------------------------------------------------------------------')
    print(f'--------- Finish time: {time.strftime("%d.%m.%Y %H:%M:%S")}')
    print('-------------------------------------------------------------------------------------------\n')


# parse arguments when run from shell
if __name__ == "__main__":
    print(f"Staring run_single_env")
    parser = argparse.ArgumentParser('Usage on single environment:')
    parser.add_argument('env_file', type=str, help=f'Path to environment file.')
    parser.add_argument('model', type=str, choices=MODELS, help=f'Available models: {MODELS}')
    parser.add_argument('output_dir', type=str, help='Output directory.')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of worker threads per environment.')

    args = vars(parser.parse_args())
    run_single_env(args['env_file'], args['output_dir'], args['model'], args['num_workers'])
    print(f"Done with run_singe_env")
