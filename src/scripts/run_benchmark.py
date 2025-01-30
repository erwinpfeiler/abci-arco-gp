import os
from datetime import datetime

##################################################
# PATHS
PROJECT = '/path/to/bci-arco-gp'
DATA = os.path.join(PROJECT, 'data')
RESULTS = os.path.join(PROJECT, 'results')
LOGS = os.path.join(PROJECT, 'logs')
CONFIGS = os.path.join(PROJECT, 'configs')

##################################################
# !!! ALWAYS CHECK !!!
##################################################
ENV = 'BarabasiAlbert'
DATA_SUBDIRS = [
    '20_nodes_100_train',
]
MODEL = 'abci-arco-gp'
SIM_TOKEN = 'some-identifier'
CONFIG = 'example-config.py'
##################################################

##################################################
# SIMULATION SETUP
CONDA_ENV = 'bci'
NUM_RUNS_PER_ENV = 1


def main():
    for data_subdir in DATA_SUBDIRS:
        # record start time
        start_time = datetime.now().strftime('%Y%m%d_%H%M%S')

        # create slurm logs and output directory
        sim_path = f'{ENV}/{data_subdir}/{start_time}_{SIM_TOKEN}'

        logs_dir = os.path.join(LOGS, sim_path)
        os.makedirs(logs_dir, exist_ok=True)

        output_dir = os.path.join(RESULTS, sim_path)
        os.makedirs(output_dir, exist_ok=True)

        # gather env files
        data_dir = os.path.join(DATA, f'{ENV}/{data_subdir}/')
        env_files = [os.path.abspath(entry) for entry in os.scandir(data_dir) if entry.is_file() and os.path.basename(
            entry)[-4:] == '.pth']

        # create working directory
        working_dir = os.path.join(PROJECT, f'bci-{SIM_TOKEN}-{start_time}')
        os.makedirs(working_dir, exist_ok=True)

        # copy src and config to working dir/output dir
        os.system(f'cp -r {os.path.join(PROJECT, "src")} {working_dir}')
        cfg_src = os.path.join(CONFIGS, CONFIG)
        cfg_dest = os.path.join(working_dir, "src/config.py")
        os.system(f'cp {cfg_src} {cfg_dest}')
        os.system(f'cp {cfg_src} {os.path.join(output_dir, "config.py")}')

        # create sbatch script
        script = os.path.join(working_dir, 'src/scripts/run_single_env.py')
        os.environ["PYTHONPATH"] = working_dir
        for i, env_file in enumerate(env_files * NUM_RUNS_PER_ENV):
            logfile = os.path.join(logs_dir, MODEL + f'-{i}.out')
            os.system(f'python {script} {env_file} {MODEL} {output_dir} >> {logfile}')


# parse arguments when run from shell
if __name__ == "__main__":
    main()
