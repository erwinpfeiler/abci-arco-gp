import os
import time
from datetime import datetime

PARTITIONS = 'normal'
EXCLUDE = 'aim-gpu[1,3-4]'
TIME = '1-23:59'

##################################################
# PATHS
ROOT = '/ceph/home/TUG/ctoth-tug'
PROJECT = os.path.join(ROOT, 'bci')
DATA = os.path.join(ROOT, 'data')
RESULTS = os.path.join(ROOT, 'results')
LOGS = os.path.join(ROOT, 'slurm-logs')
CONFIGS = os.path.join(PROJECT, 'configs')

##################################################
# !!! ALWAYS CHECK !!!
##################################################
ENV = 'BarabasiAlbert'#'ErdosRenyi'
DATA_SUBDIRS = [
    # '20_nodes_200_train',
    # '20_nodes_200_train',
    # '20_nodes_500_train',
    # '20_nodes_1000_train',
    # '50_nodes_500_train',
    # '20_nodes_100_train_linear',
    # '20_nodes_200_train_linear',
    # '20_nodes_500_train_linear',
    # '20_nodes_1000_train_linear',
    '10_nodes_50_train'
]
MODEL = 'abci-arco-gp-random' # 'abci-arco-gp-graph-info'
SIM_TOKEN = 'test-arcogp1'
CONFIG = 'example-config.py'
##################################################

##################################################
# SIMULATION SETUP
CONDA_ENV = 'clufs-bci'
NUM_GPUS = 0
MEM = 30  # in GB
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
        num_environments = len(env_files)

        # create working directory
        working_dir = os.path.join(ROOT, f'bci-{SIM_TOKEN}-{start_time}')
        os.makedirs(working_dir, exist_ok=True)

        # copy src and config to working dir/output dir
        os.system(f'cp -r {os.path.join(PROJECT, "src")} {working_dir}')
        cfg_src = os.path.join(CONFIGS, CONFIG)
        cfg_dest = os.path.join(working_dir, "src/config.py")
        os.system(f'cp {cfg_src} {cfg_dest}')
        os.system(f'cp {cfg_src} {os.path.join(output_dir, "config.py")}')

        # create sbatch script
        script = os.path.join(working_dir, 'src/scripts/run_single_env.py')
        job_path = os.path.join(working_dir, f'{SIM_TOKEN}-slurm.sh')
        with open(job_path, 'w') as jobfile:
            # write sbatch parameters
            jobfile.writelines(['#!/bin/bash\n',
                                f'#SBATCH --job-name="{ENV}-{data_subdir}-{SIM_TOKEN}"\n'
                                f'#SBATCH --output={os.path.join(logs_dir, MODEL + "-%A-%a.out")}\n',
                                '#SBATCH --open-mode=append\n',
                                '#SBATCH --ntasks=1\n'
                                '#SBATCH --cpus-per-task=1\n',
                                '#SBATCH --mail-type=END\n'
                                '#SBATCH --mail-user=christian.toth@tugraz.at\n'
                                f'#SBATCH --time={TIME}\n',
                                f'#SBATCH --partition={PARTITIONS}\n',
                                f'#SBATCH --exclude={EXCLUDE}\n',
                                f'#SBATCH --array=0-{num_environments * NUM_RUNS_PER_ENV - 1}\n',
                                f'#SBATCH --mem={MEM}G\n\n'])

            # write python init
            jobfile.writelines(["error=0; trap 'error=$(($?>$error?$?:$error))' ERR\n",
                                f'export PYTHONPATH="{working_dir}"\n',
                                f'eval "$(conda shell.bash hook)"\n'
                                f'conda activate {CONDA_ENV}\n'
                                'which python\n',
                                'echo "Job is running on ${HOSTNAME}"\n',
                                '\n'])

            # write abci script command
            jobfile.write('env_files=("' + '" "'.join(map(str, env_files * NUM_RUNS_PER_ENV)) + '")\n')
            jobfile.write(f'python {script} ' + '${env_files[$SLURM_ARRAY_TASK_ID]} ' + f'{MODEL} {output_dir} ')
            jobfile.write('\n\nexit $error\n')

        # start sbatch job
        os.system('sbatch ' + job_path)
        # os.system('rm ' + job_path)

        # delay for 1s to avoid overwriting working dirs
        time.sleep(2)


# parse arguments when run from shell
if __name__ == "__main__":
    main()
