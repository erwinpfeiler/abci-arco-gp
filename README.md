# Effective Bayesian Causal Inference with ArCO-GP

This repository contains the implementation of the ARCO-GP framework for Bayesian Causal Inference (BCI) over (non-)linear additive Gaussian noise models.
In a nutshell, we infer posterior causal quantities by marginalising over posterior Structural Causal Models (SCMs).
We do so by infering causal orders with ArCO, exactly marginalising over limited-size parent sets and marginalising over mechanisms using Gaussian Processes (GPs). 
The framework and model are described in detail in our paper [Effective Bayesian Causal Inference via Structural Marginalisation and Autoregressive Orders](https://arxiv.org/abs/2402.14781).
We provide example notebooks to illustrate the basic usage of the code base and get you started quickly.
Feel free to reach out if you have questions about the paper or code!


## Getting Started

#### Python Environment Setup

These instructions should help you set up a suitable Python environment. We recommend to use [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for easily recreating the Python environment. You can install the latest version like so:

```
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```

Once you have Miniconda installed, create a virtual environment from the included environment specification in `conda_env.yaml` like so:

```
conda env create -f conda_env.yaml
```

Finally, activate the conda environment via 
```
conda activate bci-arco-gp
```
and set your python path to the project root
```
export  PYTHONPATH="${PYTHONPATH}:/path/to/bci-arco-gp"
```

#### Project Layout

The following gives you a brief overview on the organization and contents of this repository.
Note: in general it should be clear where to change the default paths in the scripts and notebooks, but if you don't want to waste any time just use the default project structure.

```
    │
    ├── README.md           <- This readme file.
    │
    ├── conda_env.yaml      <- The Python environment specification for running the code in this project.
    │
    ├── configs	            <- Directory for custom config files.
    │
    ├── data                <- Directory for generated ground truth models.
    │
    ├── notebooks           <- Jupyter notebooks illustrating the basic usage of the code base.
    |
    ├── results             <- Simulation results.
    |
    ├── src                 <- Contains the Python source code of this project.
    │       ├── environments            <- Everything pertaining to ground truth SCMs and datasets.
    │       ├── experimental_design     <- Everything pertaining to experimental design (utility functions, optimization,...).
    │       ├── graph_models            <- Structure inference (DiBS, ArCO, ...)
    │       ├── mechanism_models        <- Mechanism inference (Gaussian Processes, ...)
    │       ├── scripts                 <- Scripts for running experiments.
    │       ├── utils                   <- Utils for plotting, metrics, baselines, handling graphs and causal orders...
    │       ├── abci_base.py            <- Active BCI base class.
    │       ├── abci_categorical_gp.py  <- ABCI with categorical distribution over graphs & GP mechanisms.
    │       ├── abci_dibs_gp.py         <- BCI with DiBS approximate graph inference & GP mechanisms.
    │       ├── abci_arco_gp.py         <- BCI with ArCO for causal order inference & GP mechanisms.
    │       ├── config.py               <- Global config file.
    │
```

#### Running the Code

You can get started and play around with the provided example notebooks in the `./src/notebooks` folder to get the gist of how to use the code base. Specifically, there are
```
    ├── notebooks
    │   ├── example_ace_estimation.ipynb    <- How to estimate average causal effects with ArCO-GP.
    │   ├── example_arco_gp.ipynb           <- Usage of ArCO-GP: inferring causal orders with ArCO, marginalising over parent sets and using GPs for mechanism inference.
    │   ├── example_arco_model.ipynb        <- Usage of the ArCO model for causal orders.
    │   ├── example_baselines.ipynb         <- Usage of the `Baseline` wrapper class.
    │   ├── example_categorical_gp.ipynb    <- Usage of Categorical-GP: categorical distribution over causal graphs and GPs for mechanism inference.
    │   ├── example_dibs_gp.ipynb           <- Usage of DiBS-GP: DiBS for approximate graph inference and GPs for mechanism inference.
    │   ├── example_load_dataset.ipynb      <- How to load a custom static dataset.
    │   ├── example_sachs.ipynb             <- Structure learning on the Sachs dataset.
    │   ├── generate_benchmark_envs.ipynb   <- How to generate a set of benchmark SCMs.
    │   ├── generate_sachs_envs.ipynb       <- How to generate random splits of the Sachs dataset.
```

To run larger simulations you first need to generate benchmark environments (e.g. with `notebooks/generate_benchmark_envs.ipynb`), or load your own datasets (see `notebooks/example_load_dataset.ipynb`).
You can then run any of our models or a baseline by using one of the scripts in `./src/scripts/`:
```
    ├── scripts
    │   ├── eval_script.py      <- Running custom evaluation scripts on already trained BCI models.
    │   ├── run_benchmark.py    <- Run a model on a batch of problem instances.
    │   ├── run_single_env.py   <- Running a BCI model or baseline of a single problem instance.
```


You can implement your own generic ground truth models by building upon the `Environment` base class in `./src/environments/environments.py`.
Note that this code base is derived from the [Active Bayesian Causal Inference](https://github.com/chritoth/active-bayesian-causal-inference) repository.
That repository uses DiBS-GP in an active learning scenario, hence certain namings like environment or abci appear in the present repository.
The present code base here does not implement the active learning policies for DiBS-GP and ArCO-GP.
An extension would in principle be possible and we might add it in the future.



