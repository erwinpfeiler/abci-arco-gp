from typing import List, Tuple, Dict, Any, Optional

import torch.nn

from src.environments.experiment import InterventionalDistributionsQuery


##############################################################################################
# GRAPH MODEL CONFIGS
##############################################################################################
class DiBSConfig:
    embedding_size: int = None  # if None will be set to number of nodes in dibs init
    num_particles: int = 10
    prior_scale: float = 1.

    def param_dict(self) -> Dict[str, Any]:
        params = {'embedding_size': self.embedding_size,
                  'num_particles': self.num_particles,
                  'prior_scale': self.prior_scale}

        return params

    def load_param_dict(self, param_dict):
        self.embedding_size = param_dict['embedding_size']
        self.num_particles = param_dict['num_particles']
        self.prior_scale = param_dict['prior_scale']


class ArCOConfig:
    map_mode: str = 'mlp'  # available 'mlp' and 'simple'

    # mlp logit map params
    prior_loc: float = 0.
    prior_scale: float = 10.
    num_hidden: int = 30

    def param_dict(self) -> Dict[str, Any]:
        params = {'map_mode': self.map_mode,
                  'prior_loc': self.prior_loc,
                  'prior_scale': self.prior_scale,
                  'num_hidden': self.num_hidden,
                  }

        return params

    def load_param_dict(self, param_dict):
        self.map_mode = param_dict['map_mode']

        self.prior_loc = param_dict['prior_loc']
        self.prior_scale = param_dict['prior_scale']
        self.num_hidden = param_dict['num_hidden']


##############################################################################################
# MECHANISM MODEL CONFIGS
##############################################################################################
class GaussianRootNodeConfig:
    mu_0: float = 0.
    kappa_0: float = 1.
    # generation setup
    # alpha_0: float = 5.
    # beta_0: float = 10.

    # inference setup (normalised data)
    alpha_0: float = 10.
    beta_0: float = 10.

    def param_dict(self) -> Dict[str, Any]:
        params = {'mu_0': self.mu_0,
                  'kappa_0': self.kappa_0,
                  'alpha_0': self.alpha_0,
                  'beta_0': self.beta_0}
        return params

    def load_param_dict(self, param_dict):
        self.mu_0 = param_dict['mu_0']
        self.kappa_0 = param_dict['kappa_0']
        self.alpha_0 = param_dict['alpha_0']
        self.beta_0 = param_dict['beta_0']


class AdditiveSigmoidsConfig:
    # prior hyperparameters
    noise_var_concentration: float = 5.
    noise_var_rate: float = 5.
    lscale_lower: float = 0.5
    lscale_upper: float = 2.
    offset_lower: float = -2.
    offset_upper: float = 2.
    outscale_concentration: float = 50.
    outscale_rate: float = 10.

    def param_dict(self) -> Dict[str, Any]:
        params = {'noise_var_concentration': self.noise_var_concentration,
                  'noise_var_rate': self.noise_var_rate,
                  'lscale_lower': self.lscale_lower,
                  'lscale_upper': self.lscale_upper,
                  'offset_lower': self.offset_lower,
                  'offset_upper': self.offset_upper,
                  'outscale_rate': self.outscale_rate,
                  'outscale_concentration': self.outscale_concentration, }
        return params

    def load_param_dict(self, param_dict):
        self.noise_var_concentration = param_dict['noise_var_concentration']
        self.noise_var_rate = param_dict['noise_var_rate']
        self.lscale_lower = param_dict['lscale_lower']
        self.lscale_upper = param_dict['lscale_upper']
        self.offset_lower = param_dict['offset_lower']
        self.offset_upper = param_dict['offset_upper']
        self.outscale_rate = param_dict['outscale_rate']
        self.outscale_concentration = param_dict['outscale_concentration']


class MLPConfig:
    hidden_layer_sizes: List[int] = [100, 100]
    activation: torch.nn.Module = torch.nn.Sigmoid

    # prior hyperparameters
    noise_var_concentration: float = 100.
    noise_var_rate: float = 10.

    def param_dict(self) -> Dict[str, Any]:
        params = {'hidden_layer_sizes': self.hidden_layer_sizes,
                  'activation': self.activation,
                  'noise_var_concentration': self.noise_var_concentration,
                  'noise_var_rate': self.noise_var_rate}
        return params

    def load_param_dict(self, param_dict):
        self.hidden_layer_sizes = param_dict['hidden_layer_sizes']
        self.activation = param_dict['activation']

        self.noise_var_concentration = param_dict['noise_var_concentration']
        self.noise_var_rate = param_dict['noise_var_rate']


class GaussianProcessConfig:
    # general setup
    per_dim_lenghtscale: bool = True
    scaling_ls_prior: bool = True
    constant_mean: bool = False  # uses constant mean prior if True, zero mean prior if false

    kernel: str = 'additive-rq'  # 'linear', 'rq', 'additive-rq', 'rff'
    num_rff_features: int = 5  # number of random Fourier features for RFF kernel
    covar_jitter: float = 1e-3  # jitter for the covar matrix of the additive RQ kernel

    # setup for sampling ground-truth mechanisms
    num_support_points: int = 50
    support_min: float = -10.
    support_max: float = 10.

    # hyper-prior params for ground-truth model generation
    noise_var_concentration: float = 50.
    noise_var_rate: float = 50.
    outscale_concentration: float = 100.
    outscale_rate: float = 10.
    lscale_concentration_multiplier: float = 30.
    lscale_rate: float = 30.
    scale_mix_concentration: float = 20.
    scale_mix_rate: float = 10.
    offset_loc: float = 0.
    offset_scale: float = 3.

    # hyper-prior params for inference (normalised data)
    # noise_var_concentration: float = 2.
    # noise_var_rate: float = 8.
    # outscale_concentration: float = 5.
    # outscale_rate: float = 1.
    # lscale_concentration_multiplier: float = 30.
    # lscale_rate: float = 30.
    # scale_mix_concentration: float = 20.
    # scale_mix_rate: float = 10.
    # offset_loc: float = 0.
    # offset_scale: float = 0.5

    def param_dict(self) -> Dict[str, Any]:
        params = {'per_dim_lenghtscale': self.per_dim_lenghtscale,
                  'scaling_ls_prior': self.scaling_ls_prior,
                  'constant_mean': self.constant_mean,
                  'kernel': self.kernel,
                  'num_rff_features': self.num_rff_features,
                  'covar_jitter': self.covar_jitter,
                  'num_support_points': self.num_support_points,
                  'support_min': self.support_min,
                  'support_max': self.support_max,
                  'noise_var_concentration': self.noise_var_concentration,
                  'noise_var_rate': self.noise_var_rate,
                  'outscale_concentration': self.outscale_concentration,
                  'outscale_rate': self.outscale_rate,
                  'lscale_concentration_multiplier': self.lscale_concentration_multiplier,
                  'lscale_rate': self.lscale_rate,
                  'scale_mix_concentration': self.scale_mix_concentration,
                  'scale_mix_rate': self.scale_mix_rate,
                  'offset_loc': self.offset_loc,
                  'offset_scale': self.offset_scale
                  }
        return params

    def load_param_dict(self, param_dict):
        self.per_dim_lenghtscale = param_dict['per_dim_lenghtscale']
        self.scaling_ls_prior = param_dict['scaling_ls_prior']
        self.constant_mean = param_dict['constant_mean']
        self.kernel = param_dict['kernel']
        self.num_rff_features = param_dict['num_rff_features']
        self.covar_jitter = param_dict['covar_jitter']
        self.num_support_points = param_dict['num_support_points']
        self.support_min = param_dict['support_min']
        self.support_max = param_dict['support_max']
        self.noise_var_concentration = param_dict['noise_var_concentration']
        self.noise_var_rate = param_dict['noise_var_rate']
        self.outscale_concentration = param_dict['outscale_concentration']
        self.outscale_rate = param_dict['outscale_rate']
        self.lscale_concentration_multiplier = param_dict['lscale_concentration_multiplier']
        self.lscale_rate = param_dict['lscale_rate']
        self.scale_mix_concentration = param_dict['scale_mix_concentration']
        self.scale_mix_rate = param_dict['scale_mix_rate']
        self.offset_loc = param_dict['offset_loc']
        self.offset_scale = param_dict['offset_scale']


class GPModelConfig:
    imll_mc_samples: int = 50  # number of ancestral mc samples to estimate an interventional mll
    opt_batch_size: int = 20  # maximum number of GP for which to update HP simulateniously to avoid out of mem
    discard_threshold_gps: int = 70000  # max number of mechanisms to keep in model
    discard_threshold_topo_orders: int = 30000  # max number of mechanisms to keep in model

    # gp hyperparam training
    num_steps: int = 100
    log_interval: int = 0
    lr: float = 5e-2
    es_threshold: float = 1e-2  # early stopping criterion threshold
    es_min_steps: int = 10  # minimum number of gradient steps to perform before checking early stopping
    es_win_size: int = 3  # window size of the running mean to compute the early stopping criterion

    def param_dict(self) -> Dict[str, Any]:
        params = {'imll_mc_samples': self.imll_mc_samples,
                  'opt_batch_size': self.opt_batch_size,
                  'num_steps': self.num_steps,
                  'log_interval': self.log_interval,
                  'lr': self.lr,
                  'es_threshold': self.es_threshold,
                  'es_min_steps': self.es_min_steps,
                  'es_win_size': self.es_win_size}
        return params

    def load_param_dict(self, param_dict):
        self.imll_mc_samples = param_dict['imll_mc_samples']
        self.opt_batch_size = param_dict['opt_batch_size']
        self.num_steps = param_dict['num_steps']
        self.log_interval = param_dict['log_interval']
        self.lr = param_dict['lr']
        self.es_threshold = param_dict['es_threshold']
        self.es_min_steps = param_dict['es_min_steps']
        self.es_win_size = param_dict['es_win_size']


##############################################################################################
# ABCI CONFIGS
##############################################################################################
class ABCIBaseConfig:
    # general config
    policy: str
    num_workers: int

    # run config
    checkpoint_interval: int
    output_dir: str
    model_name: str
    run_id: str
    num_experiments: int
    batch_size: int
    log_interval: int
    num_initial_obs_samples: int

    @classmethod
    def check_policy(cls, policy: str):
        raise NotImplementedError

    def __init__(self):
        self.check_policy(self.policy)


class ABCIFixedGraphGPConfig(ABCIBaseConfig):
    # general config
    policy: str = 'static-obs-dataset'
    num_workers: int = 1

    # run config
    checkpoint_interval: int = 10
    output_dir: str = None
    model_name: str = 'abci-fixed-graph-gp'
    run_id: str = ''
    num_experiments: int = 1
    batch_size: int = 3
    log_interval: int = 1
    num_initial_obs_samples: int = 3

    # eval parameters
    compute_distributional_stats: bool = False
    num_samples_per_graph = 10000

    @classmethod
    def check_policy(cls, policy: str):
        assert policy in {'observational', 'random', 'random-fixed-value', 'static-obs-dataset',
                          'static-intr-dataset'}, policy

    def __init__(self, param_dict: Dict[str, Any] = None):
        if param_dict is not None:
            self.load_param_dict(param_dict)
        super().__init__()

    def param_dict(self) -> Dict[str, Any]:
        params = {'policy': self.policy,
                  'num_workers': self.num_workers,
                  'checkpoint_interval': self.checkpoint_interval,
                  'output_dir': self.output_dir,
                  'model_name': self.model_name,
                  'run_id': self.run_id,
                  'num_experiments': self.num_experiments,
                  'batch_size': self.batch_size,
                  'log_interval': self.log_interval,
                  'num_initial_obs_samples': self.num_initial_obs_samples,
                  'compute_distributional_stats': self.compute_distributional_stats,
                  'num_samples_per_graph': self.num_samples_per_graph}
        return params

    def load_param_dict(self, param_dict):
        self.policy = param_dict['policy']
        self.num_workers = param_dict['num_workers']
        self.checkpoint_interval = param_dict['checkpoint_interval']
        self.output_dir = param_dict['output_dir']
        self.model_name = param_dict['model_name']
        self.run_id = param_dict['run_id']
        self.num_experiments = param_dict['num_experiments']
        self.batch_size = param_dict['batch_size']
        self.log_interval = param_dict['log_interval']
        self.num_initial_obs_samples = param_dict['num_initial_obs_samples']
        self.compute_distributional_stats = param_dict['compute_distributional_stats']
        self.num_samples_per_graph = param_dict['num_samples_per_graph']


class ABCICategoricalGPConfig(ABCIBaseConfig):
    # general config
    policy: str = 'observational'
    num_workers: int = 1

    # experimental design
    opt_strategy: str = 'gp-ucb'
    num_exp_per_graph: int = 3  # number of simulated experiments from the model posterior
    num_mc_queries: int = 3  # number of simulated queries (i.e. different intervention values) from the model posterior
    num_batches_per_query: int = 3  # number of batches per samples intervention value
    num_outer_mc_graphs: int = 5  # number of mc graphs (only for intervention-info-gain)
    outer_mc_mode: str = 'n-best'  # selection strategy of the outer mc graphs

    # run config
    checkpoint_interval: int = 10
    output_dir: str = None
    model_name: str = 'abci-categorical-gp'
    run_id: str = ''
    num_experiments: int = 10
    batch_size: int = 3
    log_interval: int = 1
    num_initial_obs_samples: int = 3

    @classmethod
    def check_policy(cls, policy: str):
        assert policy in {'observational', 'random', 'random-fixed-value', 'graph-info-gain', 'scm-info-gain',
                          'intervention-info-gain', 'oracle', 'static-obs-dataset', 'static-intr-dataset'}, policy

    def __init__(self, param_dict: Dict[str, Any] = None):
        if param_dict is not None:
            self.load_param_dict(param_dict)
        super().__init__()

    def param_dict(self) -> Dict[str, Any]:
        params = {'policy': self.policy,
                  'num_workers': self.num_workers,
                  'opt_strategy': self.opt_strategy,
                  'num_exp_per_graph': self.num_exp_per_graph,
                  'num_mc_queries': self.num_mc_queries,
                  'num_batches_per_query': self.num_batches_per_query,
                  'num_outer_mc_graphs': self.num_outer_mc_graphs,
                  'outer_mc_mode': self.outer_mc_mode,
                  'checkpoint_interval': self.checkpoint_interval,
                  'output_dir': self.output_dir,
                  'model_name': self.model_name,
                  'run_id': self.run_id,
                  'num_experiments': self.num_experiments,
                  'batch_size': self.batch_size,
                  'log_interval': self.log_interval,
                  'num_initial_obs_samples': self.num_initial_obs_samples}
        return params

    def load_param_dict(self, param_dict):
        self.policy = param_dict['policy']
        self.num_workers = param_dict['num_workers']
        self.opt_strategy = param_dict['opt_strategy']
        self.num_exp_per_graph = param_dict['num_exp_per_graph']
        self.num_mc_queries = param_dict['num_mc_queries']
        self.num_batches_per_query = param_dict['num_batches_per_query']
        self.num_outer_mc_graphs = param_dict['num_outer_mc_graphs']
        self.outer_mc_mode = param_dict['outer_mc_mode']
        self.checkpoint_interval = param_dict['checkpoint_interval']
        self.output_dir = param_dict['output_dir']
        self.model_name = param_dict['model_name']
        self.run_id = param_dict['run_id']
        self.num_experiments = param_dict['num_experiments']
        self.batch_size = param_dict['batch_size']
        self.log_interval = param_dict['log_interval']
        self.num_initial_obs_samples = param_dict['num_initial_obs_samples']


class ABCIDiBSGPConfig(ABCIBaseConfig):
    # general config
    policy: str = 'static-obs-dataset'
    num_workers: int = 1
    dibs_plus: bool = True
    num_particles: int = 10
    num_mc_graphs: int = 100
    embedding_size: int = None

    # experimental design
    opt_strategy: str = 'gp-ucb'
    num_outer_graphs: int = 5  # number of graph MC samples to approximate the outer expectation
    num_inner_graphs: int = 3  # number of graph MC samples to approximate the outer expectation
    num_exp_per_graph: int = 50  # number of simulated experiments from the model posterior
    num_mc_queries: int = 5  # number of simulated queries (i.e. different intervention values) from the model posterior
    num_batches_per_query: int = 3  # number of batches per samples intervention value

    # run config
    checkpoint_interval: int = 10
    output_dir: str = None
    model_name: str = 'abci-dibs-gp'
    run_id: str = ''
    num_experiments: int = 1
    batch_size: int = 3
    log_interval: int = 1
    num_initial_obs_samples: int = 200

    # eval parameters
    compute_distributional_stats: bool = False
    num_samples_per_graph = 10

    # training parameters
    resampling_frac: float = 0.25  # fraction of particles to keep when resampling
    resampling_threshold: float = 1e-2
    num_svgd_steps: int = 500
    svgd_log_interval: int = 25
    lr: float = 1e-1
    es_threshold: float = 1e-3  # early stopping criterion threshold
    es_min_steps: int = 50  # minimum number of gradient steps to perform before checking early stopping
    es_win_size: int = 3  # window size of the running mean to compute the early stopping criterion
    alpha_offset: float = 1.  # p(G|Z) softmax temperature linear schedule offset
    alpha_slope: float = 0.  # p(G|Z) softmax temperature linear schedule slope
    alpha_eval: float = 1.  # p(G|Z) softmax temperature @ test time
    beta_offset: float = 0.  # p(Z) acyclicty temperature linear schedule offset
    beta_slope: float = 1.  # p(Z) acyclicty temperature linear schedule slope
    beta_eval: float = 100.  # p(Z) acyclicity temperature @ test time
    num_graphs_for_ec: int = 100  # number of MC graph samples used to estimate the expected cyclicity of the particles
    max_num_mc_mechanisms: Optional[int] = 3000  # maximum number of unique mechanisms when sampling mc graphs

    @classmethod
    def check_policy(cls, policy: str):
        assert policy in {'observational', 'random', 'random-fixed-value', 'graph-info-gain', 'scm-info-gain',
                          'intervention-info-gain', 'static-obs-dataset', 'static-intr-dataset'}, policy

    def __init__(self, param_dict: Dict[str, Any] = None):
        if param_dict is not None:
            self.load_param_dict(param_dict)
        super().__init__()

    def param_dict(self) -> Dict[str, Any]:
        params = {'policy': self.policy,
                  'num_workers': self.num_workers,
                  'dibs_plus': self.dibs_plus,
                  'num_particles': self.num_particles,
                  'num_mc_graphs': self.num_mc_graphs,
                  'embedding_size': self.embedding_size,
                  'opt_strategy': self.opt_strategy,
                  'num_outer_graphs': self.num_outer_graphs,
                  'num_inner_graphs': self.num_inner_graphs,
                  'num_exp_per_graph': self.num_exp_per_graph,
                  'num_mc_queries': self.num_mc_queries,
                  'num_batches_per_query': self.num_batches_per_query,
                  'checkpoint_interval': self.checkpoint_interval,
                  'output_dir': self.output_dir,
                  'model_name': self.model_name,
                  'run_id': self.run_id,
                  'num_experiments': self.num_experiments,
                  'batch_size': self.batch_size,
                  'log_interval': self.log_interval,
                  'num_initial_obs_samples': self.num_initial_obs_samples,
                  'compute_distributional_stats': self.compute_distributional_stats,
                  'num_samples_per_graph': self.num_samples_per_graph,
                  'resampling_frac': self.resampling_frac,
                  'resampling_threshold': self.resampling_threshold,
                  'num_svgd_steps': self.num_svgd_steps,
                  'svgd_log_interval': self.svgd_log_interval,
                  'lr': self.lr,
                  'es_threshold': self.es_threshold,
                  'es_min_steps': self.es_min_steps,
                  'es_win_size': self.es_win_size,
                  'alpha_offset': self.alpha_offset,
                  'alpha_slope': self.alpha_slope,
                  'alpha_eval': self.alpha_eval,
                  'beta_offset': self.beta_offset,
                  'beta_slope': self.beta_slope,
                  'beta_eval': self.beta_eval,
                  'num_graphs_for_ec': self.num_graphs_for_ec,
                  'max_num_mc_mechanisms': self.max_num_mc_mechanisms}
        return params

    def load_param_dict(self, param_dict):
        self.policy = param_dict['policy']
        self.num_workers = param_dict['num_workers']
        self.dibs_plus = param_dict['dibs_plus']
        self.num_particles = param_dict['num_particles']
        self.num_mc_graphs = param_dict['num_mc_graphs']
        self.embedding_size = param_dict['embedding_size']
        self.opt_strategy = param_dict['opt_strategy']
        self.num_outer_graphs = param_dict['num_outer_graphs']
        self.num_inner_graphs = param_dict['num_inner_graphs']
        self.num_exp_per_graph = param_dict['num_exp_per_graph']
        self.num_mc_queries = param_dict['num_mc_queries']
        self.num_batches_per_query = param_dict['num_batches_per_query']
        self.checkpoint_interval = param_dict['checkpoint_interval']
        self.output_dir = param_dict['output_dir']
        self.model_name = param_dict['model_name']
        self.run_id = param_dict['run_id']
        self.num_experiments = param_dict['num_experiments']
        self.batch_size = param_dict['batch_size']
        self.log_interval = param_dict['log_interval']
        self.num_initial_obs_samples = param_dict['num_initial_obs_samples']
        self.compute_distributional_stats = param_dict['compute_distributional_stats']
        self.num_samples_per_graph = param_dict['num_samples_per_graph']
        self.resampling_frac = param_dict['resampling_frac']
        self.resampling_threshold = param_dict['resampling_threshold']
        self.num_svgd_steps = param_dict['num_svgd_steps']
        self.svgd_log_interval = param_dict['svgd_log_interval']
        self.lr = param_dict['lr']
        self.es_threshold = param_dict['es_threshold']
        self.es_min_steps = param_dict['es_min_steps']
        self.es_win_size = param_dict['es_win_size']
        self.alpha_offset = param_dict['alpha_offset']
        self.alpha_slope = param_dict['alpha_slope']
        self.alpha_eval = param_dict['alpha_eval']
        self.beta_offset = param_dict['beta_offset']
        self.beta_slope = param_dict['beta_slope']
        self.beta_eval = param_dict['beta_eval']
        self.num_graphs_for_ec = param_dict['num_graphs_for_ec']
        self.max_num_mc_mechanisms = param_dict['max_num_mc_mechanisms']


class ABCIArCOGPConfig(ABCIBaseConfig):
    # general config
    policy: str = 'static-obs-dataset'
    num_workers: int = 1
    max_ps_size: int = 2

    # run config
    checkpoint_interval: int = 10
    output_dir: str = None
    model_name: str = 'abci-arco-gp'
    run_id: str = ''
    num_experiments: int = 1
    batch_size: int = 1
    log_interval: int = 1
    num_initial_obs_samples: int = 200

    # eval parameters
    num_mc_cos: int = 10 # 100
    num_mc_graphs: int = 10
    compute_distributional_stats: bool = False
    num_samples_per_graph = 10

    # training parameters
    tau: float = 0.1  # score func estimator baseline decay factor
    es_threshold: float = 1e-3  # early stopping criterion threshold
    es_win_size: int = 3  # window size of the running mean to compute the early stopping criterion
    opt_log_interval: int = 20

    # co model optimisation
    num_arco_steps: int = 400
    num_cos_arco_opt: int = 100
    arco_lr: float = 1e-2  # co model optimizer learning rate
    arco_es_min_steps: int = 50  # minimum number of gradient steps to perform before checking early stopping

    # active learning
    batch_size = 10

    @classmethod
    def check_policy(cls, policy: str):
        assert policy in {'observational', 'random', 'random-fixed-value', 'static-obs-dataset',
                          'static-intr-dataset','graph-info-gain'}, policy

    def __init__(self, param_dict: Dict[str, Any] = None):
        if param_dict is not None:
            self.load_param_dict(param_dict)
        super().__init__()

    def param_dict(self) -> Dict[str, Any]:
        params = {'policy': self.policy,
                  'num_workers': self.num_workers,
                  'max_ps_size': self.max_ps_size,
                  # run config
                  'checkpoint_interval': self.checkpoint_interval,
                  'output_dir': self.output_dir,
                  'model_name': self.model_name,
                  'run_id': self.run_id,
                  'num_experiments': self.num_experiments,
                  'batch_size': self.batch_size,
                  'log_interval': self.log_interval,
                  'num_initial_obs_samples': self.num_initial_obs_samples,
                  # eval parameters
                  'num_mc_cos': self.num_mc_cos,
                  'num_mc_graphs': self.num_mc_graphs,
                  'compute_distributional_stats': self.compute_distributional_stats,
                  'num_samples_per_graph': self.num_samples_per_graph,
                  # training parameters
                  'tau': self.tau,
                  'es_threshold': self.es_threshold,
                  'es_win_size': self.es_win_size,
                  'opt_log_interval': self.opt_log_interval,
                  # co model optimisation
                  'num_arco_steps': self.num_arco_steps,
                  'num_cos_arco_opt': self.num_cos_arco_opt,
                  'arco_lr': self.arco_lr,
                  'arco_es_min_steps': self.arco_es_min_steps}
        return params

    def load_param_dict(self, param_dict):
        # general config
        self.policy = param_dict['policy']
        self.num_workers = param_dict['num_workers']
        self.max_ps_size = param_dict['max_ps_size']

        # run config
        self.checkpoint_interval = param_dict['checkpoint_interval']
        self.output_dir = param_dict['output_dir']
        self.model_name = param_dict['model_name']
        self.run_id = param_dict['run_id']
        self.num_experiments = param_dict['num_experiments']
        self.batch_size = param_dict['batch_size']
        self.log_interval = param_dict['log_interval']
        self.num_initial_obs_samples = param_dict['num_initial_obs_samples']

        # eval parameters
        self.num_mc_cos = param_dict['num_mc_cos']
        self.num_mc_graphs = param_dict['num_mc_graphs']
        self.compute_distributional_stats = param_dict['compute_distributional_stats']
        self.num_samples_per_graph = param_dict['num_samples_per_graph']

        # training parameters
        self.tau = param_dict['tau']
        self.es_threshold = param_dict['es_threshold']
        self.es_win_size = param_dict['es_win_size']
        self.opt_log_interval = param_dict['opt_log_interval']

        # co model optimisation
        self.num_arco_steps = param_dict['num_arco_steps']
        self.num_cos_arco_opt = param_dict['num_cos_arco_opt']
        self.arco_lr = param_dict['arco_lr']
        self.arco_es_min_steps = param_dict['arco_es_min_steps']


##############################################################################################
# ENVIRONMENT CONFIGS
##############################################################################################
class EnvironmentConfig:
    id_length: int = 8
    intervention_bounds: Tuple[float, float] = (-1., 1.)
    mechanism_model: str = 'gp-model'
    normalise_data: bool = True
    frac_non_intervenable_nodes: float = None
    non_intervenable_nodes: set = None

    generate_static_obs_dataset: bool = True
    num_observational_train_samples: int = 100
    num_observational_test_samples: int = 0

    generate_static_intr_dataset: bool = False
    num_train_interventions: int = 20
    num_interventional_train_samples: int = 5
    num_test_interventions: int = 40
    num_interventional_test_samples: int = 5

    generate_test_queries: bool = False
    num_test_queries: int = 30
    interventional_queries: Optional[List[InterventionalDistributionsQuery]] = None
    imll_mc_samples: int = 50  # number of ancestral mc samples to estimate an interventional mll

    def param_dict(self) -> Dict[str, Any]:
        if self.interventional_queries is not None:
            intr_query_param_dicts = [query.param_dict() for query in self.interventional_queries]
        else:
            intr_query_param_dicts = None

        params = {'id_length': self.id_length,
                  'intervention_bounds': self.intervention_bounds,
                  'mechanism_model': self.mechanism_model,
                  'normalise_data': self.normalise_data,
                  'frac_non_intervenable_nodes': self.frac_non_intervenable_nodes,
                  'non_intervenable_nodes': self.non_intervenable_nodes,
                  #
                  'generate_static_obs_dataset': self.generate_static_obs_dataset,
                  'num_observational_train_samples': self.num_observational_train_samples,
                  'num_observational_test_samples': self.num_observational_test_samples,
                  #
                  'generate_static_intr_dataset': self.generate_static_intr_dataset,
                  'num_train_interventions': self.num_train_interventions,
                  'num_interventional_train_samples': self.num_interventional_train_samples,
                  'num_test_interventions': self.num_test_interventions,
                  'num_interventional_test_samples': self.num_interventional_test_samples,
                  #
                  'generate_test_queries': self.generate_test_queries,
                  'num_test_queries': self.num_test_queries,
                  'imll_mc_samples': self.imll_mc_samples,
                  'intr_query_param_dicts': intr_query_param_dicts}
        return params

    def load_param_dict(self, param_dict):
        self.id_length = param_dict['id_length']
        self.intervention_bounds = param_dict['intervention_bounds']
        self.mechanism_model = param_dict['mechanism_model']
        self.normalise_data = param_dict['normalise_data']
        self.frac_non_intervenable_nodes = param_dict['frac_non_intervenable_nodes']
        self.non_intervenable_nodes = param_dict['non_intervenable_nodes']

        self.generate_static_obs_dataset = param_dict['generate_static_obs_dataset']
        self.num_observational_train_samples = param_dict['num_observational_train_samples']
        self.num_observational_test_samples = param_dict['num_observational_test_samples']

        self.generate_static_intr_dataset = param_dict['generate_static_intr_dataset']
        self.num_train_interventions = param_dict['num_train_interventions']
        self.num_interventional_train_samples = param_dict['num_interventional_train_samples']
        self.num_test_interventions = param_dict['num_test_interventions']
        self.num_interventional_test_samples = param_dict['num_interventional_test_samples']

        self.generate_test_queries = param_dict['generate_test_queries']
        self.num_test_queries = param_dict['num_test_queries']
        self.imll_mc_samples = param_dict['imll_mc_samples']
        self.interventional_queries = None
        if param_dict['intr_query_param_dicts'] is not None:
            self.interventional_queries = []
            for query_param_dict in param_dict['intr_query_param_dicts']:
                self.interventional_queries.append(InterventionalDistributionsQuery.load_param_dict(query_param_dict))
