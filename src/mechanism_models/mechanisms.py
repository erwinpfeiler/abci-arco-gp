import math
from itertools import chain
from typing import List, Tuple, Dict, Any

import gpytorch
import torch
import torch.distributions as dist
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RQKernel, LinearKernel, ScaleKernel, AdditiveStructureKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ZeroMean, ConstantMean
from gpytorch.models import ExactGP
from torch import Tensor, Size
from torch.nn import Module, ModuleDict
from torch.nn.utils import vector_to_parameters

from src.config import GaussianRootNodeConfig, GaussianProcessConfig, AdditiveSigmoidsConfig
from src.utils.utils import get_module_params


def get_mechanism_key(node, parents: List) -> str:
    parents = sorted(parents)
    key = str(node) + '<-' + ','.join([str(parent) for parent in parents])
    return key


def resolve_mechanism_key(key: str) -> Tuple[str, List[str]]:
    idx = key.find('<-')
    assert idx > 0, print('Invalid key: ' + key)
    node = key[:idx]
    parents = key[idx + 2:].split(',') if len(key) > idx + 2 else []
    return node, parents


class Mechanism(Module):
    """
    Class that represents a generic mechanism including a likelihood/noise model in an SCM.

    Attributes
    ----------
    in_size : int
            Number of mechanism inputs.
    """

    def __init__(self, in_size: int):
        """
        Parameters
        ----------
        in_size : int
            Number of mechanism inputs.
        """
        super().__init__()
        self.in_size = in_size

    def _check_args(self, inputs: Tensor = None, targets: Tensor = None) -> Tuple[Tensor, Tensor, Tuple[int]]:
        """
        Checks arguments (inputs and targets) shapes and compatibility. Re-shapes inputs to shape
        (num_batches, num_samples_per_batch, num_parents) and targets to shape (num_batches, num_samples_per_batch).

        Parameters
        ----------
        inputs : Tensor
            Mechanism inputs.
        targets : Tensor
            Mechanism targets, e.g., for evaluating the marginal log-likelihood.
        """
        batch_shape = None
        if inputs is not None:
            in_size = self.in_size if self.in_size > 0 else 1
            assert 2 <= inputs.dim() <= 3 and inputs.shape[-1] == in_size, print(f'Ill-shaped inputs: {inputs.shape}')
            inputs = inputs.unsqueeze(0) if inputs.dim() == 2 else inputs
            batch_shape = tuple(inputs.shape[:-1])
        if targets is not None:
            assert 1 <= targets.dim() <= 2, print(f'Ill-shaped targets: {targets.shape}')
            targets = targets.unsqueeze(0) if targets.dim() == 1 else targets
            batch_shape = tuple(targets.shape)
        if targets is not None and inputs is not None:
            assert inputs.shape[:-1] == targets.shape, print(f'Batch size mismatch: {inputs.shape} vs.'
                                                             f' {targets.shape}')

        return inputs, targets, batch_shape


class GaussianRootNode(Mechanism):
    def __init__(self, static=False, cfg: GaussianRootNodeConfig = None, param_dict: Dict[str, Any] = None):
        super().__init__(in_size=0)
        if param_dict is not None:
            self.load_param_dict(param_dict)
        else:
            self.cfg = GaussianRootNodeConfig() if cfg is None else cfg
            # init prior and posterior hyper-parameters
            self.mu_0 = self.mu_n = torch.tensor(self.cfg.mu_0)
            self.kappa_0 = self.kappa_n = torch.tensor(self.cfg.kappa_0)
            self.alpha_0 = self.alpha_n = torch.tensor(self.cfg.alpha_0)
            self.beta_0 = self.beta_n = torch.tensor(self.cfg.beta_0)
            self.lam_0 = None
            self.train_targets = None

            self.static = static
            if static:
                self.init_as_static()

    def compute_posterior_params(self, targets: Tensor, prior_mode=False):
        _, targets, batch_shape = self._check_args(targets=targets)

        full_targets = targets
        if not prior_mode and self.train_targets is not None:
            num_batches = batch_shape[0]
            full_targets = torch.cat((targets, self.train_targets.expand(num_batches, -1)), dim=-1)

        n = full_targets.shape[-1]
        empirical_means = full_targets.mean(dim=-1)

        kappa_n = self.kappa_0 + n
        mu_n = (self.kappa_0 * self.mu_0 + n * empirical_means) / kappa_n
        alpha_n = self.alpha_0 + 0.5 * n
        beta_n = self.beta_0 + 0.5 * (full_targets - empirical_means.unsqueeze(-1)).pow_(2).sum(dim=-1) + \
                 0.5 * self.kappa_0 * n * (empirical_means - self.mu_0).pow_(2) / kappa_n

        return mu_n, kappa_n.expand(mu_n.shape), alpha_n.expand(mu_n.shape), beta_n

    def init_as_static(self):
        self.lam_0 = dist.Gamma(self.alpha_0, self.beta_0).sample()
        self.mu_0 = dist.Normal(0., (self.kappa_0 * self.lam_0).pow(-0.5)).sample()

    def set_data(self, inputs: Tensor, targets: Tensor):
        _, targets, _ = self._check_args(targets=targets)
        assert targets.shape[0] == 1, print('Can only work with one batch of posterior params!')
        self.train_targets = targets
        self.mu_n, self.kappa_n, self.alpha_n, self.beta_n = self.compute_posterior_params(targets, prior_mode=True)

    def forward(self, inputs: Tensor, prior_mode=False):
        _, _, batch_shape = self._check_args(inputs)
        output_shape = batch_shape + (1,)

        if self.static or prior_mode:
            return self.mu_0 * torch.ones(output_shape)

        return self.mu_n * torch.ones(output_shape)

    def sample(self, inputs: Tensor, prior_mode=False):
        _, _, batch_shape = self._check_args(inputs)
        output_shape = batch_shape + (1,)

        if self.static:
            # sample from true distribution
            y_dist = dist.Normal(self.mu_0, self.lam_0.pow(-0.5))
            return y_dist.sample(output_shape)

        # sample from marginal likelihood
        if prior_mode:
            mu_n, kappa_n, alpha_n, beta_n = (self.mu_0, self.kappa_0, self.alpha_0, self.beta_0)
        else:
            mu_n, kappa_n, alpha_n, beta_n = (self.mu_n, self.kappa_n, self.alpha_n, self.beta_n)

        lambdas = dist.Gamma(alpha_n, beta_n).sample(output_shape[:-2])
        mus = dist.Normal(mu_n.expand_as(lambdas), (kappa_n * lambdas).pow(-0.5)).sample()

        y_dist = dist.Normal(mus, lambdas.pow(-0.5))
        samples = y_dist.sample(output_shape[-2:-1]).unsqueeze(-1).transpose(0, -1).view(output_shape)
        return samples

    def mll(self, inputs: Tensor, targets: Tensor, prior_mode=False, reduce=True):
        _, targets, batch_shape = self._check_args(targets=targets)
        output_shape = batch_shape[:-1]
        if self.static:
            # evaluate true log-likelihood
            y_dist = dist.Normal(self.mu_0, self.lam_0.pow(-0.5))
            lls = y_dist.log_prob(targets).squeeze(-1)
        else:
            if prior_mode:
                kappa_n, alpha_n, beta_n = (self.kappa_0, self.alpha_0, self.beta_0)
            else:
                kappa_n, alpha_n, beta_n = (self.kappa_n, self.alpha_n, self.beta_n)

            _, kappa_m, alpha_m, beta_m = self.compute_posterior_params(targets, prior_mode)
            lls = torch.lgamma(alpha_m) - torch.lgamma(alpha_n) + alpha_n * beta_n.log() - alpha_m * beta_m.log() + \
                  0.5 * (kappa_n.log() - kappa_m.log()) - 0.5 * targets.shape[-1] * math.log(2. * math.pi)

        assert lls.shape == output_shape, print(lls.shape)
        if reduce:
            return lls.sum()
        return lls

    def expected_noise_entropy(self, prior_mode: bool = False) -> Tensor:
        if self.static:
            return dist.Normal(self.mu_0, self.lam_0.pow(-0.5)).entropy()

        # expected noise entropy exact
        alpha, beta = (self.alpha_0, self.beta_0) if prior_mode else (self.alpha_n, self.beta_n)
        return 0.5 * (math.log(2. * math.pi * math.e) - torch.digamma(alpha) + beta.log())

        # expected noise entropy point estimate (mean variance of inverse gamma posterior)
        # return 0.5 * (2. * math.pi * beta/(alpha + 1.) * math.e).log().squeeze()

    def param_dict(self) -> Dict[str, Any]:
        params = {'in_size': 0,
                  'mu_0': self.mu_0,
                  'kappa_0': self.kappa_0,
                  'alpha_0': self.alpha_0,
                  'beta_0': self.beta_0,
                  'lam_0': self.lam_0,
                  'mu_n': self.mu_n,
                  'kappa_n': self.kappa_n,
                  'alpha_n': self.alpha_n,
                  'beta_n': self.beta_n,
                  'static': self.static,
                  'cfg_param_dict': self.cfg.param_dict()}

        return params

    def load_param_dict(self, param_dict):
        self.mu_0 = param_dict['mu_0'].float()
        self.kappa_0 = param_dict['kappa_0'].float()
        self.alpha_0 = param_dict['alpha_0'].float()
        self.beta_0 = param_dict['beta_0'].float()
        self.lam_0 = param_dict['lam_0']
        if self.lam_0 is not None:
            self.lam_0 = self.lam_0.float()
        self.mu_n = param_dict['mu_n'].float()
        self.kappa_n = param_dict['kappa_n'].float()
        self.alpha_n = param_dict['alpha_n'].float()
        self.beta_n = param_dict['beta_n'].float()
        self.static = param_dict['static']
        self.cfg = GaussianRootNodeConfig()
        self.cfg.load_param_dict(param_dict['cfg_param_dict'])


class GaussianProcess(Mechanism):
    ##########################################################################
    # Additive RQ Kernel
    class GaussianProcessAdditiveRQKernel(ExactGP):
        posterior_noise: Tensor
        posterior_outputscale: Tensor
        posterior_lengthscale: Tensor
        posterior_scale_mix: Tensor

        def __init__(self, in_size: int, cfg: GaussianProcessConfig, param_dict: Dict[str, Any] = None):
            assert in_size > 0, print(f'Invalid input dimensions {in_size}!')
            likelihood = GaussianLikelihood()
            super().__init__(None, None, likelihood)
            self.in_size = in_size
            self.cfg = cfg
            self.mean_module = ZeroMean()
            base_kernel = RQKernel(ard_num_dims=self.in_size)
            self.kernel = ScaleKernel(AdditiveStructureKernel(base_kernel, self.in_size))

            # init hp priors
            # ATTENTION: do not name the HP priors "noise_prior", "outputscale_prior" or "lengthscale_prior"
            self.noise_var_prior = dist.Gamma(cfg.noise_var_concentration, cfg.noise_var_rate)
            self.outscale_prior = dist.Gamma(cfg.outscale_concentration, cfg.outscale_rate)
            self.lscale_prior = dist.Gamma(cfg.lscale_concentration_multiplier, cfg.lscale_rate)
            self.scale_mix_prior = dist.Gamma(cfg.scale_mix_concentration, cfg.scale_mix_rate)

            if param_dict is not None:
                self.load_param_dict(param_dict)
            else:
                # init kernel parameters
                self.init_hyperparams()

            self.posterior_hp = False
            self.select_hyperparameters(posterior_hp=True)

        def forward(self, x):
            mean = self.mean_module(x)
            covar = self.kernel(x)
            return MultivariateNormal(mean, covar)

        def hyperparam_log_prior(self):
            log_prior = self.noise_var_prior.log_prob(self.likelihood.noise) + \
                        self.outscale_prior.log_prob(self.kernel.outputscale) + \
                        self.lscale_prior.log_prob(self.kernel.base_kernel.base_kernel.lengthscale).mean() + \
                        self.scale_mix_prior.log_prob(self.kernel.base_kernel.base_kernel.alpha)
            return log_prior.squeeze()

        def init_hyperparams(self):
            noise_var = self.noise_var_prior.sample()
            self.posterior_noise = noise_var if noise_var > 1e-3 else torch.tensor(1e-3)
            self.posterior_outputscale = self.outscale_prior.sample()
            self.posterior_scale_mix = self.scale_mix_prior.sample()
            if self.cfg.per_dim_lenghtscale:
                self.posterior_lengthscale = self.lscale_prior.sample(Size((self.in_size,)))
            else:
                self.posterior_lengthscale = self.lscale_prior.sample(Size((1,))).expand(self.in_size)

        def select_hyperparameters(self, posterior_hp: bool):
            if posterior_hp:
                if not self.posterior_hp:
                    # load posterior HPs
                    self.likelihood.noise = self.posterior_noise
                    self.kernel.outputscale = self.posterior_outputscale
                    self.kernel.base_kernel.base_kernel.lengthscale = self.posterior_lengthscale
                    self.kernel.base_kernel.base_kernel.alpha = self.posterior_scale_mix
                    self.posterior_hp = True
            else:
                # store posterior HPs if posterior HPs currently in use
                if self.posterior_hp:
                    self.posterior_noise = self.likelihood.noise.detach()
                    self.posterior_outputscale = self.kernel.outputscale.detach()
                    self.posterior_lengthscale = self.kernel.base_kernel.base_kernel.lengthscale.detach()
                    self.posterior_scale_mix = self.kernel.base_kernel.base_kernel.alpha.detach()
                    self.posterior_hp = False

                # draw HPs from HP prior
                self.likelihood.noise = self.noise_var_prior.sample()
                self.kernel.outputscale = self.outscale_prior.sample()
                self.kernel.base_kernel.base_kernel.alpha = self.scale_mix_prior.sample()
                if self.cfg.per_dim_lenghtscale:
                    self.kernel.base_kernel.base_kernel.lengthscale = self.lscale_prior.sample(Size((self.in_size,)))
                else:
                    self.kernel.base_kernel.base_kernel.lengthscale = self.lscale_prior.sample(Size((1,))).expand(
                        self.in_size)

        def param_dict(self) -> Dict[str, Any]:
            params = {'in_size': self.in_size,
                      'posterior_noise': self.posterior_noise,
                      'posterior_outputscale': self.posterior_outputscale,
                      'posterior_lengthscale': self.posterior_lengthscale,
                      'posterior_scale_mix': self.posterior_scale_mix}
            return params

        def load_param_dict(self, param_dict):
            self.in_size = param_dict['in_size']
            self.posterior_noise = param_dict['posterior_noise'].float()
            self.posterior_outputscale = param_dict['posterior_outputscale'].float()
            self.posterior_lengthscale = param_dict['posterior_lengthscale'].float()
            self.posterior_scale_mix = param_dict['posterior_scale_mix'].float()

    ##########################################################################
    # RQ Kernel
    class GaussianProcessRQKernel(ExactGP):
        posterior_noise: Tensor
        posterior_outputscale: Tensor
        posterior_lengthscale: Tensor
        posterior_scale_mix: Tensor

        def __init__(self, in_size: int, cfg: GaussianProcessConfig, param_dict: Dict[str, Any] = None):
            assert in_size > 0, print(f'Invalid input dimensions {in_size}!')
            likelihood = GaussianLikelihood()
            super().__init__(None, None, likelihood)
            self.in_size = in_size
            self.cfg = cfg
            self.mean_module = ZeroMean()
            ard_num_dims = in_size if self.cfg.per_dim_lenghtscale else 1
            self.covar_module = ScaleKernel(RQKernel(ard_num_dims=ard_num_dims))

            # init hp priors
            # ATTENTION: do not name the HP priors "noise_prior", "outputscale_prior" or "lengthscale_prior"
            self.noise_var_prior = dist.Gamma(cfg.noise_var_concentration, cfg.noise_var_rate)
            self.outscale_prior = dist.Gamma(cfg.outscale_concentration, cfg.outscale_rate)
            self.lscale_prior = dist.Gamma(cfg.lscale_concentration_multiplier * in_size, cfg.lscale_rate)
            self.scale_mix_prior = dist.Gamma(cfg.scale_mix_concentration, cfg.scale_mix_rate)

            if param_dict is not None:
                self.load_param_dict(param_dict)
            else:
                # init kernel parameters
                self.init_hyperparams()

            self.posterior_hp = False
            self.select_hyperparameters(posterior_hp=True)

        def forward(self, x):
            mean = self.mean_module(x)
            covar = self.covar_module(x)
            return MultivariateNormal(mean, covar)

        def hyperparam_log_prior(self):
            log_prior = self.noise_var_prior.log_prob(self.likelihood.noise) + \
                        self.outscale_prior.log_prob(self.covar_module.outputscale) + \
                        self.lscale_prior.log_prob(self.covar_module.base_kernel.lengthscale).mean() + \
                        self.scale_mix_prior.log_prob(self.covar_module.base_kernel.alpha)
            return log_prior.squeeze()

        def init_hyperparams(self):
            noise_var = self.noise_var_prior.sample()
            self.posterior_noise = noise_var if noise_var > 1e-3 else torch.tensor(1e-3)
            self.posterior_outputscale = self.outscale_prior.sample()
            ls_size = Size((self.in_size,)) if self.cfg.per_dim_lenghtscale else Size((1,))
            self.posterior_lengthscale = self.lscale_prior.sample(ls_size)
            self.posterior_scale_mix = self.scale_mix_prior.sample()

        def select_hyperparameters(self, posterior_hp: bool):
            if posterior_hp:
                if not self.posterior_hp:
                    # load posterior HPs
                    self.likelihood.noise = self.posterior_noise
                    self.covar_module.outputscale = self.posterior_outputscale
                    self.covar_module.base_kernel.lengthscale = self.posterior_lengthscale
                    self.covar_module.base_kernel.alpha = self.posterior_scale_mix
                    self.posterior_hp = True
            else:
                # store posterior HPs if posterior HPs currently in use
                if self.posterior_hp:
                    self.posterior_noise = self.likelihood.noise.detach()
                    self.posterior_outputscale = self.covar_module.outputscale.detach()
                    self.posterior_lengthscale = self.covar_module.base_kernel.lengthscale.detach()
                    self.posterior_scale_mix = self.covar_module.base_kernel.alpha.detach()
                    self.posterior_hp = False

                # draw HPs from HP prior
                self.likelihood.noise = self.noise_var_prior.sample()
                self.covar_module.outputscale = self.outscale_prior.sample()
                ls_size = Size((self.in_size,)) if self.cfg.per_dim_lenghtscale else Size((1,))
                self.covar_module.base_kernel.lengthscale = self.lscale_prior.sample(ls_size)
                self.covar_module.base_kernel.alpha = self.scale_mix_prior.sample()

        def param_dict(self) -> Dict[str, Any]:
            params = {'in_size': self.in_size,
                      'posterior_noise': self.posterior_noise,
                      'posterior_outputscale': self.posterior_outputscale,
                      'posterior_lengthscale': self.posterior_lengthscale,
                      'posterior_scale_mix': self.posterior_scale_mix}
            return params

        def load_param_dict(self, param_dict):
            self.in_size = param_dict['in_size']
            self.posterior_noise = param_dict['posterior_noise'].float()
            self.posterior_outputscale = param_dict['posterior_outputscale'].float()
            self.posterior_lengthscale = param_dict['posterior_lengthscale'].float()
            self.posterior_scale_mix = param_dict['posterior_scale_mix'].float()

    ##########################################################################
    # Linear Kernel
    class GaussianProcessLinearKernel(ExactGP):
        posterior_noise: Tensor
        posterior_outputscale: Tensor
        posterior_offset: Tensor

        def __init__(self, cfg: GaussianProcessConfig, param_dict: Dict[str, Any] = None):
            likelihood = GaussianLikelihood()
            super().__init__(None, None, likelihood)
            self.mean_module = ConstantMean()
            self.covar_module = LinearKernel()

            # init hp priors
            # ATTENTION: do not name the HP priors "noise_prior", "outputscale_prior" or "lengthscale_prior"
            self.noise_var_prior = dist.Gamma(cfg.noise_var_concentration, cfg.noise_var_rate)
            self.outscale_prior = dist.Gamma(cfg.outscale_concentration, cfg.outscale_rate)
            self.offset_prior = dist.Normal(cfg.offset_loc, cfg.offset_scale)

            if param_dict is not None:
                self.load_param_dict(param_dict)
            else:
                # draw kernel parameters
                self.init_hyperparams()

            self.posterior_hp = False
            self.select_hyperparameters(posterior_hp=True)

        def forward(self, x):
            mean = self.mean_module(x)
            covar = self.covar_module(x)
            return MultivariateNormal(mean, covar)

        def hyperparam_log_prior(self):
            log_prior = self.noise_var_prior.log_prob(self.likelihood.noise) + \
                        self.outscale_prior.log_prob(self.covar_module.variance) + \
                        self.offset_prior.log_prob(self.mean_module.constant)
            return log_prior.squeeze()

        def init_hyperparams(self):
            noise_var = self.noise_var_prior.sample()
            self.posterior_noise = noise_var if noise_var > 1e-3 else torch.tensor(1e-3)
            self.posterior_outputscale = self.outscale_prior.sample()
            self.posterior_offset = self.offset_prior.sample()

        def select_hyperparameters(self, posterior_hp: bool):
            if posterior_hp:
                if not self.posterior_hp:
                    # load posterior HPs
                    self.likelihood.noise = self.posterior_noise
                    self.covar_module.variance = self.posterior_outputscale
                    self.mean_module.constant = self.posterior_offset
                    self.posterior_hp = True
            else:
                # store posterior HPs if posterior HPs currently in use
                if self.posterior_hp:
                    self.posterior_noise = self.likelihood.noise.detach()
                    self.posterior_outputscale = self.covar_module.variance.detach()
                    self.posterior_offset = self.mean_module.constant.detach()
                    self.posterior_hp = False

                # draw HPs from HP prior
                self.likelihood.noise = self.noise_var_prior.sample()
                self.covar_module.variance = self.outscale_prior.sample()
                self.mean_module.constant = self.offset_prior.sample()

        def param_dict(self) -> Dict[str, Any]:
            params = {'posterior_noise': self.posterior_noise,
                      'posterior_outputscale': self.posterior_outputscale,
                      'posterior_offset': self.posterior_offset}
            return params

        def load_param_dict(self, param_dict):
            self.posterior_noise = param_dict['posterior_noise'].float()
            self.posterior_outputscale = param_dict['posterior_outputscale'].float()
            self.posterior_offset = param_dict['posterior_offset'].float()

    ##########################################################################
    # Main-class functions
    ##########################################################################
    def __init__(self, in_size: int, static=False, cfg: GaussianProcessConfig = None,
                 param_dict: Dict[str, Any] = None):
        super().__init__(in_size)
        if param_dict is not None:
            self.load_param_dict(param_dict)
        else:
            # load config
            self.cfg = GaussianProcessConfig() if cfg is None else cfg

            # initialize likelihood and gp model
            if self.cfg.kernel == 'linear':
                self.gp = GaussianProcess.GaussianProcessLinearKernel(self.cfg)
            elif self.cfg.kernel == 'rq':
                self.gp = GaussianProcess.GaussianProcessRQKernel(in_size, self.cfg)
            elif self.cfg.kernel == 'additive-rq':
                self.gp = GaussianProcess.GaussianProcessAdditiveRQKernel(in_size, self.cfg)
            else:
                raise NotImplementedError

            self.static = static
            if static:
                self.init_as_static()

    def init_as_static(self):
        # generate support points and sample training targets from the GP prior
        sampling_factor = 1 if self.cfg.kernel == 'additive-rq' else self.in_size
        num_train = self.cfg.num_support_points * sampling_factor
        spread = self.cfg.support_max - self.cfg.support_min
        train_x = spread * torch.rand((num_train, self.in_size)) + self.cfg.support_min
        self.eval()
        with gpytorch.settings.prior_mode(True):
            f_dist = self.gp(train_x)
            y_dist = self.gp.likelihood(f_dist)
            train_y = y_dist.sample().detach()

        # update GP data
        self.set_data(train_x, train_y)

    def set_data(self, inputs: Tensor, targets: Tensor):
        inputs, targets, _ = self._check_args(inputs, targets)
        self.gp.set_train_data(inputs, targets, strict=False)

    def forward(self, inputs: Tensor, prior_mode=False):
        inputs, _, batch_shape = self._check_args(inputs)
        output_shape = batch_shape + (1,)

        self.eval()
        with gpytorch.settings.prior_mode(prior_mode):
            f_dist = self.gp(inputs)
        return f_dist.mean.view(output_shape)

    def sample(self, inputs: Tensor, prior_mode=False):
        inputs, _, batch_shape = self._check_args(inputs)
        output_shape = batch_shape + (1,)

        self.eval()
        with gpytorch.settings.prior_mode(prior_mode):
            f_dist = self.gp(inputs)
        y_dist = self.gp.likelihood(f_dist.mean) if self.static else self.gp.likelihood(f_dist)
        return y_dist.sample().view(output_shape)

    def mll(self, inputs: Tensor, targets: Tensor, prior_mode=False, reduce=True):
        inputs, targets, batch_shape = self._check_args(inputs, targets)
        output_shape = batch_shape[:-1]
        with gpytorch.settings.prior_mode(prior_mode):
            f_dist = self.gp(inputs)

        if self.static:
            y_dist = self.gp.likelihood(f_dist.mean)
            mlls = y_dist.log_prob(targets).squeeze(-1)
        else:
            y_dist = self.gp.likelihood(f_dist)
            mlls = y_dist.log_prob(targets)
        assert mlls.shape == output_shape, print(f'Invalid shape {mlls.shape}!')

        if reduce:
            return mlls.sum()
        return mlls

    def expected_noise_entropy(self, prior_mode: bool = False) -> Tensor:
        # use point estimate with the MAP variance
        posterior_hp = self.gp.posterior_hp
        self.gp.select_hyperparameters(not prior_mode)
        entropy = 0.5 * (2. * math.pi * self.gp.likelihood.noise * math.e).log().squeeze()
        self.gp.select_hyperparameters(posterior_hp)
        return entropy

    def param_dict(self) -> Dict[str, Any]:
        gp_param_dict = self.gp.param_dict()
        params = {'in_size': self.in_size,
                  'static': self.static,
                  'gp_param_dict': gp_param_dict,
                  'cfg_param_dict': self.cfg.param_dict()}

        if self.static:
            params['train_inputs'] = self.gp.train_inputs
            params['train_targets'] = self.gp.train_targets
        return params

    def load_param_dict(self, param_dict):
        self.cfg = GaussianProcessConfig()
        self.cfg.load_param_dict(param_dict['cfg_param_dict'])
        self.in_size = param_dict['in_size']
        self.static = param_dict['static']

        if self.cfg.kernel == 'linear':
            self.gp = GaussianProcess.GaussianProcessLinearKernel(self.cfg, param_dict=param_dict['gp_param_dict'])
        elif self.cfg.kernel == 'rq':
            self.gp = GaussianProcess.GaussianProcessRQKernel(self.in_size, self.cfg,
                                                              param_dict=param_dict['gp_param_dict'])
        elif self.cfg.kernel == 'additive-rq':
            self.gp = GaussianProcess.GaussianProcessAdditiveRQKernel(self.in_size, self.cfg,
                                                                      param_dict=param_dict['gp_param_dict'])
        else:
            raise NotImplementedError

        if self.static:
            train_inputs = param_dict['train_inputs'][0].float()
            train_targets = param_dict['train_targets'].float()
            self.set_data(train_inputs, train_targets)


class SharedDataGaussianProcess(Mechanism):
    ##########################################################################
    # Shared-data GP Linear Kernel
    ##########################################################################
    class SharedDataGPLinearKernel(ExactGP):
        def __init__(self, cfg: GaussianProcessConfig, node_to_dim_map: Dict[str, int] = None,
                     param_dict: Dict[str, Any] = None):
            assert node_to_dim_map is not None or param_dict is not None
            likelihood = GaussianLikelihood()
            super().__init__(None, None, likelihood)
            self.cfg = cfg

            if param_dict is not None:
                self.load_param_dict(param_dict)
            else:
                self.node_to_dim_map = node_to_dim_map
                self.means = ModuleDict()
                self.likelihoods = ModuleDict()
                self.kernels = ModuleDict()

            # init hp priors
            # ATTENTION: do not name the HP priors "noise_prior", "outputscale_prior"
            self.noise_var_prior = dist.Gamma(cfg.noise_var_concentration, cfg.noise_var_rate)
            self.outscale_prior = dist.Gamma(cfg.outscale_concentration, cfg.outscale_rate)
            self.offset_prior = dist.Normal(cfg.offset_loc, cfg.offset_scale)

        def init_kernel(self, key: str):
            likelihood = GaussianLikelihood()
            likelihood.eval()
            self.likelihoods[key] = likelihood

            _, parents = resolve_mechanism_key(key)
            active_dims = torch.LongTensor([self.node_to_dim_map[node] for node in parents])
            kernel = LinearKernel(active_dims=active_dims)
            kernel.eval()
            self.kernels[key] = kernel

            mean = ConstantMean()
            mean.eval()
            self.means[key] = mean

        def delete_kernel(self, key: str):
            self.likelihoods.pop(key)
            self.kernels.pop(key)
            self.means.pop(key)

        def delete_kernels(self, keys: List[str]):
            likelihoods = {key: lhd for key, lhd in self.likelihoods.items() if key not in keys}
            self.likelihoods = ModuleDict(likelihoods)

            kernels = {key: kernel for key, kernel in self.kernels.items() if key not in keys}
            self.kernels = ModuleDict(kernels)

            means = {key: mean for key, mean in self.means.items() if key not in keys}
            self.means = ModuleDict(means)

        def init_hyperparams(self, key: str):
            noise_var = self.noise_var_prior.sample()
            self.likelihoods[key].noise = noise_var if noise_var > 1e-3 else torch.tensor(1e-3)
            self.kernels[key].variance = self.outscale_prior.sample()
            self.means[key].constant = self.offset_prior.sample()

        def forward(self, x, key: str):
            mean = self.means[key](x)
            covar = self.kernels[key](x)
            return MultivariateNormal(mean, covar)

        def hyperparam_log_prior(self, key: str):
            log_prior = self.noise_var_prior.log_prob(self.likelihoods[key].noise) + \
                        self.outscale_prior.log_prob(self.kernels[key].variance) + \
                        self.offset_prior.log_prob(self.means[key].constant)
            return log_prior.squeeze()

        def param_dict(self) -> Dict[str, Any]:
            # ATTENTION: does not store the training data!
            likelihood_param_dict = {k: get_module_params(m) for k, m in self.likelihoods.items()}
            kernel_param_dict = {k: get_module_params(m) for k, m in self.kernels.items()}
            mean_param_dict = {k: get_module_params(m) for k, m in self.means.items()}

            params = {'node_to_dim_map': self.node_to_dim_map,
                      'likelihood_param_dict': likelihood_param_dict,
                      'kernel_param_dict': kernel_param_dict,
                      'mean_param_dict': mean_param_dict}
            return params

        def load_param_dict(self, param_dict):
            # ATTENTION: does not load the training data!
            self.means = ModuleDict()
            self.likelihoods = ModuleDict()
            self.kernels = ModuleDict()

            self.node_to_dim_map = param_dict['node_to_dim_map']
            for key in param_dict['likelihood_param_dict']:
                self.init_kernel(key)
                likelihood_param_vec = param_dict['likelihood_param_dict'][key].float()
                kernel_param_vec = param_dict['kernel_param_dict'][key].float()
                mean_param_vec = param_dict['mean_param_dict'][key].float()

                vector_to_parameters(likelihood_param_vec, self.likelihoods[key].parameters())
                vector_to_parameters(kernel_param_vec, self.kernels[key].parameters())
                vector_to_parameters(mean_param_vec, self.means[key].parameters())

        def get_parameters(self, keys: List[str] = None):
            if keys is None:
                return self.parameters()
            else:
                mean_params = [self.means[key].parameters() for key in keys if key in self.means]
                kernel_params = [self.kernels[key].parameters() for key in keys if key in self.kernels]
                likelihood_params = [self.likelihoods[key].parameters() for key in keys if key in self.likelihoods]
                return chain(*mean_params, *kernel_params, *likelihood_params)

    ##########################################################################
    # Shared-data GP  RQ-Kernel Model
    ##########################################################################
    class SharedDataGPRQKernel(ExactGP):
        def __init__(self, cfg: GaussianProcessConfig, node_to_dim_map: Dict[str, int] = None,
                     param_dict: Dict[str, Any] = None):
            assert node_to_dim_map is not None or param_dict is not None
            likelihood = GaussianLikelihood()
            super().__init__(None, None, likelihood)
            self.cfg = cfg
            self.mean_module = ZeroMean()

            if param_dict is not None:
                self.load_param_dict(param_dict)
            else:
                self.node_to_dim_map = node_to_dim_map
                self.likelihoods = ModuleDict()
                self.kernels = ModuleDict()
                self.lscale_priors: Dict[str, dist.Gamma] = {}

            # init hp priors
            # ATTENTION: do not name the HP priors "noise_prior", "outputscale_prior" or "lengthscale_prior"
            self.noise_var_prior = dist.Gamma(cfg.noise_var_concentration, cfg.noise_var_rate)
            self.outscale_prior = dist.Gamma(cfg.outscale_concentration, cfg.outscale_rate)
            self.scale_mix_prior = dist.Gamma(cfg.scale_mix_concentration, cfg.scale_mix_rate)

        def init_kernel(self, key: str):
            likelihood = GaussianLikelihood()
            likelihood.eval()
            self.likelihoods[key] = likelihood

            _, parents = resolve_mechanism_key(key)
            active_dims = torch.LongTensor([self.node_to_dim_map[node] for node in parents])
            ard_num_dims = len(parents) if self.cfg.per_dim_lenghtscale else 1
            kernel = ScaleKernel(RQKernel(active_dims=active_dims, ard_num_dims=ard_num_dims))
            kernel.eval()
            self.kernels[key] = kernel

            self.lscale_priors[key] = dist.Gamma(self.cfg.lscale_concentration_multiplier * len(parents),
                                                 self.cfg.lscale_rate)

        def delete_kernel(self, key: str):
            self.likelihoods.pop(key)
            self.kernels.pop(key)

        def delete_kernels(self, keys: List[str]):
            likelihoods = {key: lhd for key, lhd in self.likelihoods.items() if key not in keys}
            self.likelihoods = ModuleDict(likelihoods)

            kernels = {key: kernel for key, kernel in self.kernels.items() if key not in keys}
            self.kernels = ModuleDict(kernels)

        def init_hyperparams(self, key: str):
            noise_var = self.noise_var_prior.sample()
            self.likelihoods[key].noise = noise_var if noise_var > 1e-3 else torch.tensor(1e-3)
            self.kernels[key].outputscale = self.outscale_prior.sample()
            _, parents = resolve_mechanism_key(key)
            ls_size = Size((len(parents),)) if self.cfg.per_dim_lenghtscale else Size((1,))
            self.kernels[key].base_kernel.lengthscale = self.lscale_priors[key].sample(ls_size)
            self.kernels[key].base_kernel.alpha = self.scale_mix_prior.sample()

        def forward(self, x, key: str):
            mean = self.mean_module(x)
            covar = self.kernels[key](x)
            return MultivariateNormal(mean, covar)

        def hyperparam_log_prior(self, key: str):
            log_prior = self.noise_var_prior.log_prob(self.likelihoods[key].noise) + \
                        self.outscale_prior.log_prob(self.kernels[key].outputscale) + \
                        self.lscale_priors[key].log_prob(self.kernels[key].base_kernel.lengthscale).mean() + \
                        self.scale_mix_prior.log_prob(self.kernels[key].base_kernel.alpha)
            return log_prior.squeeze()

        def param_dict(self) -> Dict[str, Any]:
            # ATTENTION: does not store the training data!
            likelihood_param_dict = {k: get_module_params(m) for k, m in self.likelihoods.items()}
            kernel_param_dict = {k: get_module_params(m) for k, m in self.kernels.items()}

            params = {'node_to_dim_map': self.node_to_dim_map,
                      'likelihood_param_dict': likelihood_param_dict,
                      'kernel_param_dict': kernel_param_dict}
            return params

        def load_param_dict(self, param_dict):
            # ATTENTION: does not load the training data!
            self.likelihoods = ModuleDict()
            self.kernels = ModuleDict()
            self.lscale_priors: Dict[str, dist.Gamma] = {}

            self.node_to_dim_map = param_dict['node_to_dim_map']
            for key in param_dict['likelihood_param_dict']:
                self.init_kernel(key)
                likelihood_param_vec = param_dict['likelihood_param_dict'][key].float()
                kernel_param_vec = param_dict['kernel_param_dict'][key].float()

                vector_to_parameters(likelihood_param_vec, self.likelihoods[key].parameters())
                vector_to_parameters(kernel_param_vec, self.kernels[key].parameters())

        def get_parameters(self, keys: List[str] = None):
            if keys is None:
                return self.parameters()
            else:
                kernel_params = [self.kernels[key].parameters() for key in keys if key in self.kernels]
                likelihood_params = [self.likelihoods[key].parameters() for key in keys if key in self.likelihoods]
                return chain(*kernel_params, *likelihood_params)

    ##########################################################################
    # Main-class functions
    ##########################################################################
    def __init__(self, in_size: int, node_to_dim_map: Dict[str, int] = None, cfg: GaussianProcessConfig = None,
                 param_dict: Dict[str, Any] = None):
        assert node_to_dim_map is not None or param_dict is not None
        super().__init__(in_size)

        if param_dict is not None:
            self.load_param_dict(param_dict)
        else:
            # load config
            self.cfg = GaussianProcessConfig() if cfg is None else cfg

            # initialize likelihood and gp model
            if self.cfg.kernel == 'linear':
                self.gp = SharedDataGaussianProcess.SharedDataGPLinearKernel(self.cfg, node_to_dim_map)
            elif self.cfg.kernel == 'rq':
                self.gp = SharedDataGaussianProcess.SharedDataGPRQKernel(self.cfg, node_to_dim_map)
            else:
                raise NotImplementedError

        self.eval()

    def set_data(self, inputs: Tensor, targets: Tensor):
        inputs, targets, _ = self._check_args(inputs, targets)
        self.gp.set_train_data(inputs, targets, strict=False)

    def init_kernel(self, key: str):
        self.gp.init_kernel(key)
        self.gp.init_hyperparams(key)

    def delete_kernel(self, key: str):
        self.gp.delete_kernel(key)

    def delete_kernels(self, keys: List[str]):
        self.gp.delete_kernels(keys)

    def init_hyperparams(self, key: str):
        self.gp.init_hyperparams(key)

    def get_keys(self):
        return list(self.gp.kernels.keys())

    def exists(self, key: str):
        return key in self.gp.kernels and key in self.gp.likelihoods

    def activate(self, key: str):
        if not self.exists(key):
            self.init_kernel(key)

        self.gp.likelihood = self.gp.likelihoods[key]
        self.gp._clear_cache()

    def hyperparam_log_prior(self, key: str):
        return self.gp.hyperparam_log_prior(key)

    def get_parameters(self, keys: List[str] = None):
        return self.gp.get_parameters(keys)

    def forward(self, inputs: Tensor, key: str, prior_mode=False):
        inputs, _, batch_shape = self._check_args(inputs)
        output_shape = batch_shape + (1,)

        self.activate(key)
        with gpytorch.settings.prior_mode(prior_mode):
            f_dist = self.gp(inputs, key=key)
        return f_dist.mean.view(output_shape)

    def sample(self, inputs: Tensor, key: str, prior_mode=False):
        inputs, _, batch_shape = self._check_args(inputs)
        output_shape = batch_shape + (1,)

        self.activate(key)
        with gpytorch.settings.prior_mode(prior_mode):
            f_dist = self.gp(inputs, key=key)
        y_dist = self.gp.likelihoods[key](f_dist)
        return y_dist.sample().view(output_shape)

    def mll(self, inputs: Tensor, targets: Tensor, key: str, prior_mode=False, reduce=True):
        inputs, targets, batch_shape = self._check_args(inputs, targets)
        output_shape = batch_shape[:-1]
        self.activate(key)
        with gpytorch.settings.prior_mode(prior_mode):
            f_dist = self.gp(inputs, key=key)
            y_dist = self.gp.likelihoods[key](f_dist)

        mlls = y_dist.log_prob(targets)
        assert mlls.shape == output_shape, print(f'Invalid shape {mlls.shape}!')

        if reduce:
            return mlls.sum()
        return mlls

    def expected_noise_entropy(self, key: str) -> Tensor:
        # use point estimate with the MAP variance
        self.activate(key)
        entropy = 0.5 * (2. * math.pi * self.gp.likelihoods[key].noise * math.e).log().squeeze()
        return entropy

    def param_dict(self) -> Dict[str, Any]:
        gp_param_dict = self.gp.param_dict()
        params = {'in_size': self.in_size,
                  'gp_param_dict': gp_param_dict,
                  'cfg_param_dict': self.cfg.param_dict()}
        return params

    def load_param_dict(self, param_dict):
        self.cfg = GaussianProcessConfig()
        self.cfg.load_param_dict(param_dict['cfg_param_dict'])

        self.in_size = param_dict['in_size']

        # initialize likelihood and gp model
        if self.cfg.kernel == 'linear':
            self.gp = SharedDataGaussianProcess.SharedDataGPLinearKernel(self.cfg,
                                                                         param_dict=param_dict['gp_param_dict'])
        elif self.cfg.kernel == 'rq':
            self.gp = SharedDataGaussianProcess.SharedDataGPRQKernel(self.cfg, param_dict=param_dict['gp_param_dict'])
        else:
            raise NotImplementedError


class AdditiveSigmoids(Mechanism):
    '''
    Static mechanism for generating ground truth environments as suggested in Buhlmann, P., Peters, J., and Ernest, J.
    "CAM: Causal additive models, high-dimensional order search and penalized regression." Annals of Statistics, 2014.
    '''
    noise: Tensor
    outscales: Tensor
    lengthscales: Tensor
    offsets: Tensor

    def __init__(self, in_size: int, cfg: AdditiveSigmoidsConfig = None, param_dict: Dict[str, Any] = None):
        super().__init__(in_size)
        if param_dict is not None:
            self.load_param_dict(param_dict)
        else:
            # load config
            self.cfg = AdditiveSigmoidsConfig() if cfg is None else cfg

            # init hp priors
            # ATTENTION: do not name the HP priors "noise_prior"
            self.noise_var_prior = dist.Gamma(self.cfg.noise_var_concentration, self.cfg.noise_var_rate)
            self.outscale_prior = dist.Gamma(self.cfg.outscale_concentration, self.cfg.outscale_rate)
            self.lscale_prior = dist.Uniform(self.cfg.lscale_lower, self.cfg.lscale_upper)
            self.offset_prior = dist.Uniform(self.cfg.offset_lower, self.cfg.offset_upper)
            self.likelihood = GaussianLikelihood()

            # initialize likelihood and parameters
            self.init_hyperparams()

    def init_hyperparams(self):
        noise_var = self.noise_var_prior.sample()
        self.noise = noise_var if noise_var > 1e-3 else torch.tensor(1e-3)
        self.outscales = self.outscale_prior.sample(Size((self.in_size,)))
        if torch.rand(1).round():
            self.outscales = -self.outscales
        self.lengthscales = self.lscale_prior.sample(Size((self.in_size,)))
        self.offsets = self.offset_prior.sample(Size((self.in_size,)))

    def forward(self, inputs: Tensor, prior_mode=False):
        inputs, _, batch_shape = self._check_args(inputs)
        output_shape = batch_shape + (1,)

        self.eval()
        tmp = self.lengthscales * (inputs + self.offsets)
        components = self.outscales * tmp / (1. + tmp.abs())
        outputs = components.sum(dim=-1, keepdim=True)
        assert outputs.shape == output_shape, print(outputs.shape)
        return outputs

    def sample(self, inputs: Tensor, prior_mode=False):
        inputs, _, batch_shape = self._check_args(inputs)
        output_shape = batch_shape + (1,)

        means = self(inputs)
        y_dist = self.likelihood(means)
        return y_dist.sample().view(output_shape)

    def mll(self, inputs: Tensor, targets: Tensor, reduce=True):
        inputs, targets, batch_shape = self._check_args(inputs, targets)
        output_shape = batch_shape[:-1]

        means = self(inputs)
        y_dist = self.likelihood(means)
        mlls = y_dist.log_prob(targets).squeeze(-1)
        assert mlls.shape == output_shape, print(f'Invalid shape {mlls.shape} vs. {output_shape}!')

        if reduce:
            return mlls.sum()
        return mlls

    def param_dict(self) -> Dict[str, Any]:
        params = {'in_size': self.in_size,
                  'noise': self.noise,
                  'outscales': self.outscales,
                  'lengthscales': self.lengthscales,
                  'offsets': self.offsets,
                  'cfg_param_dict': self.cfg.param_dict()}

        return params

    def load_param_dict(self, param_dict):
        self.cfg = AdditiveSigmoidsConfig()
        self.cfg.load_param_dict(param_dict['cfg_param_dict'])
        self.in_size = param_dict['in_size']
        self.noise = param_dict['noise']
        self.outscales = param_dict['outscales']
        self.lengthscales = param_dict['lengthscales']
        self.offsets = param_dict['offsets']
