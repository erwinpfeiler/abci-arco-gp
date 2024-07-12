from typing import Callable

import gpytorch
import torch
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.optim import Adam


class SurrogateGP(gpytorch.models.ExactGP):
    def __init__(self, train_x=None, train_y=None):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super(SurrogateGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel())

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


def gp_ucb(utility: Callable[[torch.Tensor], torch.Tensor], bounds: torch.Tensor):
    assert bounds.numel() == 2, print(bounds.shape)
    lower_bnd = bounds.squeeze()[0]
    upper_bnd = bounds.squeeze()[1]

    # parameters (still hardcoded)
    num_total_candidates = 15
    num_initial_candidates = 5
    surrogate_lr = 1e-1  # learning rate for surrogate GP
    num_opt_steps = 50  # number of optimisation steps for the surrogate GP
    beta = 5.  # ucb trade-off parameter
    num_grid_points = 100  # number of grid points for the optimisation of the acquisition function (=ucb)

    # init surrogate model
    surrogate_gp = SurrogateGP()
    optimizer = Adam(surrogate_gp.parameters(), lr=surrogate_lr)
    xgrid = torch.linspace(lower_bnd, upper_bnd, steps=num_grid_points).view(-1, 1, 1)

    # generate initial candidates
    candidates = torch.rand((num_initial_candidates, 1)) * (upper_bnd - lower_bnd) + lower_bnd
    with torch.no_grad():
        utilities = torch.tensor([utility(candidates[i, 0]).squeeze() for i in range(num_initial_candidates)])

    # search for better candidates
    try:
        for cidx in range(num_total_candidates - num_initial_candidates):
            # fit surrogate GP
            surrogate_gp.set_train_data(candidates, utilities, strict=False)
            mll = ExactMarginalLogLikelihood(surrogate_gp.likelihood, surrogate_gp)
            mll.train()

            for tidx in range(num_opt_steps):
                optimizer.zero_grad()
                predictions = surrogate_gp(candidates)
                loss = -mll(predictions, utilities)
                loss.backward()
                optimizer.step()

            # evaluate acquisition function
            surrogate_gp.eval()
            prediction = surrogate_gp(xgrid)
            ucb = prediction.mean + beta * prediction.variance

            # determine most promising candidate
            shuffled_idc = torch.randperm(ucb.numel()).long()
            argmax = ucb[shuffled_idc].argmax()
            candidate = xgrid[shuffled_idc[argmax]].view(1, 1)

            # record candidate and objective
            candidates = torch.cat((candidates, candidate), dim=0)
            with torch.no_grad():
                utilities = torch.cat((utilities, utility(candidate.squeeze()).view(-1)), dim=0)

    except Exception as e:
        print('Exception occured when running GP UCB:')
        print(e)
        print('Continuing with the best candidate from ', candidates)
        print('with utilities ', utilities)

    # return best candidate and objective values
    best_idx = utilities.argmax()
    best_candidate = candidates[best_idx].squeeze()
    best_utility = utilities[best_idx]
    return best_candidate, best_utility


def random_search(utility: callable, bounds: torch.Tensor, num_candidates=10):
    assert bounds.shape == (2, 1), print(bounds.shape)

    # generate initial candidates
    candidate_list = [torch.rand(1) * (bounds[1] - bounds[0]) + bounds[0] for _ in range(num_candidates)]
    utility_list = [utility(candidate.item()).squeeze() for candidate in candidate_list]

    # return best candidate and objective values
    best_candidate = candidate_list[torch.stack(utility_list).argmax().item()]
    best_utility = torch.stack(utility_list).max().item()
    return best_candidate, best_utility


def grid_search(utility: callable, bounds: torch.Tensor, num_candidates=10):
    assert bounds.shape == (2, 1), print(bounds.shape)

    # generate initial candidates
    candidate_list = torch.linspace(bounds[0].item(), bounds[1].item(), num_candidates)
    utility_list = [utility(candidate.item()).squeeze() for candidate in candidate_list]

    # return best candidate and objective values
    best_candidate = candidate_list[torch.stack(utility_list).argmax().item()]
    best_utility = torch.stack(utility_list).max().item()
    return best_candidate, best_utility
