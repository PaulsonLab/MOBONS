#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 09:52:22 2025

@author: kudva.7
"""

import os
import torch


tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
SMOKE_TEST = os.environ.get("SMOKE_TEST")

from botorch.test_functions.multi_objective import BraninCurrin
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.utils.transforms import unnormalize, normalize
from botorch.utils.sampling import draw_sobol_samples
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.utils.transforms import unnormalize, normalize
from botorch.utils.sampling import draw_sobol_samples
from botorch.optim.optimize import optimize_acqf, optimize_acqf_list
from botorch.acquisition.objective import GenericMCObjective
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.utils.sampling import sample_simplex
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
import time
import warnings

from botorch import fit_gpytorch_mll
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from botorch.utils.multi_objective.pareto import is_non_dominated



def generate_initial_data(n=6):
    # generate training data
    train_x = draw_sobol_samples(bounds=problem.bounds, n=n, q=1).squeeze(1)
    train_obj_true = problem(train_x)
    train_obj = train_obj_true + torch.randn_like(train_obj_true) * NOISE_SE
    return train_x, train_obj, train_obj_true


def initialize_model(train_x, train_obj):
    # define models for objective and constraint
    train_x = normalize(train_x, problem.bounds)
    models = []
    for i in range(train_obj.shape[-1]):
        train_y = train_obj[..., i : i + 1]
        train_yvar = torch.full_like(train_y, NOISE_SE[i] ** 2)
        models.append(
            SingleTaskGP(
                train_x, train_y, train_yvar, outcome_transform=Standardize(m=1)
            )
        )
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    return mll, model




def optimize_qehvi_and_get_observation(model, train_x, train_obj, sampler):
    """Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""
    # partition non-dominated space into disjoint rectangles
    with torch.no_grad():
        pred = model.posterior(normalize(train_x, problem.bounds)).mean
    partitioning = FastNondominatedPartitioning(
        ref_point=problem.ref_point,
        Y=pred,
    )
    acq_func = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=problem.ref_point,
        partitioning=partitioning,
        sampler=sampler,
    )
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=standard_bounds,
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
        sequential=True,
    )
    # observe new values
    new_x = unnormalize(candidates.detach(), bounds=problem.bounds)
    new_obj_true = problem(new_x)
    new_obj = new_obj_true + torch.randn_like(new_obj_true) * NOISE_SE
    return new_x, new_obj, new_obj_true




if __name__ == '__main__':
    
    from matplotlib.cm import ScalarMappable
    import matplotlib.pyplot as plt
    
    warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    problem = BraninCurrin(negate=True).to(**tkwargs)
    
    NOISE_SE = torch.tensor([15.19, 0.63], **tkwargs)
    
    N_BATCH = 20 if not SMOKE_TEST else 5
    MC_SAMPLES = 128 if not SMOKE_TEST else 16
    
    BATCH_SIZE = 1
    NUM_RESTARTS = 10 if not SMOKE_TEST else 2
    RAW_SAMPLES = 512 if not SMOKE_TEST else 4

    standard_bounds = torch.zeros(2, problem.dim, **tkwargs)
    standard_bounds[1] = 1
    
    verbose = True
    
    hvs_qehvi, hvs_random = [], []
    
    # call helper functions to generate initial training data and initialize model
    train_x_qparego, train_obj_qparego, train_obj_true_qparego = generate_initial_data(
        n=2 * (problem.dim + 1)
    )
   
    train_x_qehvi, train_obj_qehvi, train_obj_true_qehvi = (
        train_x_qparego,
        train_obj_qparego,
        train_obj_true_qparego,
    )
    
    train_x_random, train_obj_random, train_obj_true_random = (
        train_x_qparego,
        train_obj_qparego,
        train_obj_true_qparego,
    )
    
    mll_qehvi, model_qehvi = initialize_model(train_x_qehvi, train_obj_qehvi)

    
    # compute hypervolume
    bd = DominatedPartitioning(ref_point=problem.ref_point, Y=train_obj_true_qparego)
    volume = bd.compute_hypervolume().item()
    
    hvs_qehvi.append(volume)
    hvs_random.append(volume)
    
    # run N_BATCH rounds of BayesOpt after the initial random batch
    for iteration in range(1, N_BATCH + 1):
    
        t0 = time.monotonic()
    
        # fit the models
        fit_gpytorch_mll(mll_qehvi)
    
        # define the qEI and qNEI acquisition modules using a QMC sampler
        qehvi_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))
    
        # optimize acquisition functions and get new observations

        new_x_qehvi, new_obj_qehvi, new_obj_true_qehvi = optimize_qehvi_and_get_observation(
            model_qehvi, train_x_qehvi, train_obj_qehvi, qehvi_sampler
        )
        
        new_x_random, new_obj_random, new_obj_true_random = generate_initial_data(
            n=BATCH_SIZE
        )
    
        # update training points
    
        train_x_qehvi = torch.cat([train_x_qehvi, new_x_qehvi])
        train_obj_qehvi = torch.cat([train_obj_qehvi, new_obj_qehvi])
        train_obj_true_qehvi = torch.cat([train_obj_true_qehvi, new_obj_true_qehvi])
    
        train_x_random = torch.cat([train_x_random, new_x_random])
        train_obj_random = torch.cat([train_obj_random, new_obj_random])
        train_obj_true_random = torch.cat([train_obj_true_random, new_obj_true_random])
    
        # update progress
        for hvs_list, train_obj in zip(
            (hvs_random, hvs_qehvi),
            (   train_obj_true_random,
                train_obj_true_qehvi
            ),
        ):
            # compute hypervolume
            bd = DominatedPartitioning(ref_point=problem.ref_point, Y=train_obj)
            volume = bd.compute_hypervolume().item()
            hvs_list.append(volume)
    
        # reinitialize the models so they are ready for fitting on next iteration
        # Note: we find improved performance from not warm starting the model hyperparameters
        # using the hyperparameters from the previous iteration
        mll_qehvi, model_qehvi = initialize_model(train_x_qehvi, train_obj_qehvi)
    
        t1 = time.monotonic()
    
        if verbose:
            print(
                f"\nBatch {iteration:>2}: Hypervolume (random, qEHVI) = "
                f"({hvs_random[-1]:>4.2f}, {hvs_qehvi[-1]:>4.2f}), "
                f"time = {t1-t0:>4.2f}.",
                end="",
            )
        else:
            print(".", end="")
            
            
            
            
            
        # fig, axes = plt.subplots(1, 2, figsize=(23, 7), sharex=True, sharey=True)
        # algos = ["Sobol", "qEHVI"]
        # cm = plt.get_cmap("viridis")
        
        # batch_number = torch.cat(
        #     [
        #         torch.zeros(2 * (problem.dim + 1)),
        #         torch.arange(1, N_BATCH + 1).repeat(BATCH_SIZE, 1).t().reshape(-1),
        #     ]
        # ).numpy()
        # for i, train_obj in enumerate(
        #     (
        #         train_obj_true_random,
        #         train_obj_true_qehvi,
        #     )
        # ):
        #     sc = axes[i].scatter(
        #         train_obj[:, 0].cpu().numpy(),
        #         train_obj[:, 1].cpu().numpy(),
        #         c=batch_number,
        #         alpha=0.8,
        #     )
        #     axes[i].set_title(algos[i])
        #     axes[i].set_xlabel("Objective 1")
        # axes[0].set_ylabel("Objective 2")
        # norm = plt.Normalize(batch_number.min(), batch_number.max())
        # sm = ScalarMappable(norm=norm, cmap=cm)
        # sm.set_array([])
        # fig.subplots_adjust(right=0.9)
        # cbar_ax = fig.add_axes([0.93, 0.15, 0.01, 0.7])
        # cbar = fig.colorbar(sm, cax=cbar_ax)
        # cbar.ax.set_title("Iteration")