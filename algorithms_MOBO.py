#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 13:20:32 2025

@author: kudva.7
"""

from dataclasses import dataclass
from graph_utils import Graph
from gpytorch.constraints import Interval
import math
from typing import Callable, Optional
from botorch.models.transforms import Standardize
import torch
from torch import Tensor
from gp_network_utils import GaussianProcessNetwork
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
import time
from botorch.optim import optimize_acqf
from botorch.fit import fit_gpytorch_mll
import copy
from botorch.acquisition import qExpectedImprovement, qLogExpectedImprovement, qUpperConfidenceBound, UpperConfidenceBound, LogExpectedImprovement
from botorch.acquisition.analytic import AcquisitionFunction
from botorch.acquisition.objective import PosteriorTransform
from botorch.generation import MaxPosteriorSampling
from botorch.models.model import Model
#from botorch.sampling.pathwise.posterior_samplers import get_matheron_path_model
from botorch.utils.transforms import t_batch_mode_transform
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.acquisition.objective import MCAcquisitionObjective
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.acquisition import MCAcquisitionFunction
from botorch.utils.transforms import concatenate_pending_points, t_batch_mode_transform
from botorch.optim.initializers import gen_batch_initial_conditions
from torch.quasirandom import SobolEngine
from botorch.models import SingleTaskGP
import sys
from TSacquisition_functions import ThompsonSampleFunctionNetwork, ThompsonSampleAcq
from gpytorch.mlls import ExactMarginalLogLikelihood
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

torch.set_default_dtype(torch.float64)

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.indicators.hv import Hypervolume
from pymoo.visualization.scatter import Scatter
import numpy as np

"""
Code associated with qEHVI

"""
def initialize_model_listGP(train_x, train_obj, bounds, alg = "qEHVI"):
    # define models for objective and constraint
    if alg == "qEHVI":
        train_x = normalize(train_x, bounds)        
    
    models = []
    for i in range(train_obj.shape[-1]):
        train_y = train_obj[..., i : i + 1]
        train_yvar = torch.ones(train_obj[:,i].unsqueeze(-1).shape)*1e-4
        models.append(
            SingleTaskGP(
                train_x, train_y, train_yvar, outcome_transform=Standardize(m=1)
            )
        )
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    return mll, model




def optimize_qehvi_and_get_observation(model, 
                                       train_x,  
                                       sampler,
                                       ref_point,
                                       bounds:Tensor,
                                       BATCH_SIZE,
                                       NUM_RESTARTS,
                                       RAW_SAMPLES,
                                       MC_SAMPLES):
    
    """Optimizes the qEHVI acquisition function, and returns a new candidate """
    
    # partition non-dominated space into disjoint rectangles
    with torch.no_grad():
        pred = model.posterior(normalize(train_x, bounds)).mean

    partitioning = FastNondominatedPartitioning(
        ref_point=ref_point,
        Y=pred,
    )
    acq_func = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,
        partitioning=partitioning,
        sampler=sampler,
    )
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
        sequential=True,
    )
    # observe new values
    new_x = unnormalize(candidates.detach(), bounds=bounds)
    
    return new_x


def qEHVI(train_x_qehvi:Tensor,
          train_obj_qehvi:Tensor,
          ref_point: Tensor,
          bounds:Tensor,
          BATCH_SIZE:int = 1,
          NUM_RESTARTS:int = 10,
          RAW_SAMPLES:int = 512,
          MC_SAMPLES:int = 128):
    
    """
    This is used to wrap all the codes associated with the qEHVI acquisition function
    """
    # Generate modellistGP
    mll_qehvi, model_qehvi = initialize_model_listGP(train_x_qehvi, train_obj_qehvi, bounds)
    
    # fit the models
    fit_gpytorch_mll(mll_qehvi)

    # define the qEI and qNEI acquisition modules using a QMC sampler
    qehvi_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))

    # optimize acquisition functions and get new observations

    new_x_qehvi = optimize_qehvi_and_get_observation(model_qehvi, 
                                                     train_x_qehvi, 
                                                     qehvi_sampler,
                                                     ref_point,
                                                     bounds,
                                                     BATCH_SIZE,
                                                     NUM_RESTARTS,
                                                     RAW_SAMPLES,
                                                     MC_SAMPLES)
    
    return new_x_qehvi






"""
Code associated with qPOTS and MOBONS
"""
# TODO: 1) The max distance from existing points - done
# 2) NSGA accepting TS  - done
# 3) TS generator for qPOTS and MOBONS - todo


def max_min_gamma(surrogate_pareto_x: Tensor, 
                  train_x: Tensor) -> torch.Tensor:
    """
    Selects the next query point x_{n+1} from the surrogate Pareto front that maximizes the minimum 
    Euclidean distance to the existing training points.
    
    Args:
        surrogate_pareto_x (torch.Tensor): Candidate points from the surrogate Pareto front, shape [N, d].
        train_x (torch.Tensor): Existing evaluated points, shape [M, d].
    
    Returns:
        torch.Tensor: The selected next query point x_{n+1}, shape [1, d].
    """
    # Compute pairwise Euclidean distances between each x* in surrogate_pareto_x and each x_i in train_x
    distances = torch.cdist(surrogate_pareto_x, train_x, p=2)  # Shape: [N, M]

    # For each candidate x* in surrogate_pareto_x, find the minimum distance to the training set
    min_distances = torch.min(distances, dim=1)[0]  # Shape: [N]

    # Select the x* that has the maximum of these minimum distances
    best_index = torch.argmax(min_distances)
    
    return surrogate_pareto_x[best_index].unsqueeze(0)  # Return as a single-element tensor of shape [1, d]




def MO_TS_observation(train_x:Tensor, # x data 
                    train_obj:Tensor, # corresponding objective function value(s)
                    model, 
                    bounds:Tensor,
                    g: Graph = None, # Optional graph component which will be required for MOBONS
                    alg: str = "qPOTS"):
    
    """Optimizes the qPOTS acquisition function, and returns a new candidate """
    
    # Structure the acquisition function for easy unpacking   
    
    
    if alg == "qPOTS":
        TS_acq = []
        
        for i in range(train_obj.shape[-1]):
            train_y = train_obj[..., i : i + 1]
            TS_acq_instance = ThompsonSampleAcq(model = model.models[i],
                                                train_x = train_x,
                                                train_y = train_y)
            TS_acq.append(TS_acq_instance)
            
        
        # Define function that produces a pair of thompson samples
        def TS_obj(X):            
            X_tensor = torch.tensor(X, dtype=torch.double).unsqueeze(1)
            ts_vals = []            
            # Begin loop to evaluate the function
            for i in range(train_obj.shape[-1]):
                ts_val = TS_acq[i](X_tensor).detach().numpy()
                ts_vals.append(ts_val)          
            return np.column_stack(ts_vals)
        
        n_objective = train_obj.shape[-1]
        
    elif alg == "MOBONS":
        # Define the thompson sampling acquisition function
        TS_acq = ThompsonSampleFunctionNetwork(model)
        n_objective = g.ref_point.shape[-1]
        
        # Define function that produces a pair of thompson samples
        def TS_obj(X):            
            X_tensor = torch.tensor(X, dtype=torch.double).unsqueeze(1)
            with torch.no_grad():
                ts_vals = TS_acq(X_tensor)             
            return ts_vals.detach().numpy()
        
    else:
        print("Enter a valid method to run")
        sys.exit()
    
    
    # Define the multi-objective optimization problem
    # Note to self: Be careful while formulating the MOBONS problem
        
    class MultiObjectiveThompsonSampling(Problem):
        def __init__(self):
            super().__init__(
                n_var = bounds.shape[1],  # Number of decision variables
                n_obj= n_objective,  # Number of objectives
                n_constr=0,  # Number of constraints
                xl=bounds[0].numpy(),  # Lower bounds
                xu=bounds[1].numpy(),  # Upper bounds
            )
        
        def _evaluate(self, X, out, *args, **kwargs):
            out["F"] = TS_obj(X)  # Evaluate objective values
    
    # Define the multi-objective problem as the above function
    problem = MultiObjectiveThompsonSampling()
    
    # Define the settings that are used in the algorithm
    algorithm = NSGA2(
            pop_size=100,
            sampling=get_sampling("real_random"),
            crossover=get_crossover("real_sbx", prob=0.9, eta=15),
            mutation=get_mutation("real_pm", eta=20),
            eliminate_duplicates=True
        )
    
    # Solve the problem
    res = minimize(problem,
                   algorithm,
                   termination=("n_gen", 200),  # Terminate after 200 generations
                   seed=1,
                   verbose=False)
    
    surrogate_pareto_x = torch.tensor(res.X, dtype = torch.double)
    
    new_x = max_min_gamma(surrogate_pareto_x, train_x) 
    
    return new_x




def qPOTS(train_x:Tensor,
          train_y:Tensor,
          bounds:Tensor):
    
    # Use the same function as qEHVI for building the model
    mll, model = initialize_model_listGP(train_x, train_y, bounds, alg = "qPOTS")
    
    # Train the model based of logLikelihood
    fit_gpytorch_mll(mll)
    
    # Obtain the next value to evaluate
    new_x = MO_TS_observation(train_x = train_x, # x data 
                        train_obj = train_y, # corresponding objective function value(s)
                        model = model, 
                        bounds = bounds,
                        alg = "qPOTS")  
    
    return new_x


def MOBONS(train_x:Tensor,
          train_y:Tensor,
          bounds:Tensor,
          g:Graph):
    
    # Generate the model and train
    model = GaussianProcessNetwork(train_X = train_x,
                                   train_Y = train_y,
                                   dag = g)
    
    # Generate the Function Network TS
    new_x = MO_TS_observation(train_x = train_x, # x data 
                        train_obj = train_y, # corresponding objective function value(s)
                        model = model, 
                        bounds = bounds,
                        g = g,
                        alg = "MOBONS")  
    
    return new_x

"""
MOBO runner for all the baselines

"""


def MultiObj_BayesOpt(Ninit: int,
                  T: int,
                  g: Graph,
                  function_network: Callable,
                  seed: int = 1,
                  alg: str = "Random"):   
    
    data = {}   
    time_iter = []
    input_dim = g.nx
    bounds = torch.tensor([[0.]*(input_dim),[1.]*(input_dim)])

    torch.manual_seed(seed = (seed + 1)*2000)
    
    # Initialize the algorithms using random sampling
    if alg == "Random":
        X = torch.rand(Ninit + T, g.nx)
        Y = g.objective_function(function_network(X))        
        data = {"X": X, "Y":Y, "Y_true":Y, "time": time_iter}        
        return data 
        
    elif alg in ['qEHVI', 'qPOTS']:
        X = torch.rand(Ninit, g.nx)
        Y = g.objective_function(function_network(X))
        
    else:
        X = torch.rand(Ninit, g.nx)
        Y = function_network(X)
        
    # Begin the iterative process here
    for t in range(T):
        #print(a)  
        
        print('Iteration Number', t)
        
        if alg in ['qEHVI','qPOTS']:
            
            
            if alg == 'qEHVI':
                X_new = qEHVI(train_x_qehvi = X,
                              train_obj_qehvi = Y,
                              ref_point = -1*g.ref_point,
                              bounds = bounds)
            else:
                X_new = qPOTS(train_x = X, 
                              train_y = Y,
                              bounds = bounds)
            
            Y_new = g.objective_function(function_network(X_new))
            
        elif alg == 'MOBONS':
            X_new = MOBONS(train_x = X, 
                          train_y = Y,
                          bounds = bounds,
                          g = g)
            
            
            Y_new = function_network(X_new)          
            
        
        print('New sample X is at', X_new)
        print('Value of the Objective functions at this point is', Y_new)
        X = torch.cat([X,X_new])
        Y = torch.cat([Y,Y_new])       
    
    
    if alg == 'qEHVI': # Reverse the negation at the objective function level
        Y = -1*Y
    
    if alg is not "MOBONS":
        Y_true = Y
    else:
        Y_true = g.objective_function(Y) 
    
    data = {"X": X, "Y":Y, "Y_true": Y_true, "time": time_iter}
    
    return data      
    
    
    
    


if __name__ == "__main__":   
    
    print("Nothing is being tested")
































