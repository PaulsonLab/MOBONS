#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 10:08:07 2024

@author: kudva.7
"""

import numpy as np
import torch
from math import pi
from botorch.models.model import Model
from botorch.utils.transforms import concatenate_pending_points, t_batch_mode_transform
from torch import Tensor
from gp_network_utils import GaussianProcessNetwork
from botorch.sampling.normal import SobolQMCNormalSampler
from typing import List, Optional
from botorch.sampling.base import MCSampler
from botorch.acquisition import AnalyticAcquisitionFunction, MCAcquisitionFunction
from botorch.acquisition.objective import MCAcquisitionObjective
from botorch.optim import optimize_acqf
from botorch.optim.initializers import gen_batch_initial_conditions
from botorch.utils.safe_math import smooth_amax, smooth_amin
from graph_utils import Graph
from botorch.utils.safe_math import smooth_amax, smooth_amin
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
#from botorch.acquisition import FixedFeatureAcquisitionFunction




class PosteriorMean(MCAcquisitionFunction):
    """
    """

    def __init__(
        self,
        model: Model,
        sampler: Optional[MCSampler] = None,
        X_pending: Optional[Tensor] = None,
    ) -> None:
        r"""
        """
        super().__init__(
            model=model, sampler=sampler, objective=model.dag.objective_function, X_pending=X_pending
        )

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""
        """
        posterior = self.model.posterior(X)
        samples = self.sampler(posterior)
        obj = self.objective(samples)
        obj = obj.mean(dim=0)[..., 0]
        return obj

class EfficientThompsonSampler():
    def __init__(self, model, num_of_multistarts = 1, num_of_bases = 1024, num_of_samples = 1, train_x = None, train_y = None):
        '''
        Implementation of 'Efficiently Sampling From Gaussian Process Posteriors' by Wilson et al. (2020). It allows
        us to create approximate samples of the GP posterior, which we can optimise using gradient methods. We do this
        to generate candidates using Thompson Sampling. Link to the paper: https://arxiv.org/pdf/2002.09309.pdf .
        '''
        # GP model
        self.model = model
        # inputs
        if train_x == None:
            if type(self.model.train_x) == torch.Tensor:
                self.train_x = self.model.train_x
            else:
                self.train_x = torch.tensor(self.model.train_x)
            self.x_dim = torch.tensor(self.train_x.shape[1])
            self.train_y = self.model.train_y
            self.num_of_train_inputs = self.model.train_x.shape[0]
        else:
            if type(train_x) == torch.Tensor:
                self.train_x = train_x
            else:
                self.train_x = torch.tensor(train_x)
            self.x_dim = torch.tensor(self.train_x.shape[1])
            self.train_y = train_y  
            self.num_of_train_inputs = self.train_x.shape[0]            
        
        # thompson sampling parameters
        self.num_of_multistarts = num_of_multistarts
        self.num_of_bases = num_of_bases
        self.num_of_samples = num_of_samples
        # optimisation parameters
        self.learning_rate = 0.01
        self.num_of_epochs = 10 * self.x_dim
        # obtain the kernel parameters
        self.sigma = self.model.likelihood.noise[0].item() # assumes fixed noise value
        self.lengthscale = self.model.covar_module.base_kernel.lengthscale.detach().float()
        self.outputscale = self.model.covar_module.outputscale.item()
        # obtain the kernel
        self.kernel = self.model.covar_module
        # define the Knn matrix
        with torch.no_grad():
            self.Knn = self.kernel(self.train_x)
            self.Knn = self.Knn.evaluate()
            # precalculate matrix inverse
            self.inv_mat = torch.inverse(self.Knn + self.sigma * torch.eye(self.num_of_train_inputs))

        self.create_fourier_bases()
        self.calculate_phi()

    def create_fourier_bases(self):
        # sample thetas
        self.thetas = torch.randn(size = (self.num_of_bases, self.x_dim)) / self.lengthscale
        # sample biases
        self.biases = torch.rand(self.num_of_bases) * 2 * pi

    def create_sample(self):
        # sample weights
        self.weights = torch.randn(size = (self.num_of_samples, self.num_of_bases)).float()

    def calculate_phi(self):
        '''
        From the paper, we are required to calculate a matrix which includes the evaluation of the training set, X_train,
        at the fourier frequencies. This function calculates that matrix, Phi.
        '''
        # we take the dot product by element-wise multiplication followed by summation
        thetas = self.thetas.repeat(self.num_of_train_inputs, 1, 1)
        prod = thetas * self.train_x.unsqueeze(1)
        dot = torch.sum(prod, axis = -1)
        # add biases and take cosine to obtain fourier representations
        ft = torch.cos(dot + self.biases.unsqueeze(0))
        # finally, multiply by corresponding constants (see paper)
        self.Phi = (self.outputscale * np.sqrt(2 / self.num_of_bases) * ft).float()

    def calculate_V(self):
        '''
        From the paper, to give posterior updates we need to calculate the vector V. Since we are doing multiple samples
        at the same time, V will be a matrix. We can pre-calculate it, since its value does not depend on the query locations.
        '''
        # multiply phi matrix by weights
        # PhiW: num_of_train x num_of_samples
        PhiW = torch.matmul(self.Phi, self.weights.T)
        # add noise (see paper)
        PhiW = PhiW + torch.randn(size = PhiW.shape) * self.sigma
        # subtract from training outputs
        mat1 = self.train_y - PhiW
        # calculate V matrix by premultiplication by inv_mat = (K_nn + I_n*sigma)^{-1}
        # V: num_of_train x num_of_samples
        self.V = torch.matmul(self.inv_mat, mat1)

    def calculate_fourier_features(self, x):
        '''
        Calculate the Fourier Features evaluated at some input x
        '''
        # evaluation using fourier features
        self.posterior_update(x)
        # calculate the dot product between the frequencies, theta, and the new query points
        dot = x.matmul(self.thetas.T)
        # calculate the fourier frequency by adding bias and cosine
        ft = torch.cos(dot + self.biases.unsqueeze(0))
        # apply the normalising constants and return the output
        return self.outputscale * np.sqrt(2 / self.num_of_bases) * ft

    def sample_prior(self, x):
        '''
        Create a sample form the prior, evaluate it at x
        '''
        if type(x) is not torch.Tensor:
            x = torch.tensor(x)
        # calculate the fourier features evaluated at the query points
        out1 = self.calculate_fourier_features(x)
        # extend the weights so that we can use element wise multiplication
        weights = self.weights.repeat(self.num_of_multistarts, 1, 1)
        # return the prior
        return torch.sum(weights * out1, axis = -1)

    def posterior_update(self, x):
        '''
        Calculate the posterior update at a location x
        '''
        if type(x) is not torch.Tensor:
            x = torch.tensor(x)
        # x: num_of_multistarts x num_of_samples x dim
        self.calculate_V() # can probably pre-calculate this
        # train x: num_of_multistarts x num_of_train x dim
        train_x = self.train_x.repeat(self.num_of_multistarts, 1, 1)
        # z: num_of_multistarts x num_of_train x num_of_samples
        # z: kernel evaluation between new query points and training set
        z = self.kernel(train_x, x)
        z = z.evaluate()
        # we now repeat V the number of times necessary so that we can use element-wise multiplication
        V = self.V.repeat(self.num_of_multistarts, 1, 1)
        out = z * V
        return out.sum(axis = 1) # we return the sum across the number of training point, as per the paper

    def query_sample(self, x):
        '''
        Query the sample at a location
        '''
        prior = self.sample_prior(x)
        update = self.posterior_update(x)
        return prior + update

    def generate_candidates(self):
        '''
        Generate the Thompson Samples, this function optimizes the samples.
        '''
        # we are always working on [0, 1]^d
        bounds = torch.stack([torch.zeros(self.x_dim), torch.ones(self.x_dim)])
        # initialise randomly - there is definitely much better ways of doing this
        X = torch.rand(self.num_of_multistarts, self.num_of_samples, self.x_dim)
        X.requires_grad = True
        # define optimiser
        optimiser = torch.optim.Adam([X], lr = self.learning_rate)

        for _ in range(self.num_of_epochs):
            # set zero grad
            optimiser.zero_grad()
            # evaluate loss and backpropagate
            losses = - self.query_sample(X)
            loss = losses.sum()
            loss.backward()
            # take step
            optimiser.step()

            # make sure we are still within the bounds
            for j, (lb, ub) in enumerate(zip(*bounds)):
                X.data[..., j].clamp_(lb, ub) # need to do this on the data not X itself
        # check the final evaluations
        final_evals = self.query_sample(X)
        # choose the best one for each sample
        best_idx = torch.argmax(final_evals, axis = 0)
        # return the best one for each sample, without gradients
        X_out = X[best_idx, range(0, self.num_of_samples), :]
        return X_out.detach()
    
class GPNetworkThompsonSampler():
    def __init__(self, model, num_of_bases = 1024):
        self.model = model
        self.n_nodes = self.model.n_nodes
        self.node_TSs = [None for k in range(self.n_nodes)]
        self.outcome_transforms = [None for k in range(self.n_nodes)]
        for k in range(self.n_nodes):
            node_GP_k = self.model.node_GPs[k]
            node_GP_k.train_x, node_GP_k.train_y, node_GP_k.train_yvar = self.model.get_train_data_node_k(k)
            # need to transform data to account for the "outcome transform"
            node_GP_k.train_y, node_GP_k.train_yvar = node_GP_k.outcome_transform(node_GP_k.train_y, node_GP_k.train_yvar)
            node_TS_k = EfficientThompsonSampler(node_GP_k)
            self.node_TSs[k] = node_TS_k
            self.outcome_transforms[k] = node_GP_k.outcome_transform

    def create_sample(self):
        for k in range(self.n_nodes): self.node_TSs[k].create_sample()

    def query_sample(self, X):
        sample_shape = list(X.shape)
        sample_shape[-1] = self.n_nodes
        sample_shape = torch.Size(sample_shape)
        nodes_samples = torch.empty(sample_shape)
        nodes_samples = nodes_samples.double()
        nodes_samples_available = [False for k in range(self.n_nodes)]
        for k in self.model.root_nodes:
            if self.model.active_input_indices is not None:
                X_node_k = X[..., self.model.active_input_indices[k]]
            else:
                X_node_k = X
            outcome_transform_at_k = self.outcome_transforms[k]
            nodes_samples[..., k] = outcome_transform_at_k.untransform(self.node_TSs[k].query_sample(X_node_k))[0]
            nodes_samples_available[k] = True

        while not all(nodes_samples_available):
            for k in range(self.n_nodes):
                parent_nodes = self.model.dag.get_parent_nodes(k)
                if not nodes_samples_available[k] and all([nodes_samples_available[j] for j in parent_nodes]):
                    parent_nodes_samples_normalized = nodes_samples[..., parent_nodes].clone()
                    for j in range(len(parent_nodes)):
                        parent_nodes_samples_normalized[..., j] = (parent_nodes_samples_normalized[..., j] - self.model.normalization_constant_lower[k][j])/(self.model.normalization_constant_upper[k][j] - self.model.normalization_constant_lower[k][j])
                    X_node_k = X[..., self.model.active_input_indices[k]]
                    X_node_k = torch.cat([X_node_k, parent_nodes_samples_normalized], -1)
                    outcome_transform_at_k = self.outcome_transforms[k]
                    nodes_samples[..., k] = outcome_transform_at_k.untransform(self.node_TSs[k].query_sample(X_node_k))[0]
                    nodes_samples_available[k] = True

        return nodes_samples
    
"""

Acquisition functions

"""  
class ThompsonSampleAcq(AnalyticAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        train_x: Tensor,
        train_y: Tensor,
    ) -> None:
        # we use the AcquisitionFunction constructor, since that of
        # AnalyticAcquisitionFunction performs some validity checks that we don't want here
        kwargs = {"model": model}
        
        super(AnalyticAcquisitionFunction, self).__init__(**kwargs)
        
        self.TS_sampler = EfficientThompsonSampler(model, train_x = train_x, train_y = train_y)
        self.TS_sampler.create_sample()

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate the TS on the candidate set X

        Args:
            X: A `(b) x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.

        Returns:
            A `(b)`-dim Tensor of Upper Confidence Bound values at the given
                design points `X`.
        """
        objective_at_X = self.TS_sampler.query_sample(X)

        if len(objective_at_X.shape) == 2:
            objective_at_X = objective_at_X.squeeze(-1)
        elif len(objective_at_X.shape) == 0:
            objective_at_X = objective_at_X.unsqueeze(0)
        return objective_at_X


class ThompsonSampleFunctionNetwork(AnalyticAcquisitionFunction):
    def __init__(
        self,
        model: Model,
    ) -> None:
        # we use the AcquisitionFunction constructor, since that of
        # AnalyticAcquisitionFunction performs some validity checks that we don't want here
        kwargs = {"model": model}
        
        super(AnalyticAcquisitionFunction, self).__init__(**kwargs)
        
        self.ts_network = GPNetworkThompsonSampler(model)
        self.ts_network.create_sample()
        self.network_to_objective_transform = model.dag.objective_function

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate the TS on the candidate set X

        Args:
            X: A `(b) x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.

        Returns:
            A `(b)`-dim Tensor of Upper Confidence Bound values at the given
                design points `X`.
        """
        network_at_X = self.ts_network.query_sample(X)
        objective_at_X = self.network_to_objective_transform(network_at_X.squeeze(1))
        if len(objective_at_X.shape) == 2:
            objective_at_X = objective_at_X.squeeze(-1)
        elif len(objective_at_X.shape) == 0:
            objective_at_X = objective_at_X.unsqueeze(0)
        return objective_at_X
    

      
    

if __name__ == "__main__":
    from Objective_FN import function_network_examples as SFN
    
   
    function_network, g =  SFN('dropwave')    
    #Generate initial random data
    
    x_init = torch.rand(10, g.nx)
    y_init = function_network(x_init)    
    
    test_function_network = False
    
    if test_function_network:
        model = GaussianProcessNetwork(train_X=x_init, train_Y=y_init, dag = g)
        # Acqusition
        acquisition_function = ThompsonSampleFunctionNetwork(model)
        # Sampler
        qmc_sampler = SobolQMCNormalSampler(torch.Size([128]))
        posterior_mean_function = PosteriorMean(
            model=model,
            sampler=qmc_sampler)
        
        # batch_initial_conditions = gen_batch_initial_conditions(
        #         acq_function=acquisition_function,
        #         bounds=torch.tensor([[0. for i in range(g.nx)], [1. for i in range(g.nx)]]), 
        #         q=3,
        #         num_restarts= 100,
        #         raw_samples=1000,
        #     )
    
        x_star, _ = optimize_acqf(
            acq_function=acquisition_function,
            bounds= torch.tensor([[0. for i in range(g.nx)], [1. for i in range(g.nx)]]),
            q=1 ,
            num_restarts=1,
            raw_samples=100,
            #batch_initial_conditions= batch_initial_conditions,
            options={"batch_limit": 5},
        )
    
    else:
        
        model = SingleTaskGP(x_init, g.objective_function(y_init).unsqueeze(-1))  
        mlls = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mlls)
        # Acqusition
        acquisition_function = ThompsonSampleAcq(model, train_x = x_init, train_y = g.objective_function(y_init).unsqueeze(-1))
        # Sampler
        
        x_star, _ = optimize_acqf(
            acq_function=acquisition_function,
            bounds= torch.tensor([[0. for i in range(g.nx)], [1. for i in range(g.nx)]]),
            q=1 ,
            num_restarts=1,
            raw_samples=100,
            #batch_initial_conditions= batch_initial_conditions,
            options={"batch_limit": 5},
        )
    