#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 16:24:52 2024
Contains the GP posterior and acquisition function
@author: kudva.7
"""
from __future__ import annotations
import torch
from botorch import fit_gpytorch_model
from typing import Any, Tuple
from botorch.models.model import Model
from botorch.models import SingleTaskGP
from botorch.posteriors.posterior import Posterior
from botorch.models.transforms import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor
from gpytorch.kernels import MaternKernel, ScaleKernel

torch.set_default_dtype(torch.double)


class GaussianProcessNetwork(Model):
    """
    This class is the exact copy of 
    https://github.com/RaulAstudillo06/BOFN/tree/main based on 
    "Bayesian Optimization of Function Networks", published in NeurIPS 2021.
    """
    
    def __init__(self, train_X, train_Y, dag, train_Yvar=None, node_GPs=None, normalization_constant_lower=None, normalization_constant_upper=None) -> None:
        r"""
        """
        self.train_X = train_X
        self.train_Y = train_Y
        self.dag = dag
        self.n_nodes = dag.get_n_nodes()
        self.root_nodes = dag.get_root_nodes()
        self.active_input_indices = self.dag.active_input_indices
        self.train_Yvar = train_Yvar
        self.noise_var = 1e-5
        
        # Writing my own properties
        self.n_outs = int(self.train_Y.size()[1])
        #######################################  
        
                        
        if node_GPs is not None:
            self.node_GPs = node_GPs
            self.normalization_constant_lower = normalization_constant_lower
            self.normalization_constant_upper = normalization_constant_upper
        else:   
            self.node_GPs = [None for k in range(self.n_nodes)]
            self.node_mlls = [None for k in range(self.n_nodes)]
            self.normalization_constant_lower = [[None for j in range(len(self.dag.get_parent_nodes(k)))] for k in range(self.n_nodes)]
            self.normalization_constant_upper = [[None for j in range(len(self.dag.get_parent_nodes(k)))] for k in range(self.n_nodes)]
            self.normalization_constant_lower_std = [[None for j in range(len(self.dag.get_parent_nodes(k)))] for k in range(self.n_nodes)]
            self.normalization_constant_upper_std = [[None for j in range(len(self.dag.get_parent_nodes(k)))] for k in range(self.n_nodes)]
    
            
            for k in self.root_nodes:
                if self.active_input_indices is not None:
                    train_X_node_k = train_X[..., self.active_input_indices[k]]
                else:
                    train_X_node_k = train_X
                train_Y_node_k = train_Y[..., [k]]
                
                
                if self.dag.custom_hyperparameters:
                    self.node_GPs[k] = SingleTaskGP(train_X=train_X_node_k, train_Y=train_Y_node_k, train_Yvar=torch.ones(train_Y_node_k.shape) * self.dag.noise_level, outcome_transform=Standardize(m=1))
                    self.node_mlls[k] = ExactMarginalLogLikelihood(self.node_GPs[k].likelihood, self.node_GPs[k])
                    fit_gpytorch_model(self.node_mlls[k])
                    self.node_GPs[k].covar_module.outputscale = self.dag.output_scale
                    self.node_GPs[k].covar_module.base_kernel.lengthscale = self.dag.length_scale
                else:
                    # Covariance module
                    #covar_module = ScaleKernel(MaternKernel(nu=0.5, ard_num_dims=train_X_node_k.size()[1]))    
                    self.node_GPs[k] = SingleTaskGP(train_X=train_X_node_k, train_Y=train_Y_node_k, train_Yvar=torch.ones(train_Y_node_k.shape) * 1e-6, outcome_transform=Standardize(m=1))
                    #self.node_GPs[k] = SingleTaskGP(train_X=train_X_node_k, train_Y=train_Y_node_k,covar_module= covar_module, train_Yvar=torch.ones(train_Y_node_k.shape) * 1e-6, outcome_transform=Standardize(m=1))
                    #self.node_GPs[k] = SingleTaskGP(train_X=train_X_node_k, train_Y=train_Y_node_k,covar_module= covar_module, outcome_transform=Standardize(m=1))                      
                    self.node_mlls[k] = ExactMarginalLogLikelihood(self.node_GPs[k].likelihood, self.node_GPs[k])
                    fit_gpytorch_model(self.node_mlls[k])
                
            for k in range(self.n_nodes):
                if self.node_GPs[k] is None:
                    aux = train_Y[..., self.dag.get_parent_nodes(k)].clone()
                    for j in range(len(self.dag.get_parent_nodes(k))):
                        self.normalization_constant_lower[k][j] = torch.min(aux[..., j])
                        self.normalization_constant_upper[k][j] = torch.max(aux[..., j])
                        self.normalization_constant_lower_std[k][j] = torch.min((aux[..., j] - aux[..., j].std())/aux[..., j].mean())
                        self.normalization_constant_upper_std[k][j] = torch.max((aux[..., j] - aux[..., j].std())/aux[..., j].mean())
                        aux[..., j] = (aux[..., j] - self.normalization_constant_lower[k][j])/(self.normalization_constant_upper[k][j] - self.normalization_constant_lower[k][j])
                    train_X_node_k = torch.cat([train_X[..., self.active_input_indices[k]], aux], -1)
                    train_Y_node_k = train_Y[..., [k]]
                    aux_model =  SingleTaskGP(train_X=train_X_node_k, train_Y=train_Y_node_k, train_Yvar=torch.ones(train_Y_node_k.shape) * 1e-6, outcome_transform=Standardize(m=1, batch_shape=torch.Size([])))  
                    batch_shape = aux_model._aug_batch_shape
                    
                    if self.dag.custom_hyperparameters:
                        self.node_GPs[k] = SingleTaskGP(train_X=train_X_node_k, train_Y=train_Y_node_k, train_Yvar=torch.ones(train_Y_node_k.shape) * self.dag.noise_level, outcome_transform=Standardize(m=1, batch_shape=torch.Size([])))
                        self.node_mlls[k] = ExactMarginalLogLikelihood(self.node_GPs[k].likelihood, self.node_GPs[k])
                        fit_gpytorch_model(self.node_mlls[k])
                        self.node_GPs[k].covar_module.outputscale = self.dag.output_scale
                        self.node_GPs[k].covar_module.base_kernel.lengthscale = self.dag.length_scale
                    else:                 
                        # Covariance Module
                        #covar_module = ScaleKernel(MaternKernel(nu=0.5, ard_num_dims=train_X_node_k.size()[1]))
                        self.node_GPs[k] = SingleTaskGP(train_X=train_X_node_k, train_Y=train_Y_node_k, train_Yvar=torch.ones(train_Y_node_k.shape) * 1e-6, outcome_transform=Standardize(m=1, batch_shape=torch.Size([])))
                        #self.node_GPs[k] = SingleTaskGP(train_X=train_X_node_k, train_Y=train_Y_node_k,covar_module= covar_module, train_Yvar=torch.ones(train_Y_node_k.shape) * 1e-6, outcome_transform=Standardize(m=1, batch_shape=torch.Size([])))
                        #self.node_GPs[k] = SingleTaskGP(train_X=train_X_node_k, train_Y=train_Y_node_k, outcome_transform=Standardize(m=1, batch_shape=torch.Size([])))
                        self.node_mlls[k] = ExactMarginalLogLikelihood(self.node_GPs[k].likelihood, self.node_GPs[k])
                        fit_gpytorch_model(self.node_mlls[k])
                        
    def get_train_data_node_k(self, k):
        """
        Retrives training data for the TS

        """
        train_X = self.train_X
        train_Y = self.train_Y
        aux = train_Y[..., self.dag.get_parent_nodes(k)].clone()
        for j in range(len(self.dag.get_parent_nodes(k))):
            self.normalization_constant_lower[k][j] = torch.min(aux[..., j])
            self.normalization_constant_upper[k][j] = torch.max(aux[..., j])
            aux[..., j] = (aux[..., j] - self.normalization_constant_lower[k][j])/(self.normalization_constant_upper[k][j] - self.normalization_constant_lower[k][j])
        train_X_node_k = torch.cat([train_X[..., self.active_input_indices[k]], aux], -1)
        train_Y_node_k = train_Y[..., [k]]
        train_Yvar_node_k = torch.ones(train_Y_node_k.shape) * self.noise_var
        return train_X_node_k, train_Y_node_k, train_Yvar_node_k
                
    def posterior(self, X: Tensor, posterior_transform=None, observation_noise=False) -> MultivariateNormalNetwork:
        r"""Computes the posterior over model outputs at the provided points.
        Args:
            X: A `(batch_shape) x q x d`-dim Tensor, where `d` is the dimension
                of the feature space and `q` is the number of points considered
                jointly.
            observation_noise: If True, add the observation noise from the
                likelihood to the posterior. If a Tensor, use it directly as the
                observation noise (must be of shape `(batch_shape) x q`).
        Returns:
            A `GPyTorchPosterior` object, representing a batch of `b` joint
            distributions over `q` points. Includes observation noise if
            specified.
        """
        return MultivariateNormalNetwork(self.node_GPs, self.dag, X, self.normalization_constant_lower, self.normalization_constant_upper)
    
    def forward(self, x: Tensor) -> MultivariateNormalNetwork:
        return MultivariateNormalNetwork(self.node_GPs, self.dag, x,  self.normalization_constant_lower, self.normalization_constant_upper)
    
    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any) -> Model:
        r"""Condition the model on new observations.
        Args:
            X: A `batch_shape x n' x d`-dim Tensor, where `d` is the dimension of
                the feature space, `n'` is the number of points per batch, and
                `batch_shape` is the batch shape (must be compatible with the
                batch shape of the model).
            Y: A `batch_shape' x n' x m`-dim Tensor, where `m` is the number of
                model outputs, `n'` is the number of points per batch, and
                `batch_shape'` is the batch shape of the observations.
                `batch_shape'` must be broadcastable to `batch_shape` using
                standard broadcasting semantics. If `Y` has fewer batch dimensions
                than `X`, it is assumed that the missing batch dimensions are
                the same for all `Y`.
        Returns:
            A `Model` object of the same type, representing the original model
            conditioned on the new observations `(X, Y)` (and possibly noise
            observations passed in via kwargs).
        """
        fantasy_models = [None for k in range(self.n_nodes)]

        for k in self.root_nodes:
            if self.active_input_indices is not None:
                X_node_k = X[..., self.active_input_indices[k]]
            else:
                X_node_k = X
            Y_node_k = Y[..., [k]]
            fantasy_models[k] = self.node_GPs[k].condition_on_observations(X_node_k, Y_node_k, noise=torch.ones(Y_node_k.shape[1:]) * 1e-6)
        
        for k in range(self.n_nodes):
            if fantasy_models[k] is None:
                aux = Y[..., self.dag.get_parent_nodes(k)].clone()
                for j in range(len(self.dag.get_parent_nodes(k))):
                    aux[..., j] = (aux[..., j] - self.normalization_constant_lower[k][j])/(self.normalization_constant_upper[k][j] - self.normalization_constant_lower[k][j])
                aux_shape = [aux.shape[0]] + [1] * X[..., self.active_input_indices[k]].ndim
                X_aux = X[..., self.active_input_indices[k]].unsqueeze(0).repeat(*aux_shape)
                X_node_k = torch.cat([X_aux, aux], -1)
                Y_node_k = Y[..., [k]]
                fantasy_models[k] = self.node_GPs[k].condition_on_observations(X_node_k, Y_node_k, noise=torch.ones(Y_node_k.shape[1:]) * 1e-6)

        return GaussianProcessNetwork(dag=self.dag, train_X=X, train_Y=Y, active_input_indices=self.active_input_indices, node_GPs=fantasy_models, normalization_constant_lower=self.normalization_constant_lower, normalization_constant_upper=self.normalization_constant_upper)
        
    @property   
    def num_outputs(self):
        a = 1
        return a 

        
class MultivariateNormalNetwork(Posterior):
    def __init__(self, node_GPs, dag, X, normalization_constant_lower=None, normalization_constant_upper=None):
        self.node_GPs = node_GPs
        self.dag = dag
        self.n_nodes = dag.get_n_nodes()
        self.root_nodes = dag.get_root_nodes()
        self.X = X
        self.active_input_indices = self.dag.active_input_indices
        self.normalization_constant_lower = normalization_constant_lower
        self.normalization_constant_upper = normalization_constant_upper
        #self.posterior_transform = None
    
    @property
    def mean_sigma(self):
        self.mean, self.variance =  self._get_mean_var()
        return self.mean, self.variance.sqrt()
        
    @property
    def device(self) -> torch.device:
        r"""The torch device of the posterior."""
        return "cpu"
    
    @property
    def dtype(self) -> torch.dtype:
        r"""The torch dtype of the posterior."""
        return torch.double
    
    @property
    def event_shape(self) -> torch.Size:
        r"""The event shape (i.e. the shape of a single sample) of the posterior."""
        shape = list(self.X.shape)
        shape[-1] = self.n_nodes
        shape = torch.Size(shape)
        return shape
    
    @property
    def base_sample_shape(self) -> torch.Size:
        r"""The base shape of the base samples expected in `rsample`.
   
        Informs the sampler to produce base samples of shape
        `sample_shape x base_sample_shape`.
        """
        shape = torch.Size(list([1,1,self.n_nodes]))
        return shape
   
    @property
    def batch_range(self) -> Tuple[int, int]:
        r"""The t-batch range.
   
        This is used in samplers to identify the t-batch component of the
        `base_sample_shape`. The base samples are expanded over the t-batches to
        provide consistency in the acquisition values, i.e., to ensure that a
        candidate produces same value regardless of its position on the t-batch.
        """
        return (0, -1)
    
    def rsample_from_base_samples(self, sample_shape: torch.Size, base_samples: Tensor) -> Tensor:
        return self.rsample(sample_shape, base_samples)
   
    def rsample(self, sample_shape=torch.Size(), base_samples=None):
        #t0 =  time.time()
        nodes_samples = torch.empty(sample_shape + self.event_shape)
        nodes_samples = nodes_samples.double()
        nodes_samples_available = [False for k in range(self.n_nodes)]
        for k in self.root_nodes:
            #t0 =  time.time()
            if self.active_input_indices is not None:
                X_node_k = self.X[..., self.active_input_indices[k]]
            else:
                X_node_k = self.X
            multivariate_normal_at_node_k = self.node_GPs[k].posterior(X_node_k)
            if base_samples is not None:
                nodes_samples[..., k] = multivariate_normal_at_node_k.rsample(sample_shape, base_samples=base_samples[..., [k]])[..., 0]
            else:
                nodes_samples[..., k] = multivariate_normal_at_node_k.rsample(sample_shape)[..., 0]
            nodes_samples_available[k] = True
            #t1 = time.time()
            #print('Part A of the code took: ' + str(t1 - t0))
   
        while not all(nodes_samples_available):
            for k in range(self.n_nodes):
                parent_nodes = self.dag.get_parent_nodes(k)
                if not nodes_samples_available[k] and all([nodes_samples_available[j] for j in parent_nodes]):
                    #t0 =  time.time()
                    parent_nodes_samples_normalized = nodes_samples[..., parent_nodes].clone()
                    for j in range(len(parent_nodes)):
                        parent_nodes_samples_normalized[..., j] = (parent_nodes_samples_normalized[..., j] - self.normalization_constant_lower[k][j])/(self.normalization_constant_upper[k][j] - self.normalization_constant_lower[k][j])
                    X_node_k = self.X[..., self.active_input_indices[k]]
                    aux_shape = [sample_shape[0]] + [1] * X_node_k.ndim
                    X_node_k = X_node_k.unsqueeze(0).repeat(*aux_shape)
                    X_node_k = torch.cat([X_node_k, parent_nodes_samples_normalized], -1)
                    multivariate_normal_at_node_k = self.node_GPs[k].posterior(X_node_k)
                    if base_samples is not None:
                        #print(torch.sqrt(multivariate_normal_at_node_k.variance).shape)
                        #print(torch.flatten(base_samples[..., k]).shape)
                        my_aux = torch.sqrt(multivariate_normal_at_node_k.variance)
                        #print(my_aux.ndim)
                        if my_aux.ndim == 4:
                            nodes_samples[...,k] = (multivariate_normal_at_node_k.mean + torch.einsum('abcd,a->abcd', torch.sqrt(multivariate_normal_at_node_k.variance), torch.flatten(base_samples[..., k])))[..., 0]
                        elif my_aux.ndim == 5:
                            nodes_samples[...,k] = (multivariate_normal_at_node_k.mean + torch.einsum('abcde,a->abcde', torch.sqrt(multivariate_normal_at_node_k.variance), torch.flatten(base_samples[..., k])))[..., 0]
                        else:
                            print('error')
                    else:
                        nodes_samples[..., k] = multivariate_normal_at_node_k.rsample()[0, ..., 0]
                    nodes_samples_available[k] = True
                    #t1 = time.time()
                    #print('Part B of the code took: ' + str(t1 - t0))
        #t1 = time.time()
        #print('Taking this sample took: ' + str(t1 - t0))
        return nodes_samples
    
    def _get_mean_var(self):
        """
        For analytic acuisition function
        
        Returns
        -------
        nodes_samples : Mean
        nodes_samples_varience : Var

        """                
        # One each for mean and variance
        if self.X.dim() == 2:
            nodes_samples = torch.empty(self.event_shape).unsqueeze(1)
            nodes_samples_var = torch.empty(self.event_shape).unsqueeze(1)
        else:
            nodes_samples = torch.empty(self.event_shape).unsqueeze(2)
            nodes_samples_var = torch.empty(self.event_shape).unsqueeze(2)
        nodes_samples = nodes_samples.double()
        nodes_samples_var = nodes_samples_var.double()
        
        nodes_samples_available = [False for k in range(self.n_nodes)]
        for k in self.root_nodes:
            #t0 =  time.time()
            if self.active_input_indices is not None:
                X_node_k = self.X[..., self.active_input_indices[k]]
            else:
                X_node_k = self.X
            multivariate_normal_at_node_k = self.node_GPs[k].posterior(X_node_k)
            nodes_samples[..., k] = multivariate_normal_at_node_k.mean
            nodes_samples_var[..., k] = multivariate_normal_at_node_k.variance
            nodes_samples_available[k] = True
            #t1 = time.time()
            #print('Part A of the code took: ' + str(t1 - t0))
  
        while not all(nodes_samples_available):
            for k in range(self.n_nodes): 
                parent_nodes = self.dag.get_parent_nodes(k)
                if not nodes_samples_available[k] and all([nodes_samples_available[j] for j in parent_nodes]):
                    #t0 =  time.time()
                    parent_nodes_samples_normalized = nodes_samples[..., parent_nodes].clone()
                    for j in range(len(parent_nodes)):
                        parent_nodes_samples_normalized[..., j] = (parent_nodes_samples_normalized[..., j] - self.normalization_constant_lower[k][j])/(self.normalization_constant_upper[k][j] - self.normalization_constant_lower[k][j])
                        X_node_k = self.X[..., self.active_input_indices[k]]
                        try:
                            X_node_k = torch.cat([X_node_k, parent_nodes_samples_normalized],-1)
                        except:
                            X_node_k = torch.cat([X_node_k, parent_nodes_samples_normalized.squeeze(1)],-1)
                            
                        multivariate_normal_at_node_k = self.node_GPs[k].posterior(X_node_k)
                        nodes_samples[..., k] = multivariate_normal_at_node_k.mean
                        nodes_samples_var[..., k] = multivariate_normal_at_node_k.variance
                    
                    nodes_samples_available[k] = True
                    #t1 = time.time()
                    #print('Part B of the code took: ' + str(t1 - t0))
        #t1 = time.time()
        #print('Taking this sample took: ' + str(t1 - t0))
        return nodes_samples, nodes_samples_var # Mean and varience
    

    
    



