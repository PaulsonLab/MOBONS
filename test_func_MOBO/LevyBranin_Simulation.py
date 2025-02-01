#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 11:06:08 2025

@author: kudva.7
"""

import torch
from torch import Tensor
import numpy as np
from pytorch_pymoo_wrapper import PyTorchWrapperProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.indicators.hv import Hypervolume
from pymoo.visualization.scatter import Scatter

# Define the multi-objective problem with Levy and Branin using PyTorch
class LevyBraninProblem:
    def __init__(self, negate: bool = False):
        self.n_var = 2  # Number of decision variables
        self.n_obj = 2  # Number of objectives
        
        self.xl = torch.tensor([0 for i in range(self.n_var)])
        self.xu = torch.tensor([1 for i in range(self.n_var)])   
        self.reference_point = torch.tensor([10.,10.])
        self.hypervolume = 84.88 # Approximated from NSGA-II
        self.negate = negate
        
        # Function network values
        self.n_nodes = 5
        self.input_dim = self.n_var # Kept this variable to satisfy the legacy code 
        

    def levy(self, x: Tensor):
        x = -10 + 20*x
        w = 1 + (x - 1) / 4
        
        term1 = w[...,0]
        term2 = torch.sum((w[..., :-1] - 1)**2 * (1 + 10 * torch.sin(torch.pi * w[..., :-1] + 1)**2), dim=-1)
        term3 = w[..., -1]
        
        return torch.stack([term1, term2, term3], dim=-1)

    def branin(self, x: Tensor):
        x = -10 + 20*x
        x1, x2 = x[..., 0], x[..., 1]
        
        b = 5.1 / (4 * torch.pi**2)
        c = 5 / torch.pi
        r = 6
        
        output1 = (x2 - b * x1**2 + c * x1 - r)
        output2 = x1        
        return torch.stack([output1, output2], dim=-1)
        
    
    def function_network_evaluate(self, x: Tensor):
        f1 = self.levy(x)
        f2 = self.branin(x)
        
        
        return torch.cat((f1, f2), dim= -1)
    

    def network_to_objective_transform(self,Y:Tensor):
       """
       Used to transform our function network values to the objective function

        Parameters
        ----------
        Y : TYPE
            DESCRIPTION.

        Returns
        -------
        int
            DESCRIPTION.

        """ 
       input_shape = Y.shape
       output = torch.empty(input_shape[:-1] + torch.Size([2])) # This is the 
       a = 1
       s = 10
       t = 1 / (8 * torch.pi)
       
       output[...,0] = (torch.sin(torch.pi * Y[...,0]))**2 + Y[...,1] + (Y[...,2] - 1)**2 * (1 + torch.sin(2 * torch.pi * Y[...,2])**2)
       output[...,1] = a*Y[...,3]**2 + s*(1-t)*torch.cos(Y[...,4]) + s       
       
       if self.negate:
           return -1*output
       else:
           return output


if __name__ == "__main__":
    
    test_hypervol = True
    # Initialize the PyTorch-based problem
    torch_problem = LevyBraninProblem()
    
    # test_val = torch.zeros(2,100)
    
    # a = torch_problem.function_network_evaluate(test_val)
    
    # print(a)
    
    #print(torch_problem.network_to_objective_transform(a))   
    
    
    if test_hypervol:
    
        problem = PyTorchWrapperProblem(torch_problem)
        
        # Initialize the NSGA-II algorithm
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
                       verbose=True)
        
        # Visualize the Pareto front
        plot = Scatter(title="Pareto Front")
        plot.add(res.F, label="NSGA-II")
        plot.show()
        
        # Compute the Hypervolume
        reference_point = torch.tensor([10., 10.])  # Reference point in PyTorch
        hv = Hypervolume(ref_point=reference_point.numpy())  # Hypervolume requires numpy input
        
        # Calculate the hypervolume
        hypervolume_value = hv.do(res.F)
        print(f"Hypervolume: {hypervolume_value}")
