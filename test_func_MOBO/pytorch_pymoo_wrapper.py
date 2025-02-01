#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 11:39:17 2025

@author: kudva.7
"""

from pymoo.core.problem import Problem
import torch
import numpy as np

# Define a wrapper class for pymoo to use the PyTorch-based problem
class PyTorchWrapperProblem(Problem):
    def __init__(self, torch_problem):
        
        self.torch_problem = torch_problem
        self.n_var = self.torch_problem.n_var
        self.n_obj = self.torch_problem.n_obj
        self.xl = self.torch_problem.xl.numpy()
        self.xu = self.torch_problem.xu.numpy()
        
        super().__init__(n_var=self.n_var,  # Number of decision variables
                         n_obj=self.n_obj,  # Number of objectives
                         n_constr=0,  # No constraints
                         xl=self.xl,  # Lower bounds
                         xu=self.xu)  # Upper bounds

    # def _evaluate(self, x, out, *args, **kwargs):
    #     # Evaluate the function using the PyTorch problem
    #     x_torch = torch.tensor(x, dtype=torch.float32)  # Convert to torch tensor
    #     results = self.torch_problem.evaluate(x_torch)  # Evaluate using PyTorch
    #     with torch.no_grad():
    #         out["F"] = results.detach().numpy()  # Convert back to numpy for pymoo compatibility
    def _evaluate(self, x, out, *args, **kwargs):
        # Evaluate the function using the PyTorch problem
        x_torch = torch.tensor(x, dtype=torch.float32)  # Convert to torch tensor
        results = self.torch_problem.network_to_objective_transform(self.torch_problem.function_network_evaluate(x_torch)) # Evaluate using PyTorch
        with torch.no_grad():
            out["F"] = results.detach().numpy()  # Convert back to numpy for pymoo compatibility