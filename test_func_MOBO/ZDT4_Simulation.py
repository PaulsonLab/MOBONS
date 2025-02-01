#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 11:52:31 2025

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

# Define the multi-objective ZDT4 problem using PyTorch
class ZDT4Problem:
    def __init__(self, negate: bool = False):
        self.n_var = 10  # Number of decision variables
        self.n_obj = 2  # Number of objectives
        
        self.xl = torch.tensor([0. for i in range(self.n_var)])
        self.xu = torch.tensor([1. for i in range(self.n_var)])  
        
        self.reference_point = torch.tensor([1.0, 500.0])
        self.hypervolume = 499.655
        
        # self.reference_point = torch.tensor([1.0, 6.0])
        # self.hypervolume = 5.64  # Needs to be computed separately - Done
        self.negate = negate
        
        # Function network values
        self.n_nodes = self.n_var 
        self.input_dim = self.n_var
    
        
    def function_network_evaluate(self,x:Tensor):
        
        input_shape = x.shape
        x_scaled = x.clone()
        x_scaled[...,1:] = 20*x[...,1:] - 10 # Natural bounds of this problem is [-10,10]
        output1 = torch.empty(input_shape[:-1] + torch.Size([self.n_nodes]))
        
        output1[...,0] = x_scaled[...,0]
        
        for i in range(1,self.n_var):
            output1[...,i] = x_scaled[...,i]**2 - 10*torch.cos(4*torch.pi*x_scaled[...,i])
            
        return output1
    
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
       output = torch.empty(input_shape[:-1] + torch.Size([2])) # This is the number of objectives
       f_1 = Y[...,0]
       g_x = 1 + 10*(self.n_var - 1) + torch.sum(Y[...,1:], dim = -1)
       
       output[...,0] = f_1
       output[...,1] = (1 - torch.sqrt(torch.sqrt((f_1/g_x)**2)))*(g_x)
          
       
       if self.negate:
           return -1*output
       else:
           return output
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt    
    import pickle
    
    test_hypervol = False
    plot_figure = True
    run_GA = True
    # Initialize the PyTorch-based problem
    torch_problem = ZDT4Problem()
    
    test_val = torch.zeros(2,10)
    
    a = torch_problem.function_network_evaluate(test_val)
    
    print(a)
    
    print(torch_problem.network_to_objective_transform(a))   
    
    
    if run_GA:
    
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
                       termination=("n_gen", 150),  # 78, 100, 150
                       seed=1,
                       verbose=True)
        
        data = {"X": res.X, 'Y_true': res.F}
        with open('ZDT4_NSGA_part3.pickle', 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)   
        
        
        if plot_figure:
        
            # Visualize the Pareto front
            plot = Scatter(title="NSGA-II (20000 Function Evaluations)", figsize = (6,6))
            plot.add(res.F, label="NSGA-II", facecolors = 'maroon', edgecolors = 'black')
            plot.show()
            plt.title("NSGA-II (20000 Function Evaluations)", fontsize=15, fontweight='bold')  # Adjust title size and make it bold
            plt.grid()
            plt.xlabel("$f_{1}(x)$", fontsize = 20)
            plt.ylabel("$f_{2}(x)$", fontsize = 20)
            plt.ylim([0,10])
            plt.xlim([0,0.5])
            plt.xticks(fontsize = 20)
            plt.yticks(fontsize = 20)
            plt.show()            

    
        if test_hypervol:
            # Compute the Hypervolume
            reference_point = torch_problem.reference_point  # Reference point in PyTorch
            hv = Hypervolume(ref_point=reference_point.numpy())  # Hypervolume requires numpy input
            
            # Calculate the hypervolume
            hypervolume_value = hv.do(res.F)
            print(f"Hypervolume: {hypervolume_value}")
        
        
        
        