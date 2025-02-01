#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 11:32:32 2025
This file contains all the MOBO objective functions
@author: kudva.7
"""

import sys
sys.path.append('/home/kudva.7/Desktop/PaulsonLab/BOFN/BONS_MOBO/test_func_MOBO')
import torch
from LevyBranin_Simulation import LevyBraninProblem
from ZDT4_Simulation import ZDT4Problem
from torch import Tensor
from botorch.acquisition.objective import GenericMCObjective
from graph_utils import Graph
import sys




def function_network_examples(example, negate: bool = False):

    if example == 'levy_branin':
    
        test_function = LevyBraninProblem(negate = negate)
        input_dim = test_function.input_dim
        n_nodes = test_function.n_nodes
        
        def function_network(X: Tensor):
            return test_function.function_network_evaluate(x=X)
        
        # Underlying DAG
        g = Graph(n_nodes)
         
        active_input_indices = [[0],[0,1],[1],[0,1],[0]]
        g.register_active_input_indices(active_input_indices=active_input_indices)        
        
        # Function that maps the network output to the objective value  
        g.define_objective(test_function.network_to_objective_transform)   
        g.define_reference_point(ref_point = test_function.reference_point)
        
    elif example == 'ZDT4':
        
        test_function = ZDT4Problem(negate = negate)
        input_dim = test_function.input_dim
        n_nodes = test_function.n_nodes
        
        def function_network(X: Tensor):
            return test_function.function_network_evaluate(x=X)
        
        # Underlying DAG
        g = Graph(n_nodes)
         
        active_input_indices = [[i] for i in range(n_nodes)]
        g.register_active_input_indices(active_input_indices=active_input_indices)        
        
        # Function that maps the network output to the objective value  
        g.define_objective(test_function.network_to_objective_transform)   
        g.define_reference_point(ref_point = test_function.reference_point)
        
        
        
        
        
                
    else:
        print('Please enter a valid example problem')
        sys.exit()
        
    return function_network, g, test_function



if __name__ == '__main__':
    
    
    example_list = ['levy_branin', 'ZDT4']
    
    print('Testing testing...')
    a,b, _=  function_network_examples(example_list[1]) 
    
    test_vals = 0.5*torch.zeros(10,10)
    
    c = a(test_vals)
    #print(c)
    
    d = b.objective_function(c)