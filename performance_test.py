#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 13:39:13 2024

@author: kudva.7
"""

import pickle
import torch
from Objective_FN import function_network_examples
from algorithms import BOFN, BONS, BayesOpt
from algorithms_parttwo import CMA_ES, LBFGS_Optimize

examples = ['dropwave', 'levy','rosenbrock','griewank', 'toy_problem', 'ackley']
example = examples[5]

#alg_list = ['BONS', 'BONS-TR', 'Vanilla', 'Random', 'Turbo', 'CMAES'] 
#alg_list = ['Turbo', 'CMAES']

acq_fun = 'qEI'

print('Running ' + example)

# Extract the example and dag structure

function_network, g = function_network_examples(example = example)

#################################################################################

input_dim = g.nx  
n_outs = g.n_nodes

# Start the modeling procedure
Ninit = 2*input_dim + 1
T = 100


Nrepeats = 10

Vanilla_BO = {}
BONS_val = {}
BONS_TR_val = {}
Random_val = {}
Turbo_val = {}
CMAES_val = {}

data = {}

for alg in alg_list:
    for n in range(Nrepeats):    
        j = n + 1     
        print('Current repeat', j)  
        torch.manual_seed(seed = (j + 1)*2000)
        
        if alg == 'Random':
            x_init = torch.rand(Ninit + T, g.nx)
            y_init = function_network(x_init)
            
            rand_vals = {'X':x_init, 'Y':y_init, 'Ninit' : Ninit, 'T': T}
            
            Random_val[n] = rand_vals
            data[alg] = Random_val
            
        else:  
            
            if alg not in ['CMAES', 'LBFGS']:
                x_init = torch.rand(Ninit, g.nx)
                y_init = function_network(x_init)   
                
                if alg == 'BONS':  
                    val = BONS( x_init, y_init, g, objective = function_network, T = T, q = 1)             
                    BONS_val[n] = val 
                    data[alg] = BONS_val
                    
                if alg == 'Vanilla':
                    val = BayesOpt(x_init, y_init,g,objective = function_network,T = T,acq_type = 'logEI')       
                    Vanilla_BO[n] = val
                    data[alg] = Vanilla_BO
                    
                    
                if alg == 'Turbo':
                    val = BayesOpt(x_init, y_init,g,objective = function_network,T = T,acq_type = 'logEI', Turbo = True)       
                    Turbo_val[n] = val
                    data[alg] = Turbo_val
                    
                if alg == 'BONS-TR':  
                    val = BONS( x_init, y_init, g, objective = function_network, T = T, q = 1, Turbo = True)             
                    BONS_TR_val[n] = val 
                    data[alg] = BONS_TR_val
                    
                
            else:  
                x_init = torch.rand(1, g.nx)
                y_init = function_network(x_init)
                
                if alg == 'CMAES':
                    val = CMA_ES( x_init, y_init, g, objective_fun = function_network, T = 6) 
                    CMAES_val[n] = val
                    data[alg] = CMAES_val
                
                if alg == 'LBFGS':
                    val = LBFGS_Optimize( x_init, y_init, g, objective_fun = function_network, T = T) 
                    LBFGS_val[n] = val
                    
        with open(example + '_part2.pickle', 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)   
                    
    
# data['BONS-TR'] = BONS_TR_val




                    
                
    
     