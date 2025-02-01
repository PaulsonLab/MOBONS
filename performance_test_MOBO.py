#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 14:35:26 2025

@author: kudva.7
"""

import pickle
import torch
from Objective_FN_MOBO import function_network_examples
from algorithms_MOBO import MultiObj_BayesOpt


example_list = ['levy_branin', 'ZDT4']

example_name = example_list[1]

algo_list = ['qEHVI']


T = 100
Nrepeats = 30

data = {}

for alg in algo_list:
    
    # Get the 
    if alg == "qEHVI":
        function_network,g,_ =  function_network_examples(example_name, negate = True) 
    else:
        function_network,g,_ =  function_network_examples(example_name) 
    
    Ninit = 2*g.nx + 1
    data[alg] = {}
    
    print('Running algorithm', alg)

    for i in range(Nrepeats):
        
        print('Running repeat number', i)        

        # Generate the data
        data[alg][i] = MultiObj_BayesOpt(Ninit = Ninit,
                                         T = T,
                                         g = g,
                                         function_network = function_network,
                                         seed = i,
                                         alg = alg)
        
        
        
with open(example_name + '_part3.pickle', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)        




