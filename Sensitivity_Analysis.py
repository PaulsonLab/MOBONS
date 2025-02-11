#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 11:08:08 2025

@author: tang.1856
"""
import sys
sys.path.append('/home/tang.1856/MOBONS')
import pickle
import torch
from Objective_FN_MOBO import function_network_examples
from algorithms_MOBO import MultiObj_BayesOpt
from TSacquisition_functions import ThompsonSampleFunctionNetwork, ThompsonSampleAcq
from gp_network_utils import GaussianProcessNetwork

import numpy as np
import matplotlib.pyplot as plt
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume
from matplotlib.cm import ScalarMappable


from Objective_FN_MOBO import function_network_examples


example_list = ['ZDT4', 'ethanol']

example_name = example_list[1]

# algo_list = ['Random','qEHVI', 'qPOTS', 'MOBONS']
algo_list = ['MOBONS']

T = 50
Nrepeats = 1

data = {}

for alg in algo_list:
    
    # Get the 
    if alg == "qEHVI":
        function_network,g,_ =  function_network_examples(example_name, negate = True) 
    else:
        function_network,g,_ =  function_network_examples(example_name) 
    
    # Ninit = 2*g.nx + 1
    Ninit = 20
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

# Perform SA

# Load function network and reference point
example_list = ['levy_branin','ethanol']
example = example_list[1]

plot_repeat_number = 0 # Slect the repeat # to plot

function_network, g, test_function = function_network_examples(example) 

ref_point = -1*g.ref_point
n_objectives = ref_point.shape[-1]
Ninit = 20


plot_hypervolume = True
max_theoretical = test_function.reference_point

pareto_points_MOBONS_list = []

if plot_hypervolume:
    
    "Regret plot"

    # Storage for hypervolume results
    hv_results = {alg: np.zeros((Nrepeats, T)) for alg in data}
    
    # Compute hypervolume over iterations
    for alg in data:        
        for repeat_no in range(Nrepeats):
        # for repeat_no in [4,0,1,3,5]:
            Y_true = data[alg][repeat_no]['Y_true']  # Shape: (Ninit + T, n_objectives)
            
            # Convert to maximization (BoTorch assumes maximization)
            Y_neg = -Y_true  
            # Y_neg = Y_neg[Y_neg[:,0].argsort()]
    
            for t in range(1, T):  # Start from t=1 to avoid computing HV with just initial points
                pareto_mask = is_non_dominated(Y_neg[: Ninit + t])  # Find Pareto front
                pareto_points = Y_neg[: Ninit + t][pareto_mask]  # Extract Pareto set
                pareto_points_X = data[alg][repeat_no]['X'][: Ninit + t][pareto_mask]
                pareto_points_X = pareto_points_X[pareto_points[:,0].argsort()]
                # pareto_points = pareto_points[pareto_points[:,0].argsort()]
                
                if pareto_points.shape[0] > 0:  # Ensure Pareto set is non-empty
                    # Dynamic Reference Point: slightly worse than worst observed value
                    #ref_point_dynamic = Y_neg[: Ninit + t].min(dim=0).values - 0.1  
    
                    # Compute hypervolume
                    hv = Hypervolume(ref_point=ref_point)
                    hv_results[alg][repeat_no, t] = hv.compute(pareto_points)#.item()
                    
                    
                    
    if alg == 'MOBONS': # perform sobol sensitivity analysis
        from SALib.analyze.sobol import analyze
        from SALib.sample.sobol import sample
        percentage_bound = 0.05
        Si_list = [[],[]]
        for output_indice in range(1,2):
            # for p in range(pareto_points_X.shape[0]):
            for p in [5,12,16,20]:
                v1, v2, v3, v4, v5, v6, v7, v8 = pareto_points_X[p][0], pareto_points_X[p][1], pareto_points_X[p][2], pareto_points_X[p][3], pareto_points_X[p][4], pareto_points_X[p][5], pareto_points_X[p][6], pareto_points_X[p][7]
                problem = {
                    'num_vars': 8,
                    'names': ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8'],
                    'bounds': [[v1-percentage_bound*v1, v1+percentage_bound*v1],
                                [v2-percentage_bound*v2, v2+percentage_bound*v2],
                                [v3-percentage_bound*v3, v3+percentage_bound*v3],
                                [v4-percentage_bound*v4, v4+percentage_bound*v4],
                                [v5-percentage_bound*v5, v5+percentage_bound*v5],
                                [v6-percentage_bound*v6, v6+percentage_bound*v6],
                                [v7-percentage_bound*v7, v7+percentage_bound*v7],
                                [v8-percentage_bound*v8, v8+percentage_bound*v8]]
                }
                # Generate samples
                param_values = sample(problem, 10)  
                Y_SA_true = (g.objective_function(function_network(torch.tensor(param_values)))).numpy()[:,output_indice]
                train_x = data[alg][0]['X']
                train_y = data[alg][0]['Y']
                lb = pareto_points_X[p] - percentage_bound*pareto_points_X[p]
                ub = pareto_points_X[p] + percentage_bound*pareto_points_X[p]
                new_x = lb + (ub-lb)*torch.rand(50,8)
                new_y = function_network(new_x)   
                
                train_x = torch.cat((train_x,new_x))
                train_y = torch.cat((train_y,new_y))
                model = GaussianProcessNetwork(train_X = train_x,
                                               train_Y = train_y,
                                               dag = g)
                Y_SA = torch.zeros(param_values.shape[0])
                for _ in range(256):
                    TS_acq = ThompsonSampleFunctionNetwork(model)
                    Y_SA+=TS_acq(torch.tensor(param_values).unsqueeze(1))[:,output_indice]
                Y_SA = (Y_SA/256).detach().numpy()
                
                plt.figure()
                plt.scatter(Y_SA_true, Y_SA)
                
                # Perform analysis
                Si = analyze(problem, Y_SA, print_to_console=True)
                Si_list[output_indice].append(Si['S1'])
                # Print the first-order sensitivity indices
                print(Si['S1'])                    



# Plotting
                        
import numpy as np
import matplotlib.pyplot as plt

data_ = np.array(Si_list[1])

barWidth = 0.1
fig = plt.subplots(figsize =(12, 8)) 

br1 = np.arange(len(data_[0])) 
br2 = [x + barWidth for x in br1] 
br3 = [x + barWidth for x in br2] 
br4 = [x + barWidth for x in br3] 
br5 = [x + barWidth for x in br4] 

plt.bar(br1, data_[0], color ='r', width = barWidth, 
        edgecolor ='grey', label ='Point A') 
plt.bar(br2, data_[1], color ='g', width = barWidth, 
        edgecolor ='grey', label ='Point B') 
plt.bar(br3, data_[2], color ='b', width = barWidth, 
        edgecolor ='grey', label ='Point C') 
plt.bar(br4, data_[3], color ='orange', width = barWidth, 
        edgecolor ='grey', label ='Point D') 
plt.bar(br5, data_[4], color ='purple', width = barWidth, 
        edgecolor ='grey', label ='Point E') 

# plt.xlabel('Branch', fontweight ='bold', fontsize = 15) 
plt.ylabel('SA Indice', fontweight ='bold', fontsize = 15) 
plt.xticks([r + barWidth for r in range(len(data_[0]))], 
        ['F', 'Trxn', 'Prxn', 'P1', 'RR1', 'T2', 'RR2', 'p'])

plt.legend()
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    