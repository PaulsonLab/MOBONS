#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 22:10:15 2025

@author: kudva.7
"""
import sys
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume
from matplotlib.cm import ScalarMappable

sys.path.append('/home/kudva.7/Desktop/PaulsonLab/BOFN/BONS_MOBO/')
from Objective_FN_MOBO import function_network_examples

# Load function network and reference point
example_list = ['levy_branin','ZDT4']
example = example_list[1]

plot_repeat_number = 0 # Slect the repeat # to plot

function_network, g, test_function = function_network_examples(example) 

ref_point = -1*g.ref_point
n_objectives = ref_point.shape[-1]

# Load saved optimization data
with open(example + '.pickle', 'rb') as handle:
    data = pickle.load(handle)
    
    
algorithms = ['Random', 'qEHVI', 'qPOTS', 'MOBONS']  # Algorithm names
num_algorithms = len(algorithms)

Ninit = 2 * g.nx + 1  # Initial points
T = 100  # Iterations
repeats = 30  # Number of independent runs

plot_hypervolume = True
plot_scatter_objective = False
plot_scatter_zoom = False
max_theoretical = test_function.reference_point


if plot_hypervolume:
    
    "Regret plot"

    # Storage for hypervolume results
    hv_results = {alg: np.zeros((repeats, T)) for alg in data}
    
    # Compute hypervolume over iterations
    for alg in data:        
        for repeat_no in range(repeats):
            Y_true = data[alg][repeat_no]['Y_true']  # Shape: (Ninit + T, n_objectives)
            
            # Convert to maximization (BoTorch assumes maximization)
            Y_neg = -Y_true  
    
            for t in range(1, T):  # Start from t=1 to avoid computing HV with just initial points
                pareto_mask = is_non_dominated(Y_neg[: Ninit + t])  # Find Pareto front
                pareto_points = Y_neg[: Ninit + t][pareto_mask]  # Extract Pareto set
                
                if pareto_points.shape[0] > 0:  # Ensure Pareto set is non-empty
                    # Dynamic Reference Point: slightly worse than worst observed value
                    #ref_point_dynamic = Y_neg[: Ninit + t].min(dim=0).values - 0.1  
    
                    # Compute hypervolume
                    hv = Hypervolume(ref_point=ref_point)
                    hv_results[alg][repeat_no, t] = hv.compute(pareto_points)#.item()
    
    # Plot hypervolume results
    plt.figure(figsize=(8, 6))
    
    for i, alg in enumerate(algorithms):
        if alg == 'MOBONS':
            color_val = 'red'
            line_style = 'solid'
        elif alg == 'qEHVI':
            color_val = 'blue'
            line_style = 'dashdot'
        elif alg == 'Random':
            color_val = 'green'
            line_style = 'dotted'
        elif alg == 'qPOTS':
            color_val = 'orange'
            line_style = 'solid'
            

        mean_hv = np.mean(hv_results[alg], axis=0)  # Mean HV over repeats
        std_hv = np.std(hv_results[alg], axis=0)  # Standard deviation
    
        plt.plot(range(T), mean_hv, label=alg, color = color_val, linestyle = line_style, linewidth = 3)
        plt.fill_between(range(T), mean_hv - std_hv, mean_hv + std_hv, alpha=0.2, color = color_val)
    
    new_T_val = [25*(i) for i in range(5)]   
    fun_vals = [100*i for i in range(7)]   
    plt.axhline(y = test_function.hypervolume, linewidth = 4, linestyle = 'dashed', color = 'black')
    plt.xticks(new_T_val, fontsize = 17)
    plt.yticks(fun_vals, fontsize = 17)
    plt.xlabel("Iteration, t", fontsize = 30)
    plt.ylabel("Hypervolume", fontsize = 30)
    plt.title(example, fontsize = 30)
    plt.legend()
    plt.grid()    
    plt.xlim([0,100])
    plt.ylim([0,600])
    plt.legend(fontsize = 20, loc = 'lower right', labelspacing = 0.1)
    plt.show()
    
    

if plot_scatter_objective:
    
        # Define colormap
    cm = plt.get_cmap("jet")
    
    # Create subplots
    fig, axes = plt.subplots(1, num_algorithms, figsize=(6 * num_algorithms, 6), sharex=True, sharey=True)
    
    # Iterate over algorithms
    for i, alg in enumerate(algorithms):
        ax = axes[i]
        
        # Only consider the selected repeat number
        Y_true = data[alg][plot_repeat_number]['Y_true']  # Shape: (Ninit + T, n_objectives)
    
        # Generate batch numbers
        batch_number = np.concatenate((
            np.zeros(Ninit),  # Initial points have batch number 0
            np.arange(1, T + 1)  # Rest correspond to iterations
        ))
    
        # **Filter only iterations after Ninit + 1**
        Y_true_filtered = Y_true[Ninit + 1:]  # Skip initial points
        batch_number_filtered = batch_number[Ninit + 1:]  # Match indices
    
        # Scatter plot with iteration-based color
        sc = ax.scatter(
            Y_true_filtered[:, 0].cpu().numpy(),
            Y_true_filtered[:, 1].cpu().numpy(),
            c=batch_number_filtered,
            cmap=cm,
            alpha=0.8,
            edgecolors='black',
            s = 80,
        )
    
        ax.set_title(f"{alg}", fontsize=25, fontweight='bold')
        ax.set_xlabel("$g_{1}(x)$", fontsize=20)
        ax.tick_params(axis="x", labelsize=15)
        ax.tick_params(axis="y", labelsize=15)
        ax.grid()
    
    # Label y-axis only for the first subplot
    axes[0].set_ylabel("$g_{2}(x)$", fontsize=20)
    
    # Colorbar for iteration number
    norm = plt.Normalize(batch_number_filtered.min(), batch_number_filtered.max())
    sm = ScalarMappable(norm=norm, cmap=cm)
    sm.set_array([])
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.01, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(label='Iteration, t', size=25, weight='bold')
    cbar.ax.tick_params(labelsize=20)
    
    plt.show()

###########################################
    
if plot_scatter_zoom:
    scatter_size = 200
    # Load saved optimization data
    with open(example + '_NSGA.pickle', 'rb') as handle:
        data1 = pickle.load(handle)
        
    label_NSGA = ['NSGA II (T = 7800)','NSGA II (T = 10000)','NSGA II (T = 15000)']
        # Define colormap
    cm = plt.get_cmap("jet")
    
    # Create a figure with 4 subplots (3 for data1, 1 for MOBONS)
    fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharex=True, sharey=True)
    
    # Plot first 3 subplots with data1
    for i in range(3):
        ax = axes[i]
        
        # Extract Y_true from data1
        Y_true = data1[i]['Y_true']
        
        # Scatter plot with maroon color
        ax.scatter(
            Y_true[:, 0],
            Y_true[:, 1],
            color='maroon',
            alpha=0.8,
            edgecolors='black',
            s = scatter_size,
        )
    
        ax.set_title(label_NSGA[i], fontsize=25, fontweight='bold')
        ax.set_xlabel("$g_{1}(x)$", fontsize=20)
        ax.tick_params(axis="x", labelsize=15)
        ax.tick_params(axis="y", labelsize=15)
        ax.grid()
    
    # Set ylabel for first subplot only
    axes[0].set_ylabel("$g_{2}(x)$", fontsize=20)
    
    # Fourth subplot for MOBONS
    if 'MOBONS' in algorithms:
        ax = axes[3]  # Fourth subplot
        alg = 'MOBONS'
        
        # Extract Y_true from data
        Y_true = data[alg][plot_repeat_number]['Y_true']
        
        # Generate batch numbers
        batch_number = np.concatenate((
            np.zeros(Ninit),  # Initial points have batch number 0
            np.arange(1, T + 1)  # Iteration steps
        ))
    
        # Scatter plot with iteration-based color
        sc = ax.scatter(
            Y_true[:, 0].cpu().numpy(),
            Y_true[:, 1].cpu().numpy(),
            c=batch_number,
            cmap=cm,
            alpha=0.8,
            edgecolors='black',
            s = scatter_size,
        )
    
        ax.set_title(alg, fontsize=25, fontweight='bold')
        ax.set_xlabel("$g_{1}(x)$", fontsize=20)
        ax.tick_params(axis="x", labelsize=15)
        ax.tick_params(axis="y", labelsize=15)
        ax.grid()
    
        # Colorbar for iteration number
        norm = plt.Normalize(batch_number.min(), batch_number.max())
        sm = ScalarMappable(norm=norm, cmap=cm)
        sm.set_array([])
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.93, 0.15, 0.01, 0.7])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label(label='Iteration, t', size=25, weight='bold')
        cbar.ax.tick_params(labelsize=20)
    
    # Set axis limits for all subplots
    for ax in axes:
        ax.set_xlim([0, 0.5])
        ax.set_ylim([0, 10])
    
   
    plt.show()
   
    
        
        
        
    
