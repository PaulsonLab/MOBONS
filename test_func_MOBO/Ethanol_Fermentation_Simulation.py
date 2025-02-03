#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 10:50:18 2025

@author: tang.1856
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
from biorefineries.cane import create_sugarcane_chemicals
from biosteam.units import Fermentation
from biosteam import Stream, settings
# from biosteam import NRELFermentation
from biosteam.units import BinaryDistillation, ShortcutColumn
import biosteam as bst

# Define the multi-objective ZDT4 problem using PyTorch
class EthanolProblem:
    def __init__(self, negate: bool = False):
        self.n_var = 5  # Number of decision variables
        self.n_obj = 2  # Number of objectives
        
        self.xl = torch.tensor([10,20,1,2,2])
        self.xu = torch.tensor([300,80,5,10,10])  
        
        self.reference_point = torch.tensor([0.0, 9e10]) # just approximated 
        # self.hypervolume = 499.655
        
        # self.reference_point = torch.tensor([1.0, 6.0])
        # self.hypervolume = 5.64  # Needs to be computed separately - Done
        self.negate = negate
        
        # Function network values
        self.n_nodes = 14 
        self.input_dim = self.n_var
    
    def perform_simulation(self, x):
        rxn_T, rxn_P, RR1, RR2 = float(x[1]), float(x[2]), float(x[3]), float(x[4])
        Yeast = 100
        Water = float(x[0])
        Glucose = float(x[0])
        bst.nbtutorial()
        
        settings.set_thermo(create_sugarcane_chemicals())
        feed = Stream('Feed',
                      Water=Water,
                      DryYeast=Yeast,
                      Glucose=Glucose,
                      units='kmol/hr',
                      T=30+273.15)
        
        F1 = Fermentation('F1',
                          ins=feed, outs=('CO2', 'product'),
                          tau=8, efficiency=0.90, N=8, T=rxn_T+273.15, P=rxn_P*101325)
        F1.cell_growth_reaction.X = 0. 
        
        feed = F1 - 1
              
        # Create a distillation column and simulate
        D1 = BinaryDistillation(
            'D1', ins=feed,
            outs=('distillate', 'bottoms_product'),
            LHK=('Ethanol', 'Water'), # Light and heavy keys
            y_top=0.7, # Light key composition at the distillate
            x_bot=0.01, # Light key composition at the bottoms product
            k=RR1, # Ratio of actual reflux over minimum reflux
            is_divided=True, # Whether the rectifying and stripping sections are divided
            partial_condenser = False,
        )
        
        feed2 = D1-0
     
        D2 = BinaryDistillation(
            'D2', ins=feed2,
            outs=('distillate', 'bottoms_product'),
            LHK=('Ethanol', 'Water'), 
            y_top=0.85, 
            x_bot=0.05, 
            k=RR2, 
            is_divided=True,
            partial_condenser=False,
            
        )

        flowsheet_sys = bst.main_flowsheet.create_system('flowsheet_sys')
        flowsheet_sys.operating_hours = 8000 # Define the operating hour of the system
        flowsheet_sys.simulate()
        
        payback_period = 10    
        TAC_fermentation = F1.utility_cost + (F1.purchase_cost)/payback_period
        F_co2 = F1.outs[0].imol['CO2']
        T1 = F1.outs[1].T
        P1 = F1.outs[1].P
        F_water1 = F1.outs[1].imol['Water']
        F_glucose1 = F1.outs[1].imol['Glucose']
        F_yeast1 = F1.outs[1].imol['Yeast']
        
        TAC_distillation1 = D1.utility_cost + (D1.purchase_cost)/payback_period
        T2 = D1.outs[0].T
        P2 = D1.outs[0].P
        F_water2 = D1.outs[0].imol['Water']
        F_ethanol1 = D1.outs[0].imol['Ethanol']
        
        TAC_distillation2 = D2.utility_cost + (D2.purchase_cost)/payback_period
        F_ethanol2 = D2.outs[0].imol['Ethanol']
        
        
        return [TAC_fermentation, F_co2, T1, P1, F_water1, F_glucose1, F_yeast1, TAC_distillation1, T2, P2, F_water2, F_ethanol1, TAC_distillation2, F_ethanol2]
    
    def function_network_evaluate(self,x:Tensor):
        
        input_shape = x.shape
        x_scaled = x.clone()
        # x_scaled[...,1:] = 20*x[...,1:] - 10 # Natural bounds of this problem is [-10,10]
        x_scaled = self.xl + (self.xu-self.xl)*x
        
        output_list = []
        for i in range(input_shape[0]):
            output_list.append(self.perform_simulation(x_scaled[i]))
            
        output1 = torch.tensor(output_list)
        # output1 = torch.empty(input_shape[:-1] + torch.Size([self.n_nodes]))
        # output1[:,0] = [TAC_fermentation, F_co2, T1, P1, F_water1, F_glucose1, F_yeast1, TAC_distillation1, T2, P2, F_water2, F_ethanol1, TAC_distillation2, F_ethanol2]
        
        # output1[...,0] = x_scaled[...,0]
        
        # for i in range(1,self.n_var):
        #     output1[...,i] = x_scaled[...,i]**2 - 10*torch.cos(4*torch.pi*x_scaled[...,i])
            
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
        
       ethanol_price = 1.77 # $/gallon (https://grains.org/ethanol_report/ethanol-market-and-pricing-data-january-31-2024/)
       unit_convert = 15.42 # kmol to gallon of ethanol
       operating_hours = 8000
       
       input_shape = Y.shape
       output = torch.empty(input_shape[:-1] + torch.Size([2])) # This is the number of objectives
       # f_1 = Y[...,0]
       # g_x = 1 + 10*(self.n_var - 1) + torch.sum(Y[...,1:], dim = -1)
       
       # output[...,0] = f_1
       # output[...,1] = (1 - torch.sqrt(torch.sqrt((f_1/g_x)**2)))*(g_x)
          
       output[:,0] = ethanol_price*Y[:, -1]*unit_convert*operating_hours - (Y[:,0] + Y[:,7] + Y[:,-2]) # revenue
       
       CO2_production = Y[:,1]
       
       for i in range(CO2_production.shape[0]):
           
           if CO2_production[i]<50:
               CO2_production[i] = CO2_production[i]**0.5
           elif CO2_production[i]>=50 and CO2_production[i]<100:
               CO2_production[i] = CO2_production[i]**1
           elif CO2_production[i]>=100 and CO2_production[i]<200:
               CO2_production[i] = CO2_production[i]**2
           else:
               CO2_production[i] = CO2_production[i]**4
       
       output[:,1] = -CO2_production # GWP
        
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
    torch_problem = EthanolProblem()
    
    test_val = torch.zeros(2,5)
    
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