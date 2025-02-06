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
from biosteam import units

# Define the multi-objective Ethanol Fermentation problem using PyTorch
class EthanolProblem:
    def __init__(self, negate: bool = False):
        self.n_var = 8  # Number of decision variables
        self.n_obj = 2  # Number of objectives
        
        self.xl = torch.tensor([0. for i in range(self.n_var)])
        self.xu = torch.tensor([1. for i in range(self.n_var)])
        
        # self.reference_point = torch.tensor([-3.25e7, 13000]) # just approximated 
        self.reference_point = torch.tensor([-28.3, 13600]) # just approximated 
        self.negate = negate
        
        # Function network values
        self.n_nodes = 14
        self.input_dim = self.n_var
        
        self.operating_hours = 8000
        self.payback_period = 10  
    
    def perform_simulation(self, x):
        rxn_T, rxn_P, P1, RR1, P2, RR2, purity1 = float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5]), float(x[6]), float(x[7])
        purity2 = 0.85 # purity of ethanol solution
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
        
        # T301 = units.StorageTank('T301', F1-1, tau=4, vessel_material='Carbon steel')
        # T301.line = 'Beer tank' # Changes name on the diagram
        # # Separate 99% of yeast
        # C301 = units.SolidsCentrifuge('C301', T301-0, outs=('recycle_yeast', ''),
        #                             moisture_content=0.01,
        #                             split=(1,1, 0.99999, 0.01), # This gets reverted in the next line
        #                             order=('Ethanol', 'Water','Glucose',  'DryYeast'),
        #                             solids=('DryYeast',))
        # C301.split[:] = 1. - C301.split
        # feed = F1 - 1
        
        # create a pump
        Pump1 = units.Pump('Pump1', F1-1, P= (F1-1).P + 5*101325)
        # Pump1 = units.Pump('Pump1', C301-1, P= (C301-1).P + 5*101325)
        
        
        # Create a distillation column and simulate
        D1 = BinaryDistillation(
            'D1', ins=Pump1-0,
            outs=('distillate', 'bottoms_product'),
            LHK=('Ethanol', 'Water'), # Light and heavy keys
            y_top=purity1, # Light key composition at the distillate
            x_bot=0.01, # Light key composition at the bottoms product
            k=RR1, # Ratio of actual reflux over minimum reflux
            is_divided=True, # Whether the rectifying and stripping sections are divided
            partial_condenser = False,
            P = P1*101325
        )
        
        # feed2 = D1-0
        Pump2 = units.Pump('Pump2', D1-0, P=(D1-0).P+5*101325)
     
        D2 = BinaryDistillation(
            'D2', ins=Pump2-0,
            outs=('distillate', 'bottoms_product'),
            LHK=('Ethanol', 'Water'), 
            y_top=purity2, 
            x_bot=1-purity2, 
            k=RR2, 
            is_divided=True,
            partial_condenser=False,
            P = P2*101325
        )

        flowsheet_sys = bst.main_flowsheet.create_system('flowsheet_sys')
        flowsheet_sys.operating_hours = self.operating_hours # Define the operating hour of the system
        flowsheet_sys.simulate() # perform simulation
                
        
        TAC_fermentation = F1.utility_cost + (F1.purchase_cost)/self.payback_period # TAC of fermentation reactor
        F_co2 = F1.outs[0].imol['CO2']
        T1 = F1.outs[1].T
        P1 = F1.outs[1].P
        F_water1 = F1.outs[1].imol['Water']
        F_glucose1 = F1.outs[1].imol['Glucose']
        F_ethanol1 = F1.outs[1].imol['Ethanol']
        
        TAC_distillation1 = D1.utility_cost + (D1.purchase_cost)/self.payback_period
        T2 = D1.outs[0].T
        P2 = D1.outs[0].P
        F_water2 = D1.outs[0].imol['Water']
        F_ethanol2 = D1.outs[0].imol['Ethanol']
        
        TAC_distillation2 = D2.utility_cost + (D2.purchase_cost)/self.payback_period
        F_ethanol3 = D2.outs[0].imol['Ethanol']
        
        
        return [TAC_fermentation, F_co2, T1, P1, F_water1, F_glucose1, F_ethanol1, TAC_distillation1, T2, P2, F_water2, F_ethanol2, TAC_distillation2, F_ethanol3]
    
    def function_network_evaluate(self,x:Tensor):
        
        input_shape = x.shape
        x_scaled = x.clone()
        # xl = torch.tensor([90, 30, 1, 1, 1, 1, 2, 0.5]) # lower bound for design variables
        # xu = torch.tensor([110, 40, 3, 3, 3, 3, 4, 0.7])  # upper bound for design variables
        
        xl = torch.tensor([90, 30, 1, 1, 0.1, 1, 2, 0.5]) # lower bound for design variables
        xu = torch.tensor([110, 40, 5, 5, 10, 5, 10, 0.7])  # upper bound for design variables
        
        x_scaled = xl + (xu - xl)*x_scaled
        
        output_list = []
        for i in range(input_shape[0]):
            output_list.append(self.perform_simulation(x_scaled[i]))
            
        output1 = torch.tensor(output_list)
        
            
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
        
       ethanol_price = 1.77*0.85 # $/gallon (https://grains.org/ethanol_report/ethanol-market-and-pricing-data-january-31-2024/)
       unit_convert = 15.42 # kmol/hr to gallon/hr of ethanol (https://www.aqua-calc.com/calculate/mole-to-volume-and-weight)
       
       input_shape = Y.shape
       output = torch.empty(input_shape[:-1] + torch.Size([2])) # This is the number of objectives
       
       output[:,0] = -1e-6*(ethanol_price*Y[:, -1]*unit_convert*self.operating_hours - (Y[:,0] + Y[:,7] + Y[:,-2])) # revenue
     
       CO2_production = Y[:,1]
       
       for i in range(CO2_production.shape[0]):
           
            if CO2_production[i]<170:
                CO2_production[i] = CO2_production[i]**1.0
            elif CO2_production[i]>=170 and CO2_production[i]<175:
                CO2_production[i] = CO2_production[i]**1.5
            elif CO2_production[i]>=175 and CO2_production[i]<180:
                CO2_production[i] = CO2_production[i]**1.7
            else:
                CO2_production[i] = CO2_production[i]**1.8

                  
       output[:,1] = CO2_production # GWP
        
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
    
    test_val = torch.rand(100,8)
    
    a = torch_problem.function_network_evaluate(test_val)
    obj = torch_problem.network_to_objective_transform(a)
    
    # print(a)
    
    # print(torch_problem.network_to_objective_transform(a))   
    
    
    if run_GA:
    
        problem = PyTorchWrapperProblem(torch_problem)
        
        # Initialize the NSGA-II algorithm
        algorithm = NSGA2(
            pop_size=40,
            sampling=get_sampling("real_random"),
            crossover=get_crossover("real_sbx", prob=0.9, eta=15),
            mutation=get_mutation("real_pm", eta=20),
            eliminate_duplicates=True
        )
        
        # Solve the problem
        res = minimize(problem,
                       algorithm,
                       termination=("n_gen",20),  # 78, 100, 150
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
            # plt.ylim([0,10])
            # plt.xlim([0,0.5])
            plt.xticks(fontsize = 20)
            plt.yticks(fontsize = 20)
            plt.show()            
