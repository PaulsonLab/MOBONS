"""
Created on Thu Jan 25 11:42:08 2024

This script contains the Graph class which can be considered as the backbone of 
all the other scripts and functions in this repository

@author: kudva.7
"""

import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import sys
import torch
import collections
import itertools
from botorch.acquisition.objective import GenericMCObjective
# Compare the sets
compare = lambda x, y: collections.Counter(x) != collections.Counter(y)
# Default acquisition function: The output of final node
default_AF = lambda Y: Y[..., -1]


# Class to represent a graph
class Graph:
    """
        This class is used to define the DAG similar to the class in BOFN ...
        ... with extra utilities
    """
    def __init__(self, nodes):
            #self.graph = defaultdict(list) # dictionary containing adjacency List
            self.graph = {}
            for i in range(nodes):
                self.graph[i] = []
            self.n_nodes = nodes # No. of vertices
            self.active_input_indices = []
            self.uncertain_input_indices = []
            self.design_input_indices = []
            
            # Pre-define all nodes as root node
            self.root_nodes = [i for i in range(self.n_nodes)]
            self.parent_nodes = [[] for _ in range(self.n_nodes)]  
            
            # Added for extra features
            self.w_combinations = None
            self.w_sets = None
            self.custom_hyperparameters = False
            self.objective_function = GenericMCObjective(default_AF)
            self.noise_level = 1e-5

# function to add an edge to graph
    def addEdge(self, u, v):
        """
        This is used to define the graph and produce representations 
        described in the original class

        Parameters
        ----------
        u : Parent_node
        v : Child node
        """
        self.graph[u].append(v)
        self.calculate_parent_nodes()
        self.root_nodes = []
        for k in range(self.n_nodes):
            if len(self.parent_nodes[k]) == 0:
                self.root_nodes.append(k) 
        
        if len(self.active_input_indices) == 0:
            print('Reminder: Please provide active input indices to the problem!')



    def has_edge(self,u,v):
        """
        Used to check if a child node is connected to a given parent node

        Parameters
        ----------
        u : parent node number
        v : child node number

        Returns
        -------
        None.

        """
        try:
            if v in self.graph[u]:
                check = True
            else:
                check = False
        except:
            sys.exit(f'The parent node {u} does not exist! make sure you have defined graph properly!')

        return check


# The function to do Topological Sort. Inefficient for large networks
    def is_acyclic(self): 
        """       
        Does a topological sort obtained from:
        https://www.geeksforgeeks.org/topological-sorting/#
        
        Returns
        -------
        acyclic : Bool 

        """
        # Check if there was a cycle
        # Create a vector to store indegrees of all
        # vertices. Initialize all indegrees as 0.
        in_degree = [0]*(self.n_nodes)
        
        # Traverse adjacency lists to fill indegrees of
        # vertices. This step takes O(V + E) time
        for i in self.graph:
            for j in self.graph[i]:
                in_degree[j] += 1
    
        # Create an queue and enqueue all vertices with
        # indegree 0
        queue = []
        for i in range(self.n_nodes):
            if in_degree[i] == 0:
                queue.append(i)
    
        # Initialize count of visited vertices
        cnt = 0
    
        # Create a vector to store result (A topological
        # ordering of the vertices)
        top_order = []
    
        # One by one dequeue vertices from queue and enqueue
        # adjacents if indegree of adjacent becomes 0
        while queue:
    
            # Extract front of queue (or perform dequeue)
            # and add it to topological order
            u = queue.pop(0)
            top_order.append(u)
    
            # Iterate through all neighbouring nodes
            # of dequeued node u and decrease their in-degree
            # by 1
            for i in self.graph[u]:
                in_degree[i] -= 1
                # If in-degree becomes zero, add it to queue
                if in_degree[i] == 0:
                    queue.append(i)
    
            cnt += 1
        if cnt != self.n_nodes:
            acyclic = False
        else :
            acyclic = True        
        return acyclic
    
    def calculate_parent_nodes(self):
        """
        Parent nodes are calculated for each node
        
        """
        empty_list = [[] for _ in range(self.n_nodes)]  
        
        for i in range(len(self.graph)):
            for part in self.graph[i]:
                empty_list[part].append(i)                
        #print(empty_list)
        
        self.parent_nodes = empty_list
        
        
            
    # TODO: Generalize to get the labels of the unit operation in the figure       
    def figure(self):
        G = nx.DiGraph()
        for u, neighbors in self.graph.items():
            G.add_node(u)
            for v in neighbors:
                G.add_edge(u, v)
           
        pos = nx.spring_layout(G)
           
        nx.draw_networkx(G, pos, with_labels=True, font_weight='bold', node_size=100, node_color='skyblue', font_color='black', edge_color='gray')#, arrowsize= 30)
        plt.show()
        G.clear()
    
    # Class meta data for a more streamlined data processing
    def register_active_input_indices(self, active_input_indices: list):
        self.active_input_indices = active_input_indices
        print('Active input indices obtained')
        
    def register_uncertainty_variables(self, uncertain_input_indices: list):
        self.uncertain_input_indices = uncertain_input_indices          
        #test_list = [i for i in range(max(max(self.active_input_indices)))]
        test_list = max([max(i + [0]) for i in self.active_input_indices])
        test_list = [i for i in range(test_list + 1)]
        self.design_input_indices = [i for i in test_list if i not in self.uncertain_input_indices]       
        
        
    # Other properties
    @property
    def nw(self):
        try:
            nw = len(self.uncertain_input_indices)
        except:
            nw = 0
        return nw
    @property          
    def nz(self):
        try:
            nz = max(list(itertools.chain(*self.active_input_indices))) - len(self.uncertain_input_indices) + 1 # Number of design variables
        except:
            nz = max(list(itertools.chain(*self.active_input_indices))) + 1
        return nz
    
    @property
    def nx(self):
        return self.nz + self.nw
    
    # Discrete W values and objective function
    def register_discrete_uncertain_values(self, vals, indices):
        """
        Parameters
        ----------
        vals : list of lists
            These are the discrete values that a 
        indices : list
            index of uncertain variables with dicrete values
        Returns
        -------
        Torch tensor
        All combinations of the uncertain variables in tensor form
        
        
        TODO: saves a dictionary which saves values corresponding to index
             for problems with a combinations of continous and discrete uncertainties

        """
        
        if compare(indices,self.uncertain_input_indices) or len(self.uncertain_input_indices) == 0:
            print('uncertain variables not defined in the problem! Please try again !!!')
            sys.exit()
        elif set(indices) != set(self.uncertain_input_indices):
            print('Combination of discrete and continuous variables not supported in the current version')
            print('Contact developer, or wait for future versions')
            sys.exit()
        else:
            self.w_sets = vals
            all_combinations = itertools.product(*vals)
                    # Convert each combination to a Torch tensor
            tensors = [torch.tensor(combination) for combination in all_combinations]
            
            # Stack the tensors to create the final result
            self.w_combinations = torch.stack(tensors)
            
            self.w_num_combinations = self.w_combinations.size()[0]
            
            # self.dict_discrete = {}
            # j = 0
            # for i in indices:
            #     self.dict_discrete[i] = torch.tensor(vals[j])  
            #     j += 1
    
    
    def define_objective(self, objective, type_obj = 'white_box'):
        if type_obj == 'white_box':
            self.objective_function = GenericMCObjective(objective)
        else:
            self.objective_function = objective
    
    # TODO: Accomodate for different lengthscales for different nodes: Not a critical requirement as of now
    def set_model_hyperparameters(self, 
                          length_scale: int = 0.5,
                          output_scale: int = 1.0,
                          noise_level: int = 1e-4):
        """
        This method is onlt set for numerical experiments with FNs with
        Gaussian Processes. Not to be set under other circumstances
        
        entire FN consists of GP with same hyper-parameters is the current assumption        
        Parameters
        ----------
        length_scale : int
        output_scale : int
        """       
        
        print('Warning hyperparameters have been pre-set!')
        self.custom_hyperparameters = True
        self.length_scale = length_scale
        self.output_scale = output_scale
        self.noise_level = noise_level
    
    # For multi-objective optimization get a reference point
    def define_reference_point(self, ref_point):
        self.ref_point = ref_point
    
    
    ### DAG like methods 
    def get_n_nodes(self):
        return self.n_nodes
    
    def get_parent_nodes(self, k):
        return self.parent_nodes[k]
    
    def get_root_nodes(self):
        return self.root_nodes
        
        


if __name__ == '__main__':
    g = Graph(6)
    g.addEdge(0, 2)
    g.addEdge(0, 3)
    g.addEdge(1, 2)
    g.addEdge(1, 3)
    g.addEdge(2, 4)
    g.addEdge(2, 5)
    g.addEdge(3, 4)
    g.addEdge(3, 5)
    
    a = g.calculate_parent_nodes()
    # parent_nodes = []
    # parent_nodes.append([])
    # parent_nodes.append([])
    # parent_nodes.append([0, 1])
    # parent_nodes.append([0, 1])
    # parent_nodes.append([2, 3])
    # parent_nodes.append([2, 3])
    
    
    g.figure()
