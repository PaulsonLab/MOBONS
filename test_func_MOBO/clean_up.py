#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 10:44:24 2025

@author: kudva.7
"""

import pickle


import pickle

with open('ZDT4_NSGA_part1.pickle', 'rb') as handle:
    data1 = pickle.load(handle)
    
with open('ZDT4_NSGA_part2.pickle', 'rb') as handle:
    data2 = pickle.load(handle)
    
with open('ZDT4_NSGA_part3.pickle', 'rb') as handle:
    data3 = pickle.load(handle)
    
    
    
data = {}


data[0] = data1
data[1] = data2
data[2] = data3

    
with open('ZDT4_NSGA.pickle', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL) 
