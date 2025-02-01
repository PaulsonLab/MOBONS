#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 19:28:57 2025

@author: kudva.7
"""

import pickle

with open('ZDT4_part1.pickle', 'rb') as handle:
    data1 = pickle.load(handle)
    
with open('ZDT4_part2.pickle', 'rb') as handle:
    data2 = pickle.load(handle)
    
with open('ZDT4_part3.pickle', 'rb') as handle:
    data3 = pickle.load(handle)
    
    
    
data = {}


for i in data1:
    data[i] = data1[i]
    
    
for i in data2:
    data[i] = data2[i]

for i in data3:
    data[i] = data3[i]
    
    
with open('ZDT4.pickle', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL) 