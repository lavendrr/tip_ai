#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 09:48:38 2019

@author: rmoctezuma
"""
#Neural Networks are made up of multiple Perceptrons. It uses Perceptrons to generate new inputs for other Perceptrons.
def dot_product(u,v):
    if len(u) != len(v):
        print('Vectors are not the same length.')
        return
    dp = []
    for val in u:
        dp.append(val*v[(u.index(val))])
    return dp
#Perceptron Functions
def and_gate(u,v):
    weight1 = 1
    weight2 = 1
    weight3 = -1.5
    if (u*weight1)+(v*weight2)+weight3 > 0:
        return True
    else:
        return False
def or_gate(u,v):
    weight1 = 2
    weight2 = 2
    weight3 = -1
    if (u*weight1)+(v*weight2)+weight3 > 0:
        return True
    else:
        return False
def xor_gate(u,v):
    # 2 layered Neural Network
    #2 weights (OR)
    weight1 = 2
    weight2 = 2
    weight3 = -1
    #3 weights (NOT AND)
    weight4 = -1
    weight5 = -1
    weight6 = 1.5
    #1 weights (AND)
    weight7 = 1
    weight8 = 1
    weight9 = -1
    #output 2
    output2 = 0
    if (u*weight1)+(v*weight2)+weight3 > 0:
        output2 = 1
    #output 3
    output3 = 0
    if (u*weight4)+(v*weight5)+weight6 > 0:
        output3 = 1
    #output 1 - final check
    if (output2*weight7)+(output3*weight8)+weight9 > 0:
        return True
    else:
        return False
def matrix_mult(a,b):
    rows_a = len(a)
    cols_a = len(a[0])
    rows_b = len(b)
    cols_b = len(b[0])
    if cols_a != rows_b:
        print(cols_a)
        print(rows_b)
        print("Matrices have the wrong dimensions.")
        return
    #create blank matrix with the appropriate size
    c = [[0 for row in range(cols_b)] for col in range(rows_a)]
    #perform multiplication operations & fill out the matrix
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                c[i][j] += a[i][k] * b[k][j]
    return c
class neural_network(object):
    def __init__(self,w1,w2,h,n):
        self.weight1 = w1
        self.weight2 = w2
        self.hidden_nodes = h
        self.num_inputs = n
    
        
    