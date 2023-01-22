#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import random

def VU_init(A,k1,k2,flag,data):
    # set n to the number of rows of Attribute matrix (nodes)
    # set m to number of columns of attribute matrix (attributes)
    n = A.shape[0]
    m = A.shape[1]
    # if there is no ground truth initialize matrices randomly
    if flag == 0:
        U = np.random.random((n,k1))
        V = np.random.random((m,k2))
    else:
        # Create a string with path to the U data for k1 number
        infile = 'initialize/'+data+'_U_'+str(k1)+'.csv'
        # create an empty U list
        U = []
        # open the file
        with open(infile) as f:
            while True:
                row = []
                # read a line from the file
                line = f.readline()
                if line == '':
                    break
                # strip the newline character from the string (line)
                line = line.rstrip('\n')
                # split the string on commas (separate values)
                tmp = line.split(',')
                # add each value to the row list
                for i in tmp:
                    row.append(float(i))
                # add the row to the U list
                U.append(row)
        infile = 'initialize/'+data+'_V_'+str(k2)+'.csv'
        V = []
        with open(infile) as f:
            while True:
                row = []
                line = f.readline()
                if line == '':
                    break
                line = line.rstrip('\n')
                tmp = line.split(',')
                for i in tmp:
                    row.append(float(i))
                V.append(row)
        # convert the U and V lists to numpy arrays
        # to perform matrix calculations
        U = np.array(U)
        V = np.array(V).transpose()
    return U,V
