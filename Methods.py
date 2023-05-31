import numpy as np
import itertools
import pickle

#import matplotlib
# matplotlib.use('TkAgg')
#import matplotlib.pyplot as plt
#import pandas as pd
#from scipy.stats import bernoulli
#from scipy.stats import pareto
#import networkx as nx
#import matplotlib.pyplot as plt
import random
#import scipy.io
#import collections
#from mpl_toolkits.axes_grid1.inset_locator import (inset_axes,InsetPosition,mark_inset)
#import copy


############################################################################################################################################

"""
Function to simulate a sequence from an HMM with parameters
p0: array (initial probability vector)
P: 2-D arrray (transition probability matrix)
B: 2-D arrray (observation probability matrix)
n: integer (length of the sequence)
"""

def sim_HMM(p0, P, B, n):
    
    calX = np.arange(len(p0))
    calY = np.arange(B.shape[1])
    
    X = [np.random.choice(calX, p=p0)]
    while len(X)<n:
        X_old = X[-1]
        X.append(np.random.choice(calX, p=P[X_old,:]))
    
    Y = []
    for k in np.arange(len(X)):
        Y.append(np.random.choice(calY, p=B[X[k],:]))

    return(np.array([X,Y]))
    
    
    
############################################################################################################################################

"""
Function to estimate P and B from a given HMM
X: state sequence (array)
Y: observation sequence (array)
"""

def est_HMM(X,Y):
    calX = np.unique(X)
    calY = np.unique(Y)
    
    P_hat = np.zeros((calX.size,calX.size))  
    for x in calX:
        Indices = np.where(X == x)[0]+1 #Indices of all elements that immediately follow the element x

        if np.where(Indices==(len(X)))[0].size !=0: # the case where last element of the sequence X is x
            Indices = np.delete(Indices,np.where(Indices==(len(X)))[0][0])
            

        for s in calX: #Estimating the frequency of transitions to each state s from x
            P_hat[x,s] = np.where(X[Indices] == s)[0].size/Indices.size
           
        
    B_hat = np.zeros((calX.size,calY.size))
    for x in calX:
        Indices = np.where(X == x)[0]
        
        for y in calY: #Estimating the frequency of observation y for each state s from x
            B_hat[x,y] = np.where(Y[Indices] == y)[0].size/Indices.size

    return(P_hat,B_hat) 
        
    
    
        
############################################################################################################################################

"""
Function to find the starting and end points of all (i,j)-blocks of the sequences X (ends with i) and Y (ends with j)
"""

def find_ij_blocks(X,Y,ij_index):
    ijblock_start_indices = np.intersect1d(np.where(X == X[-ij_index])[0],np.where(Y == Y[-ij_index])[0])
    ijblock_start_end_indices = []
    
    #Creating the (ij)-blocks
    for i in np.arange(ijblock_start_indices.size-1):
        ijblock_start_end_indices.append((ijblock_start_indices[i],ijblock_start_indices[i+1])) 
    
    #If there are no (ij)-blocks return the whole sequence as the only block. If not, return all (ij)-blocks.
    if len(ijblock_start_indices) > 1:
        return ijblock_start_end_indices
    else:
        return [(0,ijblock_start_indices[0])]


############################################################################################################################################

def CP_HMM_filtering(X, Y, X_pred, Y_pred, ALPHA, calX_LEN, calY_LEN, perf = 0):
    calX = np.arange(calX_LEN)
    calY = np.arange(calY_LEN)
    
    alpha = ALPHA    
    
    T = X.size
    T1 = Y_pred.size
    Y_augmented = np.concatenate([Y,Y_pred])

    Output = []
    k = 0
    list_to_take_products = []
    while k<T1:
        list_to_take_products.append(calX)
        k+=1


    for cand_sequence in itertools.product(*list_to_take_products):
        X_augmented = np.concatenate([X,np.array(cand_sequence)]) #Step 1 of the Algorithm 1
#         print('step 1 done')

        if perf == 0:
            (P_hat,B_hat) = est_HMM(X_augmented, Y_augmented) #Estimating the parameters of the HMM (step 2 of Algorithm 1)
        if perf == 1:
            P_hat = P
            B_hat = B
#         print('step 2 done')

        #Creating all permutations of the last T1+1 ij-blocks (Step 3 of the algorithm)
        permutaions_of_ij_Blocks = []
        ij_index = 1
        while len(permutaions_of_ij_Blocks)==0 and ij_index <= T1:
            List_of_ij_Blocks = find_ij_blocks(X_augmented,Y_augmented,ij_index)
            permutaions_of_ij_Blocks = list(itertools.permutations(List_of_ij_Blocks,T1+1))
            ij_index+=1
#         print('step 3 done at ij_index = ' + str(ij_index-1))

        if len(permutaions_of_ij_Blocks) > 0:
            # Calculating the conformity score (Step 4 of the algorithm)
            List_of_Conformity_Scores = []
            for ij_block_pair in permutaions_of_ij_Blocks:

                #for each pair of ij-blocks, creating a block with the last element of the first block + second block 
                X_block = np.array([])
                Y_block = np.array([])
                for iter in np.arange(0,T1+1):
                    X_block = np.concatenate([X_block, X_augmented[ij_block_pair[iter][0]:ij_block_pair[iter][1]]])
                    Y_block = np.concatenate([Y_block, Y_augmented[ij_block_pair[iter][0]:ij_block_pair[iter][1]]])
                X_block = X_block[-(T1+1):].astype(int)  
                Y_block = Y_block[-(T1+1):].astype(int)  

                #HMM Filter
                X_block_to_Predict =  X_block[-T1:]
                Y_block_to_Predict =  Y_block[-T1:]

                normalized_filter_density_k = (calX == X_block[0]) + 0 
                HMM_filter_iteration = 0
                HMM_filter_densities = np.zeros((len(calX),T1))
                while HMM_filter_iteration<T1:
                    y_k_plus_1 = Y_block_to_Predict[HMM_filter_iteration]
                    B_hat_yk_plus_1 = np.diag(B_hat[:,y_k_plus_1])
                    unnormalized_filter_density_k_plus_1 = np.dot(B_hat_yk_plus_1,P_hat.transpose()).dot(normalized_filter_density_k)
                    normalized_filter_density_k_plus_1 = unnormalized_filter_density_k_plus_1/np.sum(unnormalized_filter_density_k_plus_1)            
                    HMM_filter_densities[:,HMM_filter_iteration] = normalized_filter_density_k_plus_1
                    normalized_filter_density_k = normalized_filter_density_k_plus_1            
                    HMM_filter_iteration+=1

                S = 0
                for iter in np.arange(0,len(X_block_to_Predict)):
                    S = S + (1-HMM_filter_densities[:,iter][X_block_to_Predict[iter]])
                List_of_Conformity_Scores.append(S)          
#             print('step 4 done')

            # Fraction of permutations (Step 5 of the algorithm)

            #take the last T1+1 elements of the augmendted sequence
            X_block = X_augmented[-(T1+1):]    
            Y_block = Y_augmented[-(T1+1):]

            #HMM Filter
            X_block_to_Predict =  X_block[-T1:]
            Y_block_to_Predict =  Y_block[-T1:]

            normalized_filter_density_k = (calX == X_block[0]) + 0 
            HMM_filter_iteration = 0
            HMM_filter_densities = np.zeros((len(calX),T1))
            while HMM_filter_iteration<T1:
                y_k_plus_1 = Y_block_to_Predict[HMM_filter_iteration]
                B_hat_yk_plus_1 = np.diag(B_hat[:,y_k_plus_1])
                unnormalized_filter_density_k_plus_1 = np.dot(B_hat_yk_plus_1,P_hat.transpose()).dot(normalized_filter_density_k)
                normalized_filter_density_k_plus_1 = unnormalized_filter_density_k_plus_1/np.sum(unnormalized_filter_density_k_plus_1)            
                HMM_filter_densities[:,HMM_filter_iteration] = normalized_filter_density_k_plus_1
                normalized_filter_density_k = normalized_filter_density_k_plus_1            
                HMM_filter_iteration+=1

            S_identity = 0
            for iter in np.arange(0,len(X_block_to_Predict)):
                S_identity = S_identity + (1-HMM_filter_densities[:,iter][X_block_to_Predict[iter]])

            if sum(List_of_Conformity_Scores>=S_identity)/len(permutaions_of_ij_Blocks) > alpha:
                Output.append(cand_sequence)
        else:
            if random.uniform(0,1)<=(1-alpha):
                Output.append(cand_sequence)
          
    return((tuple(X_pred), Output))


############################################################################################################################################

def CP_HMM_smoothing(X, Y, X_pred, Y_pred, ALPHA, calX_LEN, calY_LEN, perf = 0):
    calX = np.arange(calX_LEN)
    calY = np.arange(calY_LEN)
    
    alpha = ALPHA    
    
    T = X.size
    T1 = Y_pred.size
    Y_augmented = np.concatenate([Y,Y_pred])

    Output = []
    k = 0
    list_to_take_products = []
    while k<T1:
        list_to_take_products.append(calX)
        k+=1


    for cand_sequence in itertools.product(*list_to_take_products):
        X_augmented = np.concatenate([X,np.array(cand_sequence)]) #Step 1 of the Algorithm 1
#         print('step 1 done')
        
        if perf == 0:
            (P_hat,B_hat) = est_HMM(X_augmented, Y_augmented) #Estimating the parameters of the HMM (step 2 of Algorithm 1)
        if perf == 1:
            P_hat = P
            B_hat = B
#         print('step 2 done')

        #Creating all permutations of the last T1+1 ij-blocks (Step 3 of the algorithm)
        permutaions_of_ij_Blocks = []
        ij_index = 1
        while len(permutaions_of_ij_Blocks)==0 and ij_index <= T1:
            List_of_ij_Blocks = find_ij_blocks(X_augmented,Y_augmented,ij_index)
            permutaions_of_ij_Blocks = list(itertools.permutations(List_of_ij_Blocks,T1+1))
            ij_index+=1
#         print('step 3 done at ij_index = ' + str(ij_index-1))

        if len(permutaions_of_ij_Blocks) > 0:
            # Calculating the conformity score (Step 4 of the algorithm)
            List_of_Conformity_Scores = []
            for ij_block_pair in permutaions_of_ij_Blocks:

                #for each pair of ij-blocks, creating a block with the last element of the first block + second block 
                X_block = np.array([])
                Y_block = np.array([])
                for iter in np.arange(0,T1+1):
                    X_block = np.concatenate([X_block, X_augmented[ij_block_pair[iter][0]:ij_block_pair[iter][1]]])
                    Y_block = np.concatenate([Y_block, Y_augmented[ij_block_pair[iter][0]:ij_block_pair[iter][1]]])
                X_block = X_block[-(T1+1):].astype(int)  
                Y_block = Y_block[-(T1+1):].astype(int)  

                #HMM Filter
                X_block_to_Predict =  X_block[-T1:]
                Y_block_to_Predict =  Y_block[-T1:]

                normalized_filter_density_k = (calX == X_block[0]) + 0 
                HMM_filter_iteration = 0
                HMM_filter_densities = np.zeros((len(calX),T1))
                while HMM_filter_iteration<T1:
                    y_k_plus_1 = Y_block_to_Predict[HMM_filter_iteration]
                    B_hat_yk_plus_1 = np.diag(B_hat[:,y_k_plus_1])
                    unnormalized_filter_density_k_plus_1 = np.dot(B_hat_yk_plus_1,P_hat.transpose()).dot(normalized_filter_density_k)
                    normalized_filter_density_k_plus_1 = unnormalized_filter_density_k_plus_1/np.sum(unnormalized_filter_density_k_plus_1)            
                    HMM_filter_densities[:,HMM_filter_iteration] = normalized_filter_density_k_plus_1
                    normalized_filter_density_k = normalized_filter_density_k_plus_1            
                    HMM_filter_iteration+=1                
                    
                #HMM Smoother
                Beta = np.ones((np.shape(P_hat)[0],1))
                Beta_iteration = np.shape(Beta)[1]
                while Beta_iteration<T1:
                    y_k_plus_1 = Y_block_to_Predict[-Beta_iteration]
                    B_hat_yk_plus_1 = np.diag(B_hat[:,y_k_plus_1])
                    Beta = np.c_[np.dot(P_hat, B_hat_yk_plus_1).dot(Beta[:,-Beta_iteration]),Beta]
                    Beta_iteration+=1
                
                
                HMM_smoother_densities = np.zeros((len(calX),T1))
                Smoother_iteration = 0
                while Smoother_iteration<np.shape(HMM_smoother_densities)[1]:
                    HMM_smoother_densities[:,Smoother_iteration] = np.multiply(HMM_filter_densities[:,Smoother_iteration],Beta[:,Smoother_iteration])/(np.dot(HMM_filter_densities[:,Smoother_iteration],Beta[:,Smoother_iteration]))
                    Smoother_iteration+=1
                    
                S = 0
                for iter in np.arange(0,len(X_block_to_Predict)):
                    S = S + (1-HMM_smoother_densities[:,iter][X_block_to_Predict[iter]])
                List_of_Conformity_Scores.append(S)          
#             print('step 4 done')

            # Fraction of permutations (Step 5 of the algorithm)

            #take the last T1+1 elements of the augmendted sequence
            X_block = X_augmented[-(T1+1):]    
            Y_block = Y_augmented[-(T1+1):]

            #HMM Filter
            X_block_to_Predict =  X_block[-T1:]
            Y_block_to_Predict =  Y_block[-T1:]

            normalized_filter_density_k = (calX == X_block[0]) + 0 
            HMM_filter_iteration = 0
            HMM_filter_densities = np.zeros((len(calX),T1))
            while HMM_filter_iteration<T1:
                y_k_plus_1 = Y_block_to_Predict[HMM_filter_iteration]
                B_hat_yk_plus_1 = np.diag(B_hat[:,y_k_plus_1])
                unnormalized_filter_density_k_plus_1 = np.dot(B_hat_yk_plus_1,P_hat.transpose()).dot(normalized_filter_density_k)
                normalized_filter_density_k_plus_1 = unnormalized_filter_density_k_plus_1/np.sum(unnormalized_filter_density_k_plus_1)            
                HMM_filter_densities[:,HMM_filter_iteration] = normalized_filter_density_k_plus_1
                normalized_filter_density_k = normalized_filter_density_k_plus_1            
                HMM_filter_iteration+=1
                
            #HMM Smoother
            Beta = np.ones((np.shape(P_hat)[0],1))
            Beta_iteration = np.shape(Beta)[1]
            while Beta_iteration<T1:
                y_k_plus_1 = Y_block_to_Predict[-Beta_iteration]
                B_hat_yk_plus_1 = np.diag(B_hat[:,y_k_plus_1])
                Beta = np.c_[np.dot(P_hat, B_hat_yk_plus_1).dot(Beta[:,-Beta_iteration]),Beta]
                Beta_iteration+=1

            HMM_smoother_densities = np.zeros((len(calX),T1-1))
            Smoother_iteration = 0
            while Smoother_iteration<np.shape(HMM_smoother_densities)[1]:
                HMM_smoother_densities[:,Smoother_iteration] = np.multiply(HMM_filter_densities[:,Smoother_iteration],Beta[:,Smoother_iteration])/(np.dot(HMM_filter_densities[:,Smoother_iteration],Beta[:,Smoother_iteration]))
                Smoother_iteration+=1
            HMM_smoother_densities = np.c_[HMM_smoother_densities,HMM_filter_densities[:,-1]]
                                
            S_identity = 0
            for iter in np.arange(0,len(X_block_to_Predict)):
                S_identity = S_identity + (1-HMM_smoother_densities[:,iter][X_block_to_Predict[iter]])

            if sum(List_of_Conformity_Scores>=S_identity)/len(permutaions_of_ij_Blocks) > alpha:
                Output.append(cand_sequence)
        else:
            if random.uniform(0,1)<=(1-alpha):
                Output.append(cand_sequence)
            
    return((tuple(X_pred), Output))


