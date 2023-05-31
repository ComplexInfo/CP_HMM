# Numerical Results for "Extending Conformal Prediction to Hidden Markov Models with Exact Validity via de Finetti’s Theorem for Markov Chains"

## This file contains the codes used to obtain the numerical results contained in Sec. 4.1.
## The proposed algorithm is implemented in the "Methods.py" file and is imported from there

######################################################################################################################################################################################################
## Importing the required packages

import numpy as np
import itertools
import pickle
import os.path
from os import path
import matplotlib
import matplotlib.pyplot as plt

######################################################################################################################################################################################################
## Importing the required functions from from the Methods file (which contains the implementation of the proposed algorithm)

from Methods import *

######################################################################################################################################################################################################## 
# Parameters for numerical experiments

T_Vec = [200, 100, 50] #Calibration sequence lengths
T1_Vec = [1, 2, 3] #Prediction sequence lengths
PERF = 0 # 0: without exact P,B values, 1: with exact P,B values
Algo = 'Filter' #Filter or Smoother
ALPHA = 0.2 #Desired miscoverage level
NUM_ITER = 500 #Number of Monte-Carlo Iterations
p_diag_vec = [0.1, 0.3, 0.5, 0.7, 0.9] #Parameter of the Observation Probability Matrix (value of each diagonal element of B)
b_diag_vec = [0.5, 0.75, 0.9] #Parameter of the Observation Probability Matrix (value of each diagonal element of B)
p0 = np.array([1/2, 1/2]) #Distribution of Initial states

######################################################################################################################################################################################################## 
# Implementing the proposed Conformal Prediction for HMM framework with the above defined parameters as inputs

for T in T_Vec:
    for T1 in T1_Vec:
        for b_diag in b_diag_vec:
            for p_diag in p_diag_vec:  
                
                #Transition Probability Matrix
                P = np.array([[p_diag, 1-p_diag], [1-p_diag, p_diag]]) #HMM Setting with a 2 by 2 transition probability matrix
                calX_LEN = np.shape(P)[1] #State space {0,1} cardinality         

                #Observation Probability Matrix
                B = np.array([[b_diag, (1-b_diag)], [(1-b_diag), b_diag]])            
                calY_LEN = np.shape(B)[1] #Observation space {0,1}

                Name = 'Outputs_of_'+ 'p_' + str(p_diag) + '_b_' + str(b_diag) + '_' + Algo + '_Perf_' + str(PERF) + '_CPHMM_alpha_pt_2_T_' + str(T) + '_T1_' + str(T1) + '.pkl'

                #Creating a file if it doesnt exist
                if not path.isfile(Name):
                    with open(Name, "wb") as fp:   
                        pickle.dump([], fp)

                print(Name)

                with open(Name, 'rb') as f:
                    Outputs_of_CPHMM = pickle.load(f)

                while len(Outputs_of_CPHMM)<NUM_ITER:    
                    XY_full = sim_HMM(p0, P, B, (T+T1))

                    X = XY_full[0][:-T1]
                    Y = XY_full[1][:-T1]

                    X_pred = XY_full[0][-T1:]
                    Y_pred = XY_full[1][-T1:]


                    print("")    
                    print("*********BEGINNING A NEW ITERATION*********")   
                    print(Name)
                    print("Length:")
                    print(len(Outputs_of_CPHMM))

                    print("True State Sequence: ")    
                    print(X_pred)
                    print("")

                    if Algo == 'Smoother':
                        OP = CP_HMM_smoothing(X, Y, X_pred, Y_pred, ALPHA, calX_LEN, calY_LEN, perf = PERF)
                    if Algo == 'Filter':
                        OP = CP_HMM_filtering(X, Y, X_pred, Y_pred, ALPHA, calX_LEN, calY_LEN, perf = PERF)

                    with open(Name, 'rb') as f:
                        Outputs_of_CPHMM = pickle.load(f)

                    Outputs_of_CPHMM.append(OP)

                    with open(Name, "wb") as fp:   #Pickling
                        pickle.dump(Outputs_of_CPHMM, fp)   


                    print("*********SUMMARY OF THE FINISHED ITERATION*********")
                    print("True State Sequence: ")
                    print(Outputs_of_CPHMM[-1][0])

                    print("Output Set of Sequences: ")
                    print(Outputs_of_CPHMM[-1][1])

                    print("True State is in Prediction Set: ")
                    print(Outputs_of_CPHMM[-1][0] in Outputs_of_CPHMM[-1][1])


######################################################################################################################################################################################################## 

# Visualizing results

T_Vec = [50, 100, 200] #Calibration sequence lengths
T1_Vec = [1, 2, 3] #rediction length
PERF = 0 # 0: without exact P,B values, 1: with exact P,B values
Algo = 'Filter' #Filter or Smoother
ALPHA = 0.2 #Desired miscoverage level
NUM_ITER = 500 #Number of Monte-Carlo Iterations
p_diag_vec = [0.1, 0.3, 0.5, 0.7, 0.9] #Parameter of the Observation Probability Matrix (value of each diagonal element of B)
b_diag_vec = [0.5, 0.75, 0.9] #Parameter of the Observation Probability Matrix (value of each diagonal element of B)
p0 = np.array([1/2, 1/2]) #Distribution of Initial states


# Parameters of figures
plt.rcParams["mathtext.fontset"]
fig, ax = plt.subplots(3, 3, figsize=(7, 7))

COLS = ['g', 'm']
MARKERS = ['s', 'o', '^']
LINESTYLES = ['--', '-.', ':']
BASE_MARKER_SIZE = 6; MARKER_STEP_SIZE = 0
MARKERSIZE = [BASE_MARKER_SIZE,BASE_MARKER_SIZE-MARKER_STEP_SIZE,BASE_MARKER_SIZE-2*MARKER_STEP_SIZE]

margin_for_lines = 0.05 


subfig_row_index = 0
for T in T_Vec:
    subfig_col_index = 0
    for T1 in T1_Vec:
        STATE_SPACE_CARD = 2**T1
        
        #Plotting the results for each T and T1 in seperate subfigures
        Results_to_plot = []
        for p_diag in p_diag_vec:
            for b_diag in b_diag_vec:
                Name = 'Outputs_of_'+ 'p_' + str(p_diag) + '_b_' + str(b_diag) + '_' + Algo + '_Perf_' + str(PERF) + '_CPHMM_alpha_pt_2_T_' + str(T) + '_T1_' + str(T1) + '.pkl'

                                
                if path.isfile(Name):
                    with open(Name, 'rb') as f:
                        Outputs_of_CPHMM = pickle.load(f) 
                    if len(Outputs_of_CPHMM)!=0:
                        ITER = 0
                        Prediction_Set_Size = 0
                        TrueCount = 0    
                        while ITER < len(Outputs_of_CPHMM):
                            if Outputs_of_CPHMM[ITER][0] in Outputs_of_CPHMM[ITER][1]:
                                TrueCount = TrueCount + 1
                            Prediction_Set_Size = Prediction_Set_Size + len(Outputs_of_CPHMM[ITER][1]) 
                            ITER += 1    

                        Results_to_plot.append((p_diag, b_diag, TrueCount/len(Outputs_of_CPHMM), (Prediction_Set_Size/len(Outputs_of_CPHMM))/STATE_SPACE_CARD))
        if len(Results_to_plot) != 0:
            print(Name)
            print("Non zero length results list exist")
            print("")
            
            p_vals,b_vals,True_Frac,Pred_Size_Frac = list(zip(*Results_to_plot))          

            V = 0
            for b_diag in b_diag_vec:
                indices = list(np.where(np.array(b_vals) == b_diag)[0])
                ax[subfig_row_index,subfig_col_index].plot([p_vals[ind] for ind in indices], [True_Frac[ind] for ind in indices],COLS[0], linestyle = LINESTYLES[V], marker = MARKERS[V], markerfacecolor='none', markersize = MARKERSIZE[V], label=r'Empirical Coverage for $b = $'+str(b_diag))
                ax[subfig_row_index,subfig_col_index].plot([p_vals[ind] for ind in indices], [Pred_Size_Frac[ind] for ind in indices],COLS[1],linestyle = LINESTYLES[V], marker = MARKERS[V], markerfacecolor='none', markersize = MARKERSIZE[V], label=r'$\mathrm{\frac{\mathbb{E}\{|\mathcal{C}_{0.8}|\}}{|\mathcal{X}^{T_1}|}}$ for $b=$'+str(b_diag))
                V+=1


            ax[subfig_row_index,subfig_col_index].axhline(y=0.8, xmin=0, xmax=3, c='r', linestyle = '-', linewidth=2, zorder=0, label = r'Desired coverage $1-\alpha = 0.8$')
            ax[subfig_row_index,subfig_col_index].axhline(y=0.8 - margin_for_lines, xmin=0, xmax=3, c='r', linestyle = ':', linewidth=1, zorder=0, label = r'$\pm$' + str(margin_for_lines) + r' margins of $1-\alpha$')
            ax[subfig_row_index,subfig_col_index].axhline(y=0.8 + margin_for_lines, xmin=0, xmax=3, c='r', linestyle = ':', linewidth=1, zorder=0)    

            ax[subfig_row_index,subfig_col_index].set_xticks([0.1,0.3,0.5,0.7,0.9]) 
            ax[subfig_row_index,subfig_col_index].set_xlim(0.0,1.0)            
            ax[subfig_row_index,subfig_col_index].set_yticks([0.0,0.2,0.4,0.6,0.8,1]) 
            ax[subfig_row_index,subfig_col_index].set_ylim(0.0,1.0)            
            ax[subfig_row_index,subfig_col_index].tick_params(axis='both', which='major', labelsize=12)
            
            ax[subfig_row_index,subfig_col_index].grid()

            ax[subfig_row_index,subfig_col_index].set_title(r'$T =$' + str(T) + r', $T_1 = $' + str(T1) , fontsize = 14)
    
        subfig_col_index+=1
    subfig_row_index+=1

fig.tight_layout(pad=1)          
ax[2,1].set_xlabel(r'$p$ (probability of staying in the same state)', fontsize = 14.5, labelpad=8)
ax[1,0].set_ylabel('Empirical coverage (green) and scaled prediction set size (purple)', fontsize = 14.5, labelpad=8)
ax[0,0].legend(loc='upper center', bbox_to_anchor=(1.65, 2.25), fontsize = 13, edgecolor = 'inherit', ncol = 2)            
plt.savefig('All_Numerical_Results_2_by_2' + '.pdf', bbox_inches='tight')                                      
