#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 11:35:50 2022

@author: slava
"""

import numpy as np
from numpy.random import choice
from scipy.stats import truncnorm

######################################################################################################
# DISCRETE VARIABLE
######################################################################################################
class VarDiscrete:
    def __init__(self, num_categories, min_convergence_threshold =  0.0001, 
                 alpha = 0.7):
        
        self.num_categories = num_categories
        
        self.categories = np.array(range(0,self.num_categories))
        self.prob_vector = np.ones(self.num_categories)/self.num_categories
        
        self.min_convergence_threshold = min_convergence_threshold
        self.alpha = alpha
       
    def Generate(self):
        return choice(self.categories, 1, p=self.prob_vector)


    def GenerateN(self,N):
        return choice(self.categories, N, p=self.prob_vector)
        
        
    def IsConverged(self):
        for p in self.prob_vector:
            if( p > 0.5 and p < 1.0 - self.min_convergence_threshold ):
                return False
            if( p < 0.5 and p > self.min_convergence_threshold ):
                return False
        
        return True
        
    def UpdateParameters(self, data, iteration_num):
        
        tmp_p_vec = np.zeros(self.num_categories)
        for c in range(0,self.num_categories):
            tmp_p_vec[c] = np.count_nonzero(data == c)/len(data)
        
        self.prob_vector  = self.alpha*tmp_p_vec + (1.0-self.alpha)*self.prob_vector  
        

        
        
######################################################################################################


######################################################################################################
# CONTINUOUS VARIABLE
######################################################################################################
class VarContinuous:
    def __init__(self, init_mu, init_std, lower_bound = - np.inf, 
                 upper_bound = np.inf, std_convergence_threshold =  0.0001, 
                 is_alpha_smoothing_type = True, alpha = 0.7, betta = 0.7, q = 5):
        
        self.mu = init_mu
        self.std = init_std
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.std_convergence_threshold = std_convergence_threshold
        self.is_alpha_smoothing_type = is_alpha_smoothing_type
        self.alpha = alpha
        self.betta = betta
        self.q = q

    def Generate(self):
        if(self.lower_bound == - np.inf and self.upper_bound ==  np.inf):
            return np.random.normal(self.mu, self.std)
        else:
            return self.__TruncatedNormal(self.lower_bound, self.upper_bound, self.mu, self.std, 1)


    def GenerateN(self,N):
        if(self.lower_bound == - np.inf and self.upper_bound ==  np.inf):
            return np.random.normal(self.mu, self.std, N)
        else:
            return self.__TruncatedNormal(self.lower_bound, self.upper_bound, self.mu, self.std, N)
        
        
    def IsConverged(self):
        if(self.std <= self.std_convergence_threshold):
            return True
        else:
            return False
        
        
    def UpdateParameters(self, data, iteration_num):
        mu_t = np.mean(data)
        sigma_t = np.std(data)
            
        self.mu  = self.alpha*mu_t + (1.0-self.alpha)*self.mu
            
        if(True == self.is_alpha_smoothing_type):
            self.std = self.alpha*sigma_t + (1.0-self.alpha)*self.std
        else: 
            B_mod = self.betta - self.betta*np.power((1-1/iteration_num),self.q)
            self.std = B_mod*sigma_t + (1-B_mod)*self.std
        
    
    def __TruncatedNormal(self, clip_a, clip_b, mu, sigma, N):
        a, b = (clip_a - mu) / sigma, (clip_b - mu) / sigma    
        data = truncnorm.rvs(a, b, loc = mu, scale = sigma, size=N)
        return data
    
######################################################################################################


######################################################################################################
# CE OUTPUT OBJECT
######################################################################################################

class CEOutput:
    f_opt  = np.inf               # optimal value reached 
    opt_trj = []               # optimum trajectory reached 
    f_gamma  = np.inf              # CE gamma value reached 
    iteration_num  = 0        # current CE iteration
    is_all_convergence = False # True if all variables converged
    covergence_per_variable = [] # a list of variables with converged/not converged status
    mean_trj= []
 		
    def Print(self):
        print("---------------------------------------------------------")
        print("iteration: ",self.iteration_num)
        print("best value reached: ",self.f_opt)
        print("best trajectory reached: ",self.opt_trj)
        print("mean trajectory reached: ",self.mean_trj)
        print("gamma value reached: ",self.f_gamma)
        print("converged reached: ",self.is_all_convergence)
        print("convergence variablevize: ",self.covergence_per_variable)
        print("---------------------------------------------------------")



######################################################################################################



class CrossEntropy:

    def __StopConditionMet(self, t, max_iteration, convergence_status, gamma_array, best_perf_array, max_no_improve_iterations):
        
        if(t == max_iteration):
            return True
        
        if(True == convergence_status):
            return True 
        
        if(t >= max_no_improve_iterations and gamma_array.count(gamma_array[0]) == len(gamma_array) and 
                       best_perf_array.count(best_perf_array[0]) == len(best_perf_array)):
            return True 
          
        return False
    
    
    def CEAlgorithm(self, func, vars_list, N, rho, func_args = None,  is_minimization = True,  verbose = False,
                    max_iteration = np.inf, max_no_improve_iterations = 5, seed=None):
    
        #######################################################################################
        
        if(seed != None):
            np.random.seed(seed)
          
        # number of variables
        n = len(vars_list)    
        
        # alocate objects used in the algorithm
        t = 0                            # iteration
        trj = np.zeros(shape=(N,n))      # trajectory array
        S = np.zeros(N)                  # performance array 
        best_trj = np.array(n)           # best trajectory reached
        convergence_status = False       # variables convergence status
        if(True == is_minimization):     # best performance reached
            best_perf = np.inf
            gamma_id = int(np.ceil((rho)*N)) # index of gamma
            best_perf_id = 0
        else:
            best_perf = - np.inf    
            gamma_id = int(np.ceil((1-rho)*N)) # index of gamma
            best_perf_id = N-1
         
            
        gamma_array = []
        best_perf_array = []
        
        out = CEOutput()    
        
        ####################################################################################### 
            
        # while stop condition is not reached : TODO wtite function here
        while(False == self.__StopConditionMet(t, max_iteration, convergence_status, gamma_array, 
                                        best_perf_array, max_no_improve_iterations)):
            t = t+1 # increase iteration counter
            
            # generate trajectory
            for var_id in range(n):
                var = vars_list[var_id]
                trj[:, var_id] = var.GenerateN(N)
            
            # calculate performance
            for i in range(0,N):
                S[i] = func(trj[i], func_args)
             
            sortedids = np.argsort(S,kind = 'heapsort') # from smallest to largest
            S_sorted = S[sortedids]        
            gamma = S_sorted[gamma_id]    
            
            if(True == is_minimization):
                if(best_perf>S_sorted[best_perf_id]):
                    best_perf = S_sorted[best_perf_id]
                    best_trj = trj[sortedids[best_perf_id]].copy()
            else:
                if(best_perf<S_sorted[best_perf_id]):
                    best_perf = S_sorted[best_perf_id]
                    best_trj = trj[sortedids[best_perf_id]].copy()
             
                
            # update gamma_array and  best_perf_array
            gamma_array.append(gamma)
            best_perf_array.append(best_perf)
            
            if(t>max_no_improve_iterations):
                del gamma_array[0]
                del best_perf_array[0]
                
            # update probabilities
            
            if(True == is_minimization): 
                eliteids = sortedids[range(0,gamma_id)]           
            else:
                eliteids = sortedids[range(gamma_id,N)]           
            
            eliteTrj = trj[eliteids,:]
            
            convergence_status = True
            conv_variable = np.zeros(n)
            for var_id in range(n):
                var = vars_list[var_id]
                var.UpdateParameters(eliteTrj[:,var_id],t)
                if(False == var.IsConverged()):
                    convergence_status = False
                    conv_variable[var_id] = 0
                else: 
                    conv_variable[var_id] = 1
                    
            # update the CEOutput object
            out.f_opt = best_perf
            out.opt_trj = best_trj
            out.f_gamma = gamma
            out.iteration_num = t
            out.is_all_convergence = convergence_status
            out.covergence_per_variable = conv_variable
            
            mean_trj = []
            for v in vars_list:
                if(True == isinstance(v, VarContinuous)):
                   mean_trj.append(v.mu) 
                else:
                   mean_trj.append(v.prob_vector) 
            out.mean_trj = mean_trj
        
            if(True == verbose):
                out.Print()
            
        #######################################################################################    
       
        return out

#######################################################################################








def GuessVectorFunction(x,args):
    return np.sum(np.abs(args[0] - x))   
  
def GuessVectorFunctionMax(x,args):
    return - np.sum(np.abs(args - x))   
  
if __name__ == "__main__":  
    print("CE test")
    myargs = np.array([0,1,2,1,2.5])
    x = np.array([0,0,0,0,0])
    

    # create variable list
    v1 = VarDiscrete(2)
    v2 = VarDiscrete(3)
    v3 = VarDiscrete(4)
    v4 = VarDiscrete(5) 
    v5 = VarContinuous(0,10)
    vars_list = [v1,v2,v3,v4,v5]
    
    N = 1000
    rho = 0.1
    
    ce = CrossEntropy()
    
    #out = ce.CEAlgorithm(GuessVectorFunction, vars_list, N, rho, func_args = [myargs], verbose = True, 
    #                      is_minimization = True)
    
    out = ce.CEAlgorithm(GuessVectorFunction, vars_list, N, rho, func_args = [myargs], is_minimization = True)
    out.Print()
    
    #out = ce.CEAlgorithm(GuessVectorFunction, vars_list, N, rho, func_args = [myargs], verbose = True, seed=12345)
    
    #v5 = VarContinuous(0,10)
    #vars_list = [v1,v2,v3,v4,v5]
    
    #out = ce.CEAlgorithm(GuessVectorFunction, vars_list, N, rho, func_args = [myargs], verbose = True, seed=12345)
    
    
    
    
    
