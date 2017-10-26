# -*- coding: utf-8 -*-
"""Try to find the best combinations of features, Etienne's idea"""
import numpy as np
from logistic_regression import *

def best_combination(data, label):
    # Try to optimize these parameters **************************
    gamma = 0.7
    lambda_ = 1000
    initial_w = np.zeros((data.shape[1], 1))
    max_iter = 100
    #************************************************************
    
    numOfFeatures = 0
    the_best_combination_ind = []
    model_inProgress = []
    l_min = 1e50 # set it big to be able to replace it
    
    while numOfFeatures < 200:
        print(numOfFeatures)
        for ind_f in range(data.shape[1]):
            if ind_f not in the_best_combination_ind:
                if numOfFeatures == 0: 
                    model_temp = data[:,ind_f]
                else:
                    model_temp = model_inProgress
                    model_temp.append(data[:,ind_f])
                w_temp, loss_temp = learning_by_penalized_gradient(label, model_temp, initial_w, max_iter, gamma, lambda_)
                if loss_temp < l_min:
                    l_min = loss_temp
                    feature_indice = ind_f
        the_best_combination_ind.append(feature_indice) # To keep all the indices
        model_inProgress.append(data[:,feature_indice]) # to build the model
        numOfFeatures += 1
   
    return model_inProgress
                