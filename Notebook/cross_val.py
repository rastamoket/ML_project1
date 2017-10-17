# -*- coding: utf-8 -*-
"""implement a cross validation."""
import numpy as np
import matplotlib.pyplot as plt

from costs import compute_mse
from ridge_regression import ridge_regression
from build_polynomial import build_poly
from logistic_regression import *
from proj1_helpers import *

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)



def cross_validation(y, x, k_indices, k, gamma):
    """Cross validation
    List of the possible hyperparameters:
    - gamma
    - max iters? --> I think it is better not to be an hyperparameters but add some conditions to it (like threshold for convergence) 
    """
    # get k'th subgroup in test
    x_test = x[k_indices[k,:]]
    y_test = y[k_indices[k,:]]
    
    # Take all the others sets (except the k'th) for the training
    x_train = x
    x_train = np.delete(x_train, (k_indices[k,:]), axis=0)
    
    y_train = y
    y_train = np.delete(y_train, (k_indices[k,:]), axis=0)

    # Calculate the weights --> We might use them for initial weight ???????????????
    #w_init = ridge_regression(y_train, x_train, lambda_)
    w_init = np.zeros(x_train.shape[1])

    #loss_tr = compute_mse(y_train, x_train, w)
    #loss_te = compute_mse(y_test, x_test, w)
    
    # Definition of the different useful variables for learning_grad_descent
    
    max_iters = 1000 # Put it higher BUT add some condition for convergence (threshold)
    
    w, loss = learning_grad_descent(y_train, x_train, w_init, max_iters, gamma) 

    z = x_train.dot(w)
    sigma = sigmoid(z)

    ind_back = np.where(sigma<0.5)[0]
    ind_sign = np.where(sigma>0.5)[0]
    
        
    y_pred = predict_labels(w, x_test)
    
    train_error = np.where(y_pred == y_test)[0].shape[0]/y_test.shape[0]
    
    #prediction = np.ones(y_train.shape[0])
    #prediction[ind_back] = -1
    
    return train_error

def cross_validation_visualization(lambds, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")