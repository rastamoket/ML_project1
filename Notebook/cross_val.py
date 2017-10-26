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



def cross_validation(y, x, k_indices, k, gamma, lambda_):
    """Cross validation:
    argument:
        - y: labels
        - x: data
        - k_indices: indices of the sets 
        - k: indice of the testing set
        - gamma: step 
        - lambda_: regularize term

    """
    
    # To be sure that we don't have any -1 in our labels
    if(np.any(y == -1)):
        y[np.where(y == -1)[0]] = 0
    
    
    # get k'th subgroup in test
    x_test = x[k_indices[k,:], :]
    y_test = y[k_indices[k,:]]
    
    # Take all the others sets (except the k'th) for the training
    x_train = x
    x_train = np.delete(x_train, (k_indices[k,:]), axis=0)
    
    y_train = y
    y_train = np.delete(y_train, (k_indices[k,:]), axis=0)

    # Calculate the weights --> We might use them for initial weight ???????????????
    #w_init = ridge_regression(y_train, x_train, lambda_)
    # w_init = np.zeros(x_train.shape[1])
    w_init = np.zeros((x_train.shape[1], 1))
    
    # Definition of the different useful variables for learning_grad_descent
    
    #max_iters = 1000 # Put it higher BUT add some condition for convergence (threshold)
    
<<<<<<< HEAD
    #w, loss = learning_grad_descent(y_train, x_train, w_init, max_iters, gamma) --> PREVIOUS
    w, loss = learning_by_penalized_gradient(y_train, x_train, w_init, max_iters, gamma, lambda_)
    
    #batch_size=15000
    #max_iters = round(y_train.shape[0]/batch_size) # Put it higher BUT add some condition for convergence (threshold)
=======
        #w, loss = learning_grad_descent(y_train, x_train, w_init, max_iters, gamma) --> PREVIOUS
        #w, loss = learning_by_penalized_gradient(y_train, x_train, w_init, max_iters, gamma, lambda_)
    
    #batch_size=7000
>>>>>>> 68fb7b0c266698d7a4322c7f62004c2827856353
    #w,loss=penalized_stochastic_gradient_descent(y_train, x_train, w_init, batch_size, max_iters, gamma, lambda_)
    
    #z = x_train.dot(w)
    #sigma = sigmoid(z)

    #ind_back = np.where(sigma<0.5)[0]
    #ind_sign = np.where(sigma>0.5)[0]
   
<<<<<<< HEAD
    y_pred = np.array(predict_labels(w, x_test))
    y_test = np.reshape(y_test,(len(y_test),1))
    print(y_test.shape, y_pred.shape)
    
    validation_error = np.where(y_pred == y_test)[0].shape[0]/y_test.shape[0] # This will give the error!!!
=======
    #y_pred = predict_labels(w, x_test)
    
    #train_error = np.where(y_pred != y_test)[0].shape[0]/y_test.shape[0] # This will give the error!!!
    
    #return train_error
    #NEW VERSION from now***********************************************************
    batch_size=15000
    max_iters = round(y_train.shape[0]/batch_size) # Put it higher BUT add some condition for convergence (threshold)
    w,loss=penalized_stochastic_gradient_descent(y_train, x_train, w_init, batch_size, max_iters, gamma, lambda_)
    
   
    y_pred = np.array(predict_labels(w, x_test))
    y_test = np.reshape(y_test,(len(y_test),1))
    
    validation_error = np.where(y_pred != y_test)[0].shape[0]/y_test.shape[0] # This will give the error!!!
>>>>>>> 68fb7b0c266698d7a4322c7f62004c2827856353

    print(validation_error)
    return validation_error

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