# -*- coding: utf-8 -*-
"""implement a cross validation."""
import numpy as np
import matplotlib.pyplot as plt
from costs import compute_mse
from ridge_regression import ridge_regression
from build_polynomial import build_poly

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)



def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # get k'th subgroup in test, others in train: TODO
    # ***************************************************
    x_test = x[k_indices[k,:]]
    y_test = y[k_indices[k,:]]
    
    x_train = x
    x_train = np.delete(x_train, (k_indices[k,:]), axis=0)
    
    y_train = y
    y_train = np.delete(y_train, (k_indices[k,:]), axis=0)
    # ***************************************************
    # INSERT YOUR CODE HERE
    # form data with polynomial degree: TODO
    # ***************************************************
    # tx_train = build_poly(x_train, degree)
    # tx_test = build_poly(x_test, degree)
    # ***************************************************
    # INSERT YOUR CODE HERE
    # ridge regression: TODO
    # ***************************************************
    w = ridge_regression(y_train, x_train, lambda_)
    # ***************************************************
    # INSERT YOUR CODE HERE
    # calculate the loss for train and test data: TODO
    # ***************************************************

    #loss_tr = compute_mse(y_train, x_train, w)
    #loss_te = compute_mse(y_test, x_test, w)
    
    z = x_train.dot(w)
    sigma = np.exp(z)/(1+np.exp(z))

    ind_back = np.where(sigma<0.5)[0]
    ind_sign = np.where(sigma>0.5)[0]
    
    prediction = np.ones(y_train.shape[0])
    prediction[ind_back] = -1
    
    train_error = np.where(prediction != y_test)[0].shape[0]/y_test.shape[0]
    
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