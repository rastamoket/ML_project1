# -*- coding: utf-8 -*-
"""logistic regression."""
import numpy as np

# Steps of the algo 
""" 
- Apply sigmoid: 1/(exp(-z)+1)
- compute loss
- Compute gradient

"""

def sigmoid(x):
    sigma = 1 / (np.exp(-x) + 1)
    return sigma

def compute_loss(y, tx, w):
    return np.log(1+np.exp(tx.dot(w))) - (y.T).dot(tx.dot(w))

def compute_gradient(y, tx, w):
    sigma = sigmoid(tx.dot(w))
    tmp = sigma - y
    grad = tx.dot(tmp)
    
    return grad

def learning_grad_descent(
    