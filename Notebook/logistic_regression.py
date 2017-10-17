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
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(- loss)
    
    #return np.log(1+np.exp(tx.dot(w))) - (y.T).dot(tx.dot(w))

def compute_gradient(y, tx, w):
    sigma = sigmoid(tx.dot(w))
    tmp = sigma - y
    grad = (tx.T).dot(tmp)
    
    return grad

def learning_grad_descent(y, tx, initial_w, max_iters, gamma):
    threshold = 1e-8
    ws = [initial_w]
    losses =[]
    w = initial_w
    
    for n in range(max_iters):
        loss = compute_loss(y,tx,w)
        w = ws[n] - gamma*compute_gradient(y, tx, w)
        
        ws.append(w)
        losses.append(loss)
        if(len(losses) > 1 and abs(losses[-1] - losses[-2]) <= threshold):
            # Condition for convergence
            print("COUCOU") # DEBUG: to see if we can arrive here 
            break
    
   
    return ws[-1], losses[-1]
    