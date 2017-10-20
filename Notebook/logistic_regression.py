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
    threshold = 1e-2
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
            print("COUCOU") # DEBUG: to see if we can arrive here  --> THIS is never printed --> ISSUE with the condition: never in it, this is surely one reason of why the run is so long!!!!!!!!!!!!!!!!!!!!!!!!! CHECK THIS
            break
    
   
    return ws[-1], losses[-1]

def calculate_hessian(y, tx, w):
    """return the hessian of the loss function."""
    dia=sigmoid(tx.dot(w))*(1-sigmoid(tx.dot(w)))
    S=np.diag(dia.T[0])
    return tx.T.dot(S.dot(tx))

def learning_by_newton_method(y, tx, initial_w, max_iters):
    """
    Do one step on Newton's method.
    return the loss and updated w.
    """
    loss, gradient, hessian=logistic_regression(y, tx, w)
    w=w-np.linalg.inv(hessian).dot(gradient)
    
    
        # init parameters
    threshold = 1e-8
    lambda_ = 0.1
    losses = []

    # build tx
    tx = np.c_[np.ones((y.shape[0], 1)), x]
    w = np.zeros((tx.shape[1], 1))

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_newton_method(y, tx, w)
        # log info
        if iter % 1 == 0:
            print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    # visualization
    visualization(y, x, mean_x, std_x, w, "classification_by_logistic_regression_newton_method")
    print("loss={l}".format(l=calculate_loss(y, tx, w)))
    return loss, w


    