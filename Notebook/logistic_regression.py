# -*- coding: utf-8 -*-
"""logistic regression."""
import numpy as np
from helpers import *

# Steps of the algo 
""" 
- Apply sigmoid: 1/(exp(-z)+1)
- compute loss
- Compute gradient

"""

def sigmoid(x):
    return 1.0 / (np.exp(-x) + 1)

def compute_loss(y, tx, w):
    #pred = sigmoid(tx.dot(w))
    #loss_ = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    #loss_ = -loss_
    # Ivan's version
    loss_ = np.sum(np.log(1+np.exp(tx.dot(w))))-y.T.dot(tx.dot(w))
    return loss_
    
    #return np.log(1+np.exp(tx.dot(w))) - (y.T).dot(tx.dot(w))

def compute_gradient(y, tx, w):
    #sigma = sigmoid(tx.dot(w))
    #tmp = sigma - y
    #grad = (tx.T).dot(tmp)
    # Ivan's version
    grad=tx.T.dot(sigmoid(tx.dot(w))-np.reshape(y,(len(y),1)))
    return grad

def calculate_hessian(y, tx, w):
    """return the hessian of the loss function."""
    # Ivan's version
    dia=sigmoid(tx.dot(w))*(1-sigmoid(tx.dot(w)))
    S=np.diag(dia.T[0])
    return tx.T.dot(S.dot(tx))

def logistic_regression(y, tx, w):
    """return the loss, gradient, and hessian."""
    loss=compute_loss(y, tx, w)
    gradient=compute_gradient(y, tx, w)
    hessian=calculate_hessian(y, tx, w)
    return loss, gradient, hessian

def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient, and hessian."""
    loss, gradient, hessian=logistic_regression(y, tx, w)
    # Add the elements due to penalization
    loss=loss+(lambda_*0.5)*w.T.dot(w)
    gradient=gradient+lambda_*w
    hessian=hessian+lambda_*np.identity(hessian.shape[0])
    return loss, gradient, hessian

def learning_grad_descent(y, tx, initial_w, max_iter, gamma):
    threshold = 1e-8
    ws = [initial_w]
    losses =[]
    w = initial_w
    
    for n in range(max_iter):
        loss = compute_loss(y,tx,w)
        w = ws[n] - compute_gradient(y, tx, w)*gamma
        print(compute_gradient(y, tx, w).shape)
        ws.append(w)
        losses.append(loss)
        if(len(losses) > 1 and abs(losses[-1] - losses[-2]) <= threshold):
            # Condition for convergence
            print("COUCOU") # DEBUG: to see if we can arrive here  --> THIS is never printed --> ISSUE with the condition: never in it, this is surely one reason of why the run is so long!!!!!!!!!!!!!!!!!!!!!!!!! CHECK THIS
            break
        print("iteration={k}".format(k=n))
        print("loss (without penalization)={l}".format(l=loss))
    
   
    return ws[-1], losses[-1]

def learning_by_newton_method(y, tx, initial_w, max_iter, lambda_):
    """
    Do one step on Newton's method.
    return the loss and updated w.
    """    
    
        # init parameters
    threshold = 1e-8
    ws = [initial_w]
    losses =[]
    w = initial_w

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, gradient, hessian=logistic_regression(y, tx, w)
        w=ws[iter]-np.linalg.inv(hessian).dot(gradient)
        print(w)
        # log info
        if iter % 1 == 0:
            print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        ws.append(w)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return ws[-1], losses[-1]

def penalized_stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iter, gamma, lambda_):
    """Stochastic gradient descent algorithm."""
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iter):
        batches = batch_iter(y,tx,batch_size)
        for batchy,batchtx in batches:
            loss, gradient, hessian = penalized_logistic_regression(batchy,batchtx,w,lambda_)
            w=ws[n_iter]-gamma*np.linalg.inv(hessian).dot(gradient)
            # store w and loss
            ws.append(w)
            losses.append(loss)
            print("Gradient Descent({bi}/{ti}): loss={l}".format(bi=n_iter, ti=max_iter - 1, l=loss))
    return ws[-1], losses[-1]

def learning_by_penalized_gradient(y, tx, initial_w, max_iter, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    
    # init parameters
    threshold = 1e-5
    ws = [initial_w]
    losses =[]
    w = initial_w

    # start the logistic regression
    for n_iter in range(max_iter):
        # get loss and update w.
        loss, gradient, hessian = penalized_logistic_regression(y, tx, w, lambda_)
        w=ws[n_iter]-np.exp(-n_iter/gamma)*np.linalg.inv(hessian).dot(gradient)
        # log info
        if n_iter % 1 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        ws.append(w)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return ws[-1], losses[-1]

