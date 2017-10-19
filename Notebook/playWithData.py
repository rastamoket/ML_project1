# -*- coding: utf-8 -*-
""" Contain everything if you want to play with the data
    - Plot some stuff
    - Color for labels (signal, background)
"""
import numpy as np
import matplotlib.pyplot as plt

def create_color_vector(labels_train):
    """
    argument: labels_train, it is the labels for the training set
    return: a list of b and r, in order to have the same color for all the samples in background or in signal
    """
    
    # To create a vector with colors in order to plot features and see what samples is signal and background
    colors = []
    for t in range(labels_train.shape[0]):
        if labels_train[t] == 1:
            colors.append('b')
        else:
            colors.append('r')
    return colors

def plotFeatVsFeat(f1,f2, features_train, labels_train):
    """ 
    argument: 
        - the indices of features you want to plot against, f1, f2
        - features_train = data
        - labels_train = labels of the data
    return: nothing, but plot
    """
    plt.figure()
    colors = create_color_vector(labels_train)
    plt.scatter(features_train[:,f1], features_train[:,f2],color = colors)
    plt.show()

