# -*- coding: utf-8 -*-
""" Contain everything if you want to play with the data
    - create color vector for signal and background
    - plot one feature VS another feature
    - split the data set in one set with only signal and one with only background
    - PLot some histograms --> REMOVE OK?????????
"""

import numpy as np
import matplotlib.pyplot as plt

def create_color_vector(labels_train):
    """
    To create a vector with colors in order to plot features and see what samples is signal and background
    argument: labels_train, it is the labels for the training set
    return: a list of b and r, in order to have the same color for all the samples in background or in signal
    """
    colors = []
    for t in range(labels_train.shape[0]):
        if labels_train[t] == 1:
            colors.append('b')
        else:
            colors.append('r')
    return colors

def plotFeatVsFeat(f1,f2, features_train, labels_train):
    """ 
    To plot one feature VS another feature
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
    
def split_data(labels, features_data):
    """
    To split the data in background and signal
    argument:
        - labels: labels of the data
        - features_data: the data
    return:
        - signal: data that correspond to the signal
        - background: data that correspond to the signal
    """
    # Separate the samples that are labeled as "sample" and as "background"
    signal_good_ind = []
    background_good_ind = []

    for index_i, i in enumerate(labels):
        if i == 1:
            signal_good_ind.append(index_i)
        else:
            background_good_ind.append(index_i)

    signal = features_data[signal_good_ind,:]
    background = features_data[background_good_ind,:]
    
    return signal,background

def histo_features(data, labels): # ERROR whit this one --> OK to remove??????????????????
    """
    Just to learn some stuff about the data, plot some histogramms
    argument:
        - data: the data
        - labels: labels of the data
    return:
        - nothing, just plots
    """
    # INitiate some variables, list needed 
    mean_feat_signal = []
    mean_feat_back = []
    std_feat_signal = []
    std_feat_back = []
    diff_mean =[]
    
    signal_data, background_data = split_data(labels, data) # get the signal and the background (the 2 classes)

    for i in range(data.shape[1]):
        # Get the mean for each features for signal and background
        mean_feat_signal.append(np.mean(signal_data[:,i]))
        mean_feat_back.append(np.mean(background_data[:,i]))
        
        # Get the std for each features for signal and background
        std_feat_signal.append(np.std(signal_data[:,i]))
        std_feat_back.append(np.std(background_data[:,i]))
    
    # Calculate the difference between the mean of the signal and background for the same feature (each feature)
    for index_mean, m in enumerate(mean_feat_signal):
        diff_mean.append(abs(m - mean_feat_back[index_mean] ))

    # In order to choose only the features that have a difference between signal and background bigger than the average
    mean_thediff = np.mean(diff_mean)
    ind_featInterest = np.where(diff_mean >= mean_thediff)
    features_interst = data[:,np.where(diff_mean >= mean_thediff)[0]]

    # Try to make histograms for those features in order to see how well (or not) we can separate the two classes
    for ind in ind_featInterest[0]:
        print(ind)
        plt.figure()
        plt.hist(signal_data[:,ind],50)
        plt.hist(background_data[:,ind],50)
        plt.show()
