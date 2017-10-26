# -*- coding: utf-8 -*-
""" Contain everything that we need for pre-processed the data:
    - Data whitening
    - Remove features with at least one value = -999.00
"""
import numpy as np

def data_whitening(data):
    """ To prepare the data for further processing:
        argument:
            - data: data we want to prepare
        return:
            - data_clean: data whitened
    """
    # Keep the indices (samples = row, features = column) in which there are -999.0
    #samples_bad_ind = np.unique(np.where(data == -999.0)[0]) # indices of samples that contains at least one -999.0  NOT USEFUL
    features_bad_ind = np.unique(np.where(data == -999.0)[1]) # indices of features that contains at least one -999.0
    
    for f in range(data.shape[1]):
        if f not in features_bad_ind: # this condition is to separate the two cases: features with/without -999.0
            if np.std(data[:,f]) == 0: # This condition is to avoid the division by 0
                data[:,f] = (data[:,f] - np.mean(data[:,f])) # CORRECT to do like this???????????
            else:
                data[:,f] = (data[:,f] - np.mean(data[:,f]))/np.std(data[:,f])
        else:
            data[np.where(data[:,f] != -999.0),f] -= np.mean(data[np.where(data[:,f] != -999.0),f]) # CORRECT to do like this???????????
            if np.std(data[np.where(data[:,f] != -999.0),f]) !=0: # do the division only if standard deviation is different than 0
                data[np.where(data[:,f] != -999.0),f] = data[np.where(data[:,f] != -999.0),f]/np.std(data[np.where(data[:,f] != -999.0),f])
    
    data_clean = data
        
    return data_clean
    
    

def remove_features(data):
    """
    Find the features that contains at least one times -999.0 and remove this feature
    argument:
        - data: Data with all the features
    return:
        - data_features_removed: the data without the features we wanted to remove
    """
    features_to_remove = []
    for f in range(data.shape[1]):
        if(np.any(data[:,f] == -999.0)):
            bad_samples = np.where(data[:,f] == -999.0)[0] # To get the indices of the samples that contain a -999.0
            features_to_remove.append(f)

    data_features_removed = np.delete(data, features_to_remove, axis=1)
    
    return data_features_removed

''' 
Adding combination of features as new features at the power of choice. Ex (x1x2)^coef, (x1x3)^coef. 
x1^pow is not calculated here.
'''
def twoFeatureCombinationPower(data, coef, featNum):
    
    for i in range(featNum):
        for j in list(range(i + 1)):
            if i!=j:
                newFeature = np.multiply(data[:,i],data[:,j])
                for k in range(coef):
                    newFeaturePow = np.power(newFeature,k+1)
                    data = np.c_[data,newFeaturePow]
    return data

''' 
Adding combination of features as new features. Ex x1^2 is calculated here
'''
def FeaturePower(data, coef,featNum):
    if coef <= 1:
        print('No need to use this function you will duplicate features')
    else:    
        powers = np.linspace(2,coef,coef-1)    
        for i in range(featNum):        
            newFeature = data[:,i]
            for j in powers:
                newFeaturePow = np.power(newFeature,j)
                data = np.c_[data,newFeaturePow]
                
    return data

def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    # generate random indices
    num_row = len(y)
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_tr = indices[: index_split]
    index_te = indices[index_split:]
    # create split
    x_tr = x[index_tr]
    x_te = x[index_te]
    y_tr = y[index_tr]
    y_te = y[index_te]
    
    return x_tr, x_te, y_tr, y_te
