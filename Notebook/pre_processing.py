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
