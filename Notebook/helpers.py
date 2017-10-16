# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np


def load_data(dataSetCSVfile):
    """Load data and convert it to the metrics system."""
    path_dataset = dataSetCSVfile
    # To get the id of the samples
    id_samples = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=[0])

    if(dataSetCSVfile == "train.csv"):
        # Here we will load the data if the file given is "train.csv"
        # For the labels, we choose that s = 1, b = -1
        labels = np.genfromtxt(
            path_dataset, delimiter=",", skip_header=1, usecols=[1],
            converters={1: lambda x: -1 if b"b" in x else 1})
        # "data" will contain all the features for each samples so it will be a NxM matrix (with N = sumber of sample, M = number of features)
        data = np.genfromtxt(
            path_dataset, delimiter=",", skip_header=1, usecols=(np.arange(2,32)))
        return id_samples, labels, data
    else:
        # Here we will load the data if the file given is not "train.csv" --> so it is for "test.csv"
        # "data" will contain all the features for each samples so it will be a NxM matrix (with N = sumber of sample, M = number of features)
        data = np.genfromtxt(
            path_dataset, delimiter=",", skip_header=1, usecols=(np.arange(1,31)))
        return id_samples, data
    

    

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x


def build_model_data(data, label):
    """Form (y,tX) to get regression data in matrix form."""
    y = label
    x = data
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
