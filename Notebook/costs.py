# -*- coding: utf-8 -*-
"""A function to compute the cost."""


def compute_mse(y, tx, w):
    """compute the loss by mse."""
    e = y - tx.dot(w)
    mse = (e.T).dot(e) / (2 * len(e))
    return mse
