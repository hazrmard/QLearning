"""
Contains common utility functions.
"""

import numpy as np

def read_matrix(fname):
    """
    Reads a file of whitespace separated floats into an array.

    Args:
        fname (str): Filepath of matrix.

    Returns:
        A np.ndarray containing a matrix.
    """
    return np.loadtxt(fname)


def save_matrix(mat, fname):
    """
    Saves a ndarray into a text file of whitespace separated numbers.

    Args:
        mat (ndarray): Array to save to file.
        fname (str): Filepath where to save.
    """
    np.savetxt(fname, mat)
