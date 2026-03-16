# ------------------------------------------------------------------------------------------
# Standardization utilities.
#
# Normalizer: Container for mean and standard deviations stored as N-dimensional arrays
# fit_standardizer(): Compute mean and std of input array (feature-wise)
# transform_standardizer(): Apply standardization
# inverse_transform_standardizer(): Undo standardization
#
# Robert Pearce
# ------------------------------------------------------------------------------------------

import numpy as np
from dataclasses import dataclass


@dataclass
class Normalizer:
    """
    Container for mean and standard deviation.
    
    mean: Feature-wise mean
    std: Feature-wise standard deviation
    """
    mean: np.ndarray
    std: np.ndarray


def fit_standardizer(X: np.ndarray) -> Normalizer:
    """
    Compute feature-wise mean and standard deviation.
    
    X: N-dimensional array
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    # If std is 0 replace with 1
    std = np.where(std == 0, 1.0, std)
    return Normalizer(mean=mean, std=std)


def transform_standardizer(X: np.ndarray, norm: Normalizer) -> np.ndarray:
    """
    Apply standardization using provided Normalizer.
    
    returns: Standardized array of same shape as X
    """
    return (X - norm.mean) / norm.std


def inverse_transform_standardizer(X: np.ndarray, norm: Normalizer) -> np.ndarray:
    """
    Reverse standardization using provided Normalizer.
    
    returns: Original array of same shape as X
    """
    return (X * norm.std) + norm.mean

#-----------------------------
#         END OF FILE
#-----------------------------