import numpy as np
from enum import Enum


class Cov(Enum):
    FULL = "full"
    DIAG = "diag"
    SPHERICAL = "spherical"
    TIED_FULL = "tied"
    TIED_DIAG = "tied diag"
    TIED_SPHERICAL = "tied spherical"


def spherical_cov(data):
    total = np.concatenate([np.subtract(dim, np.mean(dim)) for dim in tuple(zip(*data))])
    var = np.cov(total)
    return var * np.identity(len(data[0]))


def full_cov(data):
    return np.cov(data, rowvar=False)


def diag_cov(data):
    total = [np.cov(dim) for dim in tuple(zip(*data))]
    return total * np.identity(len(data[0]))


def tie(clusters):
    total = np.concatenate([np.subtract(cluster, np.mean(cluster, axis=0)) for cluster in clusters])
    return total


def cov(clusters, cov_type=Cov.FULL):
    ret = []
    if cov_type == Cov.TIED_SPHERICAL or cov_type == Cov.TIED_FULL or cov_type == Cov.TIED_DIAG:
        total = tie(clusters)
        if cov_type == Cov.TIED_DIAG:
            covar = diag_cov(total)
        elif cov_type == Cov.TIED_SPHERICAL:
            covar = spherical_cov(total)
        else:
            covar = full_cov(total)
        for i in range(len(clusters)):
            ret.append(covar)
    else:
        for cluster in clusters:
            if cov_type == Cov.SPHERICAL:
                ret.append(spherical_cov(cluster))
            elif cov_type == Cov.DIAG:
                ret.append(diag_cov(cluster))
            else:
                ret.append(full_cov(cluster))
    return ret
