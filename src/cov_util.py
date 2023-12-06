import numpy as np


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


def cov(clusters, cov_type="full", tied=False):
    ret = []
    if tied:
        total = tie(clusters)
        if cov_type == "diag":
            covar = diag_cov(total)
        elif cov_type == "spherical":
            covar = spherical_cov(total)
        else:
            covar = full_cov(total)
        for i in range(len(clusters)):
            ret.append(covar)
    else:
        for cluster in clusters:
            if cov_type == "spherical":
                ret.append(spherical_cov(cluster))
            elif cov_type == "diag":
                ret.append(diag_cov(cluster))
            else:
                ret.append(full_cov(cluster))
    return ret
