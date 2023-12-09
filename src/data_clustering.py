import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from src.cov_util import cov, Cov


def perform_k_means(data, n, random_state, covariance_type=Cov.FULL):
    data = np.array(data)
    k_means = KMeans(n_clusters=n, random_state=random_state, n_init="auto", init='k-means++').fit(data)
    labels = np.array(k_means.labels_)
    unique_labels = np.unique(labels)
    points = [data[labels == label] for label in unique_labels]
    covariances = cov(points, cov_type=covariance_type)
    clusters = []

    for i in range(len(k_means.cluster_centers_)):
        clusters.append({
            "mean": k_means.cluster_centers_[i],
            "cov": covariances[i],
            "pi": len(points[i]) / len(data),
            "points": points[i]
        })
    return clusters


def perform_em(data, n, random_state, covariance_type=Cov.FULL):
    data = np.array(data)
    gm = GaussianMixture(n_components=n, covariance_type=covariance_type.value, random_state=random_state).fit(data)
    labels = gm.predict(data)
    unique_labels = np.unique(labels)
    points = [data[labels == label] for label in unique_labels]
    clusters = []
    for i in range(len(gm.means_)):
        if covariance_type == Cov.FULL:
            covar = gm.covariances_[i]
        elif covariance_type == Cov.DIAG or covariance_type == Cov.TIED_DIAG:
            covar = np.diag(gm.covariances_[i])
        elif covariance_type == Cov.SPHERICAL or covariance_type == Cov.TIED_SPHERICAL:
            covar = gm.covariances_[i] * np.identity(len(data[0]))
        else:
            covar = gm.covariances_
        clusters.append({
            "mean": gm.means_[i],
            "cov": covar,
            "pi": gm.weights_[i],
            "points": points[i]
        })
    return clusters
