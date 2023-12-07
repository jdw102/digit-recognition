import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from src.cov_util import cov


rand = np.random.RandomState(42)


def perform_k_means(data, n, covariance_type="full", tied=False):
    data = np.array(data)
    k_means = KMeans(n_clusters=n, random_state=rand, n_init="auto", init='k-means++').fit(data)
    labels = np.array(k_means.labels_)
    unique_labels = np.unique(labels)
    points = [data[labels == label] for label in unique_labels]
    covariances = cov(points, cov_type=covariance_type, tied=tied)
    clusters = []

    for i in range(len(k_means.cluster_centers_)):
        clusters.append({
            "mean": k_means.cluster_centers_[i],
            "cov": covariances[i],
            "pi": len(points[i]) / len(data),
            "points": points[i]
        })
    return clusters


def perform_em(data, n, covariance_type="full", tied=False):
    if tied:
        covariance_type = "tied " + covariance_type
    data = np.array(data)
    gm = GaussianMixture(n_components=n, covariance_type=covariance_type).fit(data)
    labels = gm.predict(data)
    unique_labels = np.unique(labels)
    points = [data[labels == label] for label in unique_labels]
    clusters = []
    for i in range(len(gm.means_)):
        if covariance_type == "full":
            covar = gm.covariances_[i]
        elif covariance_type == "diag" or covariance_type == "tied diag":
            covar = np.diag(gm.covariances_[i])
        elif covariance_type == "spherical" or covariance_type == "tied spherical":
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
