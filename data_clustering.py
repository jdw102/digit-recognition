import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from cov_util import cov
from mfccs_plot import mfccs_subplots, plotting_colors
from data_parser import training_data, phoneme_nums, separate_mfccs, extract_mfccs


rand = np.random.RandomState(42)

def mesh_gaussian_pdf(mean, cov):
    x_mesh, y_mesh = np.meshgrid(
        np.linspace(mean[0] - 5, mean[0] + 5, 100),
        np.linspace(mean[1] - 5, mean[1] + 5, 100))
    mvn = multivariate_normal(mean=mean, cov=cov)
    pdf_values = mvn.pdf(np.dstack((x_mesh, y_mesh)))
    return x_mesh, y_mesh, pdf_values


def mfccs_contours(center, cov_matrix, plotting_pairs, ax, color):
    for i, pair in enumerate(plotting_pairs):
        x, y = pair
        cov = [[cov_matrix[x][x], cov_matrix[x][y]], [cov_matrix[y][x], cov_matrix[y][y]]]
        mean = [center[x], center[y]]
        x_mesh, y_mesh, pdf = mesh_gaussian_pdf(mean, cov)
        ax[i].contour(x_mesh, y_mesh, pdf, alpha=0.4, colors=color)


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
            "pi": len(points) / len(data),
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
            cov = gm.covariances_[i]
        elif covariance_type == "diag" or covariance_type == "tied diag":
            cov = np.diag(gm.covariances_[i])
        elif covariance_type == "spherical" or covariance_type == "tied spherical":
            cov = gm.covariances_[i] * np.identity(len(data[0]))
        else:
            cov = gm.covariances_
        clusters.append({
            "mean": gm.means_[i],
            "cov": cov,
            "pi": gm.weights_[i],
            "points": points[i]
        })
    return clusters


def plot_clusters(clusters, digit):
    fig, ax = plt.subplots(1, 3, figsize=(12, 6))
    index = 0
    for cluster in clusters:
        mfccs = separate_mfccs(cluster["points"])
        mfccs_subplots(mfccs, [[1, 0], [2, 0], [2, 1]], ax, s=0.1, alpha=0.7, color=plotting_colors[index], label="phoneme cluster " + str(index + 1))
        mfccs_contours(cluster["mean"], cluster["cov"], [[1, 0], [2, 0], [2, 1]], ax, plotting_colors[index])
        index += 1
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=index, markerscale=10)
    fig.suptitle("K-Means Phoneme Clusters on MFCCs: Digit " + str(digit))
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.show()


if __name__ == "__main__":
    mfccs = extract_mfccs(training_data)
    # for digit, coeffs in enumerate(mfccs):
    #     clusters = perform_k_means(coeffs, phoneme_nums[digit])
    #     plot_clusters(clusters, digit)

    for digit, coeffs in enumerate(mfccs):
        clusters = perform_em(coeffs, phoneme_nums[digit])
        plot_clusters(clusters, digit)
