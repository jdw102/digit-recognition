import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from plot import mfccs_subplots
from data_parser import data, phoneme_nums, separate_mfccs, extract_mfccs


def mesh_gaussian_pdf(mean, cov):
    x_mesh, y_mesh = np.meshgrid(
        np.linspace(mean[0] - 5, mean[0] + 5, 100),
        np.linspace(mean[1] - 5, mean[1] + 5, 100))
    mvn = multivariate_normal(mean=mean, cov=cov)
    pdf_values = mvn.pdf(np.dstack((x_mesh, y_mesh)))
    return x_mesh, y_mesh, pdf_values


def mfccs_contours(center, mfccs, plotting_pairs, ax):
    for i, pair in enumerate(plotting_pairs):
        x, y = pair
        cov = np.cov(mfccs[x], mfccs[y])
        mean = [center[x], center[y]]
        x_mesh, y_mesh, pdf = mesh_gaussian_pdf(mean, cov)
        ax[i].contour(x_mesh, y_mesh, pdf, alpha=0.5)


def perform_k_means(mfccs, n):
    mfccs = np.array(mfccs)
    k_means = KMeans(n_clusters=n, init='k-means++').fit(mfccs)
    labels = np.array(k_means.labels_)
    unique_labels = np.unique(labels)
    clusters = [mfccs[labels == label] for label in unique_labels]
    return zip(k_means.cluster_centers_, clusters)


def plot_k_means_clusters(clusters, digit):
    fig, ax = plt.subplots(1, 3, figsize=(12, 6))
    for center, cluster in clusters:
        mfccs = separate_mfccs(cluster)
        mfccs_subplots(mfccs, [[1, 0], [2, 0], [2, 1]], ax, s=0.1, alpha=0.7)
        mfccs_contours(center, mfccs, [[1, 0], [2, 0], [2, 1]], ax)
    fig.suptitle("K-Means Phoneme Clusters on MFCCs: Digit " + str(digit))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    mfccs = extract_mfccs(data)
    for digit, coeffs in enumerate(mfccs):
        clusters = perform_k_means(coeffs, phoneme_nums[digit])
        plot_k_means_clusters(clusters, digit)
