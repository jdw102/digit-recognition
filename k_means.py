import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from plot import plot_mfccs_subplots
from data_parser import data, phoneme_nums, separate_mfccs

def extract_mfccs(data):
    mfccs = [[] for _ in range(10)]
    for digit in data:
        if digit not in mfccs:
            mfccs[digit] = []
        for block in data[digit]["male"]:
            mfccs[digit].extend(block)
        for block in data[digit]["female"]:
            mfccs[digit].extend(block)
    return mfccs

def mesh_gaussian_pdf(mean, cov):
    x_mesh, y_mesh = np.meshgrid(
        np.linspace(mean[0] - 5, mean[0] + 5, 100),
        np.linspace(mean[1] - 5, mean[1] + 5, 100))
    mvn = multivariate_normal(mean=mean, cov=cov)
    pdf_values = mvn.pdf(np.dstack((x_mesh, y_mesh)))
    return x_mesh, y_mesh, pdf_values

def plot_mfccs_contours(center, mfccs, plotting_pairs, ax):
    for i, pair in enumerate(plotting_pairs):
        x, y = pair
        cov = np.cov(mfccs[x], mfccs[y])
        mean = [center[x], center[y]]
        x_mesh, y_mesh, pdf = mesh_gaussian_pdf(mean, cov)
        ax[i].contour(x_mesh, y_mesh, pdf, alpha=0.5)

def plot_sub_dimension(x, y, center, axis, xlabel, ylabel, phon):
    cov = np.cov(x, y)

    axis.scatter(x, y, s=0.1, alpha=0.7, label="Phoneme " + str(phon))
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.legend()


def perform_k_means(mfccs, n):
    mfccs = np.array(mfccs)
    k_means = KMeans(n_clusters=n, init='k-means++').fit(mfccs)
    labels = np.array(k_means.labels_)
    unique_labels = np.unique(labels)
    clusters = [mfccs[labels == label] for label in unique_labels]
    return zip(k_means.cluster_centers_, clusters)


if __name__ == "__main__":
    mfccs = extract_mfccs(data)
    for digit, coeffs in enumerate(mfccs):
        centers = perform_k_means(coeffs, phoneme_nums[digit])
        fig, ax = plt.subplots(1, 3, figsize=(12, 6))
        for center, cluster in centers:
            mean = center[0:3]
            mfccs = separate_mfccs(cluster)
            plot_mfccs_subplots(mfccs, [[1, 0], [2, 0], [2, 1]], ax, s=0.1, alpha=0.7)
            plot_mfccs_contours(center, mfccs, [[1, 0], [2, 0], [2, 1]], ax)
        fig.suptitle("K-Means Phoneme Clusters on MFCCs: Digit " + str(digit))
        plt.tight_layout()
        plt.show()