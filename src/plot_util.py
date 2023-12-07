import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.neighbors import KernelDensity
from src.data_parser import unzip_frames
from matplotlib.colors import LinearSegmentedColormap

plotting_colors = ['red', 'green', 'blue', 'orange', 'yellow', 'brown', 'black']


def plot_analysis_window_function_single_utterance(utterance, num_coeffs, digit, gender, num):
    utterance = np.array(utterance)
    values = utterance[:, 0:num_coeffs].T

    for i in range(num_coeffs):
        plt.plot(np.arange(len(values[i])), values[i], label="MFCC" + str(i + 1))
    plt.xlabel("Analysis Frame")
    plt.ylabel("MFCC tokens")
    plt.legend()
    plt.title("Single Utterance of " + str(digit) + ", " + gender + " num " + str(num))
    plt.tight_layout()
    plt.savefig(
        "single_utterance_analysis_frame_plots/" + gender + "_num" + str(num) + "_digit" + str(digit) + "_aframe.png")
    plt.close()


def mfccs_subplots(mfccs, plotting_pairs, ax, s=10.0, alpha=1.0, color='blue', label=None):
    for i, pair in enumerate(plotting_pairs):
        x, y = pair
        ax[i].scatter(mfccs[x], mfccs[y], s=s, alpha=alpha, color=color, label=label)
        ax[i].set_xlabel("MFCC" + str(x + 1))
        ax[i].set_ylabel("MFCC" + str(y + 1))


def plot_mfccs_subplots_single_utterance(utterance, plotting_pairs, digit, gender, num):
    fig, ax = plt.subplots(1, len(plotting_pairs), figsize=(12, 6))
    mfccs_subplots(unzip_frames(utterance), plotting_pairs, ax)
    fig.suptitle("Single Utterance of " + str(digit) + ", " + gender + " num " + str(num))
    plt.tight_layout()
    fig.savefig("single_utterance_mfccs_plots/" + gender + "_num" + str(num) + "_digit" + str(digit) + "_mfccs.png")
    plt.close(fig)


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


def plot_kde(log_likelihoods, ax, digit):
    kde = KernelDensity(kernel="gaussian", bandwidth=20.0)
    kde.fit(log_likelihoods[:, np.newaxis])
    x = np.linspace(min(log_likelihoods), max(log_likelihoods), 1000)
    log_dens = kde.score_samples(x[:, np.newaxis])
    dens = np.exp(log_dens)
    ax.plot(x, dens, 'r', label='Digit' + str(digit) + ' KDE')
    ax.set_xlim(-800, -300)
    ax.legend()


def plot_clusters(clusters, digit):
    fig, ax = plt.subplots(1, 3, figsize=(12, 6))
    index = 0
    for cluster in clusters:
        mfccs = unzip_frames(cluster["points"])
        mfccs_subplots(mfccs, [[1, 0], [2, 0], [2, 1]], ax, s=0.1, alpha=0.7, color=plotting_colors[index],
                       label="phoneme cluster " + str(index + 1))
        mfccs_contours(cluster["mean"], cluster["cov"], [[1, 0], [2, 0], [2, 1]], ax, plotting_colors[index])
        index += 1
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=index, markerscale=10)
    fig.suptitle("K-Means Phoneme Clusters on MFCCs: Digit " + str(digit))
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.show()


def plot_confusion_matrix(confusion_matrix, overall_accuracy, cluster_counts, title, filename):
    fig, (ax_matrix, ax_table) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [4, 1]},
                                              figsize=(10, 6))
    gradient_cmap = LinearSegmentedColormap.from_list('custom_gradient', ["#ffffff", "#6d98ed"], N=256)
    im = ax_matrix.imshow(confusion_matrix, cmap=gradient_cmap)

    ax_matrix.set_title(f"Confusion Matrix {title} : {round(overall_accuracy * 100, 2)}% Accuracy")
    ax_matrix.set_xticks(np.arange(10))
    ax_matrix.set_yticks(np.arange(10))
    ax_matrix.set_xticklabels([str(i) for i in range(10)])
    ax_matrix.set_yticklabels([str(i) for i in range(10)])

    ax_matrix.set_xlabel("Predicted Values")
    ax_matrix.set_ylabel("True Values")

    for i in range(10):
        for j in range(10):
            ax_matrix.text(j, i, "{:.2f}".format(confusion_matrix[i, j]), ha='center', va='center', color='black')

    table_data = [["Digit", "# Clusters"]] + [[str(i), str(cluster_counts[i])] for i in range(10)]
    table = ax_table.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.5, 0.5])
    table.scale(1, 2.3)
    table.auto_set_font_size(False)
    table.set_fontsize(8)

    ax_table.set_title("Number of Clusters per Digit")
    ax_table.axis('off')

    plt.colorbar(im)
    plt.savefig(f"./data/results/confusion_matrices/{filename}.png")
    plt.show()
