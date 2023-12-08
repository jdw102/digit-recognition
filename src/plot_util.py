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
    ax.plot(x, dens)
    ax.set_ylim(0, 0.006)
    ax.set_xlim(-800, -300)
    ax.set_title(f"Digit {str(digit)} KDE")
    ax.set_xlabel("Log Likelihood")
    ax.set_ylabel("Probability Density")
    # ax.axis('equal')


def plot_kde_likelihood(log_likelihoods, title, filename):
    fig, ax = plt.subplots(5, 2, figsize=(18, 12))
    ax = ax.flatten()
    for i in range(10):
        plot_kde(np.array(log_likelihoods[i]), ax[i], i)
    # fig.text(0.5, 0.02, 'Log Likelihood', ha='center', va='center', fontsize=12)
    # fig.text(0.06, 0.5, 'Probability Density', ha='center', va='center', rotation='vertical', fontsize=12)
    fig.suptitle(title)
    plt.tight_layout()
    plt.subplots_adjust(hspace=1.0, top=0.92, bottom=0.08)
    # plt.subplots_adjust(bottom=0.1)
    plt.savefig(f"./data/results/kde_likelihood_plots/{filename}.png")


def plot_clusters(clusters, title, filename):
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
    fig.suptitle(title)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(f"./data/results/sub_dimension_clustering/{filename}.png")


def plot_confusion_matrix(confusion_matrix, cluster_counts, title, filename):
    fig, (ax_matrix, ax_table) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [4, 1]},
                                              figsize=(10, 6))
    gradient_cmap = LinearSegmentedColormap.from_list('custom_gradient', ["#ffffff", "#6d98ed"], N=256)
    im = ax_matrix.imshow(confusion_matrix, cmap=gradient_cmap)

    ax_matrix.set_title("Confusion Matrix")
    fig.suptitle(title)
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


def plot_feature_accuracy(accuracies, indices):
    write_optimal_to_txt(indices, accuracies, "./data/results/optimal-features.txt")
    x = [i + 1 for i in range(len(accuracies))]
    highest_accuracy_index = accuracies.index(max(accuracies))
    plt.scatter(highest_accuracy_index + 1, accuracies[highest_accuracy_index], color='gold', marker='*', s=200,
                label='Optimal Subset', zorder=10)
    plt.plot(x, accuracies, marker='o')
    plt.xticks(x)
    plt.title("Impact of Feature Subsets on Model Accuracy")
    plt.xlabel("Number of MFCCs Included")
    plt.ylabel("Overall Model Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./data/results/optimal-features.png")
    plt.show()


def write_optimal_to_txt(features, accuracies, file_path):
    with open(file_path, 'w') as file:
        file.write(f"F : A\n")
        for feature, accuracy in zip(features, accuracies):
            file.write(f"{feature} : {accuracy} \n")


def plot_covariance_bar_graph(em_cov, km_cov, title, filename):
    categories = list(em_cov.keys())
    em_vals = list(em_cov.values())
    km_vals = list(km_cov.values())

    bar_width = 0.35  # Width of each bar
    index = np.arange(len(categories))  # X-axis index for each category

    # Create grouped bar graph
    plt.bar(index, em_vals, width=bar_width, label='EM')
    plt.bar(index + bar_width, km_vals, width=bar_width, label='K-Means')
    for i, (v1, v2) in enumerate(zip(em_vals, km_vals)):
        plt.text(index[i] + bar_width / 20, v1, "{:.2f}".format(v1), ha='center', va='bottom')
        plt.text(index[i] + bar_width / 0.85, v2,  "{:.2f}".format(v2), ha='center', va='bottom')
    # Customize the plot
    plt.xlabel("Covariances")
    plt.ylabel("Accuracy")
    plt.title(title, fontsize=10)
    plt.suptitle("Model Accuracy Across Different Covariances", y=0.94, x=0.55)
    plt.xticks(index + bar_width / 2, categories, fontsize=8)  # Set x-axis ticks at the center of each group
    plt.legend(bbox_to_anchor=(0.15, 1.05), loc='upper center', ncol=1)
    plt.tight_layout()
    plt.savefig(f"./data/results/accuracy_cov_plots/{filename}.png")

    # Show the plot
    plt.show()


def plot_cluster_number_accuracies(digit, overall_accuracies, digit_accuracies, title, filename):
    x = [i + 1 for i in range(len(overall_accuracies))]
    plt.plot(x, overall_accuracies, label="Overall", marker='*', linestyle='--', color='gold')
    plt.xticks(x)
    for i, accuracies in enumerate(digit_accuracies):
        plt.plot(x, accuracies, label=f"{i}", marker='o')
    plt.suptitle("Impact of Changing Number of Clusters on Accuracy", y=0.94, x=0.55)
    plt.title(title, fontsize=10)
    plt.xlabel(f"Number of Phoneme Clusters for Digit {digit}")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./data/results/cluster_number_plots/{filename}.png")
    plt.show()
