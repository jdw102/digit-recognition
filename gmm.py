from data_clustering import perform_k_means, perform_em
from data_parser import phoneme_nums, extract_mfccs, training_data
from cov_util import cov
from scipy.stats import multivariate_normal
import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt


def create_gmm(digit, tokens, type="k-means", cov_type="full", tied=False):
    if type == "em":
        return perform_em(tokens, phoneme_nums[digit])
    return perform_k_means(tokens, phoneme_nums[digit])


def gmm_likelihood(gmm, utterance):
    pdf_vals = []
    for gaussian in gmm:
        mvn = multivariate_normal(mean=gaussian["mean"], cov=gaussian["cov"])
        pdf_vals.append(gaussian["pi"] * mvn.pdf(utterance))
    col_sums = np.sum(np.array(pdf_vals), axis=0)
    return np.sum(np.log(col_sums))


def digit_likelihood(digit, gmm, d):
    ret = []
    blocks = d[digit]["male"] + d[digit]["female"]
    for utterance in blocks:
        ret.append(gmm_likelihood(gmm, utterance))
    return ret


def likelihood_all_digits(digit, d):
    tokens = extract_mfccs(d)
    gmm = create_gmm(digit, tokens[digit], "full", False)
    likelihoods = []
    for num in range(10):
        likelihoods.append(digit_likelihood(num, gmm, d))
    return likelihoods


def plot_kde(log_likelihoods, ax, digit):
    kde = KernelDensity(kernel="gaussian", bandwidth=30.0)
    kde.fit(log_likelihoods[:, np.newaxis])
    x = np.linspace(min(log_likelihoods), max(log_likelihoods), 1000)
    log_dens = kde.score_samples(x[:, np.newaxis])
    dens = np.exp(log_dens)
    # plt.hist(log_likelihoods, bins=30, density=True, alpha=0.5, color='blue', label='Log Likelihoods (Hist)')
    ax.plot(x, dens, 'r', label='Digit' + str(digit) + ' KDE')
    ax.set_xlim(-800, -300)
    ax.legend()


if __name__ == "__main__":
    log_likelihoods = likelihood_all_digits(6, training_data)
    fig, ax = plt.subplots(10, sharex=True, sharey=True, figsize=(6, 12))
    for i in range(10):
        plot_kde(np.array(log_likelihoods[i]), ax[i], i)
    fig.text(0.5, 0.02, 'Log Likelihood', ha='center', va='center', fontsize=12)
    # Add a single y-axis label
    fig.text(0.06, 0.5, 'Probability Density', ha='center', va='center', rotation='vertical', fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.show()
