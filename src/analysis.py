from itertools import combinations

import numpy as np
from matplotlib import pyplot as plt

from src.data_clustering import perform_k_means, perform_em
from src.data_parser import phoneme_nums
from src.gmm import generate_model, determine_category, likelihood_all_digits
from src.plot_util import plot_clusters, plot_mfccs_subplots_single_utterance, \
    plot_analysis_window_function_single_utterance, plot_kde


def generate_mfccs_analysis(block_num, num_coeffs, gender, data):
    plotting_pairs = list(combinations(list(range(num_coeffs)), 2))
    for digit in data:
        utterance = data[digit][gender][block_num]
        plot_mfccs_subplots_single_utterance(utterance, plotting_pairs, digit, gender, block_num)
        plot_analysis_window_function_single_utterance(utterance, num_coeffs, digit, gender, block_num)


def generate_log_likelihood_analysis(digit, data, tokens, method, cov_type, tied):
    log_likelihoods = likelihood_all_digits(digit, data, tokens, method, cov_type, tied)
    fig, ax = plt.subplots(10, sharex=True, sharey=True, figsize=(6, 12))
    for i in range(10):
        plot_kde(np.array(log_likelihoods[i]), ax[i], i)
    fig.text(0.5, 0.02, 'Log Likelihood', ha='center', va='center', fontsize=12)
    fig.text(0.06, 0.5, 'Probability Density', ha='center', va='center', rotation='vertical', fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.show()


def generate_data_clustering_analysis(tokens, method):
    for digit, coeffs in enumerate(tokens):
        if method == "k-means":
            clusters = perform_k_means(coeffs, phoneme_nums[digit])
        else:
            clusters = perform_em(coeffs, phoneme_nums[digit])
        plot_clusters(clusters, digit)


def generate_confusion_matrix(tokens, testing_data, method, cov_type, tied):
    model = generate_model(tokens, method, cov_type, tied)
    ret = np.zeros((10, 10))
    total_correct = 0
    for actual in range(10):
        data = testing_data[actual]["male"] + testing_data[actual]["female"]
        for utterance in data:
            predicted = determine_category(utterance, model)
            if actual == predicted:
                total_correct += 1
            ret[actual][predicted] += 1
    print(total_correct / (220 * 10))
