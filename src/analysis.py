from itertools import combinations

import numpy as np
from matplotlib import pyplot as plt

from src.data_clustering import perform_k_means, perform_em
from src.gmm import generate_model, determine_category, likelihood_all_digits, Method
from src.plot_util import plot_clusters, plot_mfccs_subplots_single_utterance, \
    plot_analysis_window_function_single_utterance, plot_kde, plot_confusion_matrix
from src.data_parser import mfcc_indices, phoneme_nums


def generate_mfccs_analysis(block_num, num_coeffs, gender, data):
    plotting_pairs = list(combinations(list(range(num_coeffs)), 2))
    for digit in data:
        utterance = data[digit][gender][block_num]
        plot_mfccs_subplots_single_utterance(utterance, plotting_pairs, digit, gender, block_num)
        plot_analysis_window_function_single_utterance(utterance, num_coeffs, digit, gender, block_num)


def generate_log_likelihood_analysis(digit, data, tokens, method, cov_type):
    log_likelihoods = likelihood_all_digits(digit, data, tokens, method, cov_type)
    fig, ax = plt.subplots(10, sharex=True, sharey=True, figsize=(6, 12))
    for i in range(10):
        plot_kde(np.array(log_likelihoods[i]), ax[i], i)
    fig.text(0.5, 0.02, 'Log Likelihood', ha='center', va='center', fontsize=12)
    fig.text(0.06, 0.5, 'Probability Density', ha='center', va='center', rotation='vertical', fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.show()


def generate_data_clustering_analysis(tokens, method, cov_type):
    for digit, coeffs in enumerate(tokens):
        if method == Method.K_MEANS:
            clusters = perform_k_means(coeffs, phoneme_nums[digit], cov_type)
        else:
            clusters = perform_em(coeffs, phoneme_nums[digit], cov_type)
        plot_clusters(clusters, digit)


def generate_confusion_matrix(tokens, testing_data, method, cov_type):
    overall_accuracy, confusion_matrix = calculate_accuracy(tokens, testing_data, method, cov_type)
    plot_confusion_matrix(confusion_matrix, overall_accuracy, phoneme_nums,
                          f"{method.value} {cov_type.value.title()} Cov",
                          f"cm-{method.value}-{cov_type.value.title()}-{len(mfcc_indices)}coeffs-table{np.prod(np.multiply(phoneme_nums, np.arange(1, 11)))}")


def calculate_accuracy(tokens, testing_data, method, cov_type):
    model = generate_model(tokens, method, cov_type)
    confusion_matrix = np.zeros((10, 10))
    total_correct = 0
    for actual in range(10):
        data = testing_data[actual]["male"] + testing_data[actual]["female"]
        for utterance in data:
            predicted = determine_category(utterance, model)
            if actual == predicted:
                total_correct += 1
            confusion_matrix[actual][predicted] += 1
        confusion_matrix[actual] = np.divide(confusion_matrix[actual], len(data))
    overall_accuracy = total_correct / ((len(testing_data[0]["male"]) + len(testing_data[0]["female"])) * 10)
    return overall_accuracy, confusion_matrix


def determine_optimal_coeffs():
    pass
