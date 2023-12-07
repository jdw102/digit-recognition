from itertools import combinations

import numpy as np
from matplotlib import pyplot as plt
from src.data_parser import filter_data, extract_tokens
from src.data_clustering import perform_k_means, perform_em
from src.gmm import generate_model, determine_category, likelihood_all_digits, Method
from src.plot_util import plot_clusters, plot_mfccs_subplots_single_utterance, \
    plot_analysis_window_function_single_utterance, plot_kde, plot_confusion_matrix, plot_feature_accuracy


def generate_mfccs_analysis(block_num, num_coeffs, gender, data):
    plotting_pairs = list(combinations(list(range(num_coeffs)), 2))
    for digit in data:
        utterance = data[digit][gender][block_num]
        plot_mfccs_subplots_single_utterance(utterance, plotting_pairs, digit, gender, block_num)
        plot_analysis_window_function_single_utterance(utterance, num_coeffs, digit, gender, block_num)


def generate_log_likelihood_analysis(digit, test_data, training_data, method, cov_type):
    tokens = extract_tokens(test_data)
    log_likelihoods = likelihood_all_digits(digit, training_data, tokens, method, cov_type)
    fig, ax = plt.subplots(10, sharex=True, sharey=True, figsize=(6, 12))
    for i in range(10):
        plot_kde(np.array(log_likelihoods[i]), ax[i], i)
    fig.text(0.5, 0.02, 'Log Likelihood', ha='center', va='center', fontsize=12)
    fig.text(0.06, 0.5, 'Probability Density', ha='center', va='center', rotation='vertical', fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.show()


def generate_data_clustering_analysis(data, phoneme_clusters, method, cov_type):
    tokens = extract_tokens(data)
    for digit, utterances in enumerate(tokens):
        if method == Method.K_MEANS:
            clusters = perform_k_means(utterances, phoneme_clusters[digit], cov_type)
        else:
            clusters = perform_em(utterances, phoneme_clusters[digit], cov_type)
        plot_clusters(clusters, digit)


def generate_confusion_matrix(training_data, testing_data, phoneme_clusters, method, cov_type, subsampling_rate,
                              features, tag):
    table_hash = np.prod(np.multiply(phoneme_clusters, np.arange(1, 11)))
    filename = f"cm-{tag}-{method.value}-{cov_type.value}-{len(features)}coeffs-{subsampling_rate}rate-table{table_hash}"
    title = f"{Method.E_M.value} | {cov_type.value.title()} Cov | {len(features)} Features | {subsampling_rate}x Subsampling"
    filtered_training_data = filter_data(training_data, subsampling_rate, features)
    filtered_test_data = filter_data(testing_data, subsampling_rate, features)
    overall_accuracy, confusion_matrix = calculate_accuracy(filtered_training_data, filtered_test_data,
                                                            phoneme_clusters, method,
                                                            cov_type)
    plot_confusion_matrix(confusion_matrix, overall_accuracy, phoneme_clusters, title, filename)


def calculate_accuracy(training_data, testing_data, phoneme_clusters, method, cov_type):
    tokens = extract_tokens(training_data)
    model = generate_model(tokens, phoneme_clusters, method, cov_type)
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


def dummy_accuracy_calculation(training_data, testing_data, phoneme_clusters, method, cov_type):
    return np.random.uniform(0, 1), None


def determine_optimal_coeffs(training_data, testing_data, phoneme_clusters, method, cov_type, k, subsampling_rate=1):
    results = []
    ret = []
    remaining = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    while len(ret) < k:
        accuracies = []
        max_accuracy = -1
        best_feature = -1
        for feature in remaining:
            curr_features = ret + [feature]
            curr_testing_data = filter_data(testing_data, subsampling_rate, curr_features)
            curr_training_data = filter_data(training_data, subsampling_rate, curr_features)
            curr_accuracy, _ = dummy_accuracy_calculation(curr_training_data, curr_testing_data, phoneme_clusters,
                                                          method,
                                                          cov_type)
            accuracies.append(curr_accuracy)
            if curr_accuracy > max_accuracy:
                best_feature = feature
                max_accuracy = curr_accuracy
        results.append(max_accuracy)
        remaining.remove(best_feature)
        ret.append(best_feature)
    plot_feature_accuracy(results, ret)
    return ret
