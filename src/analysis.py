from itertools import combinations
import numpy as np
from src.cov_util import Cov
from src.data_parser import filter_data, extract_tokens
from src.data_clustering import perform_k_means, perform_em
from src.gmm import generate_model, determine_category, likelihood_all_digits, Method
from src.plot_util import plot_clusters, plot_mfccs_subplots_single_utterance, \
    plot_analysis_window_function_single_utterance, plot_kde, plot_confusion_matrix, plot_feature_accuracy, \
    plot_kde_likelihood, plot_covariance_bar_graph


def generate_mfccs_analysis(block_num, num_coeffs, gender, data):
    plotting_pairs = list(combinations(list(range(num_coeffs)), 2))
    for digit in data:
        utterance = data[digit][gender][block_num]
        plot_mfccs_subplots_single_utterance(utterance, plotting_pairs, digit, gender, block_num)
        plot_analysis_window_function_single_utterance(utterance, num_coeffs, digit, gender, block_num)


def generate_log_likelihood_analysis(digit, test_data, training_data, phoneme_clusters, method, cov_type):
    tokens = extract_tokens(test_data)
    log_likelihoods = likelihood_all_digits(training_data, tokens[digit], phoneme_clusters[digit], method, cov_type)
    title = f"Digit {digit} {method.value.title()} {cov_type.value.title()} Cov"
    filename = f"{method.value}-{cov_type.value}-{digit}"
    plot_kde_likelihood(log_likelihoods, title, filename)


def generate_data_clustering_analysis(data, phoneme_clusters, method, cov_type):
    tokens = extract_tokens(data)
    for digit, utterances in enumerate(tokens):
        if method == Method.K_MEANS:
            clusters = perform_k_means(utterances, phoneme_clusters[digit], cov_type)
        else:
            clusters = perform_em(utterances, phoneme_clusters[digit], cov_type)
        title = f"{method.value} Phoneme Clusters on MFCCs: Digit {digit}"
        filename = f"{method.value}-{cov_type.value}-{digit}"
        plot_clusters(clusters, title, filename)


def generate_confusion_matrix(training_data, testing_data, phoneme_clusters, method, cov_type, subsampling_rate,
                              features, tag):
    table_hash = np.prod(np.multiply(phoneme_clusters, np.arange(1, 11)))
    filename = f"cm-{tag}-{method.value}-{cov_type.value}-{len(features)}coeffs-{subsampling_rate}rate-table{table_hash}"
    filtered_training_data = filter_data(training_data, subsampling_rate, features)
    filtered_test_data = filter_data(testing_data, subsampling_rate, features)
    overall_accuracy, confusion_matrix = calculate_accuracy(filtered_training_data, filtered_test_data,
                                                            phoneme_clusters, method,
                                                            cov_type)
    title = f"{method.value} | {cov_type.value.title()} Cov | {len(features)} Features | {subsampling_rate}x Sampling | {round(overall_accuracy * 100, 2)}% Accuracy"
    plot_confusion_matrix(confusion_matrix, phoneme_clusters, title, filename)


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


# def dummy_accuracy_calculation(training_data, testing_data, phoneme_clusters, method, cov_type):
#     return np.random.uniform(0, 1), None


def determine_optimal_coeffs(training_data, testing_data, phoneme_clusters, method, cov_type, k, total_features,
                             subsampling_rate=1):
    results = []
    ret = []
    remaining = total_features.copy()
    while len(ret) < k:
        accuracies = []
        max_accuracy = -1
        best_feature = -1
        for feature in remaining:
            curr_features = ret + [feature]
            curr_testing_data = filter_data(testing_data, subsampling_rate, curr_features)
            curr_training_data = filter_data(training_data, subsampling_rate, curr_features)
            curr_accuracy, _ = calculate_accuracy(curr_training_data, curr_testing_data, phoneme_clusters,
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


def run_all_cov(training_data, testing_data, phoneme_clusters, method):
    all_cov = [Cov.TIED_SPHERICAL, Cov.SPHERICAL, Cov.TIED_DIAG, Cov.DIAG, Cov.TIED_FULL, Cov.FULL]
    ret = {}
    for cov in all_cov:
        accuracy, _ = calculate_accuracy(training_data, testing_data, phoneme_clusters, method, cov)
        ret[cov.value.title()] = accuracy
    return ret


def generate_covariance_analysis(training_data, testing_data, phoneme_clusters, subsampling_rate,
                                 features):
    filtered_training_data = filter_data(training_data, subsampling_rate, features)
    filtered_test_data = filter_data(testing_data, subsampling_rate, features)
    em_cov = run_all_cov(filtered_training_data, filtered_test_data, phoneme_clusters, Method.E_M)
    km_cov = run_all_cov(filtered_training_data, filtered_test_data, phoneme_clusters, Method.K_MEANS)
    title = f"{len(features)} Features | {subsampling_rate}x Sampling"
    filename = f"cov-{len(features)}coeffs-{subsampling_rate}rate"
    plot_covariance_bar_graph(em_cov, km_cov, title, filename)



