import numpy as np
from src.analysis import generate_log_likelihood_analysis, generate_mfccs_analysis, generate_confusion_matrix, \
    generate_data_clustering_analysis, determine_optimal_coeffs, generate_covariance_analysis, \
    generate_cluster_number_analysis
from src.data_parser import load_data
from src.gmm import Method
from src.cov_util import Cov

# Number of phonemes in each digit and the indices of the MFCCs
# initial guess
phoneme_nums = [4, 4, 5, 4, 3, 4, 4, 4, 6, 4]
mfcc_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

sampling_rate = 1
method = Method.E_M
cov = Cov.FULL
test_data_type = "test"
opt_features_num = 13

# highest accuracy:
# phoneme_nums = [4, 4, 5, 4, 3, 4, 4, 4, 6, 4]
# phoneme_nums = [4, 4, 5, 4, 3, 4, 4, 4, 6, 5]
# mfcc_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# mfcc_indices = [4, 2, 1, 7, 9, 10, 6, 8, 11]
# sampling_rate = 1
# cov = Cov.FULL
# method = Method.E_M
# test_data_type = "test"
# opt_features_num = 13


if __name__ == "__main__":
    print("Running analysis.. this may take a while.")
    if test_data_type == "train":
        test_data = load_data("./data/resources/Train_Arabic_Digit.txt", 220)
    else:
        test_data = load_data("./data/resources/Test_Arabic_Digit.txt", 220)
    training_data = load_data("./data/resources/Train_Arabic_Digit.txt", 660)
    # Generate plots for analysis frame and sub-dimension of tokens
    # generate_mfccs_analysis(0, 3, "male", training_data)

    # Generate plot for log likelihood of one GMM for all digits for each digit
    # for i in range(10):
    #     generate_log_likelihood_analysis(i, training_data, training_data, phoneme_nums, method, cov)

    # Generate plots for clustering analysis
    # generate_data_clustering_analysis(training_data, phoneme_nums, method, cov)

    # Generate confusion matrix for accuracy analysis
    generate_confusion_matrix(training_data, test_data, phoneme_nums, method, cov, sampling_rate, mfcc_indices,
                              test_data_type)

    # Generate covariance analysis
    # generate_covariance_analysis(training_data, test_data, phoneme_nums, sampling_rate, mfcc_indices)

    # Generate analysis of impact of number of clusters on accuracy
    # generate_cluster_number_analysis(2, training_data, test_data, phoneme_nums, method, cov, sampling_rate,
    #                                  mfcc_indices)

    # Find optimal features
    # determine_optimal_coeffs(training_data, test_data, phoneme_nums, method, cov, opt_features_num, mfcc_indices,
    #                          sampling_rate)
    print("Done!")
