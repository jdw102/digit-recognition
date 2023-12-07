import numpy as np
from src.analysis import generate_log_likelihood_analysis, generate_mfccs_analysis, generate_confusion_matrix, \
    generate_data_clustering_analysis, determine_optimal_coeffs
from src.data_parser import load_data
from src.gmm import Method
from src.cov_util import Cov

# Number of phonemes in each digit and the indices of the MFCCs
# highest accuracy:
phoneme_nums = [4, 4, 5, 4, 3, 4, 4, 4, 6, 4]
mfcc_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
sampling_rate = 1

# phoneme_nums = [4, 4, 5, 4, 4, 4, 4, 5, 6, 4]
# mfcc_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]


if __name__ == "__main__":
    print("Running analysis.. this may take a while.")
    training_data = load_data("./data/resources/Train_Arabic_Digit.txt", 660)
    test_data = load_data("./data/resources/Test_Arabic_Digit.txt", 220)
    # Generate plots for analysis frame and sub-dimension of tokens
    # generate_mfccs_analysis(0, 3, "male", training_data)

    # Generate plot for log likelihood of one GMM for all digits
    # generate_log_likelihood_analysis(9, training_data, training_data, Method.K_MEANS, Cov.FULL)

    # Generate plots for clustering analysis
    # generate_data_clustering_analysis(training_data, Method.K_MEANS, Cov.FULL)
    # generate_data_clustering_analysis(training_data, Method.E_M, Cov.FULL)

    # Generate confusion matrix for accuracy analysis
    generate_confusion_matrix(training_data, test_data, phoneme_nums, Method.E_M, Cov.FULL, sampling_rate, mfcc_indices,
                              "test")
    # determine_optimal_coeffs(training_data, test_data, phoneme_nums, Method.E_M, Cov.FULL, 5, 1)
    # generate_confusion_matrix(training_tokens, training_data, Method.E_M, Cov.FULL, "Training")
    # print(determine_optimal_coeffs(training_data, 6))
    print("Done!")
