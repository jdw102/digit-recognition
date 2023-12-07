from src.analysis import generate_log_likelihood_analysis, generate_mfccs_analysis, generate_confusion_matrix, generate_data_clustering_analysis
from src.data_parser import training_data, training_tokens, test_data
from src.gmm import Method
from src.cov_util import Cov

if __name__ == "__main__":
    print("Running analysis.. this may take a while.")
    # Generate plots for analysis frame and sub-dimension of tokens
    # generate_mfccs_analysis(0, 3, "male", training_data)

    # Generate plot for log likelihood of one GMM for all digits
    # generate_log_likelihood_analysis(9, training_data, training_tokens, Method.K_MEANS, Cov.FULL)

    # Generate plots for clustering analysis
    # generate_data_clustering_analysis(training_tokens, Method.K_MEANS, Cov.FULL)
    # generate_data_clustering_analysis(training_tokens, Method.E_M, Cov.FULL)

    # Generate confusion matrix for accuracy analysis
    generate_confusion_matrix(training_tokens, test_data, Method.K_MEANS, Cov.FULL)
    print("Done!")
