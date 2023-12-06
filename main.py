from src.analysis import generate_log_likelihood_analysis, generate_mfccs_analysis, generate_confusion_matrix, generate_data_clustering_analysis
from src.data_parser import training_data, training_tokens,test_data

if __name__ == "__main__":
    print("Hello world!")
    # Generate plots for analysis frame and sub-dimension of tokens
    # generate_mfccs_analysis(0, 3, "male", training_data)

    # Generate plot for log likelihood of one GMM for all digits
    generate_log_likelihood_analysis(9, training_data)

    # Generate plots for clustering analysis
    # generate_data_clustering_analysis(training_tokens, "k-means")
    # generate_data_clustering_analysis(training_tokens, "em")

    # Generate confusion matrix for accuracy analysis
    # generate_confusion_matrix(training_tokens, test_data, "em", "full", True)

