from src.data_clustering import perform_k_means, perform_em
from src.data_parser import phoneme_nums, extract_mfccs
from scipy.stats import multivariate_normal
import numpy as np


def create_gmm(digit, tokens, method="k-means", cov_type="full", tied=False):
    if method == "em":
        return perform_em(tokens, phoneme_nums[digit], covariance_type=cov_type, tied=tied)
    return perform_k_means(tokens, phoneme_nums[digit], covariance_type=cov_type, tied=tied)


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
    gmm = create_gmm(digit, tokens[digit], "k-means", "full", False)
    likelihoods = []
    for num in range(10):
        likelihoods.append(digit_likelihood(num, gmm, d))
    return likelihoods


def generate_model(tokens, method="k-means", cov_type="full", tied=False):
    return [create_gmm(i, tokens[i], method, cov_type, tied) for i in range(10)]


def determine_category(utterance, model):
    likelihoods = [gmm_likelihood(model[i], utterance) for i in range(10)]
    return np.argmax(likelihoods)
