from enum import Enum

from src.data_clustering import perform_k_means, perform_em
from src.data_parser import phoneme_nums
from scipy.stats import multivariate_normal
import numpy as np
from src.cov_util import Cov


class Method(Enum):
    K_MEANS = "K-Means"
    E_M = "EM"


def create_gmm(digit, tokens, method=Method.K_MEANS, cov_type=Cov.FULL):
    if method == Method.E_M:
        return perform_em(tokens, phoneme_nums[digit], covariance_type=cov_type)
    return perform_k_means(tokens, phoneme_nums[digit], covariance_type=cov_type)


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


def likelihood_all_digits(digit, data, tokens, method=Method.K_MEANS, cov_type=Cov.FULL):
    gmm = create_gmm(digit, tokens[digit], method, cov_type)
    likelihoods = []
    for num in range(10):
        likelihoods.append(digit_likelihood(num, gmm, data))
    return likelihoods


def generate_model(tokens, method=Method.K_MEANS, cov_type=Cov.FULL):
    return [create_gmm(i, tokens[i], method, cov_type) for i in range(10)]


def determine_category(utterance, model):
    likelihoods = [gmm_likelihood(model[i], utterance) for i in range(10)]
    return np.argmax(likelihoods)
