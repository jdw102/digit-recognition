from enum import Enum
from src.data_clustering import perform_k_means, perform_em
from scipy.stats import multivariate_normal
import numpy as np
from src.cov_util import Cov


class Method(Enum):
    K_MEANS = "K-Means"
    E_M = "EM"


def create_gmm(tokens, n_clusters, random_state, method=Method.K_MEANS, cov_type=Cov.FULL):
    if method == Method.E_M:
        return perform_em(tokens, n_clusters, random_state, covariance_type=cov_type)
    return perform_k_means(tokens, n_clusters, random_state, covariance_type=cov_type)


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


def likelihood_all_digits(data, tokens, n_clusters, method=Method.K_MEANS, cov_type=Cov.FULL):
    rand = np.random.RandomState(42)
    gmm = create_gmm(tokens, n_clusters, rand, method, cov_type)
    likelihoods = []
    for num in range(10):
        likelihoods.append(digit_likelihood(num, gmm, data))
    return likelihoods


def generate_model(tokens, cluster_nums, method=Method.K_MEANS, cov_type=Cov.FULL):
    rand = np.random.RandomState(42)
    return [create_gmm(tokens[i], cluster_nums[i], rand, method, cov_type) for i in range(10)]


def determine_category(utterance, model):
    likelihoods = [gmm_likelihood(model[i], utterance) for i in range(10)]
    return np.argmax(likelihoods)
