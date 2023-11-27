from gmm import create_gmm, gmm_likelihood
from data_parser import test_data, training_tokens
import numpy as np

def generate_model(tokens, type="k-means"):
    return [create_gmm(i, tokens[i], type=type) for i in range(10)]


def determine_category(utterance, model):
    likelihoods = [gmm_likelihood(model[i], utterance) for i in range(10)]
    return np.argmax(likelihoods)


if __name__ == "__main__":
    model = generate_model(training_tokens, "em")
    ret = np.zeros((10, 10))
    for actual in range(10):
        data = test_data[actual]["male"] + test_data[actual]["female"]
        for utterance in data:
            predicted = determine_category(utterance, model)
            ret[actual][predicted] += 1
    print("DONE")

