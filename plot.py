import matplotlib.pyplot as plt
from data_parser import parse
import numpy as np
from data_parser import separate_mfccs

def plot_analysis_window_function(utterance, num_coeffs, digit):
    utterance = np.array(utterance)
    values = utterance[:, 0:num_coeffs].T
    for i in range(num_coeffs):
        plt.plot(np.arange(len(values[i])), values[i], label="MFCC" + str(i + 1))
    plt.xlabel("Analysis Frame")
    plt.ylabel("MFCC tokens")
    plt.legend()
    plt.title("Single Utterance of " + str(digit))
    plt.tight_layout()
    plt.show()


def plot_mfccs_subplots(mfccs, plotting_pairs, ax, s=1, alpha=1):
    # fig, ax = plt.subplots(1, len(plotting_pairs), figsize=(12, 6))
    for i, pair in enumerate(plotting_pairs):
        x, y = pair
        ax[i].scatter(mfccs[x], mfccs[y], s=s, alpha=alpha)
        ax[i].set_xlabel("MFCC" + str(x + 1))
        ax[i].set_ylabel("MFCC" + str(y + 1))
    # fig.suptitle("Single Utterance of " + str(digit))
    # plt.tight_layout()
    # plt.show()



if __name__ == "__main__":
    data = parse("Train_Arabic_Digit.txt", 660)
    plot_analysis_window_function(data[0]["male"][0], 3, 0)
    plot_mfccs_subplots(separate_mfccs(data[0]["male"][0]), [[0, 1], [0, 2], [1, 2]], 0)
