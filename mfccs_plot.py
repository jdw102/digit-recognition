import matplotlib.pyplot as plt
import numpy as np
from data_parser import separate_mfccs, training_data

plotting_colors = ['red', 'green', 'blue', 'orange', 'yellow', 'brown', 'black']

def plot_analysis_window_function_single_utterance(utterance, num_coeffs, digit, gender, num):
    utterance = np.array(utterance)
    values = utterance[:, 0:num_coeffs].T

    for i in range(num_coeffs):
        plt.plot(np.arange(len(values[i])), values[i], label="MFCC" + str(i + 1))
    plt.xlabel("Analysis Frame")
    plt.ylabel("MFCC tokens")
    plt.legend()
    plt.title("Single Utterance of " + str(digit) + ", " + gender + " num " + str(num))
    plt.tight_layout()
    plt.savefig(
        "single_utterance_analysis_frame_plots/" + gender + "_num" + str(num) + "_digit" + str(digit) + "_aframe.png")
    plt.close()


def mfccs_subplots(mfccs, plotting_pairs, ax, s=10, alpha=1, color='blue', label=None):
    for i, pair in enumerate(plotting_pairs):
        x, y = pair
        ax[i].scatter(mfccs[x], mfccs[y], s=s, alpha=alpha, color=color, label=label)
        ax[i].set_xlabel("MFCC" + str(x + 1))
        ax[i].set_ylabel("MFCC" + str(y + 1))


def plot_mfccs_subplots_single_utterance(utterance, plotting_pairs, digit, gender, num):
    fig, ax = plt.subplots(1, len(plotting_pairs), figsize=(12, 6))
    mfccs_subplots(separate_mfccs(utterance), plotting_pairs, ax)
    fig.suptitle("Single Utterance of " + str(digit) + ", " + gender + " num " + str(num))
    plt.tight_layout()
    fig.savefig("single_utterance_mfccs_plots/" + gender +"_num" + str(num) + "_digit"+str(digit)+ "_mfccs.png")
    plt.close(fig)

if __name__ == "__main__":
    block_num = 0
    num_coeffs = 3
    gender = "male"
    for digit in training_data:
        utterance = training_data[digit][gender][block_num]
        plot_mfccs_subplots_single_utterance(utterance, [[0, 1], [0, 2], [1, 2]], digit, gender, block_num)
        plot_analysis_window_function_single_utterance(utterance, num_coeffs, digit, gender, block_num)
