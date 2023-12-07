import numpy as np
import copy


# Returns a general form of the data, separated by digit, then gender, and then block
def load_data(name, num_blocks):
    data = {}
    text_blocks = []
    current_block = ""
    with open(name, "r") as file:
        for line in file:
            line = line.strip()
            if line:
                current_block += line + "\n"
            else:
                if current_block:
                    text_blocks.append(current_block.strip())
                    current_block = ""
    if current_block:
        text_blocks.append(current_block.strip())
    for index, block in enumerate(text_blocks):
        digit = index // num_blocks
        gender = "male" if (index % num_blocks) < num_blocks // 2 else "female"
        if digit not in data:
            data[digit] = {}
        if gender not in data[digit]:
            data[digit][gender] = []
        entries = [[float(entry) for entry in np.array(line.split(" "))] for line in block.split("\n")]
        data[digit][gender].append(np.array(entries))
    return data


# Extracts all tokens into a single list
def extract_tokens(data):
    mfccs = [[] for _ in range(10)]
    for digit in data:
        for block in data[digit]["male"]:
            mfccs[digit].extend(block)
        for block in data[digit]["female"]:
            mfccs[digit].extend(block)
    return mfccs


def filter_data(data, frame_sampling_rate=1, features=None):
    ret = copy.deepcopy(data)
    if features is None:
        features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    for digit in data:
        for i, utterance in enumerate(ret[digit]["male"]):
            ret[digit]["male"][i] = utterance[::frame_sampling_rate, features]
        for i, utterance in enumerate(data[digit]["female"]):
            ret[digit]["female"][i] = utterance[::frame_sampling_rate, features]
    return ret


# Separates a list of tokens into a 2D tuple of MFCCs separated by their index
def unzip_frames(token):
    return tuple(zip(*token))

