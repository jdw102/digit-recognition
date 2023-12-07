import numpy as np


# Returns a general form of the data, separated by digit, then gender, and then block
def load_data(name, num_blocks, frame_sampling_rate=1, features=None):
    if features is None:
        features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
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
        entries = [[float(entry) for entry in np.array(line.split(" "))[features]] for i, line in
                   enumerate(block.split("\n"))
                   if (i + 1) % frame_sampling_rate == 0]
        data[digit][gender].append(entries)
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


# Separates a list of tokens into a 2D tuple of MFCCs separated by their index
def unzip_frames(token):
    return tuple(zip(*token))


def extract_coeffs(tokens, indices):
    tokens = np.array(tokens)
    return tokens[:, indices]



