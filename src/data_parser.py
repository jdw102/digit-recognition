import numpy as np


# Returns a general form of the data, separated by digit, then gender, and then block
def parse(name, num_blocks):
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
        if gender not in data [digit]:
            data[digit][gender] = []
        data[digit][gender].append([[float(entry) for entry in line.split(" ")] for line in block.split("\n")])
    return data


# Extracts all tokens into a single list
def extract_tokens(data):
    mfccs = [[] for _ in range(10)]
    for digit in data:
        if digit not in mfccs:
            mfccs[digit] = []
        for block in data[digit]["male"]:
            mfccs[digit].extend(block)
        for block in data[digit]["female"]:
            mfccs[digit].extend(block)
    return mfccs


# Separates a list of tokens into a 2D tuple of MFCCs separated by their index
def unzip_tokens(tokens):
    return tuple(zip(*tokens))


def extract_coeffs(tokens, indices):
    tokens = np.array(tokens)
    return tokens[:, indices]


training_data = parse("./data/resources/Train_Arabic_Digit.txt", 660)
test_data = parse("./data/resources/Test_Arabic_Digit.txt", 220)
training_tokens = unzip_tokens(training_data)
test_tokens = unzip_tokens(test_data)

# Number of phonemes in each digit and the indices of the MFCCs
phoneme_nums = [4, 4, 4, 3, 3, 4, 4, 4, 6, 3]
mfcc_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
