import matplotlib.pyplot as plt



def parse(name, num_blocks):
    data = {}
    # Initialize an empty list to store the separated blocks of text
    text_blocks = []

    # Initialize an empty string to accumulate lines between blank lines
    current_block = ""

    # Open the text file for reading
    with open(name, "r") as file:
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespace
            if line:
                current_block += line + "\n"  # Add the line to the current block
            else:
                # Encountered a blank line, add the current block to the list
                if current_block:
                    text_blocks.append(current_block.strip())  # Remove trailing newline
                    current_block = ""  # Reset the current block

    # Add the last block if it's not empty
    if current_block:
        text_blocks.append(current_block.strip())
    for index, block in enumerate(text_blocks):
        digit = index // num_blocks
        gender = "male" if (index % num_blocks) < num_blocks // 2 else "female"
        if digit not in data:
            data[digit] = {}
        if gender not in data [digit]:
            data[digit][gender] = []
        data[digit][gender].append([line.split(" ") for line in block.split("\n")])

    return data


data = parse("Train_Arabic_Digit.txt", 660)

mfccs = {}


for digit in data.keys():
    if digit not in mfccs:
        mfccs[digit] = {}
    for index, line in enumerate(data[digit]['male'][0]):
        for i in range(3):
            if i not in mfccs[digit]:
                mfccs[digit][i] = {"values": [], "indices": []}
            mfccs[digit][i]["values"].append(float(line[i]))
            mfccs[digit][i]["indices"].append(index)


def plot_first_mfccs(digit):
    plt.plot(mfccs[digit][0]["indices"], mfccs[digit][0]["values"], label="MFCC1")
    plt.plot(mfccs[digit][1]["indices"], mfccs[digit][1]["values"], label="MFCC2")
    plt.plot(mfccs[digit][2]["indices"], mfccs[digit][2]["values"], label="MFCC3")
    plt.xlabel("Analysis Frame")
    plt.ylabel("MFCC tokens")
    plt.legend()
    plt.title("Single Utterance of " + str(digit))
    plt.tight_layout()
    plt.show()

def plot_mfccs_subplots(digit):
    plt.scatter(mfccs[digit][0]["values"], mfccs[digit][1]["values"], label="x = MFCC1 y = MFCC2")
    plt.scatter(mfccs[digit][0]["values"], mfccs[digit][2]["values"], label="x = MFCC1 y = MFCC3")
    plt.scatter(mfccs[digit][1]["values"], mfccs[digit][2]["values"], label="x = MFCC2 y = MFCC3")
    plt.title("Single Utterance of " + str(digit))
    plt.legend()
    plt.tight_layout()
    plt.show()

for i in range(10):
    plot_mfccs_subplots(i)