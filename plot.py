import matplotlib.pyplot as plt




def plot_first_mfccs(digit, mfccs):
    plt.plot(mfccs[digit][0]["indices"], mfccs[digit][0]["values"], label="MFCC1")
    plt.plot(mfccs[digit][1]["indices"], mfccs[digit][1]["values"], label="MFCC2")
    plt.plot(mfccs[digit][2]["indices"], mfccs[digit][2]["values"], label="MFCC3")
    plt.xlabel("Analysis Frame")
    plt.ylabel("MFCC tokens")
    plt.legend()
    plt.title("Single Utterance of " + str(digit))
    plt.tight_layout()
    plt.show()

def plot_mfccs_subplots(digit, mfccs):
    plt.scatter(mfccs[digit][0]["values"], mfccs[digit][1]["values"], label="x = MFCC1 y = MFCC2")
    plt.scatter(mfccs[digit][0]["values"], mfccs[digit][2]["values"], label="x = MFCC1 y = MFCC3")
    plt.scatter(mfccs[digit][1]["values"], mfccs[digit][2]["values"], label="x = MFCC2 y = MFCC3")
    plt.title("Single Utterance of " + str(digit))
    plt.legend()
    plt.tight_layout()
    plt.show()


def get_mfccs(data):
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
    return mfccs


if __name__ == "__main__":
    pass
