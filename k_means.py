import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from data_parser import parse
from sklearn.cluster import KMeans

def extract_mfccs(data):
    mfccs = {}
    for digit in data:
        if digit not in mfccs:
            mfccs[digit] = []
        for block in data[digit]["male"]:
            for row in block:
                mfccs[digit].append([float(token) for token in row])
        for block in data[digit]["female"]:
            for row in block:
                mfccs[digit].append([float(token) for token in row])
    return mfccs

def create_pdf(mean, cov):
    x_mesh, y_mesh = np.meshgrid(
        np.linspace(mean[0] - 5, mean[0] + 5, 100),
        np.linspace(mean[1] - 5, mean[1] + 5, 100))
    mvn = multivariate_normal(mean=mean, cov=cov)
    pdf_values = mvn.pdf(np.dstack((x_mesh, y_mesh)))
    return x_mesh, y_mesh, pdf_values

if __name__ == "__main__":
    data = parse("Train_Arabic_Digit.txt", 660)
    mfccs = extract_mfccs(data)
    phoneme_nums = [4, 4, 4, 3, 3, 4, 4, 4, 6, 4]
    for digit in mfccs:
        k_means = KMeans(n_clusters=phoneme_nums[digit], init='k-means++').fit(mfccs[digit])
        centers = {}
        for i, label in enumerate(k_means.labels_):
            if label not in centers:
                centers[label] = []
            centers[label].append(mfccs[digit][i][0:3])
        fig, ax = plt.subplots(1, 3)
        for center in centers:
            mfcc1, mfcc2, mfcc3 = zip(*centers[center])
            mean = k_means.cluster_centers_[center][0:3]
            cov12 = np.cov(mfcc1, mfcc2)
            cov13 = np.cov(mfcc1, mfcc3)
            cov23 = np.cov(mfcc2, mfcc3)
            contour1 = create_pdf([mean[0], mean[1]], cov12)
            contour2 = create_pdf([mean[0], mean[2]], cov13)
            contour3 = create_pdf([mean[1], mean[2]], cov23)
            ax[0].contour(contour1[0], contour1[1], contour1[2], alpha=0.5)
            ax[1].contour(contour2[0], contour2[1], contour2[2], alpha=0.5)
            ax[2].contour(contour3[0], contour3[1], contour3[2], alpha=0.5)
            ax[0].scatter(mfcc1, mfcc2, s=0.5, alpha=0.7)
            ax[1].scatter(mfcc1, mfcc3, s=0.5, alpha=0.7)
            ax[2].scatter(mfcc2, mfcc3, s=0.5, alpha=0.7)
        plt.tight_layout()
        plt.show()