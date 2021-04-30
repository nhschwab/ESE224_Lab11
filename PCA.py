# This script contains two classes, one that computes the PCA Transform of the training set
# and one that computes the PCA Transform of the test set

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg

# Exercise 5


class PCA():

    def __init__(self, train, test, k):

        # training set must be a numpy array with size (10304, 360)
        if not isinstance(train, np.ndarray) or train.shape != (10304, 360):
            raise ValueError("Training set must be a numpy array with size (10304, 360)")

        # test set must be a numpy array with size (10304, 40)
        if not isinstance(test, np.ndarray) or test.shape != (10304, 40):
            raise ValueError("Test set must be a numpy array with size (10304, 40)")

        # k must be a positive integer
        if not isinstance(k, int) or k < 0:
            raise ValueError("k must be a positive integer")

        self.train = train
        self.test = test
        self.k = k
        self.M = train.shape[1]         # 360 columns
        self.N = test.shape[1]         # 40 columns
        self.mu = self.mean()
        self.P = self.unitary()

    # method to compute the mean of the training set
    def mean(self):
        return self.train.mean(axis=1)

    # method to compute covariance matrix
    def covariance(self):
        return np.cov(self.train)

    # method to compute unitary matrix from covariance matrix
    def unitary(self):

        # compute the eigenvalues and eigenvectors using SciPy method
        w, v = scipy.sparse.linalg.eigs(self.covariance(), k=self.k)

        # return the first k eigenvectors
        return v

    # method to compute the transformed training set by applying the PCA transform
    # to each column
    def training_transform(self):

        # initialize empty transform matrix
        X = np.zeros(self.train.shape)

        # iterate through each column and populate the variance matrix
        for i in range(self.M):
            X[:, i] = self.train[:, i] - self.mu

        # compute the PCA transform of the training set
        P_H = np.conjugate(np.transpose(self.P))
        D = np.matmul(P_H, X)

        return D

    # method to compute the transformed test set by applying the PCA transform
    # to each column
    def test_transform(self):

        # initialize empty transform matrix
        X = np.zeros(self.test.shape)

        # iterate through each column and populate the variance matrix
        for i in range(self.N):
            X[:, i] = self.test[:, i] - self.mu

        # compute the PCA transform of test set
        P_H = np.conjugate(np.transpose(self.P))
        D = np.matmul(P_H, X)

        return D


if __name__ == "__main__":

    # Exercise 5.1
    k_list = [1, 5, 10, 20]

    # read in training set and test set
    training_set = np.load("training_set.npy")
    test_set = np.load("test_set.npy")

    # iterate through each value of k and print the unitary and transform matrices
    for k in k_list:
        object = PCA(training_set, test_set, k)
        print("-------------------------")
        print(f"k = {k}")
        print("-------------------------")
        print("Unitary Matrix:")
        print(object.unitary().shape)
        print("-------------------------")
        print("PCA Transform of Training Set")
        print(object.training_transform().shape)
        print("-------------------------")
        print("PCA Transform of Test Set")
        print(object.test_transform().shape)










